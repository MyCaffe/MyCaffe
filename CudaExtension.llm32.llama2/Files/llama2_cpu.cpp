/*
Inference for Llama-2 Transformer model in pure Cuda.
*/

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "llama2_cpu.h"


// Each CUDA function call should be checked for errors.
#define CUCHK(err) cuda_check((err), __FILE__, __LINE__)
inline void cuda_check(cudaError_t error_code, const char* file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

long malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->k = (float*)calloc(kv_dim, sizeof(float));
    s->v = (float*)calloc(kv_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * (size_t)p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * (size_t)p->seq_len * kv_dim, sizeof(float));
    s->freq_real = (float*)calloc((size_t)p->seq_len * p->n_heads * (p->dim / p->n_heads / 2), sizeof(float));
    s->freq_imag = (float*)calloc((size_t)p->seq_len * p->n_heads * (p->dim / p->n_heads / 2), sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
        || !s->value_cache)
        return ERROR_OUTOFMEMORY;

    precompute_freqs_cis(p->dim / p->n_heads, p->seq_len, s->freq_real, s->freq_imag);

    return 0;
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(s->freq_real);
    free(s->freq_imag);
}

long memory_map_weights(TransformerWeights* w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;

    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;

    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;

    return 0;
}

long read_checkpoint(const char* checkpoint, Config* config, TransformerWeights* weights,
    int* fd, float** data, ssize_t* file_size) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file)
        return ERROR_FILE_INVALID;

    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1)
        return ERROR_FILE_INVALID;

    if (magic_number != 0x616b3432)
        return ERROR_FILE_INVALID;

    // read in the version number (uint32), has to be 1
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1)
        return ERROR_FILE_INVALID;

    if (version != 1)
        return ERROR_FILE_INVALID;

    int header_size = 256; // the header size for version 2 in bytes
    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1)
        return ERROR_FILE_INVALID;
    
    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1)
        return ERROR_FILE_INVALID;

    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    if (config->seq_len > 20480)
        config->seq_len = 20480;

    // memory map the Transformer weights into the data pointer
    *fd = _open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1)
        return ERROR_FILE_INVALID;

    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
        return ERROR_FILE_INVALID;

    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte
    return memory_map_weights(weights, config, (float*)weights_ptr, shared_classifier);
}

long TransformerCpu::build(const char* checkpoint_path)
{
    LONG lErr;
    // read in the Config and the Weights from the checkpoint
    if (lErr = read_checkpoint(checkpoint_path, &m_config, &m_weights, &fd, &data, &file_size))
        return lErr;
    // allocate the RunState buffers
    if (lErr = malloc_run_state(&m_state, &m_config))
        return lErr;

    return 0;
}

void TransformerCpu::cleanup() {
    // close the memory mapping
    if (data != MAP_FAILED)
        munmap(data, file_size);

    if (fd != -1)
        _close(fd);

    // free the RunState buffers
    free_run_state(&m_state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void rope(int num_heads, int head_size, float* sq, float* sk, float* freq_real, float* freq_imag, int nFreqOffset)
{
    freq_real += nFreqOffset;
    freq_imag += nFreqOffset;

    for (int h = 0; h < num_heads; h++)
    {
        float* q = sq + h * head_size;
        float* k = sk + h * head_size;

        for (int i = 0; i < head_size; i += 2)
        {
            float q0 = q[i];
            float q1 = q[i + 1];
            float k0 = k[i];
            float k1 = k[i + 1];
            float fcr = freq_real[i / 2];
            float fci = freq_imag[i / 2];
            q[i] = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;
            k[i] = k0 * fcr - k1 * fci;
            k[i + 1] = k0 * fci + k1 * fcr;
        }
    }
}

float* TransformerCpu::forward(int token, int pos) {

    // a few convenience variables
    Config* p = &m_config;
    TransformerWeights* w = &m_weights;
    RunState* s = &m_state;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    // pluck out the 'pos' row of freq_cis_real and freq_cis_imag
    int nFreqOffset = pos * head_size / 2;

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++)
    {
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        rope(p->n_heads, head_size, s->q, s->k, s->freq_real, s->freq_imag, nFreqOffset);

        // save key,value at this time step (pos) to our kv cache
        unsigned long long loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf((float)head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);

    return s->logits;
}

void precompute_freqs_cis(int dim, size_t seq_len, float* freq_real, float* freq_imag)
{
    float fTheta = 10000.0f;

    float* pfreqs = (float*)calloc(dim / 2, sizeof(float));
    if (pfreqs == NULL)
        throw ERROR_OUTOFMEMORY;

    for (int i = 0; i < dim; i += 2)
    {
        float fVal = i / (float)dim;
        float fDiv = powf(fTheta, fVal);
        float fFreq = (fDiv == 0.0) ? 0 : 1.0f / fDiv;
        pfreqs[i / 2] = fFreq;
    }

    for (size_t i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < dim / 2; j++)
        {
            freq_real[i * dim / 2 + j] = cosf(i * pfreqs[j]);
            freq_imag[i * dim / 2 + j] = sinf(i * pfreqs[j]);
        }
    }

    free(pfreqs);
}

// end