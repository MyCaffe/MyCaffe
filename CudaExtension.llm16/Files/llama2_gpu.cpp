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
#include "llama2_gpu.h"


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

// ----------------------------------------------------------------------------
// neural net blocks

float* TransformerGpu::forward(int token, int pos) 
{
    // a few convenience variables
    Config* p = &m_config;
    TransformerWeightsCuda* w = &m_weights;
    RunStateCuda* s = &m_state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    cudaMemcpy(x, content_row, dim * sizeof(float), cudaMemcpyDeviceToDevice);
    print("x", x, dim);

    // pluck out the 'pos' row of freq_cis_real and freq_cis_imag
    int nFreqOffset = pos * head_size / 2;

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) 
    {
        print("rms.att.wt", w->rms_att_weight + l * dim, dim, l);
        print("x", x, dim, l);

        // attention rmsnorm
        m_primitives.rmsnorm(dim, s->xb, x, w->rms_att_weight + l * dim);
        print("xb", s->xb, dim);

        // qkv matmuls for this position

        print("wq.wt", w->wq + l * dim, dim, l);
        print("wk.wt", w->wk + l * kv_dim, kv_dim, l);
        print("wv.wt", w->wv + l * kv_dim, kv_dim, l);

        m_primitives.matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim, false);
        m_primitives.matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim, false);
        m_primitives.matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        print("q", s->q, dim);
        print("k", s->k, kv_dim);
        print("v", s->v, kv_dim);

        print("freq_real", s->freq_real, kv_dim);
        print("freq_imag", s->freq_imag, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        m_primitives.rope(s->q, s->k, s->freq_real, s->freq_imag, p->n_heads, head_size, nFreqOffset);

        print("q.rope", s->q, dim);
        print("k.rope", s->k, kv_dim);

        // save key,value at this time step (pos) to our kv cache
        unsigned long long loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        cudaMemcpyAsync(key_cache_row, s->k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(value_cache_row, s->v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaStreamSynchronize(0);

        print("key_cache", key_cache_row, kv_dim);
        print("value_cache", value_cache_row, kv_dim);

        // multihead attention. iterate over all heads
        m_primitives.attention(s->xb, s->q, s->key_cache, s->value_cache, p->n_heads, head_size, loff, pos, p->seq_len, kv_dim, kv_mul);
        print("xb.att", s->xb, dim);

        // final matmul to get the output of the attention
        m_primitives.matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        print("xb2", s->xb2, dim);

        // residual connection back into x
        m_primitives.add(dim, x, s->xb2);
        print("x.resid.1", x, dim);

        // ffn rmsnorm
        m_primitives.rmsnorm(dim, s->xb, x, w->rms_ffn_weight + l * dim);
        print("xb.ffn", s->xb, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        m_primitives.matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim, false);
        m_primitives.matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        print("hb", s->hb, hidden_dim);
        print("hb2", s->hb2, hidden_dim);

        // SwiGLU non-linearity
        m_primitives.siglu(hidden_dim, s->hb, s->hb, s->hb2);
        print("hb.swiglu", s->hb, hidden_dim);

        // final matmul to get the output of the ffn
        m_primitives.matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        print("xb.ffn2", s->xb, dim);

        // residual connection
        m_primitives.add(dim, x, s->xb);
        print("x.resid.2", x, dim);
    }

    // final rmsnorm
    m_primitives.rmsnorm(dim, x, x, w->rms_final_weight);
    print("x.final", x, dim);

    // classifier into logits
    m_primitives.matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);
    print("logits", s->logits_gpu, p->vocab_size);

    // copy logits from GPU->CPU
    m_primitives.ToHost(p->vocab_size, s->logits, s->logits_gpu);
    printHost("logits_cpu", s->logits, p->vocab_size);

    return s->logits;
}

void TransformerGpu::print(const char* name, void* x1, size_t n, long long nLayer)
{
    if (!m_bDebug)
        return;

    m_primitives.Print(name, (float*)x1, n, nLayer);
}

void cuda_malloc_run_state(RunStateCuda* s, Config* p)
{
    size_t seq_len = p->seq_len;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CUCHK(cudaMalloc((void**)&s->x, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->xb, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->xb2, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->q, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->k, kv_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->v, kv_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->key_cache, p->n_layers * seq_len * kv_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->value_cache, p->n_layers * seq_len * kv_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->att, p->n_heads * seq_len * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(float)));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v
        || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    int head_size = p->dim / p->n_heads;
    size_t nCount = seq_len * (head_size / 2);
    CUCHK(cudaMalloc((void**)&s->freq_real, nCount * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->freq_imag, nCount * sizeof(float)));

    // ensure all mallocs went fine
    if (!s->freq_real || !s->freq_imag) 
    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    float* freq_real = (float*)calloc(nCount, sizeof(float));
    float* freq_imag = (float*)calloc(nCount, sizeof(float));

    if (!freq_real || !freq_imag)
    {
		printf("malloc failed!\n");
		exit(1);
	}

    precompute_freqs_cis(head_size, seq_len, freq_real, freq_imag);
    
    CUCHK(cudaMemcpyAsync(s->freq_real, freq_real, nCount * sizeof(float), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpyAsync(s->freq_imag, freq_imag, nCount * sizeof(float), cudaMemcpyHostToDevice));
	cudaStreamSynchronize(0);

    free(freq_real);
    free(freq_imag);
}

void cuda_free_run_state(RunStateCuda* s) {
    CUCHK(cudaFree(s->x));
    CUCHK(cudaFree(s->xb));
    CUCHK(cudaFree(s->xb2));
    CUCHK(cudaFree(s->hb));
    CUCHK(cudaFree(s->hb2));
    CUCHK(cudaFree(s->q));
    CUCHK(cudaFree(s->att));
    CUCHK(cudaFree(s->logits_gpu));
    free(s->logits);
    CUCHK(cudaFree(s->key_cache));
    CUCHK(cudaFree(s->value_cache));
    CUCHK(cudaFree(s->freq_real));
    CUCHK(cudaFree(s->freq_imag));
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

size_t createAndCopy(float** dst, float* src, size_t n)
{
    CUCHK(cudaMalloc((void**)dst, n * sizeof(float)));
    CUCHK(cudaMemcpyAsync(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return n;
}

void free_map_weights(TransformerWeightsCuda* w) 
{
	CUCHK(cudaFree(w->rms_att_weight));
	CUCHK(cudaFree(w->rms_ffn_weight));
	CUCHK(cudaFree(w->rms_final_weight));
	CUCHK(cudaFree(w->token_embedding_table));
	CUCHK(cudaFree(w->wq));
	CUCHK(cudaFree(w->wk));
	CUCHK(cudaFree(w->wv));
	CUCHK(cudaFree(w->wo));
	CUCHK(cudaFree(w->w1));
	CUCHK(cudaFree(w->w2));
	CUCHK(cudaFree(w->w3));
	CUCHK(cudaFree(w->wcls));
}

void memory_map_weights(TransformerGpu* t, TransformerWeightsCuda* w, Config* p, float* ptr, int shared_weights) 
{
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    size_t n_layers = p->n_layers;

    ptr += createAndCopy(&w->rms_att_weight, ptr, n_layers * p->dim);
    ptr += createAndCopy(&w->rms_ffn_weight, ptr, n_layers * p->dim);
    ptr += createAndCopy(&w->rms_final_weight, ptr, p->dim);
    ptr += createAndCopy(&w->token_embedding_table, ptr, p->vocab_size * p->dim);

    ptr += createAndCopy(&w->wq, ptr, n_layers * p->dim * (p->n_heads * head_size));
	ptr += createAndCopy(&w->wk, ptr, n_layers * p->dim * (p->n_kv_heads * head_size));
    ptr += createAndCopy(&w->wv, ptr, n_layers * p->dim * (p->n_kv_heads * head_size));
    ptr += createAndCopy(&w->wo, ptr, n_layers * (p->n_heads * head_size) * p->dim);

    ptr += createAndCopy(&w->w1, ptr, n_layers * p->dim * p->hidden_dim);
    ptr += createAndCopy(&w->w2, ptr, n_layers * p->hidden_dim * p->dim);
    ptr += createAndCopy(&w->w3, ptr, n_layers * p->dim * p->hidden_dim);

	if (shared_weights) 
		w->wcls = w->token_embedding_table;
	else 
		createAndCopy(&w->wcls, ptr, p->dim * p->vocab_size);

    CUCHK(cudaStreamSynchronize(0));
}

void read_checkpoint(TransformerGpu* t, const char* checkpoint, Config* config, TransformerWeightsCuda* weights,
    int* fd, float** data, ssize_t* file_size) 
{
    FILE* file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }
    // read in the version number (uint32), has to be 1
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 1) { fprintf(stderr, "Bad version %d, need version 1\n", version); exit(EXIT_FAILURE); }
    int header_size = 256; // the header size for version 2 in bytes
    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }

    if (config->seq_len > 20480)
        config->seq_len = 20480;

    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    long lErr;
    size_t workspace_count = config->dim * 3 * sizeof(float);
    if (workspace_count < config->seq_len + 1)
        workspace_count = config->seq_len + 1;
    if (workspace_count < config->vocab_size * 2)
        workspace_count = config->vocab_size * 2;

    if (lErr = t->m_primitives.Initialize(workspace_count, config->dim, config->n_layers, config->n_heads, config->seq_len))
    {
        printf("Primitives initialization failed!\n");
        exit(1);
    }

    // memory map the Transformer weights into the data pointer
    *fd = _open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte
    memory_map_weights(t, weights, config, (float*)weights_ptr, shared_classifier);
}

void TransformerGpu::build(const char* checkpoint_path)
{
    read_checkpoint(this, checkpoint_path, &m_config, &m_weights, &fd, &data, &file_size);
    cuda_malloc_run_state(&m_state, &m_config);
}

void TransformerGpu::cleanup() 
{
    // close the memory mapping
    if (data != MAP_FAILED) 
        munmap(data, file_size); 

    if (fd != -1) 
        _close(fd); 

    // free the RunState buffers
    cuda_free_run_state(&m_state);
    free_map_weights(&m_weights);
}

// end