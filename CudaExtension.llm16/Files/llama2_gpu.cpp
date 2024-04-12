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

    // pluck out the 'pos' row of freq_cis_real and freq_cis_imag
    int nFreqOffset = pos * head_size / 2;

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++) 
    {
        // attention rmsnorm
        m_primitives.rmsnorm(dim, s->xb, x, w->rms_att_weight + l * dim);

        // qkv matmuls for this position
        m_primitives.matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim, false);
        m_primitives.matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim, false);
        m_primitives.matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        m_primitives.rope(s->q, s->k, s->freq_real, s->freq_imag, p->n_heads, head_size, nFreqOffset);

        // save key,value at this time step (pos) to our kv cache
        unsigned long long loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        cudaMemcpyAsync(key_cache_row, s->k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(value_cache_row, s->v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaStreamSynchronize(0);

        // multihead attention. iterate over all heads
        m_primitives.attention(s->xb, s->q, s->key_cache, s->value_cache, p->n_heads, head_size, loff, pos, p->seq_len, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        m_primitives.matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // residual connection back into x
        m_primitives.add(dim, x, s->xb2);

        // ffn rmsnorm
        m_primitives.rmsnorm(dim, s->xb, x, w->rms_ffn_weight + l * dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        m_primitives.matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim, false);
        m_primitives.matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        m_primitives.siglu(hidden_dim, s->hb, s->hb, s->hb2);

        // final matmul to get the output of the ffn
        m_primitives.matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // residual connection
        m_primitives.add(dim, x, s->xb);
    }

    // final rmsnorm
    m_primitives.rmsnorm(dim, x, x, w->rms_final_weight);

    // classifier into logits
    m_primitives.matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

    // copy logits from GPU->CPU
    m_primitives.ToHost(p->vocab_size, s->logits, s->logits_gpu);

    return s->logits;
}

LONG cuda_malloc_run_state(RunStateCuda* s, Config* p)
{
    LONG lErr;
    size_t seq_len = p->seq_len;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    if (lErr = cudaMalloc((void**)&s->x, p->dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->xb, p->dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->xb2, p->dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->q, p->dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->k, kv_dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->v, kv_dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->key_cache, p->n_layers * seq_len * kv_dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->value_cache, p->n_layers * seq_len * kv_dim * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->att, p->n_heads * seq_len * sizeof(float))) 
        return lErr;
    if (lErr = cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(float))) 
        return lErr;
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v
        || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits)
        return CUDA_ERROR_OUT_OF_MEMORY;

    int head_size = p->dim / p->n_heads;
    size_t nCount = seq_len * (head_size / 2);

    if (lErr = cudaMalloc((void**)&s->freq_real, nCount * sizeof(float)))
        return lErr;
    if (lErr = cudaMalloc((void**)&s->freq_imag, nCount * sizeof(float)))
        return lErr;

    // ensure all mallocs went fine
    if (!s->freq_real || !s->freq_imag)
        return ERROR_OUTOFMEMORY;

    float* freq_real = (float*)calloc(nCount, sizeof(float));
    float* freq_imag = (float*)calloc(nCount, sizeof(float));

    if (!freq_real || !freq_imag)
        return ERROR_OUTOFMEMORY;

    precompute_freqs_cis(head_size, seq_len, freq_real, freq_imag);
    
    if (lErr = cudaMemcpyAsync(s->freq_real, freq_real, nCount * sizeof(float), cudaMemcpyHostToDevice))
        return lErr;
    if (lErr = cudaMemcpyAsync(s->freq_imag, freq_imag, nCount * sizeof(float), cudaMemcpyHostToDevice))
        return lErr;
	cudaStreamSynchronize(0);

    free(freq_real);
    free(freq_imag);
}

void cuda_free_run_state(RunStateCuda* s) 
{
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->att);
    cudaFree(s->logits_gpu);
    free(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
    cudaFree(s->freq_real);
    cudaFree(s->freq_imag);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

size_t createAndCopy(float** dst, float* src, size_t n)
{
    long lErr;
    try
    {
        if (lErr = cudaMalloc((void**)dst, n * sizeof(float)))
            throw lErr;
        if (lErr = cudaMemcpyAsync(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice))
            throw lErr;
    }
    catch (long lErr)
    {
		return lErr;
	}

    return n;
}

void free_map_weights(TransformerWeightsCuda* w) 
{
	cudaFree(w->rms_att_weight);
	cudaFree(w->rms_ffn_weight);
	cudaFree(w->rms_final_weight);
	cudaFree(w->token_embedding_table);
	cudaFree(w->wq);
	cudaFree(w->wk);
	cudaFree(w->wv);
	cudaFree(w->wo);
	cudaFree(w->w1);
	cudaFree(w->w2);
	cudaFree(w->w3);
	cudaFree(w->wcls);
}

long memory_map_weights(TransformerGpu* t, TransformerWeightsCuda* w, Config* p, float* ptr, int shared_weights) 
{
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    size_t n_layers = p->n_layers;

    try
    {
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

        return cudaStreamSynchronize(0);
    }
    catch (long lErr)
    {
        return lErr;
	}
}

long read_checkpoint(TransformerGpu* t, const char* checkpoint, Config* config, TransformerWeightsCuda* weights,
    int* fd, float** data, ssize_t* file_size) 
{
    LONG lErr;
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
    
    if (config->seq_len > 20480)
        config->seq_len = 20480;

    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1)
        return ERROR_FILE_INVALID;
    
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    size_t workspace_count = config->dim * 3 * sizeof(float);
    if (workspace_count < config->seq_len + 1)
        workspace_count = config->seq_len + 1;
    if (workspace_count < config->vocab_size * 2)
        workspace_count = config->vocab_size * 2;

    if (lErr = t->m_primitives.Initialize(workspace_count, config->dim, config->n_layers, config->n_heads, config->seq_len))
        return ERROR_INVALID_STATE;

    // memory map the Transformer weights into the data pointer
    *fd = _open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1)
        return ERROR_FILE_INVALID;

    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
        return ERROR_FILE_INVALID;

    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte

    return memory_map_weights(t, weights, config, (float*)weights_ptr, shared_classifier);
}

long TransformerGpu::build(const char* checkpoint_path)
{
    LONG lErr;
    
    if (lErr = read_checkpoint(this, checkpoint_path, &m_config, &m_weights, &fd, &data, &file_size))
        return lErr;

    if (lErr = cuda_malloc_run_state(&m_state, &m_config))
        return lErr;

    return 0;
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