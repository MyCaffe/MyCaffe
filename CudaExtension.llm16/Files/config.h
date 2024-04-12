#ifndef __CONFIG_H_
#define __CONFIG_H_

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

void precompute_freqs_cis(int dim, size_t seq_len, float* freq_real, float* freq_imag);

class Transformer
{
public:
    Config m_config;
    bool m_bDebug;

    Transformer() : m_config(), m_bDebug(false)
    {
	}

    virtual long build(const char* checkpoint_path) = 0;
    virtual float* forward(int token, int pos) = 0;
    virtual void cleanup() = 0;
};

#endif // __CONFIG_H_
