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

    Transformer() 
    {
        m_bDebug = false;
    }

    virtual void build(const char* checkpoint_path) = 0;
    virtual float* forward(int token, int pos) = 0;
    virtual void cleanup() = 0;
    virtual void print(const char* name, void* x, size_t n, long long nLayer = -1) = 0;

    void printHost(const char* name, float* x, size_t n, long long nLayer = -1)
    {
        if (!m_bDebug) return;

        if (nLayer >= 0)
			printf("Layer %d - %s: (%ld) { %lf, %lf, %lf, %lf, ...}\n", (int)nLayer, name, (int)n, x[0], x[1], x[2], x[3]);
		else
            printf("%s: (%ld) { %lf, %lf, %lf, %lf, ...}\n", name, (int)n, x[0], x[1], x[2], x[3]);
	}
};

#endif // __CONFIG_H_
