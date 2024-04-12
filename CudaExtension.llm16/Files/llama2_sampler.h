#ifndef __LLAMA2_SAMPLER_H_
#define __LLAMA2_SAMPLER_H_

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int m_vocab_size;
    ProbIndex* m_probindex; // buffer used in top-p sampling
    float m_temperature;
    float m_topp;
    unsigned long long m_rng_state;
} Sampler;

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);
int sample(Sampler* sampler, float* logits);

#endif // __LLAMA2_SAMPLER_H_

