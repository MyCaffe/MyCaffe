#ifndef __LLAMA2_TOKENIZER_H_
#define __LLAMA2_TOKENIZER_H_

typedef struct {
    char* str;
    int id;
} TokenIndex;

typedef struct {
    char** m_vocab;
    float* m_vocab_scores;
    TokenIndex* m_sorted_vocab;
    int m_vocab_size;
    unsigned int m_max_token_length;
    unsigned char m_byte_pieces[512]; // stores all single-byte strings
} Tokenizer;


void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
char* decode(Tokenizer* t, int prev_token, int token);
void safe_printf(char* piece);
void encode(Tokenizer* t, char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens);


#endif // __LLAMA2_TOKENIZER_H_