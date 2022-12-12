import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, Counter


'''
Vocabulary Class - created from article by Slinae Lin
@see [PyTorch and Tensorflow Natural Language Processing Pipeline_Data Preprocessing](https://ruolanlin.medium.com/comparing-pytorch-and-tensorflow-in-natural-language-processing-pipeline-part-1-9af01f012ff) by Slinae Lin, 2022

and github by HIT-SCIR
@see [GitHub: HIT-SCIR/plm-nlp-code](https://github.com/HIT-SCIR/plm-nlp-code/blob/main/chp4/vocab.py)
''' 

class Vocabulary:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()
        if tokens is not None:
            if "<eos>" not in tokens:
                tokens = ["<eos>"] + tokens
            if "<start>" not in tokens:
                tokens = ["<start>"] + tokens
            if "<unk>" not in tokens:
                tokens = ["<unk>"] + tokens
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.start = self.token_to_idx["<start>"]
            self.eos = self.token_to_idx["<eos>"]
            self.unk = self.token_to_idx["<unk>"]


    # Build the vocabulary from the text
    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence.split():
                token_freqs[token.lower()] += 1
        token_freqs = {val[0] : val[1] for val in sorted(token_freqs.items(), key = lambda x: (-x[1], x[0]))}
        uniq_tokens = ["<unk>"] + ["<start>"] + ["<eos>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>" and token != "<start>" and token != "<eos>"]
        return cls(uniq_tokens)

    def extend(self, v):
        for token in v.idx_to_token:
            if token not in self.idx_to_token:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        self.start = self.token_to_idx["<start>"]
        self.eos = self.token_to_idx["<eos>"]
        self.unk = self.token_to_idx["<unk>"]

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def tokenize(self, text, max_len, pad=False):
        tokens = []
        idx = 0
        for word in text.split():
            tokens.append(self.token_to_idx[word.lower()])
            idx += 1
            if idx >= max_len:
                break
        if pad:
            while len(tokens) < max_len:
                tokens.append(0)
        return tokens

    def tokenize_vector(self, phrases, max_len):
        tokens = []
        for phrase in phrases:
            tokens.append(self.tokenize(phrase, max_len))
        tokens = pad_sequence([torch.tensor(token) for token in tokens], batch_first=True, padding_value=0)
        return tokens

    def detokenize(self, indices):
        items = []
        for idx in indices:
            item = self.idx_to_token[idx]
            items.append(item)
        return items

    def save(self, path):
        with open(path, 'w') as writer:
            writer.write('\n'.join(self.idx_to_token))

    @staticmethod
    def load(path):
        with open(path, 'r') as reader:
            tokens = reader.read().split('\n')
        return Vocabulary(tokens)
