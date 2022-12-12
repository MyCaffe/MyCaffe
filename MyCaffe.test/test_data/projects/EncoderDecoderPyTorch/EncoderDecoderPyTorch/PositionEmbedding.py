import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Utility import report_memory, free_memory
from Vocabulary import Vocabulary

'''
The PositionEmbeddingFixedWeights layer is a PyTorch rewrite of the TensorFlow and Keras version by Mehreen Saeed
@see [The Transformer Positional Encoding Layer in Keras, Part 2](https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/) by Mehreen Saeed, 2022
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''

# PositionEmbeddingFixedWeights layer
class PositionEmbeddingFixedWeights(nn.Module):
    def __init__(self, sequence_length, vocab_size, output_dim, batch_size):
        super(PositionEmbeddingFixedWeights, self).__init__()
        self.token_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        self.position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)
        self.tok_embedding_layer = self.create_embedding_layer(self.token_embedding_matrix, trainable=False)
        self.pos_embedding_layer = self.create_embedding_layer(self.position_embedding_matrix, trainable=False)
        self.pos_indices = torch.arange(sequence_length).unsqueeze(0).repeat(batch_size, 1)
        
    def to(self, device):
        new_self = super(PositionEmbeddingFixedWeights, self).to(device)
        new_self.pos_indices = new_self.pos_indices.to(device)
        return new_self

    def create_embedding_layer(self, embedding_matrix, trainable=False):
        num_embeddings, embedding_dim = embedding_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix))
        emb_layer.weight.requires_grad = trainable
        return emb_layer

    def get_position_encoding(self, seq_len, d, n=10000):
        p = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                p[k, 2*i] = np.sin(k/denominator)
                p[k, 2*i+1] = np.cos(k/denominator)
        return p

    def forward(self, x):
        tok_emb = self.tok_embedding_layer(x)
        pos_emb = self.pos_embedding_layer(self.pos_indices)

        if (pos_emb.shape[1] > tok_emb.shape[1]):
            pos_emb = pos_emb[:,0:tok_emb.shape[1],:]

        emb = tok_emb + pos_emb 
        return emb.float()

    @staticmethod
    def test():
        seq_len = 5
        vocab_size = 5
        output_dim = 5
        batch_size = 2
        emb = PositionEmbeddingFixedWeights(seq_len, vocab_size, output_dim, batch_size)
        x = torch.arange(seq_len-1).unsqueeze(0).repeat(batch_size, 1)
        print(emb(x))

# PositionEmbeddingFixedWeights.test()
   
def test():
    technical_phrase = "to understand machine learning algorithms you need" +\
                       " to understand concepts such as gradient of a function " +\
                       "Hessians of a matrix and otimization etc"
    wise_phrase = "patrick henry said give me liberty of give me death " +\
                  "when he addressed the second virginia convention in march"

    total_vocabulary = 200
    sequence_length = 20
    final_output_len = 50

    phrases = [technical_phrase, wise_phrase]
    vocab = Vocabulary.build(phrases)
   
    input_data = vocab.tokenize_vector(phrases, sequence_length)
    
    fixed_weights_embedding = PositionEmbeddingFixedWeights(sequence_length, total_vocabulary, final_output_len, 1)
    fixed_embedding = fixed_weights_embedding(input_data)
    
    fig = plt.figure(figsize=(15, 5))
    title = ["Tech Phrase", "Wise Phrase"]
    for i in range(2):
        ax = plt.subplot(1, 2, 1+i)
        matrix = fixed_embedding[i, :, :]
        cax = ax.matshow(matrix)
        plt.gcf().colorbar(cax)
        plt.title(title[i], y=1.2)
    fig.suptitle("Fixed Weight Embedding")
    plt.show()

# test()