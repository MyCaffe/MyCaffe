from torch import nn
from constants import *
from layers import *

from test_base import DebugFunction

import torch


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = LogSoftmaxEx("", dim=-1)

    def save_internal_state(self, i):
        DebugFunction.set_output_path(i)
        DebugFunction.trace(self.src_embedding.weight, 'src_emb.weight');
        DebugFunction.trace(self.trg_embedding.weight, 'trg_emb.weight');
        self.encoder.save_internal_state();
        self.decoder.save_internal_state();
        DebugFunction.trace(self.output_linear.weight, 'out_linear.weight');
        DebugFunction.trace(self.output_linear.bias, 'out_linear.bias');        

    def forward(self, iter, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
        trg_input = self.trg_embedding(trg_input) # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(trg_input) # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask) # (B, L, d_model)

        d_linear = self.output_linear(d_output)
        output = self.softmax(d_linear) # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer("enc_%d" % (i)) for i in range(num_layers)])
        self.layer_norm = LayerNormalization("enc.ln")
        
    def save_internal_state(self):
        for i in range(num_layers):
            self.layers[i].save_internal_state()

    def forward(self, x, e_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer("dec_%d" % (i)) for i in range(num_layers)])
        self.layer_norm = LayerNormalization("dec.ln")

    def save_internal_state(self):
        for i in range(num_layers):
            self.layers[i].save_internal_state()
                
    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)
