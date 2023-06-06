from torch import nn
from constants import *

from test_base import DebugFunction
from test_base import mycaffe
from torch.autograd.variable import Variable

import torch
import math
import decimal
import numpy as np

from layers import MultiheadAttention
from layers import FeedFowardLayer
from layers import LayerNormalization
from layers import LayerNormEx
from layers import LinearEx
from layers import SoftmaxEx

from main import Manager
from custom_data import set_seed

tag_list = []

class LayerNormalizationEx(nn.Module):
    def __init__(self, tag, eps=1e-6):
        super().__init__()
        self.tag = tag
        self.eps = eps
        self.layer = LayerNormEx1(tag, [d_model])
        #self.layer = LayerNormEx(tag, [d_model])
        
    def forward(self, i, x):
        return self.layer(i, x)
        #return self.layer(x)
   
#
# LayerNormEx
#     
class LayerNormEx1(nn.Module):

    def __init__(self,
                 tag,
                 normal_shape,
                 gamma=False,
                 beta=False,
                 epsilon=1e-5):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNormEx1, self).__init__()
        self.tag = tag
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()  
        
    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, i, x):
        debug = DebugFunction.apply
        
        DebugFunction.trace(x, self.tag + ".1_x")
        x = debug(x)

        mean = x.mean(dim=-1, keepdim=True)
        DebugFunction.trace(mean, self.tag + ".2_mean")
        mean = debug(mean)
        
        xmu = (x - mean)
        DebugFunction.trace(xmu, self.tag + ".3_xmu")
        xmu = debug(xmu)
        
        xmusq = xmu ** 2
        DebugFunction.trace(xmusq, self.tag + ".4_xmusq")
        xmusq = debug(xmusq)
        
        var = xmusq.mean(dim=-1, keepdim=True)
        DebugFunction.trace(var, self.tag + ".5_var")
        var = debug(var)
        
        var1 = var + self.epsilon
        DebugFunction.trace(var1, self.tag + ".6_var1")
        var1 = debug(var1)
        
        std = var1.sqrt()
        DebugFunction.trace(std, self.tag + ".7_std")
        std = debug(std)
        
        y = xmu / std        
        DebugFunction.trace(y, self.tag + ".8_y")
        y = debug(y)
        
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta

        return y

class LogSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        m = x.max(axis=1, keepdims=True)[0]
        xmu = x - m
        expx = torch.exp(xmu)
        sumexp = torch.sum(expx, axis=1, keepdims=True)   
        log_z = m + torch.log(sumexp)
        y = x - log_z
        ctx.save_for_backward(y)
        print(y.detach().cpu().numpy())
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        gy = grad_output
        y, = ctx.saved_tensors
        sumgy = gy.sum(axis=1, keepdims=True)
        expy = torch.exp(y)
        grad = gy - expy * sumgy
        print("grad -> input")
        print(grad.detach().cpu().numpy())
        return grad

class LogSoftmaxEx(nn.Module):
    def __init__(self, tag, axis):
        super().__init__()
        self.axis = axis
        self.tag = tag
        
    def forward(self, x):
        softmax = LogSoftmaxFunction.apply
        return softmax(x)

class PositionalEncoderEx(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    df1 = 2 * i / d_model
                    dfPow = 10000 ** df1
                    dfPos = pos / dfPow
                    dfSin = math.sin(dfPos)
                    fSin = np.float32(dfSin)
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                    fDiff = math.fabs(fSin - pe_matrix[pos,i])
                    #print("%d,%d %.10f  %.10f  diff %.10f" % (pos, i, pe_matrix[pos,i], fSin, fDiff))
                elif i % 2 == 1:
                    df1 = 2 * i / d_model
                    dfPow = 10000 ** df1
                    dfPos = pos / dfPow
                    dfCos = math.cos(dfPos)
                    fCos = np.float32(dfCos)
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))
                    fDiff = math.fabs(fCos - pe_matrix[pos,i])
                    #print("%d,%d %.10f  %.10f  diff %.10f" % (pos, i, pe_matrix[pos,i], fCos, fDiff))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, i, x):
        debug = DebugFunction.apply
        
        DebugFunction.trace(x, self.tag + ".1_x")
        x = debug(x)
        
        x_2 = x * math.sqrt(d_model) # (B, L, d_model)

        DebugFunction.trace(x_2, self.tag + ".2_x_2")
        x_2 = debug(x_2)
        
        pos_enc = self.positional_encoding

        DebugFunction.trace(pos_enc, self.tag + ".3_pos_enc")

        x_3 = x_2 + pos_enc # (B, L, d_model)

        DebugFunction.trace(x_3, self.tag + ".3_x_3")
        x_3 = debug(x_3)
        
        return x_3

class MultiheadAttentionEx(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.inf = 1e29 #1e9

        # W^Q, W^K, W^V in the paper
        self.w_q = LinearEx(d_model, d_model)
        self.w_k = LinearEx(d_model, d_model)
        self.w_v = LinearEx(d_model, d_model)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = SoftmaxEx(tag + ".smx", dim=-1)
        
        # Final output linear transformation
        self.w_0 = LinearEx(d_model, d_model)

    def save_internal_state(self, i):
        DebugFunction.trace(self.w_q.bias, self.tag + ".w_q.bias")
        DebugFunction.trace(self.w_q.weight, self.tag + ".w_q.weight")
        DebugFunction.trace(self.w_k.bias, self.tag + ".w_k.bias")
        DebugFunction.trace(self.w_k.weight, self.tag + ".w_k.weight")
        DebugFunction.trace(self.w_v.bias, self.tag + ".w_v.bias")
        DebugFunction.trace(self.w_v.weight, self.tag + ".w_v.weight")
        DebugFunction.trace(self.w_0.bias, self.tag + ".w_o.bias")
        DebugFunction.trace(self.w_0.weight, self.tag + ".w_o.weight")

    def forward(self, i, q, k, v, mask=None):
        debug = DebugFunction.apply
        
        q1 = q.clone()
        DebugFunction.trace(q1, self.tag + ".1_q")
        q1 = debug(q1)

        k1 = k.clone()
        DebugFunction.trace(k1, self.tag + ".1_k")
        k1 = debug(k1)

        v1 = v.clone()
        DebugFunction.trace(v1, self.tag + ".1_v")
        v1 = debug(v1)

        DebugFunction.trace(mask, self.tag + ".1_mask")
        mask = debug(mask)
       
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q1 = self.w_q(q1)
        k1 = self.w_k(k1)
        v1 = self.w_v(v1)

        DebugFunction.trace(q1, self.tag + ".2_q1")
        q1 = debug(q1)
        DebugFunction.trace(k1, self.tag + ".2_k1")
        k1 = debug(k1)
        DebugFunction.trace(v1, self.tag + ".2_v1")
        v1 = debug(v1)

        q2 = q1.view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        k2 = k1.view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        v2 = v1.view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)

        DebugFunction.trace(q2, self.tag + ".3_q2")
        q2 = debug(q2)
        DebugFunction.trace(k2, self.tag + ".3_k2")
        k2 = debug(k2)
        DebugFunction.trace(v2, self.tag + ".3_v2")
        v2 = debug(v2)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q3 = q2.transpose(1, 2)
        k3 = k2.transpose(1, 2)
        v3 = v2.transpose(1, 2)

        DebugFunction.trace(q3, self.tag + ".4_q3")
        q3 = debug(q3)
        DebugFunction.trace(k3, self.tag + ".4_k3")
        k3 = debug(k3)
        DebugFunction.trace(v3, self.tag + ".4_v3")
        v3 = debug(v3)

        # Conduct self-attention
        #attn_values = self.self_attention(q3, k3, v3, mask=mask) # (B, num_heads, L, d_k)

        # Calculate attention scores with scaled dot-product attention

        k4 = k3.transpose(-2, -1)
        
        DebugFunction.trace(k4, self.tag + ".5_k4")
        k4 = debug(k4)
        
        #
        # Self attention
        #
        attn_scores1 = torch.matmul(q3, k4) # (B, num_heads, L, L)
        
        DebugFunction.trace(attn_scores1, self.tag + ".6_attn_scores1")
        attn_scores1 = debug(attn_scores1)
        attn_scores2 = attn_scores1 / math.sqrt(d_k)

        DebugFunction.trace(attn_scores2, self.tag + ".6_attn_scores2")
        
        # If there is a mask, make masked spots -INF
        mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
        attn_scores3 = attn_scores2.masked_fill_(mask == 0, -1 * self.inf)
        
        DebugFunction.trace(attn_scores3, self.tag + ".7_attn_scores3")
        attn_scores3 = debug(attn_scores3)
        
        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores3)
        
        DebugFunction.trace(attn_distribs, self.tag + ".8_attn_distribs")
        attn_distribs = debug(attn_distribs)
        
        #attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v3) # (B, num_heads, L, d_k)

        DebugFunction.trace(attn_values, self.tag + ".9_attn_values")
        attn_values = debug(attn_values)
        #
        #
        #

        concat_output1 = attn_values.transpose(1, 2);
        
        DebugFunction.trace(concat_output1, self.tag + ".10_concat_output1")
        concat_output1 = debug(concat_output1)

        concat_output2 = concat_output1.contiguous().view(input_shape[0], -1, d_model)
        
        DebugFunction.trace(concat_output2, self.tag + ".11_concat_output2")
        concat_output2 = debug(concat_output2)
        
        output = self.w_0(concat_output2)

        DebugFunction.trace(output, self.tag + ".12_output")
        output = debug(output)

        return output

class FeedFowardLayerEx(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.linear_1 = LinearEx(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = LinearEx(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(drop_out_rate)
        
    def save_internal_state(self, i):
        DebugFunction.trace(self.linear_1.weight, self.tag + ".linear_1.weight")
        DebugFunction.trace(self.linear_1.bias, self.tag + ".linear_1.bias")
        DebugFunction.trace(self.linear_2.weight, self.tag + ".linear_2.weight")
        DebugFunction.trace(self.linear_2.bias, self.tag + ".linear_2.bias")
        
    def forward(self, i, x):
        debug = DebugFunction.apply
        
        DebugFunction.trace(x, self.tag + ".x")
        x = debug(x)
        
        x1 = self.linear_1(x)
        
        DebugFunction.trace(x1, self.tag + ".x1")
        x1 = debug(x1)
        
        x2 = self.relu(x1) # (B, L, d_ff)
        
        DebugFunction.trace(x2, self.tag + ".x2")
        x2 = debug(x2)

        x3 = self.dropout(x2)

        DebugFunction.trace(x3, self.tag + ".x3")
        x3 = debug(x3)
        
        x4 = self.linear_2(x3) # (B, L, d_model)
        
        DebugFunction.trace(x4, self.tag + ".x4")
        x4 = debug(x4)

        return x4

class EncoderLayerEx(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.layer_norm_1 = LayerNormalizationEx(tag + ".ln1")
        self.multihead_attention = MultiheadAttentionEx(tag + ".mh")
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalizationEx(tag + ".ln2")
        self.feed_forward = FeedFowardLayerEx(tag + ".ff")
        self.drop_out_2 = nn.Dropout(drop_out_rate)
        
    def save_internal_state(self, i):
        self.multihead_attention.save_internal_state(i)
        self.feed_forward.save_internal_state(i)

    def forward(self, i, x, e_mask):
        debug = DebugFunction.apply
        self.save_internal_state(i)

        DebugFunction.trace(x, self.tag + ".enc.1_x")
        x = debug(x)
        
        x_1 = self.layer_norm_1(i, x) # (B, L, d_model)

        DebugFunction.trace(x_1, self.tag + ".enc.2_x_1")
        x_1 = debug(x_1)
        
        attn = self.multihead_attention(i, x_1, x_1, x_1, mask=e_mask)

        DebugFunction.trace(attn, self.tag + ".enc.3_attn")
        attn = debug(attn)

        #attn = self.drop_out_1(attn)
        xB = x + attn

        DebugFunction.trace(xB, self.tag + ".enc.4_xB")
        xB = debug(xB)
            
        x_2 = self.layer_norm_2(i, xB) # (B, L, d_model)

        DebugFunction.trace(x_2, self.tag + ".enc.5_x_2")
        x_2 = debug(x_2)
        
        ff = self.feed_forward(i, x_2)
        #ff = self.drop_out_2(ff)

        DebugFunction.trace(ff, self.tag + ".enc.6_ff")
        ff = debug(ff)
        
        xC = xB + ff # (B, L, d_model)

        DebugFunction.trace(xC, self.tag + ".enc.7_xC")
        xC = debug(xC)

        return xC # (B, L, d_model)

class DecoderLayerEx(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.layer_norm_1 = LayerNormalizationEx(tag + ".ln1")
        self.masked_multihead_attention = MultiheadAttentionEx(tag + ".mh1")
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormalizationEx(tag + ".ln2")
        self.multihead_attention = MultiheadAttentionEx(tag + ".mh2")
        self.drop_out_2 = nn.Dropout(drop_out_rate)

        self.layer_norm_3 = LayerNormalizationEx(tag + ".ln3")
        self.feed_forward = FeedFowardLayerEx(tag + ".ff")
        self.drop_out_3 = nn.Dropout(drop_out_rate)
        
    def save_internal_state(self, i):
        self.masked_multihead_attention.save_internal_state(i)
        self.multihead_attention.save_internal_state(i)
        self.feed_forward.save_internal_state(i)
                
    def forward(self, i, x, e_output, e_mask, d_mask):
        debug = DebugFunction.apply
        self.save_internal_state(i)
        
        DebugFunction.trace(e_mask, self.tag + ".dec.1_e_mask")
        DebugFunction.trace(d_mask, self.tag + ".dec.1_d_mask")

        DebugFunction.trace(x, self.tag + ".dec.1_x")
        x = debug(x)

        x_1 = self.layer_norm_1(i, x) # (B, L, d_model)
        DebugFunction.trace(x_1, self.tag + ".dec.2_x_1")
        x_1 = debug(x_1)
        
        attn1 = self.masked_multihead_attention(i, x_1, x_1, x_1, mask=d_mask)
        DebugFunction.trace(attn1, self.tag + ".dec.3_attn1")
        attn1 = debug(attn1)

        #attn = self.drop_out_1(attn)
        xB = x + attn1
        DebugFunction.trace(xB, self.tag + ".dec.4_xB")
        xB = debug(xB)
            
        x_2 = self.layer_norm_2(i, xB) # (B, L, d_model)
        DebugFunction.trace(x_2, self.tag + ".dec.5_x_2")
        x_2 = debug(x_2)

        DebugFunction.trace(e_output, self.tag + ".dec.5_e_out")
        e_output = debug(e_output)

        attn2 = self.multihead_attention(i, x_2, e_output, e_output, mask=e_mask)       
        DebugFunction.trace(attn2, self.tag + ".dec.6_attn2")
        attn2 = debug(attn2)
        
        DebugFunction.trace(xB, self.tag + ".dec.6_xB")
        xB = debug(xB)

        #attn = self.drop_out_2(attn)
        xC = xB + attn2
        DebugFunction.trace(xC, self.tag + ".dec.7_xC")
        xC = debug(xC)
        
        x_3 = self.layer_norm_3(i, xC) # (B, L, d_model)
        DebugFunction.trace(x_3, self.tag + ".dec.8_x_3")
        x_3 = debug(x_3)
                
        ff = self.feed_forward(i, x_3)
        #ff = self.drop_out_2(ff)
        DebugFunction.trace(ff, self.tag + ".dec.9_ff")
        ff = debug(ff)
        
        xD = xC + ff # (B, L, d_model)
        DebugFunction.trace(xD, self.tag + ".dec.10_xD")
        xD = debug(xD)

        return xD # (B, L, d_model)

class TransformerEx(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.tag = "transformer"
        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoderEx("pos1")
        self.encoder = EncoderEx("enc")
        self.decoder = DecoderEx("dec")
        self.output_linear = LinearEx(d_model, self.trg_vocab_size)
        self.softmax = LogSoftmaxEx(self.tag, axis=-1)

    def save_internal_state(self, i):
        DebugFunction.trace(self.src_embedding.weight, self.tag + ".src_emb.weight");
        DebugFunction.trace(self.trg_embedding.weight, self.tag + ".trg_emb.weight");
        self.encoder.save_internal_state(i);
        self.decoder.save_internal_state(i);
        DebugFunction.trace(self.output_linear.weight, self.tag + ".out_linear.weight");
        DebugFunction.trace(self.output_linear.bias, self.tag + ".out_linear.bias");        

    def forward(self, i, src_input, trg_input, e_mask=None, d_mask=None):
        debug = DebugFunction.apply
        DebugFunction.trace(src_input, self.tag + ".src_input")
        src_input = debug(src_input)
        DebugFunction.trace(trg_input, self.tag + ".trg_input")
        trg_input = debug(trg_input)

        src_input_emb = self.src_embedding(src_input) # (B, L) => (B, L, d_model)
        trg_input_emb = self.trg_embedding(trg_input) # (B, L) => (B, L, d_model)

        DebugFunction.trace(src_input_emb, self.tag + ".src_input_emb")
        src_input_emb = debug(src_input_emb)
        DebugFunction.trace(trg_input_emb, self.tag + ".trg_input_emb")
        trg_input_emb = debug(trg_input_emb)

        src_input_pos = self.positional_encoder(i, src_input_emb) # (B, L, d_model) => (B, L, d_model)
        trg_input_pos = self.positional_encoder(i, trg_input_emb) # (B, L, d_model) => (B, L, d_model)

        DebugFunction.trace(src_input_pos, self.tag + ".src_input_pos")
        src_input_pos = debug(src_input_pos)
        DebugFunction.trace(trg_input_pos, self.tag + ".trg_input_pos")
        trg_input_pos = debug(trg_input_pos)
        DebugFunction.trace(e_mask, self.tag + ".e_mask")
        DebugFunction.trace(d_mask, self.tag + ".d_mask")

        e_output = self.encoder(i, src_input_pos, e_mask) # (B, L, d_model)

        DebugFunction.trace(e_output, self.tag + ".e_output")
        e_output = debug(e_output)

        d_output = self.decoder(i, trg_input_pos, e_output, e_mask, d_mask) # (B, L, d_model)

        DebugFunction.trace(d_output, self.tag + ".d_output")
        d_output = debug(d_output)

        d_output1 = self.output_linear(d_output)

        DebugFunction.trace(d_output1, self.tag + ".d_output1")
        d_output1 = debug(d_output1)

        output = self.softmax(d_output1) # (B, L, d_model) => # (B, L, trg_vocab_size)

        DebugFunction.trace(output, self.tag + ".output")
        output = debug(output)

        return output

class EncoderEx(nn.Module):
    def __init__(self, tag):        
        super().__init__()
        self.tag = tag
        self.layers = nn.ModuleList([EncoderLayerEx(tag + ".enc%d" % (i)) for i in range(num_layers)])
        self.layer_norm = LayerNormalizationEx(tag + ".ln")

    def save_internal_state(self, it):
        for i in range(num_layers):
            self.layers[i].save_internal_state(it)
        
    def forward(self, it, x, e_mask):
        debug = DebugFunction.apply
        for i in range(num_layers):          
            DebugFunction.trace(x, self.tag + ".x%d.in" % (i))
            x = debug(x)
            
            x = self.layers[i](it, x, e_mask)

            DebugFunction.trace(x, self.tag + ".x%d.out" % (i))
            x = debug(x)
            
        DebugFunction.trace(x, self.tag + ".ln.in")
        x1 = debug(x)

        x_ln = self.layer_norm(it, x1)
        DebugFunction.trace(x_ln, self.tag + ".ln.out")
        x_ln = debug(x_ln)
        return x_ln
       
class DecoderEx(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.layers = nn.ModuleList([DecoderLayerEx(tag + ".dec%d" % (i)) for i in range(num_layers)])
        self.layer_norm = LayerNormalizationEx(tag + ".ln")

    def save_internal_state(self, it):
        for i in range(num_layers):
            self.layers[i].save_internal_state(it)
        
    def forward(self, it, x, e_output, e_mask, d_mask):
        debug = DebugFunction.apply
        for i in range(num_layers):
            DebugFunction.trace(e_output, self.tag + ".enc%d.in" % (i))
            e_output = debug(e_output)
            DebugFunction.trace(x, self.tag + ".x%d.in" % (i))
            x = debug(x)

            x = self.layers[i](it, x, e_output, e_mask, d_mask)

            DebugFunction.trace(x, self.tag + ".x%d.out" % (i))
            x = debug(x)

        DebugFunction.trace(x, self.tag + ".ln.in")
        x1 = debug(x)

        x_ln = self.layer_norm(it, x1)
        DebugFunction.trace(x_ln, self.tag + ".ln.out")
        x_ln = debug(x_ln)
        return x_ln

