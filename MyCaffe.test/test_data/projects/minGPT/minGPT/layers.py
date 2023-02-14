from torch import nn
from constants import *

from test_base import DebugFunction
from test_base import mycaffe
from torch.autograd.variable import Variable

import torch
import math
import decimal
import numpy as np

tag_list = []

class EncoderLayer(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.layer_norm_1 = LayerNormEx(tag + ".ln1")
        self.multihead_attention = MultiheadAttention(tag + ".mh")
        self.drop_out_1 = nn.Dropout(drop_out_rate)
        
        self.layer_norm_2 = LayerNormEx(tag + ".ln2")
        self.feed_forward = FeedFowardLayer(tag + ".ff")
        self.drop_out_2 = nn.Dropout(drop_out_rate)

    def save_internal_state(self):
        self.multihead_attention.save_internal_state();
        self.feed_forward.save_internal_state();
        
    def save(self, x, name):
        np.save('test/enc_' + name + '.npy', x.detach().cpu().numpy())

    def forward(self, x, e_mask):
        if save_for_testing:
            self.save(x, 'enc_in_x0')

        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2)) # (B, L, d_model)

        return x # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.layer_norm_1 = LayerNormEx(tag + ".ln1")
        self.masked_multihead_attention = MultiheadAttention(tag + ".mh1")
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.layer_norm_2 = LayerNormEx(tag + ".ln2")
        self.multihead_attention = MultiheadAttention(tag + ".mh2")
        self.drop_out_2 = nn.Dropout(drop_out_rate)
        
        self.layer_norm_3 = LayerNormEx(tag + ".ln3")
        self.feed_forward = FeedFowardLayer(tag + ".ff")
        self.drop_out_3 = nn.Dropout(drop_out_rate)
     
    def save_internal_state(self):
        self.masked_multihead_attention.save_internal_state();
        self.multihead_attention.save_internal_state();
        self.feed_forward.save_internal_state();
                
    def save(self, x, name):
        np.save('test/dec_' + name + '.npy', x.detach().cpu().numpy())

    def forward(self, x, e_output, e_mask,  d_mask):
        if save_for_testing:
            self.save(x, 'dec_in_x0')
            self.save(e_output, 'enc_out_x1')

        x_1 = self.layer_norm_1(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x) # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        ) # (B, L, d_model)
        x_3 = self.layer_norm_3(x) # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3)) # (B, L, d_model)

        return x # (B, L, d_model)
    

class MultiheadAttention(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.inf = 1e29 #1e9
        
        # W^Q, W^K, W^V in the paper
        self.w_q = LinearEx(n_embed, n_embed)
        self.w_k = LinearEx(n_embed, n_embed)
        self.w_v = LinearEx(n_embed, n_embed)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = SoftmaxEx(tag=self.tag + ".smx", dim=-1)

        # Final output linear transformation
        self.w_0 = LinearEx(n_embed, n_embed)

    def save_internal_state(self):
        DebugFunction.trace(self.w_q.weight, self.tag + ".w_q.weight")
        DebugFunction.trace(self.w_q.bias, self.tag + ".w_q.bias")
        DebugFunction.trace(self.w_k.weight, self.tag + ".w_k.weight")
        DebugFunction.trace(self.w_k.bias, self.tag + ".w_k.bias")
        DebugFunction.trace(self.w_v.weight, self.tag + ".w_v.weight")
        DebugFunction.trace(self.w_v.bias, self.tag + ".w_v.bias")
        DebugFunction.trace(self.w_0.weight, self.tag + ".w_0.weight")
        DebugFunction.trace(self.w_0.bias, self.tag + ".w_0.bias")
        
    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        if save_for_testing:
            np.save('test/q0.npy', q.detach().cpu().numpy())
            np.save('test/k0.npy', k.detach().cpu().numpy())
            np.save('test/v0.npy', v.detach().cpu().numpy())

        # Linear calculation +  split into num_heads
        #q = self.w_q(q)
        q = torch.matmul(q, self.w_q.weight.t())
        q = q + self.w_q.bias        
        q = q.view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        
        #k = self.w_k(k)
        k = torch.matmul(k, self.w_k.weight.t())
        k = k + self.w_k.bias
        k = k.view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        
        #v = self.w_v(v)
        v = torch.matmul(v, self.w_v.weight.t())
        v = v + self.w_v.bias
        v = v.view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, n_embed) # (B, L, d_model)

        output = self.w_0(concat_output)

        if save_for_testing:
            np.save('test/output0.npy', output.detach().cpu().numpy())
            # when saving for testing, break on return statement below.
            
        return output

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values

    
class FeedFowardLayer(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.linear_1 = LinearEx(n_embed, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = LinearEx(d_ff, n_embed, bias=True)
        self.dropout = nn.Dropout(drop_out_rate)
        
    def save_internal_state(self):
        DebugFunction.trace(self.linear_1.weight, self.tag + ".linear_1.weight")
        DebugFunction.trace(self.linear_1.bias, self.tag + ".linear_1.bias")
        DebugFunction.trace(self.linear_2.weight, self.tag + ".linear_2.weight")
        DebugFunction.trace(self.linear_2.bias, self.tag + ".linear_2.bias")

    def forward(self, x):
        x = self.relu(self.linear_1(x)) # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (B, L, d_model)

        return x
 

class BlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        y = mycaffe.tfb_fwd(tag, x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        return mycaffe.tfb_bwd(tag, y, grad_output)

class BlockEx(nn.Module):
    """ an unassuming Transformer block """
    
    def __init__(self, tag, config):
        super().__init__()
        self.tag = tag

    def forward(self, x):
        tfb = BlockFunction.apply
        tag_list.append(self.tag)
        y = tfb(x)
        return y


class BlockAllFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        y = mycaffe.tfb_fwd_all(tag, x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        return mycaffe.tfb_bwd_all(tag, y, grad_output)

class BlockAllEx(nn.Module):
    """ an unassuming Transformer block """
    
    def __init__(self, tag, config):
        super().__init__()
        self.tag = tag

    def forward(self, x):
        tfb = BlockAllFunction.apply
        tag_list.append(self.tag)
        y = tfb(x)
        return y
   
#
# LayerNormEx
#     
class LayerNormEx(nn.Module):
    
    def __init__(self,
                 tag,
                 normal_shape,
                 use_mycaffe=mycaffe_layernorm,
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
        super(LayerNormEx, self).__init__()
        self.tag = tag
        self.use_mycaffe = use_mycaffe
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

    def forward(self, x):
        if disable_layernorm:
            return x
        if self.use_mycaffe:
            layernorm = LayerNormFunction.apply
            tag_list.append(self.tag)
            y = layernorm(x)
        else:
            mean = x.mean(dim=-1, keepdim=True)
            xmu = (x - mean)
            xmusq = xmu ** 2
            var = xmusq.mean(dim=-1, keepdim=True)
            var1 = var + self.epsilon
            std = var1.sqrt()
            y = xmu / std
        
            if self.gamma is not None:
                y *= self.gamma
            if self.beta is not None:
                y += self.beta

        return y
    
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        y = mycaffe.layernorm_fwd(tag, x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        return mycaffe.layernorm_bwd(tag, y, grad_output)


class LinearEx(nn.Module):
    def __init__(self,
                 tag,
                 in_features,
                 out_features,
                 bias=True,
                 use_mycaffe = mycaffe_innerproduct,
                 weight_init=None,
                 bias_init=None):
        super(LinearEx, self).__init__()
        self.axis = 2
        self.tag = tag
        self.in_features = in_features
        self.out_features = out_features
        self.use_mycaffe = use_mycaffe
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(weight_init, bias_init)

    def reset_parameters(self, weight_init=None, bias_init=None):
        if weight_init is None:
            weight_init = torch.nn.init.xavier_uniform_
        if bias_init is None:
            bias_init = torch.nn.init.zeros_
        weight_init(self.weight)
        if self.bias is not None:
            bias_init(self.bias)
        if self.use_mycaffe:
            mycaffe.innerproduct_setup(self.tag, self.axis, self.in_features, self.out_features, self.weight, self.bias)

    def set_parameters(self, weight, bias):
        self.weight.data = weight
        self.bias.data = bias
        if self.use_mycaffe:
            mycaffe.innerproduct_setup(self.tag, self.axis, self.in_features, self.out_features, self.weight, self.bias)
   
    def save_internal_state(self):
        DebugFunction.trace(self.weight, self.tag + ".weight")
        DebugFunction.trace(self.bias, self.tag + ".bias")

    def save_internal_stateD(self):
        DebugFunction.traceD(self.weight, self.tag + ".weight")
        DebugFunction.traceD(self.bias, self.tag + ".bias")

    def forward(self, input):
        if self.use_mycaffe:
            innerproduct = InnerproductFunction.apply
            tag_list.append(self.tag)
            return innerproduct(input)
        elif custom_innerproduct:
            return InnerproductFunctionEx.apply(input, self.weight, self.bias)
        else:            
            x = torch.matmul(input, self.weight.t())       

            debug = DebugFunction.apply
            if save_for_testing:
                DebugFunction.trace(x, self.tag + ".x1")
                x = debug(x)

            if self.bias is not None:            
                x2 = x + self.bias
                if save_for_testing:
                    DebugFunction.trace(x2, self.tag + ".x2")
                    x2 = debug(x2)
                x = x2

        return x

# See https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
class InnerproductFunctionEx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        y = torch.matmul(x, weight.t())
        if bias is not None:
            y += bias.unsqueeze(0).expand_as(y)
        ctx.save_for_backward(x, weight, bias)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            gx = torch.matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            gyt = grad_output.transpose(-1, -2)
            wtx = torch.matmul(gyt, x)
        if ctx.needs_input_grad[2]:
            gys = grad_output.squeeze(0)
            bx = gys.sum(0)
        else:
            bx = None
        return gx, wtx, bx

class InnerproductFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        y = mycaffe.innerproduct_fwd(tag, x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        return mycaffe.innerproduct_bwd(tag, y, grad_output)
    
class SoftmaxEx(nn.Module):
    def __init__(self, tag, use_mycaffe=mycaffe_softmax, dim=-1):
        super(SoftmaxEx, self).__init__()
        self.tag = tag
        self.dim = dim
        self.use_mycaffe = use_mycaffe
        
    def forward(self, x):
        if disable_softmax:
            return x

        if save_for_testing:
            debug = DebugFunction.apply
            DebugFunction.trace(x, "softmax.x")
            x = debug(x)

        if self.use_mycaffe:
            softmax = SoftmaxFunction.apply
            tag_list.append(self.tag)
            y = softmax(x)
        else:
            maxx = x.max(dim=self.dim, keepdim=True)[0]
            x = x - maxx
            expx = torch.exp(x)
            sumexpx = expx.sum(dim=self.dim, keepdim=True)
            y = expx / sumexpx
            
        if save_for_testing:
            DebugFunction.trace(y, "softmax.y")
            y = debug(y)

        return y
    
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        y = mycaffe.softmax_fwd(tag, x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        return mycaffe.softmax_bwd(tag, y, grad_output)
        
class LogSoftmaxFunctionEx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):            
        m = x.max(axis=-1, keepdims=True)[0]
        xmu = x - m
        expx = torch.exp(xmu)
        sumexp = torch.sum(expx, axis=-1, keepdims=True)   
        log_z = m + torch.log(sumexp)
        y = x - log_z
        ctx.save_for_backward(y)
        #print(y.detach().cpu().numpy())
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        gy = grad_output
        y, = ctx.saved_tensors
        sumgy = gy.sum(axis=-1, keepdims=True)
        expy = torch.exp(y)
        grad = gy - expy * sumgy
        #print("grad -> input")
        #print(grad.detach().cpu().numpy())
        return grad

class LogSoftmaxFunctionEx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):           
        tag = tag_list[-1]
        y = mycaffe.logsoftmax_fwd(tag, x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        return mycaffe.logsoftmax_bwd(tag, y, grad_output)
    

class LogSoftmaxEx(nn.Module):
    def __init__(self, tag, use_mycaffe=mycaffe_logsoftmax, dim=-1):
        super().__init__()
        self.axis = dim
        self.tag = tag
        self.use_mycaffe = use_mycaffe
        
    def forward(self, x):
        if self.use_mycaffe:
            logsoftmax = LogSoftmaxFunctionEx.apply
            tag_list.append(self.tag)
            y = logsoftmax(x)

        else:
            debug = DebugFunction.apply
        
            if save_for_testing:
                DebugFunction.trace(x, self.tag + ".logsoftmax.x")
                x = debug(x)
        
            m = x.max(axis=-1, keepdims=True)[0]
        
            if save_for_testing:
                DebugFunction.trace(m, self.tag + ".logsoftmax.m")
                m = debug(m)
            
            xmu = x - m

            if save_for_testing:
                DebugFunction.trace(xmu, self.tag + ".logsoftmax.xmu")
                xmu = debug(xmu)
        
            expx = torch.exp(xmu)

            if save_for_testing:
                DebugFunction.trace(expx, self.tag + ".logsoftmax.expx")
                expx = debug(expx)
            
            sumexp = torch.sum(expx, axis=-1, keepdims=True)   
        
            if save_for_testing:
                DebugFunction.trace(sumexp, self.tag + ".logsoftmax.sumexp")
                sumexp = debug(sumexp)
               
            logexp = torch.log(sumexp)
        
            if save_for_testing:
                DebugFunction.trace(logexp, self.tag + ".logsoftmax.logexp")
                logexp = debug(logexp)
        
            log_z = m + logexp
        
            if save_for_testing:
                DebugFunction.trace(log_z, self.tag + ".logsoftmax.log_z")
                log_z = debug(log_z)
        
            y = x - log_z

            if save_for_testing:
                DebugFunction.trace(y, self.tag + ".logsoftmax.y")
                y = debug(y)
        
        return y        
    
class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, n_embed) # (L, d_model)

        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(n_embed):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / n_embed)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / n_embed)))

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(n_embed) # (B, L, d_model)
        pos_enc = self.positional_encoding
        x = x + pos_enc # (B, L, d_model)

        return x
