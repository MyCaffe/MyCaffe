"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import os

import clr, System
clr.AddReference("C:\\temp\\projects\\2022.10.minGPT\\OptimizerLib\\OptimizerLib\\bin\\Debug\\OptimizerLib.dll")
from OptimizerLib import *
from System import Array, Single
import ctypes
from System.Runtime.InteropServices import GCHandle, GCHandleType

from mingpt.utils import CfgNode as CN

import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional

input_dict = { None : "" }
#pid = os.getpid()
#opt = CustomOptimizer(pid)

# -----------------------------------------------------------------------------
class DebugFunction(torch.autograd.Function):
    out_path = "c:\\temp\\snap\\"
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)       
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        name = input_dict.get(input)
        
        if name == None:
            name = "unknown";

        b = grad_output.detach().cpu().numpy()
        np.savetxt(DebugFunction.out_path + "grad_" + name + ".txt", b.flatten(), fmt="%s", header=str(b.shape))
        #np.save(out_path1 + "grad_" + name, b)

        #print(name, input.shape, grad_output)
        return grad_output

#
# NewGELU
#     
class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        #debug = DebugFunction.apply
        
        #b = x.detach().numpy()
        #np.savetxt(out_path1 + "1_x.txt", b.flatten(), fmt="%s", header=str(b.shape))
             
        #input_dict.update({x: "2_x"})
        #x = debug(x)        
        
        #y = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        y = self.act(x) #_CHANGE_
        
        #b = y.detach().numpy()
        #np.savetxt(out_path1 + "2_y.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        #input_dict.update({y: "1_y"})
        #y = debug(y)        
        
        return y

#
# CausalSelfAttention
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        self.debug = False # _CHANGE_
        self.out_path = "c:\\temp\\snap\\"
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = None
        if (config.attn_pdrop > 0):
            self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = None
        if (config.resid_pdrop > 0):
            self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
    def save_internal_blobs(self, path, tag):
        b = self.c_attn.bias.detach().numpy()
        np.savetxt(path + tag + "_attn_bias.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_attn.weight.detach().numpy()
        np.savetxt(path + tag + "_attn_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_proj.bias.detach().numpy()
        np.savetxt(path + tag + "_attn_proj_bias.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_proj.weight.detach().numpy()
        np.savetxt(path + tag + "_attn_proj_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))

    def save_internal_weights(self, path, tag):
        b = self.c_attn.bias.detach().numpy()
        np.save(path + tag + "_attn_bias.npy", b)
        np.savetxt(path + tag + "_attn_bias.txt", b.shape)
        
        b = self.c_attn.weight.detach().numpy()
        np.save(path + tag + "_attn_weight.npy", b)
        np.savetxt(path + tag + "_attn_weight.txt", b.shape)
        
        b = self.c_proj.bias.detach().numpy()
        np.save(path + tag + "_attn_proj_bias.npy", b)
        np.savetxt(path + tag + "_attn_proj_bias.txt", b.shape)
        
        b = self.c_proj.weight.detach().numpy()
        np.save(path + tag + "_attn_proj_weight.npy", b)
        np.savetxt(path + tag + "_attn_proj_weight.txt", b.shape)
        
    def save_internal_grad(self, path, tag):
        b = self.c_attn.bias.grad.detach().numpy()
        np.savetxt(path + tag + "_grad_attn_bias.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_attn.weight.grad.detach().numpy()
        np.savetxt(path + tag + "_grad_attn_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_proj.bias.grad.detach().numpy()
        np.savetxt(path + tag + "_grad_attn_proj_bias.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_proj.weight.grad.detach().numpy()
        np.savetxt(path + tag + "_grad_attn_proj_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))        

    def forward(self, x):
        if self.debug:
            debug = DebugFunction.apply
            path = self.out_path + "\\iter_0\\"
            DebugFunction.out_path = path
            if not os.path.exists(path):
                os.makedirs(path)
                        
            input_dict.update({x: "x"})
            x = debug(x)

            b = x.detach().numpy()
            np.savetxt(self.out_path + "x.txt", b.flatten(), fmt="%s", header=str(b.shape))
            
            self.save_internal_blobs(self.out_path)

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        ca = self.c_attn(x)
        
        if self.debug:
            input_dict.update({ca: "ca"})
            ca = debug(ca)
        
        q, k ,v  = ca.split(self.n_embd, dim=2)

        if self.debug:
            input_dict.update({q: "q"})
            q = debug(q)
            input_dict.update({k: "k"})
            k = debug(k)
            input_dict.update({v: "v"})
            v = debug(v)
        
        val = C // self.n_head

        k = k.view(B, T, self.n_head, C // self.n_head) # (B, nh, T, hs)
        kt = k.transpose(1,2)

        if self.debug:
            input_dict.update({kt: "kt"})
            kt = debug(kt)        

        q = q.view(B, T, self.n_head, C // self.n_head)
        qt = q.transpose(1, 2) # (B, nh, T, hs)
        
        if self.debug:
            input_dict.update({qt: "qt"})
            qt = debug(qt)        
                
        v = v.view(B, T, self.n_head, C // self.n_head)
        vt = v.transpose(1, 2) # (B, nh, T, hs)
        
        if self.debug:
            input_dict.update({vt: "vt"})
            vt = debug(vt)        

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        kt1 =  kt.transpose(-2, -1)

        if self.debug:
            input_dict.update({kt1: "kt1"})
            kt1 = debug(kt1)        

            input_dict.update({qt: "qt"})
            qt = debug(qt)        
                
        qkt = qt @ kt1
        
        if self.debug:
            b = qt.detach().numpy()
            np.savetxt(self.out_path + "qt.txt", b.flatten(), fmt="%s", header=str(b.shape))

            input_dict.update({qkt: "qkt"})
            qkt = debug(qkt)        
        
        ksiz = kt.size(-1)
        scale = (1.0 / math.sqrt(ksiz))      
        att = qkt * scale

        if self.debug:
            input_dict.update({att: "att_after_scale"})
            att = debug(att)        
        
        bias1 = self.bias[:,:,:T,:T]
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        if self.debug:
            input_dict.update({att: "att_after_mask"})
            att = debug(att)        
        
        att = F.softmax(att, dim=-1)

        if self.debug:
            input_dict.update({att: "att_after_softmax"})
            att = debug(att)        
        
        if (self.attn_dropout != None):
            att = self.attn_dropout(att)

        if self.debug:
            b = att.detach().numpy()
            np.save(self.out_path + "var_att", b);
            b = vt.detach().numpy()
            np.save(self.out_path + "var_vt", b);
        
        y = att @ vt # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
       
        if self.debug:
            b = y.detach().numpy()
            np.save(self.out_path + "var_y", b);

            input_dict.update({y: "y_pre_transpose"})
            y = debug(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        if self.debug:
            input_dict.update({y: "y_post_transpose"})
            y = debug(y)

        # output projection
        y = self.c_proj(y);

        if (self.resid_dropout != None):
            y = self.resid_dropout(y)

        if self.debug:
            input_dict.update({y: "y"})
            y = debug(y)

            b = y.detach().numpy()
            np.savetxt(self.out_path + "y.txt", b.flatten(), fmt="%s", header=str(b.shape))

        return y
 
#
# LayerNormalization
#     
class LayerNormalization(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=False,
                 beta=False,
                 epsilon=1e-10):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNormalization, self).__init__()
        self.debug = False # _CHANGE_
        self.out_path = "c:\\temp\\snap\\"
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
        if self.debug:
            debug = DebugFunction.apply
        
            input_dict.update({x: "9_x"})
            x = debug(x)        
        
            b = x.detach().numpy()
            np.savetxt(self.out_path + "x.txt", b.flatten(), fmt="%s", header=str(b.shape))

        mean = x.mean(dim=-1, keepdim=True)
        
        if self.debug:
            b = mean.detach().numpy()
            np.savetxt(self.out_path + "mean.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x: "8_x"})
            x = debug(x)        
            input_dict.update({mean: "8_mean"})
            mean = debug(mean)        

        xmu = (x - mean)

        if self.debug:
            input_dict.update({xmu: "7_xmu"})
            xmu = debug(xmu)        
        
            b = xmu.detach().numpy()
            np.savetxt(self.out_path + "xmu.txt", b.flatten(), fmt="%s", header=str(b.shape))        
               
        xmusq = xmu ** 2
        
        if self.debug:
            b = xmusq.detach().numpy()
            np.savetxt(self.out_path + "xmusq.txt", b.flatten(), fmt="%s", header=str(b.shape))        
        
            input_dict.update({xmusq: "6_xmusq"})
            xmusq = debug(xmusq)        
        
        var = xmusq.mean(dim=-1, keepdim=True)
        
        if self.debug:
            input_dict.update({var: "5_var"})
            var = debug(var)        
        
        var1 = var + self.epsilon

        if self.debug:
            input_dict.update({var1: "4_var1"})
            var1 = debug(var1)        

            b = var1.detach().numpy()
            np.savetxt(self.out_path + "var1.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        std = var1.sqrt()
         
        if self.debug:
            b = std.detach().numpy()
            np.savetxt(self.out_path + "std.txt", b.flatten(), fmt="%s", header=str(b.shape))        
        
            input_dict.update({std: "1c_std"})
            std = debug(std)                
            input_dict.update({xmu: "1b_xmu"})
            xmu = debug(xmu)        
             
        y = xmu / std
        
        if self.debug:
            input_dict.update({y: "1a_y"})
            y = debug(y)        
        
            b = y.detach().numpy()
            np.savetxt(self.out_path + "y.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta

        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.weight is not None, self.bias is not None, self.epsilon,
        )

#
# LayerNormalization2 - duplicate of LayerNormalization but used to generate second LN data.
#
class LayerNormalization2(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=False,
                 beta=False,
                 epsilon=1e-10):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNormalization2, self).__init__()
        self.debug = False # _CHANGE_
        self.out_path = "c:\\temp\\snap\\"
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
        if self.debug:
            debug = DebugFunction.apply
        
            input_dict.update({x: "9_x"})
            x = debug(x)        
        
            b = x.detach().numpy()
            np.savetxt(self.out_path + "x.txt", b.flatten(), fmt="%s", header=str(b.shape))

        mean = x.mean(dim=-1, keepdim=True)
        
        if self.debug:
            b = mean.detach().numpy()
            np.savetxt(self.out_path + "mean.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x: "8_x"})
            x = debug(x)        
            input_dict.update({mean: "8_mean"})
            mean = debug(mean)        

        xmu = (x - mean)

        if self.debug:
            input_dict.update({xmu: "7_xmu"})
            xmu = debug(xmu)        
        
            b = xmu.detach().numpy()
            np.savetxt(self.out_path + "xmu.txt", b.flatten(), fmt="%s", header=str(b.shape))        
               
        xmusq = xmu ** 2
        
        if self.debug:
            b = xmusq.detach().numpy()
            np.savetxt(self.out_path + "xmusq.txt", b.flatten(), fmt="%s", header=str(b.shape))        
        
            input_dict.update({xmusq: "6_xmusq"})
            xmusq = debug(xmusq)        
        
        var = xmusq.mean(dim=-1, keepdim=True)
        
        if self.debug:
            input_dict.update({var: "5_var"})
            var = debug(var)        
        
        var1 = var + self.epsilon

        if self.debug:
            input_dict.update({var1: "4_var1"})
            var1 = debug(var1)        

            b = var1.detach().numpy()
            np.savetxt(self.out_path + "var1.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        std = var1.sqrt()
         
        if self.debug:
            b = std.detach().numpy()
            np.savetxt(self.out_path + "std.txt", b.flatten(), fmt="%s", header=str(b.shape))        
        
            input_dict.update({std: "1c_std"})
            std = debug(std)                
            input_dict.update({xmu: "1b_xmu"})
            xmu = debug(xmu)        
             
        y = xmu / std
        
        if self.debug:
            input_dict.update({y: "1a_y"})
            y = debug(y)        
        
            b = y.detach().numpy()
            np.savetxt(self.out_path + "y.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta

        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.weight is not None, self.bias is not None, self.epsilon,
        )

#
# Transformer Block
#     
class Block(nn.Module):
    """ an unassuming Transformer block """
    
    def __init__(self, config):
        super().__init__()
        self.debug = False # _CHANGE_
        self.out_path = "c:\\temp\\snap\\"
        self.ln_1 = LayerNormalization(config.n_embd) #nn.LayerNorm(config.n_embd) _CHANGE_
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNormalization2(config.n_embd) #nn.LayerNorm(config.n_embd) _CHANGE_
        #self.mlp = nn.ModuleDict(dict(
        #    c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
        #    c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
        #    act     = NewGELU(),
        #    dropout = nn.Dropout(config.resid_pdrop),
        #))
        #m = self.mlp
        #self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = NewGELU() 
        if config.resid_pdrop > 0:
            self.dropout = nn.Dropout(config.resid_pdrop)
        else:
            self.dropout = None
                    
    def save_internal_blobs(self, path, tag):
        self.attn.save_internal_blobs(path, tag)
        b = self.c_fc.bias.detach().numpy()
        np.savetxt(path + tag + "_fc_bias.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_fc.weight.detach().numpy()
        np.savetxt(path + tag + "_fc_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_proj.bias.detach().numpy()
        np.savetxt(path + tag + "_proj_bias.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_proj.weight.detach().numpy()
        np.savetxt(path + tag + "_proj_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
    def save_internal_weights(self, path, tag):
        self.attn.save_internal_weights(path, tag)
        b = self.c_fc.bias.detach().numpy()
        np.save(path + tag + "_fc_bias.npy", b)
        np.savetxt(path + tag + "_fc_bias.txt", b.shape)
        
        b = self.c_fc.weight.detach().numpy()
        np.save(path + tag + "_fc_weight.npy", b)
        np.savetxt(path + tag + "_fc_weight.txt", b.shape)
        
        b = self.c_proj.bias.detach().numpy()
        np.save(path + tag + "_proj_bias.npy", b)
        np.savetxt(path + tag + "_proj_bias.txt", b.shape)
        
        b = self.c_proj.weight.detach().numpy()
        np.save(path + tag + "_proj_weight.npy", b)
        np.savetxt(path + tag + "_proj_weight.txt", b.shape)
    
    def save_internal_grad(self, path, tag):
        self.attn.save_internal_grad(path, tag)
        b = self.c_fc.bias.grad.detach().numpy()
        np.savetxt(path + tag + "_grad_fc_bias.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_fc.weight.grad.detach().numpy()
        np.savetxt(path + tag + "_grad_fc_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_proj.bias.grad.detach().numpy()
        np.savetxt(path + tag + "_grad_proj_bias.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        b = self.c_proj.weight.grad.detach().numpy()
        np.savetxt(path + tag + "_grad_proj_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))

        
    def forward(self, x):
        if self.debug:
            debug = DebugFunction.apply
        
            b = x.detach().numpy()
            np.savetxt(self.out_path + "1_x.txt", b.flatten(), fmt="%s", header=str(b.shape))
            self.save_internal_blobs(self.out_path)
        
            input_dict.update({x: "10_x"})
            x = debug(x)        
                
        # x = x + self.attn(self.ln_1(x))
        x_ln1 = self.ln_1(x)
        
        if self.debug:
            b = x_ln1.detach().numpy()
            np.savetxt(self.out_path + "2_x_ln11.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x_ln1: "9_ln1"})
            x_ln1 = debug(x_ln1)        
        
        x_attn = self.attn(x_ln1)

        if self.debug:
            b = x_attn.detach().numpy()
            np.savetxt(self.out_path + "3_x_attn.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x_attn: "8_x_attn"})
            x_attn = debug(x_attn)        
            input_dict.update({x: "8_x"})
            x = debug(x)        
                
        x_p_attn = x + x_attn
        
        if self.debug:
            b = x_p_attn.detach().numpy()
            np.savetxt(self.out_path + "4_x_p_attn.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x_p_attn: "7_p_attn"})
            x_p_attn = debug(x_p_attn)        
            
        # x = x + self.mlpf(self.ln_2(x))
        x_ln2 = self.ln_2(x_p_attn)
        
        if self.debug:
            b = x_ln2.detach().numpy()
            np.savetxt(self.out_path + "5_x_ln2.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x_ln2: "6_x_ln2"})
            x_ln2 = debug(x_ln2)        
        
        #x5 = self.mlpf(x4)
        x_c_fc = self.c_fc(x_ln2)
        
        if self.debug:
            b = x_c_fc.detach().numpy()
            np.savetxt(self.out_path + "6_x_c_fc.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x_c_fc: "5_x_c_fc"})
            x_c_fc = debug(x_c_fc)        
        
        x_act = self.act(x_c_fc)
        
        if self.debug:
            b = x_act.detach().numpy()
            np.savetxt(self.out_path + "7_x_act.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x_act: "4_x_act"})
            x_act = debug(x_act)        
        
        x_c_proj = self.c_proj(x_act)

        if self.debug:
            b = x_c_proj.detach().numpy()
            np.savetxt(self.out_path + "8_x_c_proj.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x_c_proj: "3_x_c_proj"})
            x_c_proj = debug(x_c_proj)        
                
        if self.dropout is not None:
            x_dropout = self.dropout(x_c_proj)
        else:
            x_dropout = x_c_proj

        if self.debug:
            b = x_dropout.detach().numpy()
            np.savetxt(self.out_path + "9_x_dropout.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({x_p_attn: "2_x_p_attn"})
            x_p_attn = debug(x_p_attn)        
            input_dict.update({x_dropout: "2_x_dropout"})
            x_dropout = debug(x_dropout)        
        
        y = x_p_attn + x_dropout
        
        if self.debug:
            input_dict.update({y: "1_y"})
            y = debug(y)        

            b = y.detach().numpy()
            np.savetxt(self.out_path + "10_y.txt", b.flatten(), fmt="%s", header=str(b.shape))
                
        return y

#
# Custom AdamW2 - customized AdamW optimizer originally from the PyTorch Source on Github,
# distributed under the PyTorch license at https://github.com/pytorch/pytorch/blob/master/LICENSE
# This altered version of the Adam2 optimizer allows for using the C# AdamW optimizer implemented
# in the OptimizerLib project.
#
class AdamW2(Optimizer):
    r"""Implements AdamW algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        capturable (bool, optional): whether this instance is safe to capture in a CUDA graph.
            Passing True can impair ungraphed performance, so if you don't intend to
            graph capture this instance, leave it False (default: False)

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable)
        super(AdamW2, self).__init__(params, defaults) 

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

            adamw(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  state_steps,
                  amsgrad=amsgrad,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'],
                  capturable=group['capturable'])

        return loss


def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[Tensor],
          # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
          # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
          foreach: bool = None,
          capturable: bool = False,
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable)


def _single_tensor_adamw(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         amsgrad: bool,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = step_t.item()

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                             
            #param2 = param.clone()
            #param2.addcdiv_(exp_avg, denom, value=-step_size)
                
            #nexpected = param.numpy().flatten()                
            nexp_avg = exp_avg.numpy().flatten()
            ndenom = denom.numpy().flatten()
            nparam = param.numpy().flatten()
                        
            nN = param.shape[0];
            nC = 1
            if len(param.shape) > 1:
                nC = param.shape[1]

            rgParam = Array[Single](nparam)
            rgExpAvg = Array[Single](nexp_avg)
            rgDenom = Array[Single](ndenom)
            #rgExpected = Array[Single](nexpected)

            rgOut = opt.addcdiv(nN, nC, rgParam, rgExpAvg, rgDenom, -step_size, None)
            outData = asNumpyArray(rgOut)
            temp = torch.from_numpy(outData)
            temp.detach();
            temp = temp.view(param.shape)
            param.copy_(temp)

#
# Simplified portion of AdamW used
#
def _single_tensor_adamw_OriginalSimple(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         amsgrad: bool,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = step_t.item()

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = math.sqrt(bias_correction2)

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                         
        param.addcdiv_(exp_avg, denom, value=-step_size)
            
#
# Requires OptimilerLib C# Project for custom AdamW in C#    
#
def _single_tensor_adamw_Custom(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_sqs: List[Tensor],
                         max_exp_avg_sqs: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         amsgrad: bool,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float,
                         eps: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        opt.update_step(lr, weight_decay, beta1, beta2, step_t, eps)

        rgW = Array[Single](param.numpy().flatten())
        rgG = Array[Single](grad.numpy().flatten())
        rgM = Array[Single](exp_avg.numpy().flatten())
        rgV = Array[Single](exp_avg_sq.numpy().flatten())            
        rgW = opt.step(rgW, rgG, rgM, rgV)     
        
        w = asNumpyArray(rgW)
        tensorW = torch.from_numpy(w)
        tensorW.detach();
        tensorW = tensorW.view(param.shape)
        param.copy_(tensorW)
        
        m = asNumpyArray(opt.m)
        tensorM = torch.from_numpy(m)
        tensorM.detach()
        tensorM = tensorM.view(exp_avg.shape)
        exp_avg.copy_(tensorM)
        
        v = asNumpyArray(opt.v)
        tensorV = torch.from_numpy(v)
        tensorV.detach()
        tensorV = tensorV.view(exp_avg_sq.shape)
        exp_avg_sq.copy_(tensorV)
        

_MAP_NP_NET = {
    np.dtype('float32'): System.Single,
    np.dtype('float64'): System.Double,
    np.dtype('int8')   : System.SByte,
    np.dtype('int16')  : System.Int16,
    np.dtype('int32')  : System.Int32,
    np.dtype('int64')  : System.Int64,
    np.dtype('uint8')  : System.Byte,
    np.dtype('uint16') : System.UInt16,
    np.dtype('uint32') : System.UInt32,
    np.dtype('uint64') : System.UInt64,
    np.dtype('bool')   : System.Boolean,
}
_MAP_NET_NP = {
    'Single' : np.dtype('float32'),
    'Double' : np.dtype('float64'),
    'SByte'  : np.dtype('int8'),
    'Int16'  : np.dtype('int16'), 
    'Int32'  : np.dtype('int32'),
    'Int64'  : np.dtype('int64'),
    'Byte'   : np.dtype('uint8'),
    'UInt16' : np.dtype('uint16'),
    'UInt32' : np.dtype('uint32'),
    'UInt64' : np.dtype('uint64'),
    'Boolean': np.dtype('bool'),
}

def asNumpyArray(netArray):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

    try: # Memmove 
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__['data'][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated: sourceHandle.Free()
    return npArray

def _multi_tensor_adamw(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool):
    if len(params) == 0:
        return

    if capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]
    exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avg_sqs]
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]

    # update steps
    torch._foreach_add_(state_steps, 1)

    # Perform stepweight decay
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)

    if capturable:
        # TODO: use foreach_pow if/when foreach_pow is added
        bias_correction1 = [torch.pow(beta1, step) for step in state_steps]
        bias_correction2 = [torch.pow(beta2, step) for step in state_steps]
        # foreach_sub doesn't allow a scalar as the first arg
        torch._foreach_sub_(bias_correction1, 1)
        torch._foreach_sub_(bias_correction2, 1)
        torch._foreach_neg_(bias_correction1)
        torch._foreach_neg_(bias_correction2)

        # foreach_div doesn't allow a scalar as the first arg
        step_size = torch._foreach_div(bias_correction1, lr)
        torch._foreach_reciprocal_(step_size)
        torch._foreach_neg_(step_size)

        bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
            torch._foreach_div_(max_exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps_over_step_size)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, torch._foreach_mul(bias_correction2_sqrt, step_size))
            eps_over_step_size = torch._foreach_div(step_size, eps)
            torch._foreach_reciprocal_(eps_over_step_size)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)

        torch._foreach_addcdiv_(params, exp_avgs, denom)
    else:
        bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
        bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]

        step_size = [(lr / bc) * -1 for bc in bias_correction1]

        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)

            # Use the max. for normalizing running avg. of gradient
            max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
        else:
            exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

        torch._foreach_addcdiv_(params, exp_avgs, denom, step_size)

# 
# GPT
#     
class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.0 # 0.1 _CHANGED_
        C.resid_pdrop = 0.0 # 0.1 _CHANGED_
        C.attn_pdrop = 0.0 # 0.1 _CHANGED_
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.config = config
        self.debug = False # _CHANGE_
        self.iter_num = 0
        self.out_path = "c:\\temp\\snap\\"
        torch.set_printoptions(precision=10)

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-mini1':     dict(n_layer=2, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
                # testing only
                'gpt-pico':     dict(n_layer=1, n_head=1, n_embd=3),
                'gpt-pico3':     dict(n_layer=1, n_head=3, n_embd=3),
                'gpt-picoB':     dict(n_layer=1, n_head=1, n_embd=3),
                'gpt-pico3B':     dict(n_layer=1, n_head=3, n_embd=3),
                'gpt-pico3B5':     dict(n_layer=1, n_head=3, n_embd=3)
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNormalization(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_next_index(self, max_idx):
        return opt.get_next_index(max_idx)

    def set_iter(self, iter_num, force):
        self.iter_num = iter_num
        self.out_path = "c:\\temp\\snap\\iter_%d\\" % iter_num
        DebugFunction.out_path = self.out_path
        if force or self.debug or self.transformer.h[0].debug:
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
            if not os.path.exists(self.out_path + "\\after_step"):
                os.makedirs(self.out_path + "\\after_step")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias') and not k.endswith('ln_1.weight') and not k.endswith('ln_1.bias') and not k.endswith('ln_2.weight') and not k.endswith('ln_2.bias') and not k.endswith('ln_f.weight') and not k.endswith('ln_f.bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                k1 = k
                if ".mlp." in k:
                    k1 = k.replace(".mlp.", ".")
                assert sd_hf[k].shape[::-1] == sd[k1].shape
                with torch.no_grad():
                    sd[k1].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                k1 = k
                if ".mlp." in k:
                    k1 = k.replace(".mlp.", ".")
                assert sd_hf[k].shape == sd[k1].shape
                with torch.no_grad():
                    sd[k1].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        sdecay = sorted(list(decay))
        sno_decay = sorted(list(no_decay))

        # create the pytorch optimizer object
        optim_groups = [ 
            {"params": [param_dict[pn] for pn in sdecay], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sno_decay], "weight_decay": 0.0},
        ]
        #optimizer = AdamW2(optim_groups, lr=train_config.learning_rate, betas=train_config.betas) #_CHANGE_
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas) #_CHANGE_
        #optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate, momentum=0.9) 
        return optimizer
    
    def save_internal_blobs(self):                    
        b = self.transformer.wte.weight.detach().numpy()
        np.savetxt(self.out_path + "gpt_wte_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
                
        b = self.transformer.wpe.weight.detach().numpy()
        np.savetxt(self.out_path + "gpt_wpe_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
        tfb = 1
        for block in self.transformer.h:
            block.save_internal_blobs(self.out_path, "tfb_" + str(tfb))
            tfb += 1

        b = self.lm_head.weight.detach().numpy()
        np.savetxt(self.out_path + "gpt_lm_head_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
    def save_internal_weights(self):                    
        b = self.transformer.wte.weight.detach().numpy()
        np.save(self.out_path + "gpt_wte_weight.npy", b)
        np.savetxt(self.out_path + "gpt_wte_weight.txt", b.shape)
                
        b = self.transformer.wpe.weight.detach().numpy()
        np.save(self.out_path + "gpt_wpe_weight.npy", b)
        np.savetxt(self.out_path + "gpt_wpe_weight.txt", b.shape)
        
        tfb = 1
        for block in self.transformer.h:
            block.save_internal_weights(self.out_path, "tfb_" + str(tfb))
            tfb += 1

        b = self.lm_head.weight.detach().numpy()
        np.save(self.out_path + "gpt_lm_head_weight.npy", b)
        np.savetxt(self.out_path + "gpt_lm_head_weight.txt", b.shape)

        
    def save_internal_grad(self, extrapath = None):
        path = self.out_path
        if extrapath != None:
            path += extrapath + "\\"
            
        b = self.transformer.wte.weight.grad.detach().numpy()
        np.savetxt(path + "grad_gpt_wte_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
                
        b = self.transformer.wpe.weight.grad.detach().numpy()
        np.savetxt(path + "grad_gpt_wpe_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))

        tfb = 1
        for block in self.transformer.h:
            block.save_internal_grad(path, "tfb_" + str(tfb))
            tfb += 1
        
        b = self.lm_head.weight.grad.detach().numpy()
        np.savetxt(path + "grad_gpt_lm_head_weight.txt", b.flatten(), fmt="%s", header=str(b.shape))
                    
    def forward(self, idx, targets=None):        
        if self.debug:
            debug = DebugFunction.apply
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        if self.debug:
            self.save_internal_blobs()
            b = idx.detach().numpy()
            np.savetxt(self.out_path + "1_idx.txt", b.flatten(), fmt="%s", header=str(b.shape))
            b = pos.detach().numpy()
            np.savetxt(self.out_path + "1_pos.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({idx: "7_idx"})
            idx = debug(idx)                
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if self.debug:
            b = tok_emb.detach().numpy()
            np.savetxt(self.out_path + "2_tok_emb.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({pos: "7_pos"})
            pos = debug(pos)                
        
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)

        if self.debug:
            b = pos_emb.detach().numpy()
            np.savetxt(self.out_path + "3_pos_emb.txt", b.flatten(), fmt="%s", header=str(b.shape))
        
            input_dict.update({pos_emb: "6_pos_emb"})
            pos_emb = debug(pos_emb)                
            input_dict.update({tok_emb: "6_tok_emb"})
            tok_emb = debug(tok_emb)        
        
        if self.config.embd_pdrop > 0:
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = tok_emb + pos_emb
        
        if self.debug:
            b = x.detach().numpy()
            np.savetxt(self.out_path + "4_x.txt", b.flatten(), fmt="%s", header=str(b.shape))        

            input_dict.update({x: "5_x"})
            x = debug(x)        
        
        tfb = 1
        for block in self.transformer.h:
            x = block(x)

            if self.debug:
                b = x.detach().numpy()
                filename = self.out_path + "5_x_tfb_" + str(tfb) + ".txt"
                np.savetxt(filename, b.flatten(), fmt="%s", header=str(b.shape))        

                input_dict.update({x: "4_x_tfb_" + str(tfb)})
                x = debug(x)
            tfb += 1
            
        x = self.transformer.ln_f(x)

        if self.debug:
            b = x.detach().numpy()
            np.savetxt(self.out_path + "6_x.txt", b.flatten(), fmt="%s", header=str(b.shape))        
        
            input_dict.update({x: "3_x"})
            x = debug(x)        
        
        logits = self.lm_head(x)
        
        if self.debug:
            b = logits.detach().numpy()
            np.savetxt(self.out_path + "7_logits.txt", b.flatten(), fmt="%s", header=str(b.shape))        
        
            b = targets.detach().numpy()
            np.savetxt(self.out_path + "7_targets.txt", b.flatten(), fmt="%s", header=str(b.shape))        

            input_dict.update({logits: "2_logits"})
            logits = debug(logits)        
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            logits1 = logits.view(-1, logits.size(-1))
            targets1 = targets.view(-1)
            loss = F.cross_entropy(logits1, targets1, ignore_index=-1)
    
        if self.debug:    
            b = loss.detach().numpy()
            np.savetxt(self.out_path + "7_loss.txt", b.flatten(), fmt="%s", header=str(b.shape))        
            
            input_dict.update({loss: "1_loss"})
            loss = debug(loss)        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
