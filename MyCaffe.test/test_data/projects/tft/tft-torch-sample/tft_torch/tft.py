import copy
import math
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
from tft_torch.base_blocks import TimeDistributed, NullTransform
from utility import DebugFunction
from tft_torch.mycaffe import MyCaffe

mycaffe = MyCaffe(False)
#full_mycaffe = MyCaffe(True)

# Softmax
#
tag_list = []
smx_axis = []
ip_axis = []
ip_bias = []
ip_numout = []
lstm_num = []
lstm_state = []
lstm_ctx = {}

last_y = None
last_x_grad = None

# Set the devie to CUDA if available
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

#
# @see [Parameters in Tensorflow Keras RNN and CUDNN RNN](https://kaixih.github.io/keras-cudnn-rnn/) by Kaixi Hou, 2021
# @see [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) by PyTorch
#
def get_cudnn_lstm_weights(lstm):
    num_layers = lstm.num_layers
    hidden_size = lstm.hidden_size
    wts = []
    for param in lstm.state_dict():
        param_val = lstm.state_dict()[param]
        wts.append(param_val)
    
    all_wts = []
    idx = 0
    for i in range(0, num_layers):
        wtii = wts[idx][:hidden_size,:]
        wtif = wts[idx][hidden_size:hidden_size*2,:]
        wtig = wts[idx][hidden_size*2:hidden_size*3,:]
        wtio = wts[idx][hidden_size*3:hidden_size*4,:]
        idx = idx + 1

        wthi = wts[idx][:hidden_size,:]
        wthf = wts[idx][hidden_size:hidden_size*2,:]
        wthg = wts[idx][hidden_size*2:hidden_size*3,:]
        wtho = wts[idx][hidden_size*3:hidden_size*4,:]
        idx = idx + 1

        bii = wts[idx][:hidden_size]
        bif = wts[idx][hidden_size:hidden_size*2]
        big = wts[idx][hidden_size*2:hidden_size*3]
        bio = wts[idx][hidden_size*3:hidden_size*4]
        idx = idx + 1
    
        bhi = wts[idx][:hidden_size]
        bhf = wts[idx][hidden_size:hidden_size*2]
        bhg = wts[idx][hidden_size*2:hidden_size*3]
        bho = wts[idx][hidden_size*3:hidden_size*4]
        idx = idx + 1

        wts1 = [wtii, wtif, wtio, wtig, wthi, wthf, wtho, wthg]
        b1 = [bii, bif, bio, big, bhi, bhf, bho, bhg]
    
        shape = [-1]
        weights = [torch.reshape(torch.transpose(x, 0, 1), shape) for x in wts1]
        biases = [torch.reshape(x, shape) for x in b1]
        cudnnwts = torch.concat(weights + biases, axis=0)
        all_wts.append(cudnnwts)
    
    full_cudnnwts = torch.concat(all_wts, axis=0) if num_layers > 1 else all_wts[0]
    return full_cudnnwts

class MyCaffeModel(nn.Module):
    def __init__(self, tag, path=""):
        super(MyCaffeModel, self).__init__()
        self.tag = tag
        self.path = path

    def forward(self, x1, t1):
        modelFn = ModelFunction.apply
        return modelFn(x1, t1)

    def forward2(self, x1, x2, t1):
        modelFn = ModelFunction2.apply
        return modelFn(x1, x2, t1)

    def forward3(self, x1, x2, x3, t1):
        modelFn = ModelFunction3.apply
        return modelFn(x1, x2, x3, t1)

    def forward4(self, x1, x2, x3, x4, t1):
        modelFn = ModelFunction4.apply
        return modelFn(x1, x2, x3, x4, t1)

    def forward5(self, x1, x2, x3, x4, x5, t1):
        modelFn = ModelFunction5.apply
        return modelFn(x1, x2, x3, x4, x5, t1)

    def forward6(self, x1, x2, x3, x4, x5, x6, t1):
        modelFn = ModelFunction6.apply
        return modelFn(x1, x2, x3, x4, x5, x6, t1)

    def forward7(self, x1, x2, x3, x4, x5, x6, x7, t1):
        modelFn = ModelFunction7.apply
        return modelFn(x1, x2, x3, x4, x5, x6, x7, t1)

    def forward_direct_full(self):
        res = full_mycaffe.model_fwd_full()
        return res

    def forward_direct(self, s1, s2, h1, h2, f1, f2, trg):
        return full_mycaffe.model_fwd(s1, s2, h1, h2, f1, f2, trg)

    def backward_direct(self, y):
        return full_mycaffe.model_bwd(y)

    def update(self, nIter):
        full_mycaffe.model_update(nIter)

class ModelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, t1):
        res = full_mycaffe.model_fwd(x1, t1)
        ctx.save_for_backward(x1, t1)              
        y = torch.from_numpy(res[0]).to(device)
        y.requires_grad = True
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, t1 = ctx.saved_tensors
        res = full_mycaffe.model_bwd(grad_output)

        x1_grad = t1_grad = None
        if ctx.needs_input_grad[0]:
            x1_grad = torch.from_numpy(res[0]).to(device)
        if ctx.needs_input_grad[1]:
            t1_grad = torch.from_numpy(res[1]).to(device)
        return x1_grad, t1_grad

class ModelFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, t1):
        res = full_mycaffe.model_fwd2(x1, x2, t1)
        ctx.save_for_backward(x1, x2, t1)              
        y = torch.from_numpy(res[0]).to(device)
        y.requires_grad = True
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, t1 = ctx.saved_tensors
        res = full_mycaffe.model_bwd(grad_output)

        x1_grad = x2_grad = t1_grad = None
        if ctx.needs_input_grad[0]:
            x1_grad = torch.from_numpy(res[0]).to(device)
        if ctx.needs_input_grad[1]:
            x2_grad = torch.from_numpy(res[1]).to(device)
        if ctx.needs_input_grad[2]:
            t1_grad = torch.from_numpy(res[2]).to(device)
        return x1_grad, x2_grad, t1_grad

class ModelFunction3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, x3, t1):
        res = full_mycaffe.model_fwd3(x1, x2, x3, t1)
        ctx.save_for_backward(x1, x2, x3, t1)              
        y = torch.from_numpy(res[0]).to(device)
        y.requires_grad = True
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, x3, t1 = ctx.saved_tensors
        res = full_mycaffe.model_bwd(grad_output)

        x1_grad = x2_grad = x3_grad = t1_grad = None
        if ctx.needs_input_grad[0]:
            x1_grad = torch.from_numpy(res[0]).to(device)
        if ctx.needs_input_grad[1]:
            x2_grad = torch.from_numpy(res[1]).to(device)
        if ctx.needs_input_grad[2]:
            x3_grad = torch.from_numpy(res[2]).to(device)
        if ctx.needs_input_grad[3]:
            t1_grad = torch.from_numpy(res[3]).to(device)
        return x1_grad, x2_grad, x3_grad, t1_grad

class ModelFunction4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, x3, x4, t1):
        res = full_mycaffe.model_fwd4(x1, x2, x3, x4, t1)
        ctx.save_for_backward(x1, x2, x3, x4, t1)              
        y = torch.from_numpy(res[0]).to(device)
        y.requires_grad = True
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, x3, x4, t1 = ctx.saved_tensors
        res = full_mycaffe.model_bwd(grad_output)

        x1_grad = x2_grad = x3_grad = x4_grad = t1_grad = None
        if ctx.needs_input_grad[0]:
            x1_grad = torch.from_numpy(res[0]).to(device)
        if ctx.needs_input_grad[1]:
            x2_grad = torch.from_numpy(res[1]).to(device)
        if ctx.needs_input_grad[2]:
            x3_grad = torch.from_numpy(res[2]).to(device)
        if ctx.needs_input_grad[3]:
            x4_grad = torch.from_numpy(res[3]).to(device)
        if ctx.needs_input_grad[4]:
            t1_grad = torch.from_numpy(res[4]).to(device)
        return x1_grad, x2_grad, x3_grad, x4_grad, t1_grad

class ModelFunction5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, x3, x4, x5, t1):
        res = full_mycaffe.model_fwd5(x1, x2, x3, x4, x5, t1)
        ctx.save_for_backward(x1, x2, x3, x4, x5, t1)              
        y = torch.from_numpy(res[0]).to(device)
        y.requires_grad = True
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, x3, x4, x5, t1 = ctx.saved_tensors
        res = full_mycaffe.model_bwd(grad_output)

        x1_grad = x2_grad = x3_grad = x4_grad = x5_grad = t1_grad = None
        if ctx.needs_input_grad[0]:
            x1_grad = torch.from_numpy(res[0]).to(device)
        if ctx.needs_input_grad[1]:
            x2_grad = torch.from_numpy(res[1]).to(device)
        if ctx.needs_input_grad[2]:
            x3_grad = torch.from_numpy(res[2]).to(device)
        if ctx.needs_input_grad[3]:
            x4_grad = torch.from_numpy(res[3]).to(device)
        if ctx.needs_input_grad[4]:
            x5_grad = torch.from_numpy(res[4]).to(device)
        if ctx.needs_input_grad[5]:
            t1_grad = torch.from_numpy(res[5]).to(device)
        return x1_grad, x2_grad, x3_grad, x4_grad, x5_grad, t1_grad

class ModelFunction6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, x3, x4, x5, x6, t1):
        res = full_mycaffe.model_fwd6(x1, x2, x3, x4, x5, x6, t1)
        ctx.save_for_backward(x1, x2, x3, x4, x5, x6, t1)              
        y = torch.from_numpy(res[0]).to(device)
        y.requires_grad = True
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, x3, x4, x5, x6, t1 = ctx.saved_tensors
        res = full_mycaffe.model_bwd(grad_output)

        x1_grad = x2_grad = x3_grad = x4_grad = x5_grad = x6_grad = t1_grad = None
        if ctx.needs_input_grad[0]:
            x1_grad = torch.from_numpy(res[0]).to(device)
        if ctx.needs_input_grad[1]:
            x2_grad = torch.from_numpy(res[1]).to(device)
        if ctx.needs_input_grad[2]:
            x3_grad = torch.from_numpy(res[2]).to(device)
        if ctx.needs_input_grad[3]:
            x4_grad = torch.from_numpy(res[3]).to(device)
        if ctx.needs_input_grad[4]:
            x5_grad = torch.from_numpy(res[4]).to(device)
        if ctx.needs_input_grad[5]:
            x6_grad = torch.from_numpy(res[5]).to(device)
        if ctx.needs_input_grad[6]:
            t1_grad = torch.from_numpy(res[6]).to(device)
        return x1_grad, x2_grad, x3_grad, x4_grad, x5_grad, x6_grad, t1_grad

class ModelFunction7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, x3, x4, x5, x6, x7, t1):
        res = full_mycaffe.model_fwd7(x1, x2, x3, x4, x5, x6, x7, t1)
        ctx.save_for_backward(x1, x2, x3, x4, x5, x6, x7, t1)              
        y = torch.from_numpy(res[0]).to(device)
        y.requires_grad = True
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, x3, x4, x5, x6, x7, t1 = ctx.saved_tensors
        res = full_mycaffe.model_bwd(grad_output)

        x1_grad = x2_grad = x3_grad = x4_grad = x5_grad = x6_grad = x7_grad = t1_grad = None
        if ctx.needs_input_grad[0]:
            x1_grad = torch.from_numpy(res[0]).to(device)
        if ctx.needs_input_grad[1]:
            x2_grad = torch.from_numpy(res[1]).to(device)
        if ctx.needs_input_grad[2]:
            x3_grad = torch.from_numpy(res[2]).to(device)
        if ctx.needs_input_grad[3]:
            x4_grad = torch.from_numpy(res[3]).to(device)
        if ctx.needs_input_grad[4]:
            x5_grad = torch.from_numpy(res[4]).to(device)
        if ctx.needs_input_grad[5]:
            x6_grad = torch.from_numpy(res[5]).to(device)
        if ctx.needs_input_grad[6]:
            x7_grad = torch.from_numpy(res[6]).to(device)
        if ctx.needs_input_grad[7]:
            t1_grad = torch.from_numpy(res[7]).to(device)
        return x1_grad, x2_grad, x3_grad, x4_grad, x5_grad, x6_grad, x7_grad, t1_grad

class LstmEx(nn.Module):
    def __init__(self, tag, use_mycaffe, state=64, num=1, debug=False, path=""):
        super(LstmEx, self).__init__()
        self.tag = tag
        self.num = num
        self.state = state
        self.debug = debug
        self.path = path
        self.use_mycaffe = use_mycaffe
        if self.use_mycaffe != True:
            self.lstm1 = nn.LSTM(input_size = state, hidden_size = state, num_layers = num, dropout = 0, batch_first = True)

    def save_wts(self, tag="",path=""):
        if self.use_mycaffe:
            data = mycaffe.lstm_wts(self.tag)
            i = 0
            for data1 in data:
                d = torch.from_numpy(data1).float()
                DebugFunction.trace(d, tag + "ZZZ." + self.tag + ".lstm.wt" + str(i), path)
                i = i + 1 
        else:
            data = get_cudnn_lstm_weights(self.lstm1)
            DebugFunction.trace(data, tag + "ZZZ.pytorch." + self.tag + ".lstm.wt", path)

    def save_grad(self, tag="",path=""):
        if self.use_mycaffe:
            data = mycaffe.lstm_grad(self.tag)
            i = 0
            for data1 in data:
                d = torch.from_numpy(data1).float()
                DebugFunction.trace(d, tag + "ZZZ." + self.tag + ".lstm.wt.grad" + str(i), path)
                i = i + 1

    def update_wts(self, lr, decay, beta1, beta2, nT, eps):
        mycaffe.lstm_update_wts(self.tag, lr, decay, beta1, beta2, nT, eps)

    def forward(self, x, h=None):
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + ".lstm.x")
            x = debug(x)
            if h != None:
                DebugFunction.trace(h[0], self.tag + ".lstm.h0")
                DebugFunction.trace(h[1], self.tag + ".lstm.c0")

        if self.use_mycaffe:
            lstm = LstmFunction.apply
            tag_list.append(self.tag)
            lstm_num.append(self.num)
            lstm_state.append(self.state)
            y, h1, c1 = lstm(x, h[0], h[1])
            h2 = (h1,c1)
        else:
            y, h2 = self.lstm1(x, h)

        if self.debug:
            DebugFunction.trace(y, self.tag + ".lstm.y")
            y = debug(y)
            DebugFunction.trace(h2[0], self.tag + ".lstm.h1")
            DebugFunction.trace(h2[1], self.tag + ".lstm.c1")
        return y, h2

class LstmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, c):
        tag = tag_list[-1]
        num = lstm_num[-1]
        state = lstm_state[-1]
        res = mycaffe.lstm_fwd(tag, x, h, c, state, num)
        ctx.save_for_backward(x, h, c)      
        
        y = res[0]
        h1 = res[1]
        c1 = res[2]
        return y, h1, c1

    @staticmethod
    def backward(ctx, grad_output, grad_h, grad_c):
        x, h, c = ctx.saved_tensors
        tag = tag_list.pop()
        num = lstm_num.pop()
        state = lstm_state.pop()
        res = mycaffe.lstm_bwd(tag, x, h, c, grad_output, grad_h, grad_c)

        x_grad = h_grad = c_grad = None
        if ctx.needs_input_grad[0]:
            x_grad = res[0]
        if ctx.needs_input_grad[1]:
            h_grad = res[1]
        if ctx.needs_input_grad[2]:
            c_grad = res[2]

        return x_grad, h_grad, c_grad

class SoftmaxEx(nn.Module):
    def __init__(self, tag, use_mycaffe, dim=-1, debug=False, path=""):
        super(SoftmaxEx, self).__init__()
        self.tag = tag
        self.dim = dim
        self.debug = debug
        self.path = path
        self.use_mycaffe = use_mycaffe
        
    def forward(self, x):
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x, "softmax.x")
            x = debug(x)

        if self.use_mycaffe:
            softmax = SoftmaxFunction.apply
            tag_list.append(self.tag)
            smx_axis.append(self.dim)
            y = softmax(x)
        else:
            maxx = x.max(dim=self.dim, keepdim=True)[0]
            x = x - maxx
            expx = torch.exp(x)
            sumexpx = expx.sum(dim=self.dim, keepdim=True)
            y = expx / sumexpx
            
        if self.debug:
            DebugFunction.trace(y, "softmax.y")
            y = debug(y)
        return y
    
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        axis = smx_axis[-1]
        y = mycaffe.softmax_fwd(tag, x, axis)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        axis = smx_axis.pop()
        return mycaffe.softmax_bwd(tag, y, grad_output)

class SigmoidEx(nn.Module):
    def __init__(self, tag, use_mycaffe=False, debug=False):
        super(SigmoidEx, self).__init__()
        self.tag = tag + ".elu" if tag != None else ""
        self.debug = debug
        self.use_mycaffe = use_mycaffe
        if self.use_mycaffe == False:
            self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + ".x")
            x = debug(x)

        if self.use_mycaffe:
            sigmoid = SigmoidFunction.apply
            tag_list.append(self.tag)
            y = sigmoid(x)
        else:
            y = self.sigmoid(x)
            
        if self.debug:
            DebugFunction.trace(y, self.tag + ".y")
            y = debug(y)
        return y
    
class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        y = mycaffe.sigmoid_fwd(tag, x)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        return mycaffe.sigmoid_bwd(tag, y, grad_output)


class ELUEx(nn.Module):
    def __init__(self, tag, use_mycaffe=False, debug=False):
        super(ELUEx, self).__init__()
        self.tag = tag + ".elu" if tag != None else ""
        self.debug = debug
        self.use_mycaffe = use_mycaffe
        if self.use_mycaffe == False:
            self.elu1 = nn.ELU()
        
    def forward(self, x):
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + ".x")
            x = debug(x)

        if self.use_mycaffe:
            elu = ELUFunction.apply
            tag_list.append(self.tag)
            y = elu(x)
        else:
            y = self.elu1(x)
            
        if self.debug:
            DebugFunction.trace(y, self.tag + ".y")
            y = debug(y)
        return y
    
class ELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        y = mycaffe.elu_fwd(tag, x)
        ctx.save_for_backward(y, x)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y,x = ctx.saved_tensors
        tag = tag_list.pop()
        return mycaffe.elu_bwd(tag, y, grad_output, x)

#
# LinearEx
#     
class LinearEx(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_init=None,
                 bias_init=None, debug: Optional[bool] = False, tag = None, path = "", use_mycaffe: Optional[bool] = False, axis=None):
        super(LinearEx, self).__init__()
        self.debug = debug
        self.path = path
        self.tag = tag if tag != None else ""
        self.in_features = in_features
        self.out_features = out_features
        self.use_mycaffe = use_mycaffe
        self.biasval = bias
        self.numout = out_features
        self.axis = axis
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
   
    def forward(self, input):
        if self.debug:
            debug = DebugFunction.apply

        if self.use_mycaffe:
            ip = InnerProductFunction.apply
            tag_list.append(self.tag)
            x2 = ip(input, self.biasval, self.numout, self.axis)

        else:
            x1 = torch.matmul(input, self.weight.t())

            if self.debug:
                DebugFunction.trace(x1, self.tag + ".ip.x1")
                x1 = debug(x1)

            if self.bias is not None:
                x2 = x1 + self.bias
            else:
                x2 = x1

        if self.debug:
            DebugFunction.trace(x2, self.tag + ".ip.x2")
            x2 = debug(x2)

        return x2

class InnerProductFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tag = tag_list[-1]
        bias = ip_bias[-1]
        numout = ip_numout[-1]
        axis = ip_axis[-1]
        y = mycaffe.innerproduct_fwd(tag, x, bias, numout, axis)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        tag = tag_list.pop()
        axis = ip_axis.pop()
        bias = ip_bias.pop()
        numout = ip_numout.pop()
        return mycaffe.innerproduct_bwd(tag, y, grad_output)

#
# LayerNormEx
#     
class LayerNormEx(nn.Module):
    
    def __init__(self,
                 normal_shape,
                 gamma=False,
                 beta=False,
                 epsilon=1e-10,
                 debug=False,
                 tag=None,
                 use_mycaffe=False, path="", enable_pass_through=False):
        """Layer normalization layer

        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super(LayerNormEx, self).__init__()
        self.use_mycaffe = use_mycaffe
        self.debug = debug
        self.path = path
        self.enable_pass_through = enable_pass_through
        self.tag = tag + ".layernorm" if tag != None else ""

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
        debug1 = False
        if self.enable_pass_through:
            return x

        if self.debug or debug1:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + ".x")
            x = debug(x)
        
        if self.use_mycaffe and debug1 == False:
            layernorm = LayerNormFunction.apply
            tag_list.append(self.tag)
            y = layernorm(x)

        else:
            mean = x.mean(dim=-1, keepdim=True)

            if self.debug or debug1:
                DebugFunction.trace(mean, self.tag + ".mean")
                mean = debug(mean)

            xmu = (x - mean)

            if self.debug or debug1:
                DebugFunction.trace(xmu, self.tag + ".xmu")
                xmu = debug(xmu)

            xmusq = xmu ** 2

            if self.debug or debug1:
                DebugFunction.trace(xmusq, self.tag + ".xmusq")
                xmusq = debug(xmusq)

            var = xmusq.mean(dim=-1, keepdim=True)

            if self.debug or debug1:
                DebugFunction.trace(var, self.tag + ".var")
                var = debug(var)

            var1 = var + self.epsilon

            if self.debug or debug1:
                DebugFunction.trace(var1, self.tag + ".var1")
                var1 = debug(var1)

            std = var1.sqrt()

            if self.debug or debug1:
                DebugFunction.trace(std, self.tag + ".std")
                std = debug(std)

            y = xmu / std

            if self.debug or debug1:
                DebugFunction.trace(y, self.tag + ".y")
                y = debug(y)
        
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

class GatedLinearUnit(nn.Module):
    """
    This module is also known as  **GLU** - Formulated in:
    `Dauphin, Yann N., et al. "Language modeling with gated convolutional networks."
    International conference on machine learning. PMLR, 2017
    <https://arxiv.org/abs/1612.08083>`_.

    The output of the layer is a linear projection (X * W + b) modulated by the gates **sigmoid** (X * V + c).
    These gates multiply each element of the matrix X * W + b and control the information passed on in the hierarchy.
    This unit is a simplified gating mechanism for non-deterministic gates that reduce the vanishing gradient problem,
    by having linear units coupled to the gates. This retains the non-linear capabilities of the layer while allowing
    the gradient to propagate through the linear unit without scaling.

    Parameters
    ----------
    input_dim: int
        The embedding size of the input.
    """

    def __init__(self, input_dim: int, debug: Optional[bool] = False, use_mycaffe: Optional[bool] = False, tag = None, path=""):
        super(GatedLinearUnit, self).__init__()

        # Two dimension-preserving dense layers
        #self.fc1 = nn.Linear(input_dim, input_dim)
        #self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc1 = LinearEx(input_dim, input_dim)
        self.fc2 = LinearEx(input_dim, input_dim)
        self.tag = tag + "_" if tag != None else ""
        self.sigmoid = SigmoidEx(use_mycaffe=use_mycaffe, debug=debug, tag=self.tag)
        self.debug = debug
        self.path = path

    def forward(self, x):
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + "glu_x")
            x = debug(x)

        if self.debug:
            DebugFunction.trace(self.fc1.weight, self.tag + "glu.internal.fc1.weight");
            DebugFunction.trace(self.fc1.bias, self.tag + "glu.internal.fc1.bias");

        x1 = self.fc1(x);

        if self.debug:
            DebugFunction.trace(x1, self.tag + "glu_x1");
            x1 = debug(x1)

        sig = self.sigmoid(x1);

        if self.debug:
            DebugFunction.trace(sig, self.tag + "glu_sig")
            sig = debug(sig)

            DebugFunction.trace(self.fc2.weight, self.tag + "glu.internal.fc2.weight");
            DebugFunction.trace(self.fc2.bias, self.tag + "glu.internal.fc2.bias");

        x2 = self.fc2(x);

        if self.debug:
            DebugFunction.trace(x2, self.tag + "glu_x2")
            x2 = debug(x2)

        y = torch.mul(sig, x2)

        if self.debug:
            DebugFunction.trace(y, self.tag + "glu_y")
            y = debug(y)

        return y

class GatedResidualNetwork(nn.Module):
    """
    This module, known as **GRN**, takes in a primary input (x) and an optional context vector (c).
    It uses a ``GatedLinearUnit`` for controlling the extent to which the module will contribute to the original input
    (x), potentially skipping over the layer entirely as the GLU outputs could be all close to zero, by that suppressing
    the non-linear contribution.
    In cases where no context vector is used, the GRN simply treats the context input as zero.
    During training, dropout is applied before the gating layer.

    Parameters
    ----------
    input_dim: int
        The embedding width/dimension of the input.
    hidden_dim: int
        The intermediate embedding width.
    output_dim: int
        The embedding width of the output tensors.
    dropout: Optional[float]
        The dropout rate associated with the component.
    context_dim: Optional[int]
        The embedding width of the context signal expected to be fed as an auxiliary input to this component.
    batch_first: Optional[bool]
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: Optional[float] = 0.05,
                 context_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True,
                 debug: Optional[bool] = False,
                 tag = None,
                 use_mycaffe: Optional[bool] = False, activation_relu: Optional[bool] = False, path=""):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.debug = debug
        self.path = path
        self.tag = tag + "_" if tag != None else ""

        # =================================================
        # Input conditioning components (Eq.4 in the original paper)
        # =================================================
        # for using direct residual connection the dimension of the input must match the output dimension.
        # otherwise, we'll need to project the input for creating this residual connection
        self.project_residual: bool = self.input_dim != self.output_dim
        if self.project_residual:
            self.skip_layer = TimeDistributed(LinearEx(self.input_dim, self.output_dim))

        # A linear layer for projecting the primary input (acts across time if necessary)
        self.fc1 = TimeDistributed(LinearEx(self.input_dim, self.hidden_dim), batch_first=batch_first)

        # In case we expect context input, an additional linear layer will project the context
        if self.context_dim is not None:
            self.context_projection = TimeDistributed(LinearEx(self.context_dim, self.hidden_dim, bias=False),
                                                      batch_first=batch_first)
        # non-linearity to be applied on the sum of the projections
        if activation_relu == True:
            self.act = nn.ReLU()
        else:
            self.act = ELUEx(use_mycaffe=use_mycaffe, debug=debug, tag=self.tag) # nn.ELU()

        # ============================================================
        # Further projection components (Eq.3 in the original paper)
        # ============================================================
        # additional projection on top of the non-linearity
        self.fc2 = TimeDistributed(LinearEx(self.hidden_dim, self.output_dim), batch_first=batch_first)

        # ============================================================
        # Output gating components (Eq.2 in the original paper)
        # ============================================================
        self.dropout = nn.Dropout(self.dropout)
        tag1 = tag + "_grn" if tag != None else ""
        self.gate = TimeDistributed(GatedLinearUnit(self.output_dim, debug=debug, use_mycaffe=use_mycaffe, tag=tag1, path=self.path), batch_first=batch_first)
        #self.layernorm = TimeDistributed(nn.LayerNorm(self.output_dim), batch_first=batch_first)
        self.layernorm = TimeDistributed(LayerNormEx(self.output_dim,debug=self.debug, tag=self.tag, use_mycaffe=use_mycaffe, path=self.path), batch_first=batch_first)

    def forward(self, x, context=None):
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + ".grn_x")
            x = debug(x)
            if context != None:
                DebugFunction.trace(context, self.tag + ".grn_context")
                context = debug(context)

        # compute residual (for skipping) if necessary
        if self.project_residual:
            residual = self.skip_layer(x)
        else:
            residual = x
        # ===========================
        # Compute Eq.4
        # ===========================
        if self.debug:
            DebugFunction.trace(residual, self.tag + ".grn_residual")
            residual = debug(residual)
            DebugFunction.trace(x, self.tag + ".grn_xa")
            x = debug(x)

        x1 = self.fc1(x)

        if self.debug:
            DebugFunction.trace(x1, self.tag + ".grn_x1")
            x1 = debug(x1)

        if context is not None:
            context1 = self.context_projection(context)
            if self.debug:
                DebugFunction.trace(context1, self.tag + ".grn_context1")
                context1 = debug(context1)
            x1b = x1 + context1

            if self.debug:
                DebugFunction.trace(x1b, self.tag + ".grn_x1b")
                x1b = debug(x1b)
        else:
            x1b = x1

        if self.debug:
            DebugFunction.trace(x1b, self.tag + ".grn_x1b")
            x1b = debug(x1b)

        # compute eta_2 (according to paper)
        x2 = self.act(x1b)

        if self.debug:
            DebugFunction.trace(x2, self.tag + ".grn_x2")
            x2 = debug(x2)

        # ===========================
        # Compute Eq.3
        # ===========================
        # compute eta_1 (according to paper)
        x3 = self.fc2(x2)

        if self.debug:
            DebugFunction.trace(x3, self.tag + ".grn_x3")
            x3 = debug(x3)

        # ===========================
        # Compute Eq.2
        # ===========================
        x4 = self.dropout(x3)

        if self.debug:
            DebugFunction.trace(x4, self.tag + ".grn_x4")
            x4 = debug(x4)

        x5 = self.gate(x4)

        if self.debug:
            DebugFunction.trace(x5, self.tag + ".grn_x5")
            x5 = debug(x5)

        # perform skipping using the residual
        x6 = x5 + residual

        if self.debug:
            DebugFunction.trace(x6, self.tag + ".grn_x6")
            x6 = debug(x6)

        # apply normalization layer
        y = self.layernorm(x6)

        if self.debug:
            DebugFunction.trace(y, self.tag + ".grn_y")
            y = debug(y)

        return y


class VariableSelectionNetwork(nn.Module):
    """
    This module is designed to handle the fact that the relevant and specific contribution of each input variable
    to the  output is typically unknown. This module enables instance-wise variable selection, and is applied to
    both the static covariates and time-dependent covariates.

    Beyond providing insights into which variables are the most significant oones for the prediction problem,
    variable selection also allows the model to remove any unnecessary noisy inputs which could negatively impact
    performance.

    Parameters
    ----------
    input_dim: int
        The attribute/embedding dimension of the input, associated with the ``state_size`` of th model.
    num_inputs: int
        The quantity of input variables, including both numeric and categorical inputs for the relevant channel.
    hidden_dim: int
        The embedding width of the output.
    dropout: float
        The dropout rate associated with ``GatedResidualNetwork`` objects composing this object.
    context_dim: Optional[int]
        The embedding width of the context signal expected to be fed as an auxiliary input to this component.
    batch_first: Optional[bool]
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    """

    def __init__(self, input_dim: int, num_inputs: int, hidden_dim: int, dropout: float,
                 context_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True,
                 debug: Optional[bool] = False,
                 use_mycaffe: Optional[bool] = False,
                 tag = None, path=""):
        super(VariableSelectionNetwork, self).__init__()

        self.use_mycaffe = use_mycaffe
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context_dim = context_dim
        self.debug = debug
        self.path = path
        self.tag = tag + "_" if tag != None else ""

        # A GRN to apply on the flat concatenation of the input representation (all inputs together),
        # possibly provided with context information
        tag1 = self.tag + "grn1_" if self.tag != None else ""
        self.flattened_grn = GatedResidualNetwork(input_dim=self.num_inputs * self.input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  output_dim=self.num_inputs,
                                                  dropout=self.dropout,
                                                  context_dim=self.context_dim,
                                                  batch_first=batch_first,
                                                  debug = debug,
                                                  tag = tag1,
                                                  use_mycaffe=use_mycaffe, path=self.path)
        # activation for transforming the GRN output to weights
        self.softmax = SoftmaxEx(self.tag + ".smx", use_mycaffe=use_mycaffe,dim=1)

        # In addition, each input variable (after transformed to its wide representation) goes through its own GRN.
        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            tag1 = self.tag + "grn1_%d_" % i if self.tag != None else ""
            self.single_variable_grns.append(
                GatedResidualNetwork(input_dim=self.input_dim,
                                     hidden_dim=self.hidden_dim,
                                     output_dim=self.hidden_dim,
                                     dropout=self.dropout,
                                     batch_first=batch_first,
                                     debug=debug,
                                     tag=tag1,
                                     use_mycaffe=use_mycaffe, path=self.path))

    def forward(self, flattened_embedding, context=None):
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(flattened_embedding, self.tag + "varsel_flattened_embedding")
            flattened_embedding = debug(flattened_embedding)
            if context != None:
                DebugFunction.trace(context, self.tag + "varsel_context")
                context = debug(context)
        # ===========================================================================
        # Infer variable selection weights - using the flattened representation GRN
        # ===========================================================================
        # the flattened embedding should be of shape [(num_samples * num_temporal_steps) x (num_inputs x input_dim)]
        # where in our case input_dim represents the model_dim or the state_size.
        # in the case of static variables selection, num_temporal_steps is disregarded and can be thought of as 1.
        sparse_weights = self.flattened_grn(flattened_embedding, context)
        if self.debug:
            DebugFunction.trace(sparse_weights, self.tag + "varsel_sparse_weights")
            sparse_weights = debug(sparse_weights)

        sparse_weights_smx = self.softmax(sparse_weights).unsqueeze(2)
        if self.debug:
            DebugFunction.trace(sparse_weights_smx, self.tag + "varsel_sparse_weights_smx")
            sparse_weights_smx = debug(sparse_weights_smx)

        # After that step "sparse_weights" is of shape [(num_samples * num_temporal_steps) x num_inputs x 1]

        if self.debug:
            DebugFunction.trace(flattened_embedding, "varsel_flattened_embedding_1b");
            flattened_embedding = debug(flattened_embedding)

        # Before weighting the variables - apply a GRN on each transformed input
        processed_inputs = []
        for i in range(self.num_inputs):
            # select slice of embedding belonging to a single input - and apply the variable-specific GRN
            # (the slice is taken from the flattened concatenated embedding)
            single_grn_x = flattened_embedding[..., (i * self.input_dim): (i + 1) * self.input_dim]
            if self.debug:
                DebugFunction.trace(single_grn_x, self.tag + "varsel_single_grn_x_%d" % i)
                single_grn_x = debug(single_grn_x)

            single_grn_x1 = self.single_variable_grns[i](single_grn_x)
            if self.debug:
                DebugFunction.trace(single_grn_x1, self.tag + "varsel_single_grn_x1_%d" % i)
                single_grn_x1 = debug(single_grn_x1)

            processed_inputs.append(single_grn_x1)

        # each element in the resulting list is of size: [(num_samples * num_temporal_steps) x state_size],
        # and each element corresponds to a single input variable

        # combine the outputs of the single-var GRNs (along an additional axis)
        processed_inputs1 = torch.stack(processed_inputs, dim=-1)
        if self.debug:
            DebugFunction.trace(processed_inputs1, self.tag + "varsel_processed_inputs1")
            processed_inputs1 = debug(processed_inputs1)
        # Dimensions:
        # processed_inputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]

        # weigh them by multiplying with the weights tensor viewed as
        # [(num_samples * num_temporal_steps) x 1 x num_inputs]
        # so that the weight given to each input variable (for each time-step/observation) multiplies the entire state
        # vector representing the specific input variable on this specific time-step
        sparse_weights_smx_t = sparse_weights_smx.transpose(1, 2)
        if self.debug:
            DebugFunction.trace(sparse_weights_smx_t, self.tag + "varsel_sparse_weights_smx_t")
            sparse_weights_smx_t = debug(sparse_weights_smx_t)
            
        outputs = processed_inputs1 * sparse_weights_smx_t
        if self.debug:
            DebugFunction.trace(outputs, self.tag + "varsel_outputs")
            outputs = debug(outputs)

        # Dimensions:
        # outputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]

        # and finally sum up - for creating a weighted sum representation of width state_size for every time-step
        #if self.use_mycaffe == True:
        #    outputs_sum = mycaffe.sumEx(outputs, -1)
        #else:
        outputs_sum = outputs.sum(axis=-1)
        if self.debug:
            DebugFunction.trace(outputs_sum, self.tag + "varsel_outputs_sum")
            outputs_sum = debug(outputs_sum)

        # Dimensions:
        # outputs: [(num_samples * num_temporal_steps) x state_size]

        return outputs_sum, sparse_weights


class InputChannelEmbedding(nn.Module):
    """
    A module to handle the transformation/embedding of an input channel composed of numeric tensors and categorical
    tensors.
    It holds a NumericInputTransformation module for handling the embedding of the numeric inputs,
    and a CategoricalInputTransformation module for handling the embedding of the categorical inputs.

    Parameters
    ----------
    state_size : int
        The state size of the model, which determines the embedding dimension/width of each input variable.
    num_numeric : int
        The quantity of numeric input variables associated with the input channel.
    num_categorical : int
        The quantity of categorical input variables associated with the input channel.
    categorical_cardinalities: List[int]
        The quantity of categories associated with each of the categorical input variables.
    time_distribute: Optional[bool]
        A boolean indicating whether to wrap the composing transformations using the ``TimeDistributed`` module.
    """

    def __init__(self, state_size: int, num_numeric: int, num_categorical: int, categorical_cardinalities: List[int],
                 time_distribute: Optional[bool] = False, debug: Optional[bool] = False, tag = None, path=""):
        super(InputChannelEmbedding, self).__init__()

        self.tag = tag if tag != None else ""
        self.debug = debug
        self.path = path
        self.state_size = state_size
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities
        self.time_distribute = time_distribute

        if (num_numeric + num_categorical) < 1:
            raise ValueError(f"""At least a single input variable (either numeric or categorical) should be included
            as part of the input channel.
            According to the provided configuration:
            num_numeric + num_categorical = {num_numeric} + {num_categorical} = {num_numeric + num_categorical} < 1
            """)

        if self.time_distribute:
            self.numeric_transform = TimeDistributed(
                NumericInputTransformation(num_inputs=num_numeric, state_size=state_size, debug=self.debug, tag = self.tag + ".numeric",path=self.path), return_reshaped=False)
            self.categorical_transform = TimeDistributed(
                CategoricalInputTransformation(num_inputs=num_categorical, state_size=state_size,
                                               cardinalities=categorical_cardinalities, debug=self.debug, tag = self.tag + ".categorical",path=self.path), return_reshaped=False)
        else:
            self.numeric_transform = NumericInputTransformation(num_inputs=num_numeric, state_size=state_size, debug=self.debug, tag = self.tag + ".numeric",path=self.path)
            self.categorical_transform = CategoricalInputTransformation(num_inputs=num_categorical,
                                                                        state_size=state_size,
                                                                        cardinalities=categorical_cardinalities, debug=self.debug, tag = self.tag + ".categorical",path=self.path)

        # in case some input types are not expected there is no need in the type specific transformation.
        # instead the "transformation" will return an empty list
        if num_numeric == 0:
            self.numeric_transform = NullTransform()
        if num_categorical == 0:
            self.categorical_transform = NullTransform()

    def forward(self, x_numeric, x_categorical) -> torch.tensor:
        batch_shape = x_numeric.shape if x_numeric.nelement() > 0 else x_categorical.shape

        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x_numeric, self.tag + ".x_numeric")
            x_numeric = debug(x_numeric)
            DebugFunction.trace(x_categorical, self.tag + ".x_categorical")
            x_categorical = debug(x_categorical)

        processed_numeric = self.numeric_transform(x_numeric)

        if self.debug:
            for i in range(len(processed_numeric)):
                DebugFunction.trace(processed_numeric[i], self.tag + ".processed_numeric_%d" % (i))
                processed_numeric[i] = debug(processed_numeric[i])

        processed_categorical = self.categorical_transform(x_categorical)

        if self.debug:
            for i in range(len(processed_categorical)):
                DebugFunction.trace(processed_categorical[i], self.tag + ".processed_categorical_%d" % (i))
                processed_categorical[i] = debug(processed_categorical[i])

        # Both of the returned values, "processed_numeric" and "processed_categorical" are lists,
        # with "num_numeric" elements and "num_categorical" respectively - each element in these lists corresponds
        # to a single input variable, and is represent by its embedding, shaped as:
        # [(num_samples * num_temporal_steps) x state_size]
        # (for the static input channel, num_temporal_steps is irrelevant and can be treated as 1

        # the resulting embeddings for all the input varaibles are concatenated to a flattened representation
        merged_transformations = torch.cat(processed_numeric + processed_categorical, dim=1)
        # Dimensions:
        # merged_transformations: [(num_samples * num_temporal_steps) x (state_size * total_input_variables)]
        # total_input_variables stands for the amount of all input variables in the specific input channel, i.e
        # num_numeric + num_categorical

        # for temporal data we return the resulting tensor to its 3-dimensional shape
        if self.time_distribute:
            merged_transformations = merged_transformations.view(batch_shape[0], batch_shape[1], -1)
            # In that case:
            # merged_transformations: [num_samples x num_temporal_steps x (state_size * total_input_variables)]

        if self.debug:
            DebugFunction.trace(merged_transformations, self.tag + ".merged_transformations")
            merged_transformations = debug(merged_transformations)
        
        return merged_transformations


class NumericInputTransformation(nn.Module):
    """
    A module for transforming/embeddings the set of numeric input variables from a single input channel.
    Each input variable will be projected using a dedicated linear layer to a vector with width state_size.
    The result of applying this module is a list, with length num_inputs, that contains the embedding of each input
    variable for all the observations and time steps.

    Parameters
    ----------
    num_inputs : int
        The quantity of numeric input variables associated with this module.
    state_size : int
        The state size of the model, which determines the embedding dimension/width.
    """

    def __init__(self, num_inputs: int, state_size: int, use_mycaffe: Optional[bool] = False, debug: Optional[bool] = False, tag = None, path=""):
        super(NumericInputTransformation, self).__init__()
        self.debug = debug
        self.path = path
        self.tag = tag if tag != None else ""
        self.num_inputs = num_inputs
        self.state_size = state_size

        self.numeric_projection_layers = nn.ModuleList()
        for i in range(self.num_inputs):
            self.numeric_projection_layers.append(LinearEx(1, self.state_size, use_mycaffe=use_mycaffe, debug=debug, tag=self.tag + ".ip%d" % (i), path=self.path))

    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        # every input variable is projected using its dedicated linear layer,
        # the results are stored as a list

        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + ".x")
            x = debug(x)

        projections = []
        for i in range(self.num_inputs):
            x1 = x[:, [i]]

            if self.debug:
                DebugFunction.trace(x1, self.tag + ".x1_%d" % i)
                x1 = debug(x1)

            x2 = self.numeric_projection_layers[i](x1)

            if self.debug:
                DebugFunction.trace(x2, self.tag + ".x2_%d" % i)
                x2 = debug(x2)

            projections.append(x2)

        return projections


class CategoricalInputTransformation(nn.Module):
    """
    A module for transforming/embeddings the set of categorical input variables from a single input channel.
    Each input variable will be projected using a dedicated embedding layer to a vector with width state_size.
    The result of applying this module is a list, with length num_inputs, that contains the embedding of each input
    variable for all the observations and time steps.

    Parameters
    ----------
    num_inputs : int
        The quantity of categorical input variables associated with this module.
    state_size : int
        The state size of the model, which determines the embedding dimension/width.
    cardinalities: List[int]
        The quantity of categories associated with each of the input variables.
    """

    def __init__(self, num_inputs: int, state_size: int, cardinalities: List[int], debug: Optional[bool] = False, tag = None,path=""):
        super(CategoricalInputTransformation, self).__init__()
        self.debug = debug
        self.path = path
        self.tag = tag if tag != None else ""
        self.num_inputs = num_inputs
        self.state_size = state_size
        self.cardinalities = cardinalities

        # layers for processing the categorical inputs
        self.categorical_embedding_layers = nn.ModuleList()
        for idx, cardinality in enumerate(self.cardinalities):
            self.categorical_embedding_layers.append(nn.Embedding(cardinality, self.state_size))

    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + ".x")
            x = debug(x)

        # every input variable is projected using its dedicated embedding layer,
        # the results are stored as a list
        embeddings = []
        for i in range(self.num_inputs):
            x1 = x[:, i]

            if self.debug:
                DebugFunction.trace(x1, self.tag + ".x1_%d" % i)
                x1 = debug(x1)

            x2 = self.categorical_embedding_layers[i](x1)

            if self.debug:
                DebugFunction.trace(x2, self.tag + ".x2_%d" % i)
                x2 = debug(x2)

            embeddings.append(x2)

        return embeddings


class GateAddNorm(nn.Module):
    """
    This module encapsulates an operation performed multiple times across the TemporalFusionTransformer model.
    The composite operation includes:
    a. A *Dropout* layer.
    b. Gating using a ``GatedLinearUnit``.
    c. A residual connection to an "earlier" signal from the forward pass of the parent model.
    d. Layer normalization.

    Parameters
    ----------
    input_dim: int
        The dimension associated with the expected input of this module.
    dropout: Optional[float]
        The dropout rate associated with the component.
    """

    def __init__(self, input_dim: int, dropout: Optional[float] = None, debug: Optional[bool] = False, tag = None, path="", disable_layer_norm = False, use_mycaffe: Optional[bool] = False):
        super(GateAddNorm, self).__init__()
        self.dropout_rate = dropout
        self.debug = debug
        self.path = path
        self.tag = tag + "_" if tag != None else ""

        if dropout:
            self.dropout_layer = nn.Dropout(self.dropout_rate)
        tag1 = tag + "_gan" if tag != None else ""
        self.gate = TimeDistributed(GatedLinearUnit(input_dim, debug=debug, use_mycaffe=use_mycaffe, tag=tag1, path=self.path), batch_first=True)
        #self.layernorm = TimeDistributed(nn.LayerNorm(input_dim), batch_first=True)
        self.layernorm = TimeDistributed(LayerNormEx(input_dim, use_mycaffe=use_mycaffe, debug=debug, tag=tag1,path=self.path, enable_pass_through=disable_layer_norm), batch_first=True)

    def forward(self, x, residual=None):
        debug1 = False
        if self.debug or debug1:
            debug = DebugFunction.apply
            DebugFunction.trace(x, self.tag + ".gan_x")
            x = debug(x)

        if self.dropout_rate:
            x1 = self.dropout_layer(x)
        else:
            x1 = x

        if self.debug or debug1:
            DebugFunction.trace(x1, self.tag + ".gan_x1")
            x1 = debug(x1)

        x2 = self.gate(x1)

        if self.debug or debug1:
            DebugFunction.trace(x2, self.tag + ".gan_x2")
            x2 = debug(x2)

        # perform skipping
        if residual is not None:
            if self.debug or debug1:
                DebugFunction.trace(residual, self.tag + ".gan_residual")
                residual = debug(residual)
            x3 = x2 + residual
        else:
            x3 = x2

        if self.debug or debug1:
            DebugFunction.trace(x3, self.tag + ".gan_x3")
            x3 = debug(x3)

        # apply normalization layer
        y = self.layernorm(x3)

        if self.debug or debug1:
            DebugFunction.trace(y, self.tag + ".gan_y")
            y = debug(y)

        return y


class InterpretableMultiHeadAttention(nn.Module):
    """
    The mechanism implemented in this module is used to learn long-term relationships across different time-steps.
    It is a modified version of multi-head attention, for enhancing explainability. On this modification,
    as opposed to traditional versions of multi-head attention, the "values" signal is shared for all the heads -
    and additive aggregation is employed across all the heads.
    According to the paper, each head can learn different temporal patterns, while attending to a common set of
    input features which can be interpreted as  a simple ensemble over attention weights into a combined matrix, which,
    compared to the original multi-head attention matrix, yields an increased representation capacity in an efficient
    way.

    Parameters
    ----------
    embed_dim: int
        The dimensions associated with the ``state_size`` of th model, corresponding to the input as well as the output.
    num_heads: int
        The number of attention heads composing the Multi-head attention component.
    """

    def __init__(self, embed_dim: int, num_heads: int, debug: Optional[bool] = False, tag = None, use_mycaffe: Optional[bool] = False, path=""):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.debug = debug
        self.path = path
        self.tag = tag + "_" if tag != None else ""
        self.d_model = embed_dim  # the state_size (model_size) corresponding to the input and output dimension
        self.num_heads = num_heads  # the number of attention heads
        self.all_heads_dim = embed_dim * num_heads  # the width of the projection for the keys and queries

        self.w_q = LinearEx(embed_dim, self.all_heads_dim)  # multi-head projection for the queries
        self.w_k = LinearEx(embed_dim, self.all_heads_dim)  # multi-head projection for the keys
        self.w_v = LinearEx(embed_dim, embed_dim)  # a single, shared, projection for the values

        self.softmax = SoftmaxEx(self.tag + ".smx", use_mycaffe=use_mycaffe, dim=-1, path=self.path)

        # the last layer is used for final linear mapping (corresponds to W_H in the paper)
        self.out = LinearEx(self.d_model, self.d_model, path=self.path)
       

    def forward(self, q, k, v, mask=None):
        debug1 = False
        debug = None
        if self.debug or debug1:
            debug = DebugFunction.apply
            DebugFunction.trace(q, self.tag + "imha_q")
            q = debug(q)
            DebugFunction.trace(k, self.tag + "imha_k")
            k = debug(k)
            DebugFunction.trace(v, self.tag + "imha_v")
            v = debug(v)
            if mask != None:
                DebugFunction.trace(mask, self.tag + "imha_mask")                

        num_samples = q.size(0)

        # Dimensions:
        # queries tensor - q: [num_samples x num_future_steps x state_size]
        # keys tensor - k: [num_samples x (num_total_steps) x state_size]
        # values tensor - v: [num_samples x (num_total_steps) x state_size]

        # perform linear operation and split into h heads
        q_proj1 = self.w_q(q) 
        if self.debug or debug1:
            DebugFunction.trace(q_proj1, self.tag + "imha_q_proj1")
            q_proj1 = debug(q_proj1)

        q_proj2 = q_proj1.view(num_samples, -1, self.num_heads, self.d_model)
        if self.debug or debug1:
            DebugFunction.trace(q_proj2, self.tag + "imha_q_proj2")
            q_proj2 = debug(q_proj2)

        k_proj1 = self.w_k(k)
        if self.debug or debug1:
            DebugFunction.trace(k_proj1, self.tag + "imha_k_proj1")
            k_proj1 = debug(k_proj1)

        k_proj2 = k_proj1.view(num_samples, -1, self.num_heads, self.d_model)
        if self.debug or debug1:
            DebugFunction.trace(k_proj2, self.tag + "imha_k_proj2")
            k_proj2 = debug(k_proj2)

        v_proj1 = self.w_v(v)
        if self.debug or debug1:
            DebugFunction.trace(v_proj1, self.tag + "imha_v_proj1")
            v_proj1 = debug(v_proj1)

        v_proj2a = v_proj1.repeat(1, 1, self.num_heads);
        if self.debug or debug1:
            DebugFunction.trace(v_proj2a, self.tag + "imha_v_proj2a")
            v_proj2a = debug(v_proj2a)

        v_proj2 = v_proj2a.view(num_samples, -1, self.num_heads, self.d_model)
        if self.debug or debug1:
            DebugFunction.trace(v_proj2, self.tag + "imha_v_proj2")
            v_proj2 = debug(v_proj2)

        # transpose to get the following shapes
        q_proj3 = q_proj2.transpose(1, 2)  # (num_samples x num_future_steps x num_heads x state_size)
        if self.debug or debug1:
            DebugFunction.trace(q_proj3, self.tag + "imha_q_proj3")
            q_proj3 = debug(q_proj3)

        k_proj3 = k_proj2.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)
        if self.debug or debug1:
            DebugFunction.trace(k_proj3, self.tag + "imha_k_proj3")
            k_proj3 = debug(k_proj3)

        v_proj3 = v_proj2.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)
        if self.debug or debug1:
            DebugFunction.trace(v_proj3, self.tag + "imha_v_proj3")
            v_proj3 = debug(v_proj3)

        # calculate attention using function we will define next
        attn_outputs_all_heads, attn_scores_all_heads = self.attention(q_proj3, k_proj3, v_proj3, debug, mask)
        # Dimensions:
        # attn_scores_all_heads: [num_samples x num_heads x num_future_steps x num_total_steps]
        # attn_outputs_all_heads: [num_samples x num_heads x num_future_steps x state_size]

        if self.debug or debug1:
            DebugFunction.trace(attn_outputs_all_heads, self.tag + "imha_attn_output_all_heads")
            attn_outputs_all_heads = debug(attn_outputs_all_heads)
            DebugFunction.trace(attn_scores_all_heads, self.tag + "imha_attn_scores_all_heads")
            attn_scores_all_heads = debug(attn_scores_all_heads)

        # take average along heads
        attention_scores = attn_scores_all_heads.mean(dim=1)
        attention_outputs = attn_outputs_all_heads.mean(dim=1)
        # Dimensions:
        # attention_scores: [num_samples x num_future_steps x num_total_steps]
        # attention_outputs: [num_samples x num_future_steps x state_size]

        if self.debug or debug1:
            DebugFunction.trace(attention_scores, self.tag + "imha_attention_scores")
            attention_scores = debug(attention_scores)
            DebugFunction.trace(attention_outputs, self.tag + "imha_attention_outputs")
            attention_outputs = debug(attention_outputs)

        # weigh attention outputs
        output = self.out(attention_outputs)
        # output: [num_samples x num_future_steps x state_size]

        if self.debug or debug1:
            DebugFunction.trace(output, self.tag + "imha_output")
            output = debug(output)

        return output, attention_outputs, attention_scores

    def attention(self, q, k, v, debug, mask=None):
        # Applying the scaled dot product
        k1 = k.transpose(-2, -1)
        if self.debug or debug != None:
            DebugFunction.trace(k1, self.tag + "imha_k1")
            k1 = debug(k1)

        attention_scores1 = torch.matmul(q, k1)

        if self.debug or debug != None:
            DebugFunction.trace(attention_scores1, self.tag + "imha_attention_scores1")
            attention_scores1 = debug(attention_scores1)

        scale1 = math.sqrt(self.d_model)
        scale = 1. / math.sqrt(self.d_model)
        attention_scores2 = attention_scores1 / math.sqrt(self.d_model)
        if self.debug or debug != None:
            DebugFunction.trace(attention_scores2, self.tag + "imha_attention_scores2")
            attention_scores2 = debug(attention_scores2)

        # Dimensions:
        # attention_scores: [num_samples x num_heads x num_future_steps x num_total_steps]

        # Decoder masking is applied to the multi-head attention layer to ensure that each temporal dimension can only
        # attend to features preceding it
        if mask is not None:
            # the mask is broadcast along the batch(dim=0) and heads(dim=1) dimensions,
            # where the mask==True, the scores are "cancelled" by setting a very small value
            attention_scores3 = attention_scores2.masked_fill(mask, -1e29)
        else:
            attention_scores3 = attention_scores2

        if self.debug or debug != None:
            DebugFunction.trace(attention_scores3, self.tag + "imha_attention_scores3")
            attention_scores3 = debug(attention_scores3)

        # still part of the scaled dot-product attention (dimensions are kept)
        #attention_scores4 = F.softmax(attention_scores3, dim=-1)
        attention_scores4 = self.softmax(attention_scores3)
        if self.debug or debug != None:
            DebugFunction.trace(attention_scores4, self.tag + "imha_attention_scores4")
            attention_scores4 = debug(attention_scores4)

        # matrix multiplication is performed on the last two-dimensions to retrieve attention outputs
        attention_outputs = torch.matmul(attention_scores4, v)
        # Dimensions:
        # attention_outputs: [num_samples x num_heads x num_future_steps x state_size]

        if self.debug or debug != None:
            DebugFunction.trace(attention_outputs, self.tag + "imha_attention_outputs1")
            attention_outputs = debug(attention_outputs)

        return attention_outputs, attention_scores4


class TemporalFusionTransformer(nn.Module):
    """
    This class implements the Temporal Fusion Transformer model described in the paper
    `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    <https://arxiv.org/abs/1912.09363>`_.

    Parameters
    ----------
    config: DictConfig
        A mapping describing both the expected structure of the input of the model, and the architectural specification
        of the model.
        This mapping should include a key named ``data_props`` in which the dimensions and cardinalities (where the
        inputs are categorical) are specified. Moreover, the configuration mapping should contain a key named ``model``,
        specifying ``attention_heads`` , ``dropout`` , ``lstm_layers`` , ``output_quantiles`` and ``state_size`` ,
        which are required for creating the model.
    """

    def __init__(self, config: DictConfig, debug: Optional[bool] = False, tag = "tft", use_mycaffe: Optional[bool] = False, lstm_use_mycaffe: Optional[bool] = False, path="", use_mycaffe_model: Optional[bool] = False):
        super().__init__()

        self.debug = debug;
        self.tag = tag
        self.use_mycaffe = use_mycaffe
        self.lstm_use_mycaffe = lstm_use_mycaffe
        self.config = config
        self.path = path

        # ============
        # data props
        # ============
        data_props = config['data_props']
        self.num_historical_numeric = data_props.get('num_historical_numeric', 0)
        self.num_historical_categorical = data_props.get('num_historical_categorical', 0)
        self.historical_categorical_cardinalities = data_props.get('historical_categorical_cardinalities', [])

        self.num_static_numeric = data_props.get('num_static_numeric', 0)
        self.num_static_categorical = data_props.get('num_static_categorical', 0)
        self.static_categorical_cardinalities = data_props.get('static_categorical_cardinalities', [])

        self.num_future_numeric = data_props.get('num_future_numeric', 0)
        self.num_future_categorical = data_props.get('num_future_categorical', 0)
        self.future_categorical_cardinalities = data_props.get('future_categorical_cardinalities', [])

        self.historical_ts_representative_key = 'historical_ts_numeric' if self.num_historical_numeric > 0 \
            else 'historical_ts_categorical'
        self.future_ts_representative_key = 'future_ts_numeric' if self.num_future_numeric > 0 \
            else 'future_ts_categorical'

        # ============
        # model props
        # ============
        self.task_type = config.task_type
        self.attention_heads = config.model.attention_heads
        self.dropout = config.model.dropout
        self.lstm_layers = config.model.lstm_layers
        self.target_window_start_idx = (config.target_window_start - 1) if config.target_window_start is not None else 0
        if self.task_type == 'regression':
            self.output_quantiles = config.model.output_quantiles
            self.num_outputs = len(self.output_quantiles)
        elif self.task_type == 'classification':
            self.output_quantiles = None
            self.num_outputs = 1
        else:
            raise ValueError(f"unsupported task type: {self.task_type}")
        self.state_size = config.model.state_size

        # =====================
        # Input Transformation
        # =====================
        self.static_transform = InputChannelEmbedding(state_size=self.state_size,
                                                      num_numeric=self.num_static_numeric,
                                                      num_categorical=self.num_static_categorical,
                                                      categorical_cardinalities=self.static_categorical_cardinalities,
                                                      time_distribute=False, debug=self.debug, tag = self.tag + ".static", path=self.path)

        self.historical_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_historical_numeric,
            num_categorical=self.num_historical_categorical,
            categorical_cardinalities=self.historical_categorical_cardinalities,
            time_distribute=True, debug=self.debug, tag = self.tag + ".historical", path=self.path)

        self.future_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_future_numeric,
            num_categorical=self.num_future_categorical,
            categorical_cardinalities=self.future_categorical_cardinalities,
            time_distribute=True, debug=self.debug, tag = self.tag + ".future", path=self.path)

        # =============================
        # Variable Selection Networks
        # =============================
        # %%%%%%%%%% Static %%%%%%%%%%%
        self.static_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_static_numeric + self.num_static_categorical,
            hidden_dim=self.state_size, dropout=self.dropout, use_mycaffe=use_mycaffe, debug=self.debug, tag=self.tag + ".static", path=self.path)

        self.historical_ts_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_historical_numeric + self.num_historical_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout,
            context_dim=self.state_size, use_mycaffe=use_mycaffe, debug=self.debug, tag = self.tag + ".historical", path=self.path)

        self.future_ts_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_future_numeric + self.num_future_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout,
            context_dim=self.state_size, use_mycaffe=use_mycaffe, debug=self.debug, tag = self.tag + ".future", path=self.path)

        # =============================
        # static covariate encoders
        # =============================
        static_covariate_encoder = GatedResidualNetwork(input_dim=self.state_size,
                                                        hidden_dim=self.state_size,
                                                        output_dim=self.state_size,
                                                        dropout=self.dropout, use_mycaffe=use_mycaffe, path=self.path, debug=self.debug)
        self.static_encoder_selection = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_selection.tag = self.tag + ".static_encoder_selection"
        self.static_encoder_enrichment = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_enrichment.tag = self.tag + ".static_encoder_enrichment"
        self.static_encoder_sequential_cell_init = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_sequential_cell_init.tag = self.tag + ".static_encoder_sequential_cell_init"
        self.static_encoder_sequential_state_init = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_sequential_state_init.tag = self.tag + ".static_encoder_sequential_state_init"

        # ============================================================
        # Locality Enhancement with Sequence-to-Sequence processing
        # ============================================================
        if use_mycaffe == False and self.lstm_use_mycaffe == False:
            self.past_lstm = nn.LSTM(input_size=self.state_size,
                                     hidden_size=self.state_size,
                                     num_layers=self.lstm_layers,
                                     dropout=self.dropout,
                                     batch_first=True)

            self.future_lstm = nn.LSTM(input_size=self.state_size,
                                       hidden_size=self.state_size,
                                       num_layers=self.lstm_layers,
                                       dropout=self.dropout,
                                       batch_first=True)
        else:
            self.past_lstm = LstmEx(tag="YYY.past_lstm", 
                               state=self.state_size,
                               num=self.lstm_layers,
                               debug=self.debug,
                               use_mycaffe=True, path=self.path)
        
            self.future_lstm = LstmEx(tag="YYY.future_lstm", 
                               state=self.state_size,
                               num=self.lstm_layers,
                               debug=self.debug,
                               use_mycaffe=True, path=self.path)

        self.post_lstm_gating = GateAddNorm(input_dim=self.state_size, dropout=self.dropout, use_mycaffe=use_mycaffe, debug=self.debug, tag=self.tag + ".plg.", path=self.path)

        # ============================================================
        # Static enrichment
        # ============================================================
        self.static_enrichment_grn = GatedResidualNetwork(input_dim=self.state_size,
                                                          hidden_dim=self.state_size,
                                                          output_dim=self.state_size,
                                                          context_dim=self.state_size,
                                                          dropout=self.dropout, use_mycaffe=use_mycaffe, debug=self.debug, tag=self.tag + ".statenr_grn", path=self.path)

        # ============================================================
        # Temporal Self-Attention
        # ============================================================
        self.multihead_attn = InterpretableMultiHeadAttention(embed_dim=self.state_size, num_heads=self.attention_heads, use_mycaffe=use_mycaffe, path=self.path, debug=self.debug)
        self.post_attention_gating = GateAddNorm(input_dim=self.state_size, dropout=self.dropout, use_mycaffe=use_mycaffe, path=self.path, debug=self.debug, tag="pag")

        # ============================================================
        # Position-wise feed forward
        # ============================================================
        self.pos_wise_ff_grn = GatedResidualNetwork(input_dim=self.state_size,
                                                    hidden_dim=self.state_size,
                                                    output_dim=self.state_size,
                                                    dropout=self.dropout, path=self.path, use_mycaffe=use_mycaffe, debug=debug, tag="pwff_grn")
        self.pos_wise_ff_gating = GateAddNorm(input_dim=self.state_size, dropout=None, debug=self.debug, use_mycaffe=use_mycaffe, path=self.path, tag="pwff")

        # ============================================================
        # Output layer
        # ============================================================
        self.output_layer = LinearEx(self.state_size, self.num_outputs)

        if use_mycaffe_model:
            self.mycaffe_model = MyCaffeModel(self.tag, self.path)
        else:
            self.mycaffe_model = None

    def apply_temporal_selection(self, temporal_representation: torch.tensor,
                                 static_selection_signal: torch.tensor,
                                 temporal_selection_module: VariableSelectionNetwork,
                                 tag = None
                                 ) -> Tuple[torch.tensor, torch.tensor]:
        num_samples, num_temporal_steps, _ = temporal_representation.shape

        full_tag = self.tag
        if tag != None:
            full_tag += "." + tag

        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(temporal_representation, full_tag + ".temporal_representation");
            temporal_representation = debug(temporal_representation)
            DebugFunction.trace(static_selection_signal, full_tag + ".static_selection_signal");
            static_selection_signal = debug(static_selection_signal)

        # replicate the selection signal along time
        time_distributed_context = self.replicate_along_time(static_signal=static_selection_signal,
                                                             time_steps=num_temporal_steps)
        # Dimensions:
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]
        # temporal_representation: [num_samples x num_temporal_steps x (total_num_temporal_inputs * state_size)]

        # for applying the same selection module on all time-steps, we stack the time dimension with the batch dimension
        temporal_flattened_embedding = self.stack_time_steps_along_batch(temporal_representation)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # Dimensions:
        # temporal_flattened_embedding: [(num_samples * num_temporal_steps) x (total_num_temporal_inputs * state_size)]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

        if self.debug:
            DebugFunction.trace(temporal_flattened_embedding, full_tag + ".temporal_flattened_embedding")
            temporal_flattened_embedding = debug(temporal_flattened_embedding);
            DebugFunction.trace(time_distributed_context, full_tag + ".time_distributed_context")
            time_distributed_context = debug(time_distributed_context);

        # applying the selection module across time
        temporal_selection_output, temporal_selection_weights = temporal_selection_module(
            flattened_embedding=temporal_flattened_embedding, context=time_distributed_context)
        # Dimensions:
        # temporal_selection_output: [(num_samples * num_temporal_steps) x state_size]
        # temporal_selection_weights: [(num_samples * num_temporal_steps) x (num_temporal_inputs) x 1]

        if self.debug:
            DebugFunction.trace(temporal_selection_output, full_tag + ".temporal_selection_output")
            temporal_selection_output = debug(temporal_selection_output);
            DebugFunction.trace(temporal_selection_weights, full_tag + ".temporal_selection_weights")
            temporal_selection_weights = debug(temporal_selection_weights);

        # Reshape the selection outputs and selection weights - to represent the temporal dimension separately
        temporal_selection_output = temporal_selection_output.view(num_samples, num_temporal_steps, -1)
        temporal_selection_weights = temporal_selection_weights.squeeze(-1).view(num_samples, num_temporal_steps, -1)
        # Dimensions:
        # temporal_selection_output: [num_samples x num_temporal_steps x state_size)]
        # temporal_selection_weights: [num_samples x num_temporal_steps x num_temporal_inputs)]

        if self.debug:
            DebugFunction.trace(temporal_representation, full_tag + ".temporal_selection_output");
            temporal_selection_output = debug(temporal_selection_output)
            DebugFunction.trace(static_selection_signal, full_tag + ".temporal_selection_weights");
            static_selection_signal = debug(static_selection_signal)

        return temporal_selection_output, temporal_selection_weights

    @staticmethod
    def replicate_along_time(static_signal: torch.tensor, time_steps: int) -> torch.tensor:
        """
        This method gets as an input a static_signal (non-temporal tensor) [num_samples x num_features],
        and replicates it along time for 'time_steps' times,
        creating a tensor of [num_samples x time_steps x num_features]

        Args:
            static_signal: the non-temporal tensor for which the replication is required.
            time_steps: the number of time steps according to which the replication is required.

        Returns:
            torch.tensor: the time-wise replicated tensor
        """
        time_distributed_signal = static_signal.unsqueeze(1).repeat(1, time_steps, 1)
        return time_distributed_signal

    @staticmethod
    def stack_time_steps_along_batch(temporal_signal: torch.tensor) -> torch.tensor:
        """
        This method gets as an input a temporal signal [num_samples x time_steps x num_features]
        and stacks the batch dimension and the temporal dimension on the same axis (dim=0).

        The last dimension (features dimension) is kept as is, but the rest is stacked along dim=0.
        """
        return temporal_signal.view(-1, temporal_signal.size(-1))

    def transform_inputs(self, batch: Dict[str, torch.tensor]) -> Tuple[torch.tensor, ...]:
        """
        This method processes the batch and transform each input channel (historical_ts, future_ts, static)
        separately to eventually return the learned embedding for each of the input channels

        each feature is embedded to a vector of state_size dimension:
        - numeric features will be projected using a linear layer
        - categorical features will be embedded using an embedding layer

        eventually the embedding for all the features will be concatenated together on the last dimension of the tensor
        (i.e. dim=1 for the static features, dim=2 for the temporal data).

        """
        if self.debug:
            debug = DebugFunction.apply

        empty_tensor = torch.empty((0, 0))

        x_static_numeric = x_numeric=batch.get('static_feats_numeric', empty_tensor)
        x_static_categorical = x_categorical=batch.get('static_feats_categorical', empty_tensor)

        x_hist_numeric = x_numeric=batch.get('historical_ts_numeric', empty_tensor)
        x_hist_categorical = x_categorical=batch.get('historical_ts_categorical', empty_tensor)
        
        x_fut_numeric = x_numeric=batch.get('future_ts_numeric', empty_tensor)
        x_fut_categorical = x_categorical=batch.get('future_ts_categorical', empty_tensor)

        if self.debug:
            DebugFunction.trace(x_static_numeric, "ti.x_static_numeric")
            DebugFunction.trace(x_static_categorical, "ti.x_static_categorical")
            DebugFunction.trace(x_hist_numeric, "ti.x_hist_numeric")
            DebugFunction.trace(x_hist_categorical, "ti.x_hist_categorical")
            DebugFunction.trace(x_fut_numeric, "ti.x_fut_numeric")
            DebugFunction.trace(x_fut_categorical, "ti.x_fut_categorical")

        static_rep = self.static_transform(x_numeric=x_static_numeric, x_categorical=x_static_categorical)
        historical_ts_rep = self.historical_ts_transform(x_numeric=x_hist_numeric, x_categorical=x_hist_categorical)
        future_ts_rep = self.future_ts_transform(x_numeric=x_fut_numeric, x_categorical=x_fut_categorical)

        if self.debug:
            DebugFunction.trace(historical_ts_rep, "ti.historical_ts_rep")
            historical_ts_rep = debug(historical_ts_rep)
            DebugFunction.trace(future_ts_rep, "ti.future_ts_rep")
            future_ts_rep = debug(future_ts_rep)
            DebugFunction.trace(static_rep, "ti.static_rep")
            static_rep = debug(static_rep)

        return future_ts_rep, historical_ts_rep, static_rep

    def get_static_encoders(self, selected_static: torch.tensor) -> Tuple[torch.tensor, ...]:
        """
        This method processes the variable selection results for the static data, yielding signals which are designed
        to allow better integration of the information from static metadata.
        Each of the resulting signals is generated using a separate GRN, and is eventually wired into various locations
        in the temporal fusion decoder, for allowing static variables to play an important role in processing.

        c_selection will be used for temporal variable selection
        c_seq_hidden & c_seq_cell will be used both for local processing of temporal features
        c_enrichment will be used for enriching temporal features with static information.
        """
        c_selection = self.static_encoder_selection(selected_static)
        c_enrichment = self.static_encoder_enrichment(selected_static)
        c_seq_hidden = self.static_encoder_sequential_state_init(selected_static)
        c_seq_cell = self.static_encoder_sequential_cell_init(selected_static)
        return c_enrichment, c_selection, c_seq_cell, c_seq_hidden

    def apply_sequential_processing(self, selected_historical: torch.tensor, selected_future: torch.tensor,
                                    c_seq_hidden: torch.tensor, c_seq_cell: torch.tensor, pre_gan = False) -> torch.tensor:
        """
        This part of the model is designated to mimic a sequence-to-sequence layer which will be used for local
        processing.
        On that part the historical ("observed") information will be fed into a recurrent layer called "Encoder" and
        the future information ("known") will be fed into a recurrent layer called "Decoder".
        This will generate a set of uniform temporal features which will serve as inputs into the temporal fusion
        decoder itself.
        To allow static metadata to influence local processing, we use "c_seq_hidden" and "c_seq_cell" context vectors
        from the static covariate encoders to initialize the hidden state and the cell state respectively.
        The output of the recurrent layers is gated and fused with a residual connection to the input of this block.
        """
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(selected_historical, self.tag + ".asp.selected_historical");
            selected_historical = debug(selected_historical)
            DebugFunction.trace(selected_future, self.tag + ".asp.selected_future");
            selected_future = debug(selected_future)
            DebugFunction.trace(c_seq_hidden, self.tag + ".asp.c_seq_hidden");
            c_seq_hidden = debug(c_seq_hidden)
            DebugFunction.trace(c_seq_cell, self.tag + ".asp.c_seq_cell");
            c_seq_cell = debug(c_seq_cell)

        # concatenate the historical (observed) temporal signal with the futuristic (known) temporal singal, along the
        # time dimension
        lstm_input = torch.cat([selected_historical, selected_future], dim=1)

        if self.debug:
            DebugFunction.trace(lstm_input, self.tag + ".asp.lstm_input")
            lstm_input = debug(lstm_input)

        # the historical temporal signal is fed into the first recurrent module
        # using the static metadata as initial hidden and cell state
        # (initial cell and hidden states are replicated for feeding to each layer in the stack)
        past_lstm_output, hidden = self.past_lstm(selected_historical,
                                                  (c_seq_hidden.unsqueeze(0).repeat(self.lstm_layers, 1, 1),
                                                   c_seq_cell.unsqueeze(0).repeat(self.lstm_layers, 1, 1)))

        if self.debug:
            DebugFunction.trace(past_lstm_output, self.tag + ".asp.past_lstm_output")
            past_lstm_output = debug(past_lstm_output)
            hidden0 = hidden[0]
            DebugFunction.trace(hidden0, self.tag + ".asp.hidden_0")
            hidden0 = debug(hidden0)
            hidden1 = hidden[1]
            DebugFunction.trace(hidden1, self.tag + ".asp.hidden_1")
            hidden1 = debug(hidden1)
            hidden = (hidden0, hidden1)

        # the future (known) temporal signal is fed into the second recurrent module
        # using the latest (hidden,cell) state of the first recurrent module
        # for setting the initial (hidden,cell) state.
        future_lstm_output, _ = self.future_lstm(selected_future, hidden)

        if self.debug:
            DebugFunction.trace(future_lstm_output, self.tag + ".asp.future_lstm_output")
            future_lstm_output = debug(future_lstm_output)

        # concatenate the historical recurrent output with the futuristic recurrent output, along the time dimension
        lstm_output = torch.cat([past_lstm_output, future_lstm_output], dim=1)

        if self.debug:
            DebugFunction.trace(lstm_output, self.tag + ".asp.lstm_output")
            lstm_output = debug(lstm_output)

        if pre_gan:
            return lstm_output, lstm_input

        val = torch.zeros(1).to(device)
        lstm_output = lstm_output + val
        lstm_input = lstm_input + val

        if self.debug:
            DebugFunction.trace(lstm_output, self.tag + ".asp.lstm_output.XX")
            lstm_output = debug(lstm_output)
            DebugFunction.trace(lstm_input, self.tag + ".asp.lstm_input.XX")
            lstm_input = debug(lstm_input)

        # perform gating to the recurrent output signal, using a residual connection to input of this block
        gated_lstm_output = self.post_lstm_gating(lstm_output, residual=lstm_input)

        if self.debug:
            DebugFunction.trace(gated_lstm_output, self.tag + ".asp.gated_lstm_output")
            gated_lstm_output = debug(gated_lstm_output)

        return gated_lstm_output

    def apply_static_enrichment(self, gated_lstm_output: torch.tensor,
                                static_enrichment_signal: torch.tensor) -> torch.tensor:
        """
        This static enrichment stage enhances temporal features with static metadata using a GRN.
        The static enrichment signal is an output of a static covariate encoder, and the GRN is shared across time.
        """
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(gated_lstm_output, self.tag + ".statenr.gated_lstm_output");
            gated_lstm_output = debug(gated_lstm_output)
            DebugFunction.trace(static_enrichment_signal, self.tag + ".statenr.static_enrichment_signal");
            static_enrichment_signal = debug(static_enrichment_signal)

        num_samples, num_temporal_steps, _ = gated_lstm_output.shape

        # replicate the selection signal along time
        time_distributed_context = self.replicate_along_time(static_signal=static_enrichment_signal,
                                                             time_steps=num_temporal_steps)
        # Dimensions:
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]

        # for applying the same GRN module on all time-steps, we stack the time dimension with the batch dimension
        flattened_gated_lstm_output = self.stack_time_steps_along_batch(gated_lstm_output)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # Dimensions:
        # flattened_gated_lstm_output: [(num_samples * num_temporal_steps) x state_size]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

        # applying the GRN using the static enrichment signal as context data
        enriched_sequence = self.static_enrichment_grn(flattened_gated_lstm_output,
                                                       context=time_distributed_context)
        # Dimensions:
        # enriched_sequence: [(num_samples * num_temporal_steps) x state_size]

        # reshape back to represent temporal dimension separately
        enriched_sequence = enriched_sequence.view(num_samples, -1, self.state_size)
        # Dimensions:
        # enriched_sequence: [num_samples x num_temporal_steps x state_size]

        if self.debug:
            DebugFunction.trace(enriched_sequence, self.tag + ".statenr.enriched_sequence")
            enriched_sequence = debug(enriched_sequence)

        return enriched_sequence

    def apply_self_attention(self, enriched_sequence: torch.tensor,
                             num_historical_steps: int,
                             num_future_steps: int, pre_gan = False):
        debug1 = False
        if self.debug or debug1:
            debug = DebugFunction.apply
            DebugFunction.trace(enriched_sequence, self.tag + ".ada.enriched_sequence");
            enriched_sequence = debug(enriched_sequence)

        # create a mask - so that future steps will be exposed (able to attend) only to preceding steps
        output_sequence_length = num_future_steps - self.target_window_start_idx
        mask = torch.cat([torch.zeros(output_sequence_length,
                                      num_historical_steps + self.target_window_start_idx,
                                      device=enriched_sequence.device),
                          torch.triu(torch.ones(output_sequence_length, output_sequence_length,
                                                device=enriched_sequence.device),
                                     diagonal=1)], dim=1)
        if self.debug or debug1:
            DebugFunction.trace(mask, self.tag + ".ada.mask");

        # Dimensions:
        # mask: [output_sequence_length x (num_historical_steps + num_future_steps)]

        # apply the InterpretableMultiHeadAttention mechanism
        val = torch.zeros(1).to(device)

        enriched_sequence1 = enriched_sequence + val

        if self.debug or debug1:
            DebugFunction.trace(enriched_sequence1, self.tag + ".ada.enriched_sequence1");
            enriched_sequence1 = debug(enriched_sequence1)

        q1 = enriched_sequence1[:, (num_historical_steps + self.target_window_start_idx):, :]
        k1 = enriched_sequence1 + val
        v1 = enriched_sequence1 + val

        if self.debug or debug1:
            DebugFunction.trace(q1, self.tag + ".ada.q1");
            q1 = debug(q1)
            DebugFunction.trace(k1, self.tag + ".ada.k1");
            k1 = debug(k1)
            DebugFunction.trace(v1, self.tag + ".ada.v1");
            v1 = debug(v1)

        post_attention, attention_outputs, attention_scores = self.multihead_attn(q=q1, k=k1, v=v1, mask = mask.bool())

        if self.debug or debug1:
            DebugFunction.trace(post_attention, self.tag + ".ada.post_attention");
            post_attention = debug(post_attention)

            DebugFunction.trace(attention_scores, self.tag + ".ada.attention_scores");
            attention_scores = debug(attention_scores)

            DebugFunction.trace(attention_outputs, self.tag + ".ada.attention_outputs");
            attention_outputs = debug(attention_outputs)
        # Dimensions:
        # post_attention: [num_samples x num_future_steps x state_size]
        # attention_outputs: [num_samples x num_future_steps x state_size]
        # attention_scores: [num_samples x num_future_steps x num_total_steps]

        if pre_gan == True:
            return post_attention, attention_scores

        # Apply gating with a residual connection to the input of this stage.
        # Because the output of the attention layer is only for the future time-steps,
        # the residual connection is only to the future time-steps of the temporal input signal

        gated_post_attention = self.post_attention_gating(
            x=post_attention,
            residual=enriched_sequence[:, (num_historical_steps + self.target_window_start_idx):, :])

        # Dimensions:
        # gated_post_attention: [num_samples x num_future_steps x state_size]

        if self.debug or debug1:
            DebugFunction.trace(gated_post_attention, self.tag + ".ada.gated_post_attention");
            gated_post_attention = debug(gated_post_attention)
            DebugFunction.trace(enriched_sequence, self.tag + ".ada.enriched_sequence_p");
            enriched_sequence = debug(enriched_sequence)

        return gated_post_attention, attention_scores

    def update(self, nIter):
        if self.mycaffe_model != None:
            self.mycaffe_model.update(nIter)

    def backward_direct(self):
        y = np.array([1])
        y = torch.from_numpy(y).to(device)
        self.mycaffe_model.backward_direct(y)

    def forward(self, batch):
        if self.debug:
            debug = DebugFunction.apply
            DebugFunction.trace(batch['static_feats_numeric'], self.tag + ".static_feats_numeric")
            DebugFunction.trace(batch['static_feats_categorical'], self.tag + ".static_feats_categorical")
            DebugFunction.trace(batch['historical_ts_numeric'], self.tag + ".historical_ts_numeric")
            DebugFunction.trace(batch['historical_ts_categorical'], self.tag + ".historical_ts_categorical")
            DebugFunction.trace(batch['future_ts_numeric'], self.tag + ".future_ts_numeric")
            DebugFunction.trace(batch['future_ts_categorical'], self.tag + ".future_ts_categorical")
            DebugFunction.trace(batch['target'], self.tag + ".target")

        # infer batch structure
        num_samples, num_historical_steps, _ = batch[self.historical_ts_representative_key].shape
        num_future_steps = batch[self.future_ts_representative_key].shape[1]
        # define output_sequence_length : num_future_steps - self.target_window_start_idx

        if self.mycaffe_model != None:
            #x_num_stat = batch['static_feats_numeric']
            #x_cat_stat = batch['static_feats_categorical']
            #x_num_hist = batch['historical_ts_numeric']
            #x_cat_hist = batch['historical_ts_categorical']
            #x_num_fut = batch['future_ts_numeric']
            #x_cat_fut = batch['future_ts_categorical']
            x_target = batch['target']

            # ok
            #gated_lstm_output, lstm_input = self.apply_sequential_processing(selected_historical=selected_historical,
            #                                                     selected_future=selected_future,
            #                                                     c_seq_hidden=c_seq_hidden,
            #                                                     c_seq_cell=c_seq_cell, pre_gan=True)

            return (self.mycaffe_model.forward_direct_full(), None)
            #return (self.mycaffe_model.forward_direct(x_cat_stat, x_num_hist, x_cat_hist, x_num_fut, x_cat_fut, x_target), None)
            #return (self.mycaffe_model.forward3(gated_lstm_output, lstm_input, c_enrichment, x_target), None)
            #return (self.mycaffe_model.forward6(historical_ts_rep, future_ts_rep, c_selection, c_seq_hidden, c_seq_cell, c_enrichment, x_target), None)
        else:
            # =========== Transform all input channels ==============
            future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)

            if self.debug:
                DebugFunction.trace(future_ts_rep, self.tag + ".future_ts_rep")
                future_ts_rep = debug(future_ts_rep)
                DebugFunction.trace(historical_ts_rep, self.tag + ".historical_ts_rep")
                historical_ts_rep = debug(historical_ts_rep)
                DebugFunction.trace(static_rep, self.tag + ".static_rep")
                static_rep = debug(static_rep)

            # Dimensions:
            # static_rep: [num_samples x (total_num_static_inputs * state_size)]
            # historical_ts_rep: [num_samples x num_historical_steps x (total_num_historical_inputs * state_size)]
            # future_ts_rep: [num_samples x num_future_steps x (total_num_future_inputs * state_size)]

            # =========== Static Variables Selection ==============
            selected_static, static_weights = self.static_selection(static_rep)

            if self.debug:
                DebugFunction.trace(selected_static, self.tag + ".selected_static")
                selected_static = debug(selected_static)

            # Dimensions:
            # selected_static: [num_samples x state_size]
            # static_weights: [num_samples x num_static_inputs x 1]

            # =========== Static Covariate Encoding ==============
            c_enrichment, c_selection, c_seq_cell, c_seq_hidden = self.get_static_encoders(selected_static)
            # each of the static encoders signals is of shape: [num_samples x state_size]

            val = torch.zeros(1).to(device)
            c_selection = c_selection + val

            val = torch.zeros(1).to(device)
            c_selection_h = c_selection + val
            c_selection_f = c_selection + val

            if self.debug:
                DebugFunction.trace(c_enrichment, self.tag + ".c_enrichment.XX")
                c_enrichment = debug(c_enrichment)
                DebugFunction.trace(c_selection, self.tag + ".c_selection.XX")
                c_selection = debug(c_selection)
                DebugFunction.trace(c_seq_hidden, self.tag + ".c_seq_hidden.XX")
                c_seq_hidden = debug(c_seq_hidden)
                DebugFunction.trace(c_seq_cell, self.tag + ".c_seq_cell.XX")
                c_seq_cell = debug(c_seq_cell)

            # =========== Historical variables selection ==============
            if self.debug:
                DebugFunction.trace(c_selection_h, self.tag + ".c_selection_h")
                c_selection_h = debug(c_selection_h)

            selected_historical, historical_selection_weights = self.apply_temporal_selection(
                temporal_representation=historical_ts_rep,
                static_selection_signal=c_selection_h,
                temporal_selection_module=self.historical_ts_selection, tag = "hist")

            if self.debug:
                DebugFunction.trace(selected_historical, self.tag + ".selected_historical")
                selected_historical = debug(selected_historical)

            # Dimensions:
            # selected_historical: [num_samples x num_historical_steps x state_size]
            # historical_selection_weights: [num_samples x num_historical_steps x total_num_historical_inputs]

            # =========== Future variables selection ==============
            if self.debug:
                DebugFunction.trace(c_selection_f, self.tag + ".c_selection_f")
                c_selection_f = debug(c_selection_f)

            selected_future, future_selection_weights = self.apply_temporal_selection(
                temporal_representation=future_ts_rep,
                static_selection_signal=c_selection_f,
                temporal_selection_module=self.future_ts_selection, tag = "future")
            # Dimensions:
            # selected_future: [num_samples x num_future_steps x state_size]
            # future_selection_weights: [num_samples x num_future_steps x total_num_future_inputs]

            if self.debug:
                DebugFunction.trace(selected_future, self.tag + ".selected_future")
                selected_future = debug(selected_future)

            # =========== Locality Enhancement - Sequential Processing ==============
            gated_lstm_output = self.apply_sequential_processing(selected_historical=selected_historical,
                                                                 selected_future=selected_future,
                                                                 c_seq_hidden=c_seq_hidden,
                                                                 c_seq_cell=c_seq_cell)
            if self.debug:
                DebugFunction.trace(gated_lstm_output, self.tag + ".gated_lstm_output")
                gated_lstm_output = debug(gated_lstm_output)

            # Dimensions:
            # gated_lstm_output : [num_samples x (num_historical_steps + num_future_steps) x state_size]

            # =========== Static enrichment ==============
            enriched_sequence = self.apply_static_enrichment(gated_lstm_output=gated_lstm_output,
                                                             static_enrichment_signal=c_enrichment)
            if self.debug:
                DebugFunction.trace(enriched_sequence, self.tag + ".enriched_sequence")
                enriched_sequence = debug(enriched_sequence)
            # Dimensions:
            # enriched_sequence: [num_samples x (num_historical_steps + num_future_steps) x state_size]

            # =========== self-attention ==============
            gated_post_attention, attention_scores = self.apply_self_attention(enriched_sequence=enriched_sequence,
                                                                               num_historical_steps=num_historical_steps,
                                                                               num_future_steps=num_future_steps)
            if self.debug:
                DebugFunction.trace(gated_post_attention, self.tag + ".gated_post_attention")
                gated_post_attention = debug(gated_post_attention)
                DebugFunction.trace(attention_scores, self.tag + ".attention_scores")
                attention_scores = debug(attention_scores)

            # =========== position-wise feed-forward ==============
            # Applying an additional non-linear processing to the outputs of the self-attention layer using a GRN,
            # where its weights are shared across the entire layer
            post_poswise_ff_grn = self.pos_wise_ff_grn(gated_post_attention)

            if self.debug:
                DebugFunction.trace(post_poswise_ff_grn, self.tag + ".post_poswise_ff_grn")
                post_poswise_ff_grn = debug(post_poswise_ff_grn)

            # Also applying a gated residual connection skipping over the
            # attention block (using sequential processing output), providing a direct path to the sequence-to-sequence
            # layer, yielding a simpler model if additional complexity is not required
            gated_poswise_ff = self.pos_wise_ff_gating(
                post_poswise_ff_grn,
                residual=gated_lstm_output[:, (num_historical_steps + self.target_window_start_idx):, :])

            if self.debug:
                DebugFunction.trace(gated_poswise_ff, self.tag + ".gated_poswise_ff")
                gated_poswise_ff = debug(gated_poswise_ff)

            # Dimensions:
            # gated_poswise_ff: [num_samples x output_sequence_length x state_size]

            # =========== output projection ==============
            # Each predicted quantile has its own projection weights (all gathered in a single linear layer)
            predicted_quantiles = self.output_layer(gated_poswise_ff)
            # Dimensions:
            # predicted_quantiles: [num_samples x num_future_steps x num_quantiles]

            if self.debug:
                DebugFunction.trace(predicted_quantiles, self.tag + ".predicted_quantiles")
                predicted_quantiles = debug(predicted_quantiles)
                DebugFunction.trace(attention_scores, self.tag + ".attention_scores")
                attention_scores = debug(attention_scores)

            return {
                'predicted_quantiles': predicted_quantiles,  # [num_samples x output_sequence_length x num_quantiles]
                'static_weights': static_weights.squeeze(-1),  # [num_samples x num_static_inputs]
                'historical_selection_weights': historical_selection_weights,
                # [num_samples x num_historical_steps x total_num_historical_inputs]
                'future_selection_weights': future_selection_weights,
                # [num_samples x num_future_steps x total_num_future_inputs]
                'attention_scores': attention_scores
                # [num_samples x output_sequence_length x (num_historical_steps + num_future_steps)]
            }
