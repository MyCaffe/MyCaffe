from torch import nn
from constants import *

import os
import torch
import math
import numpy as np
import zipfile
import random
from mycaffe import MyCaffe

input_dict = { None : "" }
mycaffe = MyCaffe(False)

class DebugFunction(torch.autograd.Function):
    out_path = "test/"

    @staticmethod
    def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def set_output_path(i):
        DebugFunction.out_path = "test/iter_%d/" % i
        if not os.path.exists(DebugFunction.out_path):
            os.makedirs(DebugFunction.out_path)

    @staticmethod
    def trace(t, name):
        if not os.path.exists(DebugFunction.out_path):
            os.makedirs(DebugFunction.out_path)
        input_dict.update({t: name})
        np.save(DebugFunction.out_path + name + ".npy", t.detach().cpu().numpy())

    @staticmethod
    def traceD(t, name):
        if not os.path.exists(DebugFunction.out_path):
            os.makedirs(DebugFunction.out_path)
        input_dict.update({t: name})
        np.save(DebugFunction.out_path + name + ".d.npy", t.detach().type(torch.cuda.DoubleTensor).cpu().numpy())

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
            
        if name == "15_loss":
            grad_output = grad_output * loss_weight

        if save_for_testing:
            np.save(DebugFunction.out_path + "grad_" + name, grad_output.detach().cpu().numpy())
        return grad_output
    
    @staticmethod
    def load(name):
        filename = DebugFunction.out_path + name + ".npy"
        itemnpy = np.load(filename)
        return torch.from_numpy(itemnpy)