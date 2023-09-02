import os
import numpy as np
import torch
import random
from typing import Dict

input_dict = { None : "" }
loss_weight = 1

class DebugFunction(torch.autograd.Function):
    out_path = "test/"

    @staticmethod
    def seed_everything(seed=42):
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    @staticmethod
    def set_output_path(path, i):
        if path == "":
            print("DebugFunction missing path")
        DebugFunction.out_path = "test/" + path + "/iter_%d/" % i
        if not os.path.exists(DebugFunction.out_path):
            os.makedirs(DebugFunction.out_path)
        return DebugFunction.out_path

    @staticmethod
    def trace(t, name, subpath=None):
        path = DebugFunction.out_path
        if subpath != None:
            path += subpath + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        input_dict.update({t: name})
        filename = path + name + ".npy"
        np.save(filename, t.detach().cpu().numpy())

    @staticmethod
    def trace_ex(strPath, t, name):
        if not os.path.exists(strPath):
            os.makedirs(strPath)
        input_dict.update({t: name})
        filename = strPath + name + ".npy"
        np.save(filename, t.detach().cpu().numpy())

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

        #print("bwd: " + name)
        np.save(DebugFunction.out_path + name + ".grad", grad_output.detach().cpu().numpy())
        return grad_output

