from torch import nn
from constants import *

import os
import torch
import math
import numpy as np
import zipfile
from layers_ex import LogSoftmaxEx
from test_base import DebugFunction

subPath = "logsmx"

DebugFunction.set_output_path(subPath, 0)

np.set_printoptions(precision=12, suppress=False)
    
torch.manual_seed(1701)
input = torch.randn(3, 5, requires_grad=True)
print("input")
print(input.detach().cpu().numpy())

debug = DebugFunction.apply

nllgrad = torch.zeros(3, 5)
target = torch.tensor([1, 0, 4])
print("target")
print(target.detach().cpu().numpy())
loss = nn.NLLLoss()

softmaxEx = LogSoftmaxEx(axis=-1)
print("sm")
sm = softmaxEx(input)

DebugFunction.trace(sm, "sm")
sm = debug(sm)

output = loss(sm, target)
output.backward()

softmaxPy = nn.LogSoftmax(dim=-1)

DebugFunction.trace(input, "input")
input = debug(input)

smpy = softmaxPy(input)
print("smpy")
print(smpy.detach().cpu().numpy())
output = loss(smpy, target)
print("smpy grad")
output.backward()

print("done!")
    
