from torch import nn
from constants import *

import os
import torch
import math
import numpy as np
import zipfile

from test_base import DebugFunction

d = torch.nn.Dropout()

inp = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)
inp.requires_grad = True

debug = DebugFunction.apply
DebugFunction.trace(inp, "input");
inp = debug(inp)

print(d(inp))

loss = torch.sum(inp)

DebugFunction.trace(loss, "loss");
loss = debug(loss)

loss.backward()

