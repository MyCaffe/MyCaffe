import os
cwd = os.getcwd()
import torch
from torch import nn
import numpy as np
from utility import DebugFunction
from tft_torch.tft import LayerNormEx

os.chdir(cwd)
print(os.getcwd())

strPathWt = "data/favorita/weights/"
strPath = "test/iter_0/"
DebugFunction.set_output_path(0)
debug = DebugFunction.apply

x = np.load(strPath + "test4_gan.layernorm.x.npy")
x = torch.from_numpy(x).double()
x.requires_grad = True
N = x.shape[0]
D = x.shape[1]

mean = x.mean(axis = 1)
mean = mean.reshape((mean.shape[0], 1))
xmu = x - mean

var1 = xmu * xmu
var = var1.mean(axis = 1)
var = var.reshape((var.shape[0], 1))

std = torch.sqrt(var)

dx = xmu / std
dxsum = dx.sum()

layernorm = LayerNormEx([D],epsilon=0)
dx2 = layernorm(x)

dx2sum = dx2.sum()

layernorm2 = nn.LayerNorm([D], eps=0)
dx3 = layernorm(x)

dx3sum = dx3.sum()

diff1 = dxsum - dx2sum
diff2 = dxsum - dx3sum

dy = np.load(strPath + "test4_gan.layernorm.y.grad.npy")
dy = torch.from_numpy(dy).double()

rstd = 1.0 / std
rd = 1.0 / float(D)

xhat = (x - mean) * rstd
DebugFunction.trace(xhat, "LLL.xhat")

c1 = (xhat * dy)
DebugFunction.trace(c1, "LLL.c1a")

c1 = c1.sum(axis = 1)
c1 = c1 * rd
c1 = c1.reshape(c1.shape[0],1)
DebugFunction.trace(c1, "LLL.c1b")


c2 = dy.sum(axis = 1)
c2 = c2 * rd
c2 = c2.reshape(c2.shape[0],1)
DebugFunction.trace(c2, "LLL.c2a")

dx1 = xhat * c1
DebugFunction.trace(dx1, "LLL.dx1")

dx2 = dx1 + c2
DebugFunction.trace(dx2, "LLL.dx2")

dx3 = dy - dx2
DebugFunction.trace(dx3, "LLL.dx3")

dx4 = dx3 * rstd
DebugFunction.trace(dx4, "LLL.dx4")

dx = (dy - (xhat * c1 + c2)) * rstd

dxdiffA = dx4 = dx
dxdiffASum = dxdiffA.sum()

dx = dx * x



dx2.backward(dy)

graddiff = x.grad - dx
gradsum = graddiff.sum()

xgradsum = x.grad.sum()
dxsum = dx.sum()
graddiff = xgradsum - dxsum

print("done!")
