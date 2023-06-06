import torch
import numpy as np
from utility import DebugFunction

arr = [[8, 2, 9, 4, 8.2, 7],
       [1, 10, 12, 0.01, 4, 5],
       [2, 4, 5, 6, 7, 8]]
print(np.percentile(arr, 25, axis=0)) # 2
print(np.percentile(arr, 50, axis=0)) # 5.5
print(np.percentile(arr, 75, axis=0)) # 8

np.save("c:\\temp\\foo.npy", arr)
arr = np.load("c:\\temp\\foo.npy")

def percentile(x, q):
    k = 1 + round(.01 * float(q) * (len(x) - 1))
    result = x[k]
    return result

print(percentile(arr[0], 25))

debug = DebugFunction.apply

x = torch.ones((2,3), requires_grad=True)
DebugFunction.trace(x, "x")
x = debug(x)

x1 = x[1:,:]
DebugFunction.trace(x1, "x1")
x1 = debug(x1)

sum1 = x1.sum()
sum1.backward()


x = torch.FloatTensor([[100,500,700],[1000,1020,1300]])
x.requires_grad = True

DebugFunction.trace(x, "x")
x = debug(x)

mean = x.mean(dim=-1, keepdim=True)

DebugFunction.trace(mean, "mean")
mean = debug(mean)

xmu = (x - mean)

DebugFunction.trace(xmu, "xmu")
xmu = debug(xmu)

xmusq = xmu * xmu

DebugFunction.trace(xmusq, "xmusq")
xmusq = debug(xmusq)

var = xmusq.mean(dim=-1, keepdim=True)

DebugFunction.trace(var, "var")
var = debug(var)

std = var.sqrt()

DebugFunction.trace(std, "std")
std = debug(std)

y = xmu / std

DebugFunction.trace(y, "y")
y = debug(y)

y_sum = y.sum()

DebugFunction.trace(y_sum, "y_sum")
y_sum = debug(y_sum)

loss = y_sum + 1000

DebugFunction.trace(loss, "loss")
loss = debug(loss)

loss.backward()


