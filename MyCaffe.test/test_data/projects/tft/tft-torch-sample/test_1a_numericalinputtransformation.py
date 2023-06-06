import os
cwd = os.getcwd()
import numpy as np
import torch
from torch import nn
from typing import Dict
from utility import load_batch, DebugFunction
from tft_torch.tft import NumericInputTransformation
from tft_torch.base_blocks import TimeDistributed, NullTransform

os.chdir(cwd)
print(os.getcwd())

strSubPath = "numtrx"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPathWt = "data/favorita/weights/hist_ts_transform/"
strPath = "data/favorita/batch256/batch_0"
batch = load_batch(strPath)

empty_tensor = torch.empty((0, 0))
x_numeric = batch.get('historical_ts_numeric', empty_tensor)

num_categorical = 7
categorical_cardinalities = [2,3,8,13,72,6,28]

num_numeric = 4
state_size = 64
numeric_transform = TimeDistributed(NumericInputTransformation(num_inputs=num_numeric, state_size=state_size), return_reshaped=False)


wt0 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.0.weight.npy")
b0 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.0.bias.npy")
numeric_transform.module.numeric_projection_layers[0].weight = nn.Parameter(torch.from_numpy(wt0))
numeric_transform.module.numeric_projection_layers[0].bias = nn.Parameter(torch.from_numpy(b0))
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[0].weight, "proj_layer.0.weight")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[0].bias, "proj_layer.0.bias")

wt1 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.1.weight.npy")
b1 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.1.bias.npy")
numeric_transform.module.numeric_projection_layers[1].weight = nn.Parameter(torch.from_numpy(wt1))
numeric_transform.module.numeric_projection_layers[1].bias = nn.Parameter(torch.from_numpy(b1))
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[1].weight, "proj_layer.1.weight")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[1].bias, "proj_layer.1.bias")

wt2 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.2.weight.npy")
b2 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.2.bias.npy")
numeric_transform.module.numeric_projection_layers[2].weight = nn.Parameter(torch.from_numpy(wt2))
numeric_transform.module.numeric_projection_layers[2].bias = nn.Parameter(torch.from_numpy(b2))
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[2].weight, "proj_layer.2.weight")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[2].bias, "proj_layer.2.bias")

wt3 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.3.weight.npy")
b3 = np.load(strPathWt + "numeric_transform.module.numeric_projection_layers.3.bias.npy")
numeric_transform.module.numeric_projection_layers[3].weight = nn.Parameter(torch.from_numpy(wt3))
numeric_transform.module.numeric_projection_layers[3].bias = nn.Parameter(torch.from_numpy(b3))
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[3].weight, "proj_layer.3.weight")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[3].bias, "proj_layer.3.bias")

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
numeric_transform.to(device)
x_numeric = x_numeric.to(device)
x_numeric.requires_grad = True;

DebugFunction.trace(x_numeric, "x_numeric")
x_numeric = debug(x_numeric)

processed_numeric = numeric_transform(x_numeric)

DebugFunction.trace(processed_numeric[0], "processed_numeric_0")
DebugFunction.trace(processed_numeric[1], "processed_numeric_1")
DebugFunction.trace(processed_numeric[2], "processed_numeric_2")
DebugFunction.trace(processed_numeric[3], "processed_numeric_3")
processed_numeric[0] = debug(processed_numeric[0])
processed_numeric[1] = debug(processed_numeric[1])
processed_numeric[2] = debug(processed_numeric[2])
processed_numeric[3] = debug(processed_numeric[3])

p0 = processed_numeric[0].clone()
p1 = processed_numeric[1].clone()
p2 = processed_numeric[2].clone()
p3 = processed_numeric[3].clone()

p0 = p0 * 0 + 1
p1 = p1 * 0 + 1
p2 = p2 * 0 + 1
p3 = p3 * 0 + 1

loss = (processed_numeric[0] - p0).sum() + (processed_numeric[1] - p1).sum() + (processed_numeric[2] - p2).sum() + (processed_numeric[3] - p3).sum()

DebugFunction.trace(loss, "loss")
loss = debug(loss)

loss.backward()

DebugFunction.trace(numeric_transform.module.numeric_projection_layers[0].weight.grad, "proj_layer.0.weight.grad")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[0].bias.grad, "proj_layer.0.bias.grad")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[1].weight.grad, "proj_layer.1.weight.grad")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[1].bias.grad, "proj_layer.1.bias.grad")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[2].weight.grad, "proj_layer.2.weight.grad")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[2].bias.grad, "proj_layer.2.bias.grad")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[3].weight.grad, "proj_layer.3.weight.grad")
DebugFunction.trace(numeric_transform.module.numeric_projection_layers[3].bias.grad, "proj_layer.3.bias.grad")

print("done!");


