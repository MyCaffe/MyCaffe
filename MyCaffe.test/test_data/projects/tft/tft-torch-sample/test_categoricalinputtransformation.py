import numpy as np
import torch
from torch import nn
from typing import Dict
from utility import load_batch, DebugFunction
from tft_torch.tft import CategoricalInputTransformation
from tft_torch.base_blocks import TimeDistributed, NullTransform

debug = DebugFunction.apply
DebugFunction.set_output_path(0)

strPathWt = "data/favorita/weights"
strPath = "data/favorita/batch256/batch_0"
batch = load_batch(strPath)

empty_tensor = torch.empty((0, 0))
x_categorical = batch.get('historical_ts_categorical', empty_tensor)

num_categorical = 7
categorical_cardinalities = [2,3,8,13,72,6,28]

num_numeric = 4
state_size = 64

categorical_transform = TimeDistributed(
                CategoricalInputTransformation(num_inputs=num_categorical, state_size=state_size,
                                               cardinalities=categorical_cardinalities), return_reshaped=False)

wt0 = np.load(strPathWt + "/17_historical_ts_transform.categorical_transform.module.categorical_embedding_layers.0.weight.npy")
categorical_transform.module.categorical_embedding_layers[0].weight = nn.Parameter(torch.from_numpy(wt0))
DebugFunction.trace(categorical_transform.module.categorical_embedding_layers[0].weight, "emb_layer.0.weight")

wt1 = np.load(strPathWt + "/18_historical_ts_transform.categorical_transform.module.categorical_embedding_layers.1.weight.npy")
categorical_transform.module.categorical_embedding_layers[1].weight = nn.Parameter(torch.from_numpy(wt1))
DebugFunction.trace(categorical_transform.module.categorical_embedding_layers[1].weight, "emb_layer.1.weight")

wt2 = np.load(strPathWt + "/19_historical_ts_transform.categorical_transform.module.categorical_embedding_layers.2.weight.npy")
categorical_transform.module.categorical_embedding_layers[2].weight = nn.Parameter(torch.from_numpy(wt2))
DebugFunction.trace(categorical_transform.module.categorical_embedding_layers[2].weight, "emb_layer.2.weight")

wt3 = np.load(strPathWt + "/20_historical_ts_transform.categorical_transform.module.categorical_embedding_layers.3.weight.npy")
categorical_transform.module.categorical_embedding_layers[3].weight = nn.Parameter(torch.from_numpy(wt3))
DebugFunction.trace(categorical_transform.module.categorical_embedding_layers[3].weight, "emb_layer.3.weight")

is_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if is_cuda else "cpu")
categorical_transform.to(device)
x_categorical = x_categorical.to(device)

DebugFunction.trace(x_categorical, "x_categorical")

processed_categorical = categorical_transform(x_categorical)

DebugFunction.trace(processed_categorical[0], "processed_categorical_0")
DebugFunction.trace(processed_categorical[1], "processed_categorical_1")
DebugFunction.trace(processed_categorical[2], "processed_categorical_2")
DebugFunction.trace(processed_categorical[3], "processed_categorical_3")
processed_categorical[0] = debug(processed_categorical[0])
processed_categorical[1] = debug(processed_categorical[1])
processed_categorical[2] = debug(processed_categorical[2])
processed_categorical[3] = debug(processed_categorical[3])

p0 = processed_categorical[0].clone()
p1 = processed_categorical[1].clone()
p2 = processed_categorical[2].clone()
p3 = processed_categorical[3].clone()

p0 = p0 * 0 + 1
p1 = p1 * 0 + 1
p2 = p2 * 0 + 1
p3 = p3 * 0 + 1

loss = (processed_categorical[0] - p0).sum() + (processed_categorical[1] - p1).sum() + (processed_categorical[2] - p2).sum() + (processed_categorical[3] - p3).sum()

DebugFunction.trace(loss, "loss")
loss = debug(loss)

loss.backward()

DebugFunction.trace(categorical_transform.module.categorical_embedding_layers[0].weight.grad, "emb_layer.0.weight_grad")
DebugFunction.trace(categorical_transform.module.categorical_embedding_layers[1].weight.grad, "emb_layer.1.weight_grad")
DebugFunction.trace(categorical_transform.module.categorical_embedding_layers[2].weight.grad, "emb_layer.2.weight_grad")
DebugFunction.trace(categorical_transform.module.categorical_embedding_layers[3].weight.grad, "emb_layer.3.weight_grad")

print("done!");


