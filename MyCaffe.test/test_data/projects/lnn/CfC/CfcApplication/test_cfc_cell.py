import os
cwd = os.getcwd()

import torch
import torch.nn as nn
import numpy as np
from utility import DebugFunction
from torch_cfc import CfcCell

DebugFunction.seed_everything()

input_size = 82
batch_size = 128
hidden_size = 256
no_gate = False

os.chdir(cwd)
print(os.getcwd())

if no_gate:
    strSubPath = "cfc_cell_no_gate"
else:
    strSubPath = "cfc_cell_gate"

_debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

x = torch.randn(batch_size, input_size, requires_grad=True)
hx = torch.randn(batch_size, hidden_size, requires_grad=True)
ts = torch.zeros(batch_size, requires_grad=True)

BEST_DEFAULT = {
    "epochs": 57,
    "class_weight": 11.69,
    "clipnorm": 0,
    "hidden_size": 256,
    "base_lr": 0.002,
    "decay_lr": 0.9,
    "backbone_activation": "silu",
    "backbone_units": 64,
    "backbone_dr": 0.0  ,
    "backbone_layers": 2,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.5,
    "batch_size": 128,
    "use_mixed": False,
    "no_gate": no_gate,
    "minimal": False,
    "use_ltc": False,
    "debug": True
}

DebugFunction.trace(x, "x")
DebugFunction.trace(hx, "hx")
DebugFunction.trace(ts, "ts")

x = _debug(x)
hx = _debug(hx)
ts = _debug(ts)

cfc_cell = CfcCell(input_size, hidden_size, BEST_DEFAULT)

idx = 0
for i in range(0, len(cfc_cell.layer_list)):
    if isinstance(cfc_cell.layer_list[i], nn.Linear):
        DebugFunction.trace(cfc_cell.layer_list[i].weight, "bb_" + str(idx) + ".weight", "weights")
        DebugFunction.trace(cfc_cell.layer_list[i].bias, "bb_" + str(idx) + ".bias", "weights")
        idx += 1

DebugFunction.trace(cfc_cell.ff1.weight, "ff1.weight", "weights")
DebugFunction.trace(cfc_cell.ff1.bias, "ff1.bias", "weights")
DebugFunction.trace(cfc_cell.ff2.weight, "ff2.weight", "weights")
DebugFunction.trace(cfc_cell.ff2.bias, "ff2.bias", "weights")
DebugFunction.trace(cfc_cell.time_a.weight, "time_a.weight", "weights")
DebugFunction.trace(cfc_cell.time_a.bias, "time_a.bias", "weights")
DebugFunction.trace(cfc_cell.time_b.weight, "time_b.weight", "weights")
DebugFunction.trace(cfc_cell.time_b.bias, "time_b.bias", "weights")

h_state = cfc_cell(x, hx, ts)

DebugFunction.trace(h_state, "h_state")
h_state = _debug(h_state)

h_state.sum().backward()

print("done!")

