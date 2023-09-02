import os
cwd = os.getcwd()

import torch
import torch.nn as nn
import numpy as np
from utility import DebugFunction
from torch_cfc import LTCCell

DebugFunction.seed_everything()

input_size = 82
batch_size = 128
hidden_size = 256
no_gate = False

os.chdir(cwd)
print(os.getcwd())

if no_gate:
    strSubPath = "ltc_cell_no_gate"
else:
    strSubPath = "ltc_cell_gate"

_debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

x = torch.randn(batch_size, input_size, requires_grad=True)
hx = torch.randn(batch_size, hidden_size, requires_grad=True)
ts = torch.zeros(batch_size, requires_grad=True)

BEST_LTC = {
    "optimizer": "adam",
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "forget_bias": 2.4,
    "epochs": 80,
    "class_weight": 8,
    "clipnorm": 0,
    "hidden_size": 64,
    "backbone_units": 64,
    "backbone_dr": 0.1,
    "backbone_layers": 3,
    "weight_decay": 0,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 64,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
    "debug" : True,
}

DebugFunction.trace(x, "x")
DebugFunction.trace(hx, "hx")
DebugFunction.trace(ts, "ts")

x = _debug(x)
hx = _debug(hx)
ts = _debug(ts)

ltc_cell = LTCCell(input_size, hidden_size, BEST_LTC)

idx = 0
ltc_cell.save_weights()

h_state = ltc_cell(x, hx, ts)

DebugFunction.trace(h_state, "h_state")
h_state = _debug(h_state)

h_state.sum().backward()

print("done!")

