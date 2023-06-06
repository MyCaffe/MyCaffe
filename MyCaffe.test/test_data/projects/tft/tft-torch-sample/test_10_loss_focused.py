import copy
import os
cwd = os.getcwd()

import numpy as np
import torch
from utility import load_batch, DebugFunction
from tft_torch.loss import get_quantiles_loss_and_q_risk

strSubPath = "loss"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

os.chdir(cwd)
print("\n" + os.getcwd())

strPathWt = "data/favorita/weights/"
strPath = "test/iter_0.base_set/"

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

outputs = np.load(strPath + "tft.loss.outputs.npy")
outputs = torch.from_numpy(outputs).to(device)
outputs.requires_grad = True

targets = np.load(strPath + "tft.loss.targets.npy")
targets = torch.from_numpy(targets).to(device)

desired_quantiles = np.array([0.1, 0.5, 0.9])
desired_quantiles = torch.from_numpy(desired_quantiles).to(device)

DebugFunction.trace(outputs, "outputs")
outputs = debug(outputs)
DebugFunction.trace(targets, "targets")

q_loss, q_risk, _ = get_quantiles_loss_and_q_risk(outputs, targets, desired_quantiles=desired_quantiles, debug_opt=True, path=strSubPath)

DebugFunction.trace(q_loss, "q_loss")
q_loss = debug(q_loss)
DebugFunction.trace(q_risk, "q_risk")

grad = np.load(strPath + "tft.loss.q_loss.grad.npy")
grad = torch.from_numpy(grad).to(device)

q_loss.backward(grad)

print("done!");


