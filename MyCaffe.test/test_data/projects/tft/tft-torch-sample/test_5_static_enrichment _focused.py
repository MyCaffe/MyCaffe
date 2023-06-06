import os
cwd = os.getcwd()

import copy
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork, GateAddNorm
from tft_torch.base_blocks import TimeDistributed, NullTransform
from tft_helper import apply_static_enrichment

os.chdir(cwd)
print(os.getcwd())

random_seed = 1 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

strSubPath = "statenr"
debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

strPathWt = "data/favorita/weights/"



num_categorical = 9
categorical_cardinalities = [2,3,8,13,72,6,28]
dropout = 0.0
num_numeric = 0
state_size = 64
lstm_layers = 2

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

static_enrichment_grn = GatedResidualNetwork(input_dim=state_size,
                                                          hidden_dim=state_size,
                                                          output_dim=state_size,
                                                          context_dim=state_size,
                                                          dropout=dropout, debug=True, tag="tft.statenr1", use_mycaffe=True, path=strSubPath)

idx = 0
for param in static_enrichment_grn.state_dict():
        strFile = strPathWt + "static_enrichment_grn/" + param + ".npy"
        data = np.load(strFile)
        static_enrichment_grn.state_dict()[param] = torch.from_numpy(data).to(device)        
        idx = idx + 1
static_enrichment_grn.to(device)

gated_lstm_output = np.load("test/iter_0.base_set/tft.statenr.gated_lstm_output.npy")
gated_lstm_output = torch.from_numpy(gated_lstm_output).to(device)
gated_lstm_output.requires_grad = True

DebugFunction.trace(gated_lstm_output, "tft.statenr.gated_lstm_output.val")
gated_lstm_output = debug(gated_lstm_output)

c_enrichment = np.load("test/iter_0.base_set/tft.statenr.static_enrichment_signal.npy")
c_enrichment = torch.from_numpy(c_enrichment).to(device)
c_enrichment.requires_grad = True

DebugFunction.trace(c_enrichment, "tft.statenr.c_enrichment.val")
c_enrichment = debug(c_enrichment)

for param in static_enrichment_grn.state_dict():
        DebugFunction.trace(static_enrichment_grn.state_dict()[param], "tft.stateenr." + param)    

enriched_sequence = apply_static_enrichment(gated_lstm_output=gated_lstm_output,
                                                         static_enrichment_signal=c_enrichment,
                                                         static_enrichment_grn=static_enrichment_grn, state_size=state_size, path=strSubPath)


DebugFunction.trace(enriched_sequence, "enriched_sequence")
enriched_sequence = debug(enriched_sequence)

grad = np.load("test/iter_0.base_set/tft.statenr.enriched_sequence.grad.npy")
grad = torch.from_numpy(grad).to(device)

enriched_sequence.backward(grad)

print("done!");


