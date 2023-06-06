import copy
import os
cwd = os.getcwd()

import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple
from utility import load_batch, DebugFunction
from tft_torch.tft import InputChannelEmbedding, VariableSelectionNetwork, GatedResidualNetwork, GatedLinearUnit, GateAddNorm, InterpretableMultiHeadAttention
from tft_torch.base_blocks import TimeDistributed, NullTransform
from tft_helper import apply_temporal_selection, apply_self_attention

os.chdir(cwd)
print("\n" + os.getcwd())

strSubPath = "imha"
strPathWt = "data/favorita/weights/"
strPath = "test/iter_0.base_set/"

debug = DebugFunction.apply
DebugFunction.set_output_path(strSubPath, 0)

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

state_size = 64
attention_heads = 4
dropout = 0

multihead_attn = InterpretableMultiHeadAttention(embed_dim=state_size, num_heads=attention_heads, tag="tft.asa.imha", debug=True, use_mycaffe=True)
post_attention_gating = GateAddNorm(input_dim=state_size, dropout=dropout, tag="tft.asa.gan", debug=True, use_mycaffe=True, disable_layer_norm=True)

for param in multihead_attn.state_dict():
        strFile = strPathWt + "multihead_attn/" + param + ".npy"
        data = np.load(strFile)
        multihead_attn.state_dict()[param] = torch.from_numpy(data).to(device)        
multihead_attn.to(device)

for param in post_attention_gating.state_dict():
        strFile = strPathWt + "post_attention_gating/" + param + ".npy"
        data = np.load(strFile)
        post_attention_gating.state_dict()[param] = torch.from_numpy(data).to(device)        
post_attention_gating.to(device)

for param in multihead_attn.state_dict():
    DebugFunction.trace(multihead_attn.state_dict()[param], "tft.test.multihead_attn." + param, "weights")

for param in post_attention_gating.state_dict():
    DebugFunction.trace(post_attention_gating.state_dict()[param], "tft.test.post_attention_gating_attn." + param, "weights")

enriched_sequence = np.load(strPath + "tft.ada.enriched_sequence.npy")
enriched_sequence = torch.from_numpy(enriched_sequence).to(device)
enriched_sequence.requires_grad = True

nun_historical_steps = 90
num_future_steps = 30
target_window_start_idx = 0

DebugFunction.trace(enriched_sequence, "tft.ada.enriched_sequence1")
enriched_sequence = debug(enriched_sequence)

gated_post_attention, attention_scores = apply_self_attention(enriched_sequence=enriched_sequence, num_historical_steps=nun_historical_steps, num_future_steps=num_future_steps, multihead_attn=multihead_attn, post_attention_gating=post_attention_gating, target_window_start_idx=target_window_start_idx, path=strSubPath)

DebugFunction.trace(gated_post_attention, "tft.ada.gated_post_attention")
DebugFunction.trace(attention_scores, "tft.ada.attention_scores")
attention_scores = debug(attention_scores)

grad = np.load(strPath + "tft.ada.gated_post_attention.grad.npy")
grad = torch.from_numpy(grad).to(device)

grad = grad * 1000
DebugFunction.trace(grad, "tft.ada.gated_post_attention.grad")

gated_post_attention.backward(grad)

DebugFunction.trace(multihead_attn.w_q.weight.grad, "tft.test.multihead_attn.w_q.weight.grad", "weights")
DebugFunction.trace(multihead_attn.w_q.bias.grad, "tft.test.multihead_attn.w_q.bias.grad", "weights")
DebugFunction.trace(multihead_attn.w_k.weight.grad, "tft.test.multihead_attn.w_k.weight.grad", "weights")
DebugFunction.trace(multihead_attn.w_k.bias.grad, "tft.test.multihead_attn.w_k.bias.grad", "weights")
DebugFunction.trace(multihead_attn.w_v.weight.grad, "tft.test.multihead_attn.w_v.weight.grad", "weights")
DebugFunction.trace(multihead_attn.w_v.bias.grad, "tft.test.multihead_attn.w_v.bias.grad", "weights")
DebugFunction.trace(multihead_attn.out.weight.grad, "tft.test.multihead_attn.out.weight.grad", "weights")
DebugFunction.trace(multihead_attn.out.bias.grad, "tft.test.multihead_attn.out.bias.grad", "weights")

DebugFunction.trace(post_attention_gating.gate.module.fc1.weight.grad, "tft.test.post_attention_gating_attn.gate.module.fc1.weight.grad", "weights")
DebugFunction.trace(post_attention_gating.gate.module.fc1.bias.grad, "tft.test.post_attention_gating_attn.gate.module.fc1.bias.grad", "weights")
DebugFunction.trace(post_attention_gating.gate.module.fc2.weight.grad, "tft.test.post_attention_gating_attn.gate.module.fc2.weight.grad", "weights")
DebugFunction.trace(post_attention_gating.gate.module.fc2.bias.grad, "tft.test.post_attention_gating_attn.gate.module.fc2.bias.grad", "weights")

print("done!");


