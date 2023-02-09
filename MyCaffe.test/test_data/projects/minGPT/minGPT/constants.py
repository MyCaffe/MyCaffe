# USE CUDA GPU 0 if available
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        

import torch

mycaffe_layernorm = False
mycaffe_softmax = False
mycaffe_logsoftmax = False
mycaffe_innerproduct = False
act_relu = True
clip_grad = False

num_workers = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 5e-4
save_for_testing = False

drop_out_rate = 0.0 if save_for_testing else 0.1

seq_len = 128
batch_size = 64
loss_weight = 1
working_dir = './out/chargpt'

pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3

#'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
# GPT-2 configs
#'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
#'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
#'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
#'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
# Gophers
#'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
# (there are a number more...)
# I made these tiny models up
#'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
#'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
#'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),

model_type = "gpt-mini"
