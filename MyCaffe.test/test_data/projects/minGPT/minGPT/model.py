"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from pickle import TRUE
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import CfgNode as CN

from adamw import AdamW2
from layers import LinearEx
from layers import LayerNormEx
from layers import SoftmaxEx
from layers import LogSoftmaxEx
from layers import BlockEx
from layers import BlockAllEx
from test_base import DebugFunction
from constants import *

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
            
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        if act_relu:
            return self.relu(x)
        else:
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, tag, config):
        super().__init__()
        self.tag = tag
        assert config.n_embd % config.n_head == 0
        self.softmax = SoftmaxEx(tag + ".smx", dim=-1)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = LinearEx(tag + ".c_attn", config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = LinearEx(tag + ".c_proj", config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
    def save_internal_state(self):
        DebugFunction.trace(self.c_attn.weight, self.tag + ".c_attn.weight")
        DebugFunction.trace(self.c_attn.bias, self.tag + ".c_attn.bias")
        DebugFunction.trace(self.c_proj.weight, self.tag + ".c_proj.weight")
        DebugFunction.trace(self.c_proj.bias, self.tag + ".c_proj.bias")
        
    def load_internal_state(self):
        self.c_attn.weight = DebugFunction.load(self.tag + ".c_attn.weight")
        self.c_attn.bias = DebugFunction.load(self.tag + ".c_attn.bias")
        self.c_proj.weight = DebugFunction.load(self.tag + ".c_proj.weight")
        self.c_proj.bias = DebugFunction.load(self.tag + ".c_proj.bias")

    def forward(self, x):
        if save_for_testing:
            self.save_internal_state()
        
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        inf = 1e+29
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, -inf) # float('-inf'))
        att = self.softmax(att) 
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """
    
    def __init__(self, tag, config, use_mycaffe=False):
        super().__init__()
        self.tag = tag
        self.ln_1 = LayerNormEx(tag + ".ln1", config.n_embd)
        self.attn = CausalSelfAttention(tag + ".attn", config)
        self.ln_2 = LayerNormEx(tag + ".ln2", config.n_embd)

        self.c_fc = LinearEx(tag + ".c_fc", config.n_embd, 4 * config.n_embd, use_mycaffe = mycaffe_innerproduct_fc)
        self.c_proj = LinearEx(tag + ".c_proj", 4 * config.n_embd, config.n_embd, use_mycaffe=mycaffe_innerproduct_proj)
        self.act = NewGELU()
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        #self.mlp = nn.ModuleDict(dict(
        #    c_fc    = LinearEx(tag + ".c_fc", config.n_embd, 4 * config.n_embd),
        #    c_proj  = LinearEx(tag + ".c_proj", 4 * config.n_embd, config.n_embd, use_mycaffe=mycaffe_innerproduct_proj),
        #    act     = NewGELU(),
        #    dropout = nn.Dropout(config.resid_pdrop),
        #))
        #m = self.mlp
        #self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forwardOriginal(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

    def forward(self, x):
        x1 = x + self.attn(self.ln_1(x))
        ln2_x = self.ln_2(x1)
        fc_x = self.c_fc(ln2_x)
        act_x = self.act(fc_x)
        proj_x = self.c_proj(act_x)
        dropout_x = self.dropout(proj_x)
        x2 = x1 + dropout_x
        return x2


class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = drop_out_rate
        C.resid_pdrop = drop_out_rate
        C.attn_pdrop = drop_out_rate
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini1':     dict(n_layer=1, n_head=6, n_embd=192),
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        if mycaffe_transformerblock_all:        
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.embd_pdrop),
                h = BlockAllEx("blkall", config),
                ln_f = LayerNormEx("ln_f", config.n_embd),
            ))
        elif mycaffe_transformerblock:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.embd_pdrop),
                h = nn.ModuleList([BlockEx("blk%d" % (i), config) for i in range(config.n_layer)]),
                ln_f = LayerNormEx("ln_f", config.n_embd),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.embd_pdrop),
                h = nn.ModuleList([Block("blk%d" % (i), config) for i in range(config.n_layer)]),
                ln_f = LayerNormEx("ln_f", config.n_embd),
            ))
        self.lm_head = LinearEx("lm_head", config.n_embd, config.vocab_size, bias=False)
        self.softmax = LogSoftmaxEx("f_smx", dim = -1)
        self.criterion = nn.NLLLoss()

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (LinearEx, )
        blacklist_weight_modules = (LayerNormEx, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        if mycaffe_adamw:
            optimizer = AdamW2(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def calculate_accuracy(self, output, trg):
        output = output.detach().cpu().numpy()
        trg = trg.detach().cpu().numpy()
        
        output = np.argmax(output, axis=-1)        

        correct = 0;
        total = 0;
        for i in range(len(output)):
            for j in range(len(output[i])):
                if output[i][j] == trg[i][j]:
                    correct += 1
                total += 1
        accuracy = correct / total

        return accuracy
    
    def forward(self, idx, targets=None):
        debug = DebugFunction.apply
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        save_for_testing1 = False
        if save_for_testing and targets != None:
            save_for_testing1 = True

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        if save_for_testing1:
            DebugFunction.trace(idx, "1_x")
            DebugFunction.trace(x, "1_x_emb")
            x = debug(x)
            DebugFunction.trace(targets, "1_targets")

        idx = 0
        if mycaffe_transformerblock_all:
            x = self.transformer.h(x)
            if save_for_testing1:
                DebugFunction.trace(x, "2_x")
                x = debug(x)
        else:
            for block in self.transformer.h:
                x = block(x)
                if save_for_testing1:
                    DebugFunction.trace(x, "%d_blk_x" % (idx))
                    x = debug(x)
                    idx = idx + 1
            
        ln_x = self.transformer.ln_f(x)
        if save_for_testing1:
            DebugFunction.trace(ln_x, "12_ln_x")
            ln_x = debug(ln_x)
        
        logits = self.lm_head(ln_x)
        if save_for_testing1:
            DebugFunction.trace(logits, "13_logits")
            logits = debug(logits)
        
        # if we are given some desired targets also calculate the loss
        loss = None
        acc = None
        if targets is not None:
            prob = self.softmax(logits)
            if save_for_testing1 or True:
                DebugFunction.trace(prob, "14_prob")
                prob = debug(prob)
            
            loss = self.criterion(prob.view(-1, prob.size(-1)), targets.view(-1))
            if save_for_testing1 or True:
                DebugFunction.trace(loss, "15_loss")
                loss = debug(loss)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
