# USE CUDA GPU 0 if available
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        

import torch

# Path or parameters for data
DATA_DIR = f'en_fr/data' # f'en_fr/data'
SP_DIR = f'{DATA_DIR}/sp'
SRC_DIR = 'src'
TRG_DIR = 'trg'
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
TRAIN_NAME = 'train.txt'
VALID_NAME = 'valid.txt'
TEST_NAME = 'test.txt'

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
sp_vocab_size = 16000 #en_fr: 16000, chatbot: 3168
character_coverage = 1.0
model_type = 'unigram'

# Parameters for Transformer & training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 1e-4
loss_weight = 1
save_for_testing = True

max_load = 1000 if save_for_testing else 300000 # 30000000

mycaffe_layernorm = True
mycaffe_softmax = True
mycaffe_logsoftmax = True
run_validation = True

batch_size = 3 if save_for_testing else 40
seq_len = 200
num_heads = 8
num_layers = 6
d_model = 512
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.0 # if save_for_testing else 0.1
num_epochs = 30
beam_size = 8
ckpt_dir = 'en_fr/saved_model' # 'en_fr/saved_model'
