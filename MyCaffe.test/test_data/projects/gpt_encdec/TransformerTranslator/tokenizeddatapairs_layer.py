﻿import os
import clr
clr.AddReference("System")

import sentencepiece as spm
import numpy as np
import torch

import sentencepiece as spm

strDataDir = "C:/ProgramData/MyCaffe/test_data/data/text/encdec/";
cacheDir = strDataDir + "cache/"
loadLimit = 0xFFFFFFFF
DATA_DIR = strDataDir + f'en_fr/data' 
SP_DIR = f'{DATA_DIR}/sp'
SRC_DIR = 'src'
TRG_DIR = 'trg'
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
TRAIN_NAME = 'train.txt'
VALID_NAME = 'valid.txt'
TEST_NAME = 'test.txt'

pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
sp_vocab_size = 16000 

seq_len = 200

src_sp = spm.SentencePieceProcessor()
trg_sp = spm.SentencePieceProcessor()
src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

def pad_or_truncate(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text

def process_src(text_list):
    tokenized_list = []
    idx = 0
    for text in text_list:
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))
        idx = idx + 1
        if idx == loadLimit:
            break;

    return tokenized_list

def process_trg(text_list):
    input_tokenized_list = []
    output_tokenized_list = []
    idx = 0
    for text in text_list:
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized
        trg_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(trg_input))
        output_tokenized_list.append(pad_or_truncate(trg_output))
        idx = idx + 1
        if idx == loadLimit:
            break;
        
    return input_tokenized_list, output_tokenized_list

def make_mask(enc, dec):
    e_mask = (enc != pad_id).unsqueeze(1) # (num_samples, 1, L)
    d_mask = (dec != pad_id).unsqueeze(1) # (num_samples, 1, L)
    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
    d_mask = d_mask & nopeak_mask # (num_samples, L, L) padding false
    return e_mask, d_mask

def get_data(file_name):
    print(f"Getting source/target {file_name}...")
    with open(f"{DATA_DIR}/{SRC_DIR}/{file_name}", 'r', encoding="utf8") as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TRG_DIR}/{file_name}", 'r', encoding="utf8") as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list) # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(trg_text_list) # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    enc = np.asarray(src_list)
    dec = np.asarray(input_trg_list)
    trg = np.asarray(output_trg_list)
          
    num = enc.shape[0]
    channels = enc.shape[1]

    enc1 = torch.from_numpy(enc)
    dec1 = torch.from_numpy(dec)
    e_mask, d_mask = make_mask(enc1, dec1)
    del enc1
    del dec1
    
    emsk = e_mask.detach().numpy()
    dmsk = d_mask.detach().numpy()
    
    del e_mask
    del d_mask

    return num, channels, enc, dec, trg, emsk, dmsk

def get_full_data():
    print("Loading vocabs...")
    src_i2w = {}
    trg_i2w = {}

    with open(f"{SP_DIR}/{src_model_prefix}.vocab", encoding="utf8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        src_i2w[i] = word

    with open(f"{SP_DIR}/{trg_model_prefix}.vocab", encoding="utf8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        trg_i2w[i] = word
        
    print(f"The size of src vocab is {len(src_i2w)} and that of trg vocab is {len(trg_i2w)}.")
    
    print("Load the data...")
    num, channels, enc, dec, trg, emsk, dmsk = get_data(TRAIN_NAME)

    if not os.path.exists(cacheDir):
        os.makedirs(cacheDir)
    
    encFile = f"{cacheDir}train_enc.npy";
    np.save(encFile, enc)
    decFile = f"{cacheDir}train_dec.npy"
    np.save(decFile, dec)
    trgFile = f"{cacheDir}train_trg.npy"
    np.save(trgFile, trg)
    emskFile = f"{cacheDir}train_emsk.npy"
    np.save(emskFile, emsk)
    dmskFile = f"{cacheDir}train_dmsk.npy"
    np.save(dmskFile, dmsk)
     
    return {'num': num, 'channels': channels, 'encfile': encFile, 'decfile': decFile, 'trgFile':trgFile, 'emskFile':emskFile, 'dmskFile':dmskFile }    

res = get_full_data()
