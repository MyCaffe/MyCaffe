# Code to generate the trainingset for TokenizedDataPairsLayerPy.
# @see [GitHub:devjwsong/transformer-translator-pytorch](https://github.com/devjwsong/transformer-translator-pytorch/tree/master/src) by Jaewoo (Kyle) Song, 2021, 
# distributed under MIT License https://github.com/devjwsong/transformer-translator-pytorch/blob/master/LICENSE
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import torch
import sys, os
import numpy as np
import random
import sentencepiece as spm

strDataDir = f'C:/ProgramData/MyCaffe/test_data/data/text/encdec'
DATA_DIR = strDataDir + f'/en_fr/data' 
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
sp_vocab_size = 16000 

batch_size = 40
seq_len = 200

src_sp = spm.SentencePieceProcessor()
trg_sp = spm.SentencePieceProcessor()
src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_data_loader(file_name):
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

    set_seed(1701)

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return dataloader


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
    for text in tqdm(text_list):
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))

    return tokenized_list


def process_trg(text_list):
    input_tokenized_list = []
    output_tokenized_list = []
    for text in tqdm(text_list):
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized
        trg_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(trg_input))
        output_tokenized_list.append(pad_or_truncate(trg_output))

    return input_tokenized_list, output_tokenized_list


class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(input_trg_list), "The shape of src_list and input_trg_list are different."
        assert np.shape(input_trg_list) == np.shape(output_trg_list), "The shape of input_trg_list and output_trg_list are different."

    def make_mask(self):
        e_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        d_mask = (self.input_trg_data != pad_id).unsqueeze(1) # (num_samples, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask # (num_samples, L, L) padding false

        return e_mask, d_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]

    
class Manager():
    def __init__(self):
        # Load vocabs
        print("Loading vocabs...")
        self.src_i2w = {}
        self.trg_i2w = {}

    def initialize(self):
        if not os.path.exists(f"{SP_DIR}/{src_model_prefix}.vocab"):
            print(f"You must first run src/sentencepiece_train.py as discussed at:")
            print(f"https://github.com/devjwsong/transformer-translator-pytorch")
            return False

        with open(f"{SP_DIR}/{src_model_prefix}.vocab", encoding="utf8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_i2w[i] = word

        with open(f"{SP_DIR}/{trg_model_prefix}.vocab", encoding="utf8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.trg_i2w[i] = word

        print(f"The size of src vocab is {len(self.src_i2w)} and that of trg vocab is {len(self.trg_i2w)}.")
        
        # Load Transformer model & Adam optimizer
        set_seed(1701)
        self.train_loader = get_data_loader(TRAIN_NAME)
        self.valid_loader = get_data_loader(VALID_NAME) 
        return True
                
    def generate_data(self, strType, strSubDir):
        print(f"Creating {strType} data...")

        if seq_len == 200:        
            cachePath = strDataDir + f"/cache"
        else:
            cachePath = strDataDir + f"/cache{seq_len}"
        if not os.path.exists(cachePath):
            os.mkdir(cachePath)

        cachePath = cachePath + f"/{strSubDir}"            
        if not os.path.exists(cachePath):
            os.mkdir(cachePath)

        nTotal = 40155 if strSubDir == "train" else 10039
        nIdx = 0

        loader = self.train_loader if strSubDir == "train" else self.valid_loader

        for i, batch in tqdm(enumerate(loader)):
            nIdx += 1
            pct = (nIdx/nTotal) * 100

            strEncFile = cachePath + f"/{nIdx}_enc.npy"
            strDecFile = cachePath + f"/{nIdx}_dec.npy"
            strTrgFile = cachePath + f"/{nIdx}_trg.npy"

            enc, dec, trg = batch
                
            np.save(strEncFile, enc.detach().cpu().numpy())
            np.save(strDecFile, dec.detach().cpu().numpy())
            np.save(strTrgFile, trg.detach().cpu().numpy())

            if (nIdx % 100 == 0):
                print(f" Saving data at {pct} %")

            del enc, dec, trg
            torch.cuda.empty_cache()                    
        print(f"Items in {strType} loader = {nIdx}")

if __name__=='__main__':
    manager = Manager()
    if manager.initialize():
        manager.generate_data("training", "train")
        manager.generate_data("validation", "valid")
        print(f"finished creating data!")
        print(f"Training data in: {strDataDir}/cache/train directory.")
        print(f"Validation data in: {strDataDir}/cache/valid directory.")
