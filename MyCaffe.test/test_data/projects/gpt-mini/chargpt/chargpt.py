
"""
Trains a character-level language model.
"""

from cmath import e
import os
import sys
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import mingpt.model
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # model
    C.model = GPT.get_default_config()

    #C.model.model_type = 'gpt-mini1' # _CHANGE_
    C.model.model_type = 'gpt-mini'
    
    # data
    C.data = CharDataset.get_default_config(C.model.model_type)
    
    # trainer
    C.trainer = Trainer.get_default_config(C.model.model_type)
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config(model_type):
        C = CN()
        C.block_size = 128
               
        if model_type == "gpt-pico" or model_type == "gpt-pico3":
            C.block_size = 4
        if model_type == "gpt-picoB" or model_type == "gpt-pico3B":
            C.block_size = 4
        if model_type == "gpt-pico3B5":
            C.block_size = 4
            
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        self.max_idx = data_size - (config.block_size + 1)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size
    
    def __getitem__(self, idx):
        # Used to get same data ordering
        #idx = model.get_next_index(self.max_idx)
        idx = random.randint(0, self.max_idx)
        
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)      
        #print('idx = %d' % idx)
        #file1 = open("c:\\temp\\snap\\idx.txt", "a")
        #file1.write('idx = %d\n' % idx)
        #file1.close()
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)
    
    # construct the training dataset
    if config.model.model_type == "gpt-mini" or config.model.model_type == "gpt-mini1" or config.model.model_type == "gpt2":
        text = open('input.txt', 'r').read()
    else:
        text = open('test_input.txt', 'r').read() # don't worry we won't run out of file handles _CHANGE_
    train_dataset = CharDataset(config.data, text)
    
    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    #model = GPT.from_pretrained(config.model.model_type)
    #model.save_internal_weights()
    
    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset) 
    
    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        #file1 = open("c:\\temp\\snap\\idx.txt", "a")
        #file1.write(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}\n")
        #file1.close()
        
        if trainer.iter_num > 0 and trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
