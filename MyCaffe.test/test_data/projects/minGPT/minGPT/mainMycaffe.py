"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from model import GPT
from trainerMycaffe import TrainerMyCaffe
from utils import set_seed, setup_logging, CfgNode as CN
from constants import *

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = working_dir

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = model_type

    # trainer
    C.trainer = TrainerMyCaffe.get_default_config()
    C.trainer.learning_rate = learning_rate # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = seq_len
        return C

    def __init__(self, config, data):
        self.config = config
        self.training_loss = []
        self.training_acc = []

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
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
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if save_for_testing:
        test_path = "test"
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)
    
    training_loss = []
    training_acc = []
    training_acc1 = []

    # construct the model
    #config.model.vocab_size = train_dataset.get_vocab_size()
    #config.model.block_size = train_dataset.get_block_size()
    #model = GPT(config.model)

    # construct the trainer object
    trainer = TrainerMyCaffe(config.trainer, train_dataset)
    
    def update_loss_acc(trainer):
        training_loss.append(trainer.loss())
        training_acc1.append(trainer.accuracy())
        training_acc.append(trainer.calculate_accuracy())
        
        if len(training_loss) > 100:
            training_loss.pop(0)
            training_acc1.pop(0)
            training_acc.pop(0)
        
        return sum(training_loss) / len(training_loss), sum(training_acc1) / len(training_acc1), sum(training_acc) / len(training_acc)
        

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            ave_loss, ave_acc1, ave_acc = update_loss_acc(trainer)
            ave_acc1 = ave_acc * 100
            loss1 = trainer.loss()
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {loss1:.5f}, ave loss {ave_loss:.5f}; ave acc1 {ave_acc1:.5f}, ave acc {ave_acc:.5f} ({ave_acc1:.2f}%)")

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
