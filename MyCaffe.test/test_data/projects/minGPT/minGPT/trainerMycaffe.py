"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

from constants import *

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN
from test_base import DebugFunction
from mycaffe import MyCaffe

class TrainerMyCaffe:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = num_workers
        # optimizer parameters
        C.max_iters = None
        C.batch_size = batch_size
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, train_dataset):
        self.config = config
        #self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.last_output = None
        self.last_target = None
        self.mycaffe = MyCaffe(True)
        
        # determine the device we'll train on
        #self.model = self.model.to(device)
        print("running on device", device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def loss(self):
        return self.mycaffe.current_loss()

    def accuracy(self):
        return self.mycaffe.current_accuracy()

    def calculate_accuracy(self):
        output = self.last_output.detach().cpu().numpy()
        trg = self.last_target.detach().cpu().numpy()
        
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

    def run(self):
        config = self.config;
        #model, config = self.model, self.config

        # setup the optimizer
        #self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        
        #model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        while True:
            if save_for_testing:
                DebugFunction.set_output_path(self.iter_num)
            
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(device) for t in batch]
            x, y = batch

            # forward the model
            prob = self.mycaffe.step(self.iter_num, x, y) 
            #logits, self.loss = model(x, y)            

            # backprop and update the parameters
            #model.zero_grad(set_to_none=True)
            #self.loss.backward()
            #if clip_grad:
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            #self.optimizer.step()

            self.last_output = prob
            self.last_target = y

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
