import torch
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os.path
from os import path

from time import time
from pickle import dump, load
from matplotlib.pylab import plt
from numpy import arange
from collections import defaultdict
from Utility import free_memory
import os
import gc

from TransformerModel import TransformerModel
from PrepareDataset import PrepareDataset

'''
The Training module is a PyTorch rewrite of the TensorFlow and Keras version by Stefania Cristina
@see [Plotting the Training and Validation Loss Curves for the Transformer Model](https://machinelearningmastery.com/plotting-the-training-and-validation-loss-curves-for-the-transformer-model/) by Stefania Cristina, 2022
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''

# USE CUDA GPU 0 if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model parameters
h = 8 # Number of self-attention heads
d_k = 64 # Dimension of the queries and keys
d_v = 64 # Dimension of the values
d_ff = 2048 # Demension of the fully connected mpl
d_model = 512 # Dimension of the model
n = 6 # Number of encoder and decoder blocks

# Define the training parameters
batch_size = 64
epochs = 18
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

dataset = PrepareDataset()
trainX, trainY, valX, valY, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length = dataset('data/english-german-both.pkl')

print(enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size)

model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate, batch_size).to(device)

# Prepare the dataset batches
train_dataset = TensorDataset(trainX, trainY)
# Prepare the validation dataset
val_dataset = TensorDataset(valX, valY)

class LRScheduler():
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
                        
class Trainer:
    def __init__(self, model, optimizer, train_dataset, val_dataset, device = None):
        self.device = device if device != None else torch.device('cpu')
        self.model = model
        self.optimizer = optimizer
        self.train_loss = {}
        self.val_loss = {}
        self.train_acc = {}
        self.val_acc = {}
        self.train_time = {}
        self.val_time = {}
        self.epoch = 0
        self.best_val_loss = 1e9
        self.best_val_acc = 0
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self.model.to(self.device)
        print("running on device ", self.device)
        
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss = None

    def train_step(self, x_enc, x_dec, y_dec):
        logits, self.loss, accuracy = self.model(x_enc, x_dec, y_dec)
        self.model.zero_grad(set_to_none=True)
        self.loss.backward()
        self.optimizer.step()
        return logits, self.loss, accuracy
      
    # Train the model
    def train(self, epochs):    
        train_iter = iter(self.train_dataloader)
        val_iter = iter(self.val_dataloader)
        
        # Load previously saved checkpoint 
        ckpt_path = 'checkpoints/checkpoint.pt'
        if path.exists(ckpt_path):
            print("Loading checkpoint")
            checkpoint = torch.load(ckpt_path)            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['loss']

        for epoch in range(epochs):
            start_time = time()
            self.model.train()

            loss_total = 0
            loss_count = 0
            accuracy_total = 0
            accuracy_count = 0
            step = 0

            while True:
                try:
                    train_batchX, train_batchY = next(train_iter)
                    train_batchX = train_batchX.type(torch.LongTensor)  
                    train_batchY = train_batchY.type(torch.LongTensor)
                    
                    # Maintain fixed sized batches
                    if (train_batchX.shape[0] < batch_size):
                        train_iter = iter(self.train_dataloader)
                        break;

                    #Define the encoder decoder inputs, and the decoder output
                    encoder_input = train_batchX[:, 1:].to(self.device)
                    decoder_input = train_batchY[:, :-1].to(self.device)
                    decoder_output = train_batchY[:, 1:].to(self.device)

                    _, loss, accuracy = self.train_step(encoder_input, decoder_input, decoder_output)
                    loss_total += loss
                    loss_count += 1
                    accuracy_total += accuracy
                    accuracy_count += 1

                    print("Epoch %d Step %d, Loss %.4f, Accuracy %.4f" % ((epoch + 1), step, loss_total/loss_count, accuracy_total/accuracy_count))
                    step += 1

                    free_memory(encoder_input)
                    free_memory(decoder_input)
                    free_memory(decoder_output)
                        
                except StopIteration:
                    train_iter = iter(self.train_dataloader)
                    break

            train_loss_mean = loss_total/loss_count
            self.train_loss[epoch] = train_loss_mean.float()
            train_acc_mean = accuracy_total/accuracy_count
            self.train_acc[epoch] = train_acc_mean.float()
                    
            # Run a validation setp after every epoch
            self.model.eval()
            loss_total = 0
            loss_count = 0
            
            while True:
                try:
                    val_batchX, val_batchY = next(val_iter)
                    val_batchX = val_batchX.type(torch.LongTensor)  
                    val_batchY = val_batchY.type(torch.LongTensor)
                    
                    # Maintain fixed sized batches
                    if (val_batchX.shape[0] < batch_size):
                        val_iter = iter(self.val_dataloader)
                        break;
                    
                    #Define the encoder and decoder inputs, and the decoder output
                    encoder_input = val_batchX[:, 1:].to(self.device)
                    decoder_input = val_batchY[:, :-1].to(self.device)
                    decoder_output = val_batchY[:, 1:].to(self.device)
        
                    # Generate a prediction
                    _, loss, accuracy = self.model(encoder_input, decoder_input, decoder_output)
                    loss_total += loss
                    loss_count += 1
                    accuracy_total += accuracy
                    accuracy_count += 1

                    free_memory(encoder_input)
                    free_memory(decoder_input)
                    free_memory(decoder_output)
             
                except StopIteration:
                    val_iter = iter(self.val_dataloader)
                    break

            n = gc.collect()
            print("Number of unreachable objects collected by GC: ", n)
            
            if (self.device.type == 'cuda'):
                torch.cuda.empty_cache()

            val_loss_mean = loss_total/loss_count
            self.val_loss[epoch] = val_loss_mean.float()
            val_acc_mean = accuracy_total/accuracy_count
            self.val_acc[epoch] = val_acc_mean.float()
                        
            # Print epoch number and loss value at the end of each epoch
            print("Epoch %d: Training Loss %0.4f, Training Accuracy %0.4f, Validation Loss %0.4f, Validation Accuracy %0.4f" % (epoch + 1, train_loss_mean, train_acc_mean, val_loss_mean, val_acc_mean))
            print("Total time taken: %.2fs" % (time() - start_time))
            
            # Save a checkpoint
            if epoch%10 == 0:
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss,
                    }, 'checkpoints/checkpoint_' + str(epoch) + '.pt')
                
        # Save final checkpoint     
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            }, 'checkpoints/checkpoint.pt')                

        # Save the training loss values
        with open('data/train_loss.pkl', 'wb') as file:
            dump(self.train_loss, file)
    
        # Save the validation loss values
        with open('data/val_loss.pkl', 'wb') as file:
            dump(self.val_loss, file)
      
    # Plot the Train/Val loss
    def plot(self):
        # Plot the Training and Validation Loss Curves
        train_loss = load(open('data/train_loss.pkl', 'rb'))
        val_loss = load(open('data/val_loss.pkl', 'rb'))
        
        # Retrieve the training and validation loss values
        train_loss_values = train_loss.values()
        val_loss_values = val_loss.values()
        
        train_loss_list = []
        for value in train_loss_values:
            train_loss_list.append(value.detach().cpu().numpy())

        val_loss_list = []
        for value in val_loss_values:
            val_loss_list.append(value.detach().cpu().numpy())
            
        # Generate a sequence of integers to represent the epochs
        epochs = range(1, len(train_loss_values) + 1)
        
        # Plot the training and validation loss values
        plt.plot(epochs, train_loss_list, label='Training Loss')
        plt.plot(epochs, val_loss_list, label='Validation Loss')
        
        # Add in a title an axes labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        # Set the tick locations
        plt.xticks(epochs)
        
        # Display the plot
        plt.legend(loc='best')
        plt.show(block=True)    
        
optimizer = LRScheduler(torch.optim.AdamW(model.parameters(), betas=(beta_1, beta_2), eps=epsilon), d_model)
trainer = Trainer(model, optimizer, train_dataset, val_dataset, device)     
trainer.train(epochs)
trainer.plot()
              

