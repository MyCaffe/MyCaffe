import torch
import torch.utils
import torch.nn.functional as F
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
from Vocabulary import Vocabulary

'''
The Training module is a PyTorch rewrite of the TensorFlow and Keras version by Stefania Cristina
@see [Inferencing the Transformer Model](https://machinelearningmastery.com/inferencing-the-transformer-model/) by Stefania Cristina, 2022
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
batch_size = 1 

# Define the dataset parameters
enc_seq_length = 8 # Encoder sequence length
dec_seq_length = 12 # Decoder sequence length
enc_vocab_size = 2405 # Encoder vocabulary size
dec_vocab_size = 3864 # Decoder vocabulary size

# Create the model
model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0, batch_size).to(device)

class Translate:
    def __init__(self, model):
        super(Translate, self).__init__()
        self.transformer = model

    def load_tokenizer(self, filename):
        return load(open(filename, 'rb'))

    def __call__(self, sentence):
        # Append start and end of string tokens to the input sentence
        sentence = "<START> " + sentence[0] + " <EOS>"
        
        # Load encoder and decoder tokenizers
        enc_tokenizer = Vocabulary.load('encoder_tokenizer.txt')
        dec_tokenizer = Vocabulary.load('decoder_tokenizer.txt')
        
        # Prepare the input sentence by tokenizing, padding and converting to tensor
        encoder_input = enc_tokenizer.tokenize(sentence, enc_seq_length-1, True)
        encoder_input = torch.tensor(encoder_input, device=device).view(1, -1).long()

        # Prepare the decoder input by adding the start token and converting to tensor
        output_start = dec_tokenizer.tokenize('<START>', dec_seq_length-1)
        
        # Prepare the output <EOS> token by tokenizing and converting to tensor
        output_end = dec_tokenizer.tokenize('<EOS>', dec_seq_length)

        # Prepare the output <UNK> token by tokenizing and converting to tensor
        output_unk = dec_tokenizer.tokenize('<UNK>', dec_seq_length)
        
        # Prepare the output array of dynamic size
        output = output_start
        decoder_output = torch.tensor(output, device=device).view(1, -1).long()
                
        for i in range(dec_seq_length):
            prediction = self.transformer(encoder_input, decoder_output)
            prediction = prediction[0][:, -1, :]

            # Select the prediction with the highest score
            prediction_id = prediction.argmax(dim = -1)
                        
            # Break if the predicted word is <EOS>
            pred_id = prediction_id[0].item()
            if pred_id == output_end[0] or pred_id == output_unk[0]:
                break

            # Append the predicted word to the output array  
            decoder_output = torch.cat((decoder_output, prediction_id.view(1, -1)), dim = 1)
                        
        # Decode the predicted tokens into an output string
        output_str = dec_tokenizer.detokenize(decoder_output.detach().cpu().numpy()[0])
        return output_str
        
    @staticmethod
    def test():           
        # Load the trained model's weights
        ckpt_path = 'checkpoints/checkpoint.pt'
        if path.exists(ckpt_path):
            print("Loading checkpoint")
            checkpoint = torch.load(ckpt_path)        
            state = checkpoint['model_state_dict']
            model.load_state_dict(state)

        # Create the trnaslator
        translator = Translate(model)
        
        # Sentence to translate
        sentence = ["im stuffed"]
        print (sentence)
        print(translator(sentence))

        sentence = ["leave us alone"]
        print(sentence)
        print(translator(sentence))

        sentence = ["congratulations"]
        print(sentence)
        print(translator(sentence))

        sentence = ["i want a martini"]
        print(sentence)
        print(translator(sentence))
            
Translate.test()

            
            