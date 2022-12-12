import torch
import torch.nn as nn
from torch.nn import functional as F
from TransformerEncoder import TransformerEncoder
from TransformerDecoder import TransformerDecoder
from Utility import free_memory

'''
The TransformerModel layer is a PyTorch rewrite of the TensorFlow and Keras version by Stefania Cristina
@see [Joining the Transformer Encoder and Decoder Plus Masking](https://machinelearningmastery.com/joining-the-transformer-encoder-and-decoder-and-masking/) by Stefania Cristina, 2022
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''
class TransformerModel(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, num_heads, d_k, d_v, d_model, d_ff_inner, num_layers, rate, batch_size):
        super(TransformerModel, self).__init__()
        # Setup the encoder
        self.encoder = TransformerEncoder(enc_vocab_size, enc_seq_length, num_heads, d_k, d_v, d_model, d_ff_inner, num_layers, rate, batch_size)
        
        # Setup the decoder
        self.decoder = TransformerDecoder(dec_vocab_size, dec_seq_length, num_heads, d_k, d_v, d_model, d_ff_inner, num_layers, rate, batch_size)
        
        # Setup the final layer
        self.final_layer = nn.Linear(d_model, dec_vocab_size)

        # Save the device we are running on
        self.device = self.final_layer.weight.device

        # Set the target ones and zeros
        self.tgt_ones = None
        self.tgt_zeros = None
        
    def to(self, device):
        new_self = super(TransformerModel, self).to(device)
        new_self.encoder = new_self.encoder.to(device)
        new_self.decoder = new_self.decoder.to(device)
        new_self.final_layer = new_self.final_layer.to(device)
        new_self.device = device
        return new_self
    
    def padding_mask(self, input):
        # Create a mask where the padding tokens are 0 and the rest are 1
        return torch.where(input == 0, 1, 0)

    def lookahead_mask(self, shape):
        # Create a mask to mask out the future tokens
        if (shape[0] == 1):
            mask = torch.ones(shape[1]).to(self.device)
            mask[0] = 0
        else:
            tgt_ones = torch.ones(shape[0], shape[1]).to(self.device)        
            mask = (torch.triu(tgt_ones) == 1).float()
            mask = torch.cat((mask[1:], mask[-1:]))             
            free_memory(tgt_ones)        
        return mask
    
    # Masked Loss function
    def loss_fcn(self, targets, logits):        
        # Mask out the logits that are masked out targets
        mask = torch.where(targets == 0, 0, 1)
        mask = mask.view(mask.shape[0], mask.shape[1], 1)
        mask = mask.expand(logits.shape)
        logits = logits * mask
        # Reshape for cross entropy
        logits1 = logits.view(-1, logits.size(-1))
        targets1 = targets.flatten()
        # Compute a cross entropy loss, ignore index 0
        loss = F.cross_entropy(logits1, targets1, ignore_index=0)
        return loss

    # Accuracy function with mask
    def accuracy_fcn(self, target, prediction):
        # Create mask so that the zero padding values are not included in the accuracy calculation.
        if (self.tgt_ones == None):
            self.tgt_ones = torch.ones(target.shape).to(self.device)
        if (self.tgt_zeros == None):
            self.tgt_zeros = torch.zeros(target.shape).to(self.device)
        padding_mask = torch.where(target != 0, self.tgt_ones, self.tgt_zeros)
        pad_sum = padding_mask.sum()

        # Find equal prediction and target values and apply the padding mask
        idx = torch.argmax(prediction, axis=2)
        full = torch.full(idx.shape, float('Inf')).to(self.device)
        idx = torch.where(padding_mask != 0, idx, full)
        target_sum = torch.sum(target == idx)

        free_memory(idx)
        free_memory(full)
        free_memory(padding_mask)

        return target_sum / pad_sum

    def forward(self, enc_input, dec_input, targets=None):
        # Create the padding mask
        enc_padding_mask = self.padding_mask(enc_input)
        dec_padding_mask = self.padding_mask(dec_input)
        
        # Create the lookahead mask
        look_ahead_mask = self.lookahead_mask(dec_input.shape)
        
        # Create the combined mask
        combined_mask = torch.maximum(dec_padding_mask, look_ahead_mask.to(self.device))
        
        # Encode the input
        enc_output = self.encoder(enc_input, enc_padding_mask)
        
        # Decode the input
        dec_output = self.decoder(dec_input, enc_output, combined_mask, enc_padding_mask)

        # Final layer
        final_output = self.final_layer(dec_output)
        
        # Calculate the loss
        loss = None
        accuracy = None
        if targets != None:        
            loss = self.loss_fcn(targets, final_output)
            accuracy = self.accuracy_fcn(targets, final_output)

        free_memory(combined_mask)
        free_memory(look_ahead_mask)
        free_memory(enc_padding_mask)
        free_memory(dec_padding_mask)
        free_memory(enc_output)
        free_memory(dec_output)

        return final_output, loss, accuracy

    @staticmethod
    def test():
        h = 8 # Number of self-attention heads
        d_k = 64 # Dimension of the queries and keys
        d_v = 64 # Dimension of the values
        d_ff = 2048 # Demension of the fully connected mpl
        d_model = 512 # Dimension of the model
        n = 6 # Number of encoder layers
        batch_size = 64 # Batch size
        dropout_rate = 0.1 # Frequency of dropout
        enc_vocab_size = 20 # Size of the encoder vocabulary
        dec_vocab_size = 20 # Size of the decoder vocabulary
        enc_seq_length = 5 # Length of the input sequence
        dec_seq_length = 5 # Length of the target sequence

        # Create the model
        training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
       
        
# TransformerModel.test()
        