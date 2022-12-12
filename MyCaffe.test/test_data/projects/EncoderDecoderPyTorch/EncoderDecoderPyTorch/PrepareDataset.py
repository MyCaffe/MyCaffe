from pickle import load, dump, HIGHEST_PROTOCOL
from numpy.random import shuffle
from numpy import savetxt
import torch
from torch.nn.utils.rnn import pad_sequence
from Vocabulary import Vocabulary

'''
The PrepareDataset class is a PyTorch rewrite of the TensorFlow and Keras version by Stefania Cristina
@see [Training the Transformer Model](https://machinelearningmastery.com/training-the-transformer-model/) by Stefania Cristina, 2022
@see [Plotting the Training and Validation Loss Curves for the Transformer Model](https://machinelearningmastery.com/plotting-the-training-and-validation-loss-curves-for-the-transformer-model/) by Stefania Cristina, 2022
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''

# PrepareDataset class used to preprocess the data
class PrepareDataset:
    def __init__(self):
        super(PrepareDataset, self).__init__()
        self.n_sentences = 10000 # Number of sentences to include in the dataset
        self.train_split = 0.8 # Ratio of training data split
        self.val_split = 0.1 # Ratio of the validation data split        
        
    def find_seq_length(self, dataset):
        return max(len(line.split()) for line in dataset)
        
    def encode_pad(self, dataset, tokenizer, seq_length):
        x = []
        y = torch.empty(len(dataset), seq_length)
        idx = 0
        for line in dataset:
            tokenizedline = tokenizer.tokenize(line, seq_length)
            paddedLine = []
            
            nLen = max(len(tokenizedline), seq_length)
            for i in range(nLen):
                if i < len(tokenizedline):
                    paddedLine.append(tokenizedline[i])
                else:
                    paddedLine.append(0)
            
            y[idx] = torch.tensor(paddedLine, dtype=torch.long)
            idx = idx + 1
        return y
    
    def __call__(self, filename):
        # Load a clean dataset
        clean_dataset = load(open(filename, 'rb'))
        
        # Reduce dataset size
        dataset = clean_dataset[:self.n_sentences, :]
        
        # Include start and end of string tokens
        for i in range(dataset[:, 0].size):
            dataset[i,0] = "<start> " + dataset[i,0] + " <eos>"
            dataset[i,1] = "<start> " + dataset[i,1] + " <eos>"

        # Random shuffle the dataset
        shuffle(dataset)

        # Split into input and output
        train = dataset[:int(self.n_sentences * self.train_split)]
        val = dataset[int(self.n_sentences * self.train_split):int(self.n_sentences * (1-self.val_split))]
        test = dataset[int(self.n_sentences * (1 - self.val_split)):]
        
        # Prepare encoder tokenizer
        encoder_tokenizer = Vocabulary.build(dataset[:, 0])
        enc_seq_length = self.find_seq_length(dataset[:, 0]) + 1
        enc_vocab_size = len(encoder_tokenizer)

        # Prepare decoder tokenizer
        decoder_tokenizer = Vocabulary.build(dataset[:, 1])
        dec_seq_length = self.find_seq_length(dataset[:, 1])
        dec_vocab_size = len(decoder_tokenizer)

        # Encode the pad training sequences
        trainX = self.encode_pad(train[:, 0], encoder_tokenizer, enc_seq_length)
        trainY = self.encode_pad(train[:, 1], decoder_tokenizer, dec_seq_length)

        # Encode the pad validation sequences
        valX = self.encode_pad(val[:, 0], encoder_tokenizer, enc_seq_length)
        valY = self.encode_pad(val[:, 1], decoder_tokenizer, dec_seq_length)
        
        # Save the encoder and decoder tokenizers
        encoder_tokenizer.save('encoder_tokenizer.txt')
        decoder_tokenizer.save('decoder_tokenizer.txt')
        
        return trainX, trainY, valX, valY, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length
        
    @staticmethod
    def test():
        print('PrepareDataset.test()')
        prepare_dataset = PrepareDataset()
        trainX, trainY, valX, valY, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length = prepare_dataset('data\english-german.pkl')
        
        print('Encoder vocab size: %d, sequence length: %d' % (enc_vocab_size, enc_seq_length))
        print('Decoder vocab size: %d, sequence length: %d' % (dec_vocab_size, dec_seq_length))

# PrepareDataset.test()