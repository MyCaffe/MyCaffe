import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array

'''
The PrepareTextData functions are originally by Jason Brownlee and modified for the Encoder/Decoder model written in PyTorch.  This script only needs to be
run once on the file 'data/deu.txt' to create the file 'data/english-german.pkl' which is used throughout this sample for training.

@see [How to Develop a Neural Machine Translation System from Scratch](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/) by Jason Brownlee, 2018
@see [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
'''

# Load the document into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# Split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs

# Clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)

# Save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)
    
# Load dataset
filename = 'data/deu.txt'
doc = load_doc(filename)
# Split into english-german pairs
pairs = to_pairs(doc)
# Clean sentences
clean_pairs = clean_pairs(pairs)
# Save clean pairs to file
save_clean_data(clean_pairs, 'data/english-german.pkl')
# Spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
