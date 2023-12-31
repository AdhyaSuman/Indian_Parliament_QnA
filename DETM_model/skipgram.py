import gensim
import pickle
import os
import numpy as np
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--data_file', type=str, default='', help='a .txt file containing the corpus')
parser.add_argument('--vocab_file', type=str, default='', help='a .txt vocab file for the data post processing, in order to save embeddings only for the words needed')
parser.add_argument('--emb_file', type=str, default='embeddings.txt', help='file to save the word embeddings')
parser.add_argument('--dim_rho', type=int, default=300, help='dimensionality of the word embeddings')
parser.add_argument('--min_count', type=int, default=2, help='minimum term frequency (to define the vocabulary)')
parser.add_argument('--sg', type=int, default=1, help='whether to use skip-gram')
parser.add_argument('--workers', type=int, default=25, help='number of CPU cores')
parser.add_argument('--negative_samples', type=int, default=10, help='number of negative samples')
parser.add_argument('--window_size', type=int, default=4, help='window size to determine context')
parser.add_argument('--iters', type=int, default=5, help='number of iterationst')

args = parser.parse_args()

# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        with open(self.filename) as infile:
            for line in infile:
                yield line.split()

# Gensim code to obtain the embeddings
sentences = MySentences(args.data_file) # a memory-friendly iterator
print('Model training begins')
model = gensim.models.Word2Vec(sentences, min_count=args.min_count, sg=args.sg, vector_size=args.dim_rho, 
    epochs=args.iters, workers=args.workers, negative=args.negative_samples, window=args.window_size, )
print('Model trained')
vocab = [line.strip() for line in open(args.vocab_file, 'r').readlines()]
# Write the embeddings to a file
model_vocab = list(model.wv.index_to_key)
print(len(model_vocab))
del sentences
n = 0
f = open(args.emb_file, 'w')
for v in tqdm(model_vocab):
    if v in vocab:
        vec = list(model.wv.__getitem__(v))
        f.write(v + ' ')
        vec_str = ['%.9f' % val for val in vec]
        vec_str = " ".join(vec_str)
        f.write(vec_str + '\n')
        n += 1
f.close()
print('DONE! - saved embeddings for ' + str(n) + ' words.')
