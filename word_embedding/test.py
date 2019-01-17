# from glove import Glove
import numpy as np
import argparse
import time


# Load the model file
path = 'glove.model'
model = Glove.load(path)


# function using model
def using_model():
    start = time.time()

    # Input the argument word
    parser = argparse.ArgumentParser('input')
    parser.add_argument('-word', '-w', help='the most similar words will be printed', type=str)
    arg = parser.parse_args()

    print(model.most_similar(arg.word, 2))

    print('time taken by glove', time.time()-start)


# Load the dictionary
d = np.load('my_dict.npy').item()


# function using dictionary
def using_dictionary():
    start = time.time()

    # Input the argument word
    parser = argparse.ArgumentParser('input')
    parser.add_argument('-word', '-w', help='the most similar words will be printed', type=str)
    arg = parser.parse_args()

    print(d[arg.word])

    print('time taken by dictionary', time.time() - start)


# using_model()

using_dictionary()


