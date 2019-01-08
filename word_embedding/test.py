from glove import Glove
import argparse


# Load the model file
path = 'C:/Users/admmin/PycharmProjects/NLP/word_embedding/glove.model'
model = Glove.load(path)

# Input the argument word
parser = argparse.ArgumentParser('input')
parser.add_argument('word', help='the most similar words will be printed', type=str)
arg = parser.parse_args()

print(model.most_similar(arg.word, 2))
