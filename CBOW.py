"""This module is a CBOW implementation

"""

from gensim.models import Word2Vec


corpus = []

# Read data from a file
for f in open('text3.txt', 'r'):
    corpus.append(f.split())

# train model
model = Word2Vec(corpus,sg=0, min_count=1)  # sg=0 for CBOW and 1 skip-gram

# summarize the loaded model
print(model)

# summarize vocabulary
words = list(model.wv.vocab)
print(words)

#finding similar words
print(model.wv.most_similar(positive=['from','place']))


