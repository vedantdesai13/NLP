"""
This module is used to define a tokenizing class

"""

import BOW as b  # File containing pre-processing functions
import numpy as np


class Tokenizer:
    """
    Used for defining various functions for tokenization.
    """

    # convert strings into list
    def text_to_word(self, s):
        return s.split()

    # returns the number of unique words present
    def number_of_unique_words(self, s):
        return len(set(s))

    # returns word count of each words in descending order
    def word_count(self, s):
        dc = {}
        for w in s:
            if w not in dc:
                dc[w] = 1
            else:
                dc[w] += 1
        return sorted(dc.items(), key=lambda x: x[1], reverse=True)

    # creating bag of words for a given input from the corpus
    def bag_of_words(self, words, vocab):
        words = b.rem_special_chars(words)
        words = b.convert_to_lowcase(words)
        words = b.rem_stopwords(words)
        words = b.lemma(words)
        words = self.text_to_word(words)

        bag = np.zeros(len(vocab))
        for w in words:
            for i, x in enumerate(vocab):
                if x == w:
                    bag[i] += 1
        return np.array(bag)


# Read data from a file
f = open('text2.txt', 'r')
corpus = f.read()

corpus = b.rem_special_chars(corpus)
corpus = b.convert_to_lowcase(corpus)
corpus = b.rem_stopwords(corpus)
corpus = b.lemma(corpus)

t = Tokenizer()
s = t.text_to_word(corpus)
print("word count=", t.word_count(s))
print("number of unique words", t.number_of_unique_words(s))

inp = "autonomous individuals mutual aid self governance"
print("input = ", inp)
print("bag of words = ", t.bag_of_words(inp, s))
