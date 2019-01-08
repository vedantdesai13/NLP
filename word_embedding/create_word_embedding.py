from glove import Corpus, Glove
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re
from nltk.corpus import wordnet
import PyPDF2
import collections


# Used for pre-processing the text
def text_pre_process(c):

    t = [word_tokenize(i) for i in c]  # converting into list of list
    print('tokenized')
    # For removing special characters
    t = [[re.sub('[^a-zA-Z]+', '-', w) for w in z] for z in t]
    spec_char = ['-']
    t = [[w.lower() for w in z if w not in spec_char] for z in t]
    print('removed')
    # Lemmatizing
    lema = WordNetLemmatizer()
    line = [[lema.lemmatize(w) for w in z] for z in t]
    print('lemmatzed removed')
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    line = [[w for w in z if w not in stop_words] for z in line]
    print('stop words removed')
    return line


# Training the words into glove
def train_model(line):
    corpus = Corpus()
    corpus.fit(line)
    glove = Glove(no_components=5, learning_rate=0.05, random_state=0)
    glove.fit(corpus.matrix, epochs=10, no_threads=100, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')
    return glove


# Used to return words with their feature values
def show_word_embeddings(glove):
    x = [k for k in glove.dictionary.keys()]
    w = dict(zip(x, glove.word_vectors))
    d = pd.DataFrame.from_dict(w)
    d.to_html('glove_word_embedding.html', index=True)


# Shows similarity between words
def show_similar_words(glove):
    wd = []
    sim_wd = []
    for w in glove.dictionary:
        wd.append(w)
        sim_wd.append(glove.most_similar(w, 6))

    s = dict(zip(wd, sim_wd))
    df = pd.DataFrame.from_dict(s)
    df.to_html('glove_word_mapping.html', index=True)


# visualize the words
def visualize(vector, vocab):
    # fit a 2d PCA model to the vectors
    X = vector
    pca = PCA(n_components=3)
    result = pca.fit_transform(X)

    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1], c=result[:, 2])

    # Label the words
    for i, word in enumerate(vocab):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


# Create word mappings
def word_mappings(model):
    wd = []
    sim_wd = []
    for w in model.wv.vocab:
        wd.append(w)
        sim_wd.append(model.wv.most_similar(positive=w, topn=5))

    s = dict(zip(wd, sim_wd))
    df = pd.DataFrame.from_dict(s)
    df.to_html('word2vec_word_mapping.html', index=True)


# Create synonym word matrix
def word_syn(model):
    wd = []
    syn = []
    for w in model.wv.vocab:
        wd.append(w)
        a = []
        for s in wordnet.synsets(w):
            for l in s.lemmas():
                a.append(l.name())
        syn.append(a)

    print(type(syn))

    diction = dict(zip(wd, syn))
    print(diction)
    df = pd.DataFrame.from_dict(diction, orient='index')
    df = df.transpose()
    df.to_html('word_syn.html', index=True)


# Implement word2vec
def imp_word2vec(corpus):
    model = Word2Vec(corpus, sg=0, min_count=1, window=2)
    model.train(corpus, epochs=10, total_examples=len(corpus))
    # print(model.wv.most_similar(positive='jvm'))
    # print(model[model.wv.vocab])

    x = [k for k in model.wv.vocab]
    m = dict(zip(x, model.wv.vectors))
    d = pd.DataFrame.from_dict(m)
    d.to_html('word2vec_word_embedding.html', index=True)

    word_mappings(model)

    word_syn(model)

    visualize(model.wv.vectors, model.wv.vocab)


# Implement GloVe
def imp_glove(lines):
    glove = train_model(lines)

    # show_similar_words(glove, 'program')  # Print the words most similar to run

    # Return the word embedding into a file
    show_word_embeddings(glove)

    show_similar_words(glove)

    # Plot
    visualize(glove.word_vectors, glove.dictionary.keys())



# Reading the file
f = open('java2.txt', errors='ignore')
co = f.readlines()
print('read')
lines = text_pre_process(co)
# print(lines)

imp_glove(lines)

imp_word2vec(lines)





