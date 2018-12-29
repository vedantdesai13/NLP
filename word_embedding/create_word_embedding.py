# from glove import Corpus, Glove
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


# Used for pre-processing the text
def text_pre_process(c):

    t = [word_tokenize(i) for i in c]  # converting into list of list

    # For removing special characters
    filter_words = ['(', ')', '.', '*', ',', '?', '**']
    t = [[w.lower() for w in z if w not in filter_words] for z in t]

    # Lemmatizing
    lema = WordNetLemmatizer()
    line = [[lema.lemmatize(w) for w in z] for z in t]

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    line = [[w for w in z if w not in stop_words] for z in line]
    return line


'''
# Training the words into glove
def train_model(line):
    corpus = Corpus()
    corpus.fit(line)
    glov = Glove(no_components=10, learning_rate=0.05)
    glov.fit(corpus.matrix, epochs=200, no_threads=10, verbose=True)
    glov.add_dictionary(corpus.dictionary)
    glov.save('glove.model')
    return glov



# Used to return words with their feature values
def show_word_embeddings(glove):
    x = [k for k in glove.dictionary.keys()]
    m = dict(zip(x, glove.word_vectors))
    return m


# Shows similarity between words
def show_similar_words(glove, word):
    print(glove.most_similar(word, 20))
'''


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


# Implement word2vec
def imp_word2vec(corpus):
    model = Word2Vec(corpus, sg=1, min_count=10, window=10)
    model.train(corpus, epochs=100, total_examples=len(corpus))
    print(model.wv.most_similar(positive='jvm'))
    # print(model[model.wv.vocab])

    x = [k for k in model.wv.vocab]
    m = dict(zip(x, model.wv.vectors))
    d = pd.DataFrame.from_dict(m)
    d.to_html('word2vec_output.html', index=True)

    visualize(model.wv.vectors, model.wv.vocab)


# Reading the file
f = open('java.md', 'r')
co = f.readlines()

lines = text_pre_process(co)
# print(lines)
'''
glove = train_model(lines)

show_similar_words(glove, 'program')  # Print the words most similar to run

# Return the word embedding into a file
word_emb = show_word_embeddings(glove)
d = pd.DataFrame.from_dict(word_emb)
d.to_html('output.html', index=True)

# Plot
visualize(glove.word_vectors, glove.dictionary.keys())
'''

imp_word2vec(lines)




