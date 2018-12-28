from glove import Corpus, Glove
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd


# Used for pre-processing the text
def text_pre_process(c):

    t = [word_tokenize(i) for i in c]  # converting into list of list

    # For removing special characters
    filter_words = ['(', ')', '.', '*', ',', '?', '**', '1']
    t = [[w.lower() for w in z if w not in filter_words] for z in t]

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    line = [[w for w in z if w not in stop_words] for z in t]

    # Lematizing
    lema = WordNetLemmatizer()
    line = [[lema.lemmatize(w) for w in z] for z in line]

    return line


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

    print(glove.most_similar(word))


# Reading the file
f = open('java.md', 'r')
co = f.readlines()

lines = text_pre_process(co)
# print(lines)
glove = train_model(lines)

show_similar_words(glove, 'run')  # Print the words most similar to run

# Return the word embedding into a file
word_emb = show_word_embeddings(glove)
d = pd.DataFrame.from_dict(word_emb)
d.to_html('output.html', index=True)





