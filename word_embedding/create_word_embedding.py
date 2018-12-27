from glove import Corpus, Glove
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def text_pre_process(c):

    t = [word_tokenize(i) for i in c]

    filter_words = ['(', ')', '.', '*', ',', '?', '**', ]
    t = [[w.lower() for w in z if w not in filter_words] for z in t]
    #print(filtered)
    stop_words = set(stopwords.words('english'))

    lines = [[w for w in z if w not in stop_words] for z in t]

    lema = WordNetLemmatizer()
    lines = [[lema.lemmatize(w) for w in z] for z in lines]

    return lines


def train_model(lines):

    corpus = Corpus()
    corpus.fit(lines)

    glove = Glove(no_components=5, learning_rate=0.1)

    glove.fit(corpus.matrix, epochs=150, no_threads=10, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')
    return glove


def show_word_embeddings(glove):

    x = [k for k in glove.dictionary.keys()]
    m = dict(zip(x, glove.word_vectors))
    print(m)


def show_similar_words(glove, word):

    print(glove.most_similar(word))


f = open('java.md', 'r')
co = f.readlines()
lines = text_pre_process(co)
# print(lines)
glove = train_model(lines)
# show_word_embeddings(glove)
show_similar_words(glove, 'run')



