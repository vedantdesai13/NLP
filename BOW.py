"""
This module is for the creation of bag of words from scratch

"""

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from keras.preprocessing import text


# Read data from a file
f = open('text2.txt', 'r')
corpus = f.read()


# Function to remove special characters
def rem_special_chars(s):
    st = re.sub("^\W+", '', s)
    return st


# Function to convert string to lower case
def convert_to_lowcase(st):
    st = st.lower()
    return st


# Function to remove stop words
def rem_stopwords(s):
    stop_words = set(stopwords.words('english'))
    '''
    wd = ""
    for i in s.split():
        if i not in stop_words:
            wd += i
            wd += ' '
    return wd
    '''
    return " ".join([w for w in s.split() if w not in stop_words])


# Function to implement Lemmatization
def lemma(s):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in s.split()])


'''
# Function to tokenize the words
def con_to_token(s):
    s = text.text_to_word_sequence(s)
    t = text.Tokenizer()
    t.fit_on_texts(s)
    return t



corpus = rem_special_chars(corpus)
corpus = convert_to_lowcase(corpus)
corpus = rem_stopwords(corpus)
corpus = lemma(corpus)
tokenizer = con_to_token(corpus)


print("word count = ", tokenizer.word_counts)
print("document count = ", tokenizer.document_count)
print("word index = ", tokenizer.word_index)
print("word document id = ", tokenizer.word_docs)


count = 0
for i in corpus.split():
    print(i)
    count += 1

print(count)'''
