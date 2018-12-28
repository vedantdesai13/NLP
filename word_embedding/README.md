The 'create_word_embedding.py' is used to create a word embedded matrix of words with their features.
This is done using the GloVe library.
Input is taken from the file 'java.md'.
The word embedded output can seen from the file 'output.html'.
And the glove model is present in 'glove.model' file.


REQUIREMENTS:-

- glove library
- nltk library
- pandas


Steps to install glove library:-

First download the code from 'https://github.com/maciejkula/glove-python', then change the line :
libraries=["stdc++"],
in setup.py to :
libraries=[], (or maybe libraries=[""],)
then open CMD in current folder and write : python setup.py install, it will install glove-python for you.
(Make sure pip version is 18.1)