import numpy as np
from word_embedding import create_word_embedding as cwb

# reading the file adn converting each line to a list
f = open('java3.txt')
lines = f.readlines()
print(lines)
lines = cwb.text_pre_process(lines)
print(lines)

d = {}
for l in lines:
    for w in l:
        d[w] = [f for f in l if f not in w]


print(d['run'])
# print(d)
np.save('my_dict', d)
