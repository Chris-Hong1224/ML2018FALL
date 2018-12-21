import sys
import numpy as np
from gensim.models import Word2Vec
from keras.models import load_model
import jieba

jieba.load_userdict(sys.argv[2])

x = []

with open(sys.argv[1], 'r') as R:
    count = 0
    R.readline()
    for i in R.readlines():
        pos = i.find(',')
        c = i[pos+1:-1]
        c = list(jieba.cut(c))
        x.append(c)
        count += 1

data = x

count = len(data)
w2v_model = Word2Vec.load('w2v.model')
tmp = np.zeros((count, 64, 200), dtype=float)
for i in range(count):
    if(len(data[i]) > 64):
        data[i] = data[i][:64]
    for j in range(len(data[i])):
        tmp[i, j, :] = w2v_model.wv[data[i][j]]

x = tmp

model = load_model('hw4.h5')
y = model.predict(x)

with open(sys.argv[3], 'w') as W:
    W.write('id,label\r\n')
    for i in range(count):
        if(y[i] > 0.5):
            W.write('%d,1\r\n'%i)
        else:
            W.write('%d,0\r\n'%i)





