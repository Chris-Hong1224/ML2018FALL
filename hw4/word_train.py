import sys
from gensim.models import Word2Vec
import jieba

jieba.load_userdict(sys.argv[3])

x = []

with open(sys.argv[1], newline='') as R:
    R.readline()
    for i in R.readlines():
        c = i.split(',', 1)
        c = list(jieba.cut(c[1]))
        x.append(c)

with open(sys.argv[2], newline='') as R:
    R.readline()
    for i in R.readlines():
        c = i.split(',', 1)
        c = list(jieba.cut(c[1]))
        x.append(c)

model = Word2Vec(x, size=200, min_count=1, iter=10)
model.save('w2v.model')

