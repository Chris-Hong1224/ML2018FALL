wget https://www.dropbox.com/s/van9br9955457hf/w2v.model.trainables.syn1neg.npy?dl=1 -O w2v.model.trainables.syn1neg.npy
wget https://www.dropbox.com/s/wjn1874uwoaugq1/w2v.model.wv.vectors.npy?dl=1 -O w2v.model.wv.vectors.npy
python3.6 test.py $1 $2 $3
