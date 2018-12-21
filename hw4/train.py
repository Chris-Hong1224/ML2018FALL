import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LeakyReLU, LSTM, Embedding
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
import jieba

jieba.load_userdict(sys.argv[3])

x = []
y = []

with open(sys.argv[1], newline='') as R:
    count = 0
    R.readline()
    for i in R.readlines():
        c = i.split(',', 1)
        c = list(jieba.cut(c[1]))
        x.append(c)
        count += 1

with open(sys.argv[2], newline='') as R:
    R.readline()
    for i in R.readlines():
        c = i.split(',')
        y.append(int(c[1]))

x = np.array(x)
y = np.array(y)

w2v_model = Word2Vec.load('w2v.model')

for i in range(count):
    if(len(x[i]) > 64):
        x[i] = x[i][:64]

tmp = np.zeros((count, 64, 200), dtype=float)
for i in range(count):

    for j in range(len(x[i])):
        tmp[i, j, :] = w2v_model.wv[x[i][j]]

x = tmp

x_valid = x[-6666:]
y_valid = y[-6666:]
x = x[:-6666]
y = y[:-6666]

model = Sequential()

#model.add(Embedding(120000-6666, 128, input_length=64, trainable=False))

model.add(LSTM(units = 256, input_shape=(64, 200), dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(LSTM(units = 128, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

print('Training ----------------------')
cp = ModelCheckpoint(filepath='hw4.h5',monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

model.fit(x, y, validation_data=(x_valid, y_valid), epochs=100, batch_size=64, callbacks=[cp])





