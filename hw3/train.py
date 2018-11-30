import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

x = []
y = []

with open(sys.argv[1], newline='') as R:
	count = 0
	R.readline()
	for i in R.readlines():
		c = i.split(',')
		y.append([0,0,0,0,0,0,0])
		y[count][int(c[0])] = 1
		c = c[1].split(' ')
		c = list(map(float, c))
		c = np.reshape(c, (48, 48))
		x.append(c)
		count+=1

	x = np.array(x)
	x = x.reshape(-1, 48, 48, 1)/255
	y = np.array(y)

x_valid = x[-3333:]
y_valid = y[-3333:]
x = x[:-3333]
y = y[:-3333]

model = Sequential()

model.add(Convolution2D(filters=64, input_shape=(48, 48, 1), padding='same', kernel_size=(5, 5)))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Convolution2D(128, padding='same', kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Convolution2D(128, padding='same', kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Convolution2D(256, padding='same', kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Dropout(0.40))
model.add(MaxPooling2D(2, 2))

model.add(Convolution2D(512, padding='same', kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Dropout(0.45))
model.add(Convolution2D(512, padding='same', kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print('Training ----------------------')
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, shear_range=0.2, zoom_range=0.2)
cp = ModelCheckpoint(filepath='hw3.h5',monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

model.fit_generator(datagen.flow(x, y, batch_size=128), steps_per_epoch=3*len(x)/128, epochs=200, callbacks = [cp], validation_data=(x_valid, y_valid))

#model.save('hw3.h5')




