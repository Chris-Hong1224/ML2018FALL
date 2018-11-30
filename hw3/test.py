import sys
import numpy as np
from keras.models import load_model

x = []

with open(sys.argv[1], newline='') as R:
	count = 0
	R.readline()
	for i in R.readlines():
		c = i.split(',')
		c = c[1].split(' ')
		c = list(map(float, c))
		c = np.reshape(c, (48, 48))
		x.append(c)
		count+=1

	x = np.array(x)
	x = x.reshape(-1, 48, 48, 1)/255

model = load_model('hw3.h5')
y = model.predict(x)

with open(sys.argv[2], 'w') as W:
	W.write('id,label\r\n')
	for i in range(count):
		W.write('%d,%d\r\n'%(i, np.argmax(y[i])))



