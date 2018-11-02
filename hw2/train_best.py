#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy
import pandas

def f(a):
	return 1/(1+np.exp(-a))

def gradient(x, y, w):
	dlw = np.zeros(len(w))
	for i in range(len(x)):
		error = f(w.T.dot(x[i]))
		dlw -= (y[i] - error) * x[i]
	dlw -= (w*w)/10
	return dlw / len(x)



features = 30
nonuse = 17
datas = 20000
data = np.zeros((datas,features), dtype=np.float64)
x = np.zeros((datas,features-nonuse), dtype=np.float64)
y = np.zeros(datas, dtype=np.float64)
lr = 30


with open('data/train_x.csv', encoding='big5', newline='') as R:
	count = 0
	R.readline()
	for i in R.readlines():
		c = i.split(',')
		flag = False
		for j in range(11,23):
			if (float(c[j]) < 0):
				flag = True
				break
		if flag:
			pass
		
		data[count][0] = float(c[0])
		for j in range(4,23):
			data[count][j+6] = float(c[j])

		if(float(c[1])==1):
			data[count][1] = 1
			data[count][2] = 0
		elif(float(c[1])==2):
			data[count][1] = 0
			data[count][2] = 1
		else:
			data[count][1] = 0
			data[count][2] = 1

		if(float(c[2])==1):
			data[count][3] = 1
			data[count][4] = 0
			data[count][5] = 0
			data[count][6] = 0
		elif(float(c[2])==2):
			data[count][3] = 0
			data[count][4] = 1
			data[count][5] = 0
			data[count][6] = 0
		elif(float(c[2])==3):
			data[count][3] = 0
			data[count][4] = 0
			data[count][5] = 1
			data[count][6] = 0
		elif(float(c[2])==4):
			data[count][3] = 0
			data[count][4] = 0
			data[count][5] = 0
			data[count][6] = 1
		else:
			data[count][3] = 0
			data[count][4] = 0
			data[count][5] = 0
			data[count][6] = 1

		if(float(c[3])==1):
			data[count][7] = 1
			data[count][8] = 0
			data[count][9] = 0
		elif(float(c[3])==2):
			data[count][7] = 0
			data[count][8] = 1
			data[count][9] = 0
		elif(float(c[3])==3):
			data[count][7] = 0
			data[count][8] = 0
			data[count][9] = 1
		else:
			data[count][7] = 0
			data[count][8] = 0
			data[count][9] = 1

		count += 1
		if (count == datas):
			break
	print (count)

with open('data/train_y.csv', encoding='big5', newline='') as R:
	count = 0
	R.readline()
	for i in R.readlines():
		y[count] = float(i)
		count += 1
		if (count == datas):
			break

w = np.zeros(features-nonuse, dtype=np.float64)
iteration = 0
for i in range(0, 6):
	w[i] = -1
#w, iteration = np.load('args_LR.npy')

xmax = [0 for i in range(features)]
xmin = [1e10 for i in range(features)]
for i in range(len(data)):
	for j in range(len(data[i])):
		if (data[i][j] > xmax[j]):
			xmax[j] = data[i][j]
		if (data[i][j] < xmin[j]):
			xmin[j] = data[i][j]

for i in range(len(data)):
	#for j in range(0,1):
	#	x[i][j] = (data[i][j]-xmin[j])/(xmax[j]-xmin[j])
	#for j in range(1,10):
	#	x[i][j] = data[i][j]
	#for j in range(10,11):
	#	x[i][j] = (data[i][j]-xmin[j])/(xmax[j]-xmin[j])
	for j in range(11,17):
		x[i][j-11] = data[i][j]
	for j in range(23,29):
		x[i][j-17] = (data[i][j]-xmin[j])/(xmax[j]-xmin[j])
	x[i][12] = 1



while iteration < 5000:

	dlw = gradient(x, y, w)

	w -= lr/(iteration**0.5+1) * dlw

	iteration += 1
	print ('iteration: %d '%(iteration), end='')
	print ([w[i] for i in range(7)])

	np.save("args_LR.npy", (w, iteration))

print (np.load("args_LR.npy"))


















