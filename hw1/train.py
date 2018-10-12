#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy
import pandas

data = []
x = []
y = []
start_hour = 3
hours = 9
months = 0

with open('data/train.csv', encoding='big5', newline='') as R:
	for i in R.readlines():
		c = i.split(',')
		if c[2] == 'PM2.5':
			if months%20 == 0:
				data.append([])
			for j in range(3, 27):
				data[int(months/20)].append(float(c[j]))
			months += 1

months = int(months/20)
for i in range(months):
	for j in range(len(data[i])-10):
		flag = False
		for k in range(10):
			if data[i][j+k]	<= 0:
				flag = True
				break
			if j+k < len(data[i]):
				if abs(data[i][j+k]-data[i][j+k+1]) > 60:
					flag = True
					break
		if flag:
			continue

		x.append([])
		for k in range(9):
			x[len(y)].append(data[i][j+k])
		x[len(y)].append(1)
		y.append(data[i][j+9])

test = np.matrix(x).transpose()
test2 = np.matrix(y).transpose()
test3 = (test * test.transpose()).I * test * test2
w = test3.A1.tolist()
print (w)
np.save('args.npy', w)

