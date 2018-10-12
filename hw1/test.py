#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy
import pandas
import sys

data = []
hours = 9
count = 0
ans = []

w = np.load("args.npy")
b = w[9]

with open(sys.argv[1], encoding='big5', newline='') as R:
	for i in R.readlines():
		c = i.split(',')
		if c[1] == 'PM2.5':
			data.append([])
			for j in range(2,2+hours):
				if float(c[j]) <= 0:
					if j == 2:
						data[count].append(float(c[j+1]))
					elif j == 1+hours:
						data[count].append(float(c[j-1]))
					else:
						data[count].append((float(c[j-1]) + float(c[j+1]))/2)
				else:
					data[count].append(float(c[j]))
			count += 1

# f(x) = w1x1 + w2x2 + ... + w9x9 + b
for i in range(count):
	tmp = 0
	for j in range(hours):
		tmp += w[j] * data[i][j]
	tmp += b
	if tmp < 0:
		tmp = 0
	ans.append('id_' + str(i) + ',' + str(tmp) + '\r\n')


with open(sys.argv[2], 'w') as W:
	W.write('id,value\r\n')
	for i in ans:
		W.write(i)
	W.close()

