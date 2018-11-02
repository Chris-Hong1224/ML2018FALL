#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy
import pandas
import sys

def p(x, u, sigma):
	return ( np.exp( (-np.asmatrix(x-u).dot(sigma).dot(np.asmatrix(x-u).T)) / 2 ) / ((2*np.pi)**len(u)*np.linalg.det(sigma))**0.5 )

test_file = sys.argv[3]
output_file = sys.argv[4]
u0, u1, sigma, count0, count1 = np.load('args_GM.npy')

count = 0
features = 30
nonuse = 17
datas = 10000
data = np.zeros((datas,features), dtype=np.float64)
x = np.zeros((datas,features-nonuse), dtype=np.float64)

with open(test_file, encoding='big5', newline='') as R:
	R.readline()
	for i in R.readlines():
		c = i.split(',')

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

xmax = [0 for i in range(features)]
xmin = [0 for i in range(features)]
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

one_cnt = 0

with open(output_file, 'w') as W:
	W.write('id,Value\r\n')
	for i in range(count):
		p0 = p(x[i], u0, sigma)*count0
		p1 = p(x[i], u1, sigma)*count1
		if (p1/(p0+p1) >= 0.5):
			W.write('id_%d,1\r\n'%(i))
			one_cnt += 1
		else:
			W.write('id_%d,0\r\n'%(i))

print (one_cnt)

