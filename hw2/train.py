#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import scipy
import pandas

features = 30
nonuse = 17
datas = 20000
data = np.zeros((datas,features), dtype=np.float64)
x = np.zeros((datas,features-nonuse), dtype=np.float64)
y = np.zeros(datas, dtype=np.float64)
lr = 30
onecnt = 0


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
		if y[count] == 1:
			onecnt += 1
		count += 1
		if (count == datas):
			break

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


x0 = np.zeros((count-onecnt,features-nonuse), dtype=np.float64)
x1 = np.zeros((onecnt,features-nonuse), dtype=np.float64)
count0 = 0
count1 = 0

for i in range(count):
	if y[i] == 1:
		x1[count1] = x[i]
		count1 += 1
	else:
		x0[count0] = x[i]
		count0 += 1

u1 = np.mean(x1, axis=0)
x1 = x1+u1
sigma1 = np.asmatrix(x1[0]).T.dot(np.asmatrix(x1[0]))
for i in range(1, count1):
	sigma1 += np.asmatrix(x1[i]).T.dot(np.asmatrix(x1[i]))
sigma1 /= count1

u0 = np.mean(x0, axis=0)
x0 = x0+u0
sigma0 = np.asmatrix(x0[0]).T.dot(np.asmatrix(x0[0]))
for i in range(1, count0):
	sigma0 += np.asmatrix(x0[i]).T.dot(np.asmatrix(x0[i]))
sigma0 /= count0

sigma = (sigma0*count0+sigma1*count1)/count

np.save("args_GM.npy", (u0, u1, sigma, count0, count1))

print (np.load("args_GM.npy"))


















