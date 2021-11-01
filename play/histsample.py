from matplotlib import pyplot as plt
import numpy as np
import pickle
from random import *
with open("sample.dat", "rb") as f:
    data = []
    while True:
        try:
            temp = pickle.load(f)

            if type(temp) is not list:
                temp = np.ndarray.tolist(temp)

            data = data + temp
        except EOFError:
            break

right = [];left = [];wheel = [];pic=[];joy_values=[];screenshot=[];new_data = [];order = []
tmp = [];tmpj=[];tmps=[];bin=np.arange(-0.5,0.5,0.05)
for i in range(20):
    ##리스트만들기
    tmp.append([])
    tmpj.append([])
    tmps.append([])
    for j in range(len(data)):
        if bin[i]+0.05>=data[j][1]:
            if bin[i]<data[j][1]:
                tmpj[i].append(data[j][1])
                tmps[i].append(data[j][0])
                tmp[i].append([data[j][0], data[j][1]])

    ##리스트 자르기
    order.append(len(tmpj[i]))
order.sort()
print(order)
for i in range(19):
    t=0
    if order[i]>800:
        if order[i+1]>order[i]*3:
            t=order[i]*3
            break
for i in range(20):
    joy_values.append([])
    screenshot.append([])
    if len(tmp[i])>t:
        sa=sample(tmp[i],t)
        for j in range(t):
            joy_values[i].append(sa[j][1])
            screenshot[i].append(sa[j][0])
    else:
        joy_values[i]=tmpj[i]
        screenshot[i]=tmps[i]
    wheel = wheel + joy_values[i]
    pic = pic+screenshot[i]
##데이터셋
data=[]
for i in range(len(wheel)):
    data.append([pic[i],wheel[i]])
with open('sample1.dat', 'wb') as fw:
    pickle.dump(data, fw)
#히스토그램
for i in range(10):
    right=right+joy_values[i]
    left=left+joy_values[i+10]

pright='+ : '+str(len(right)/len(wheel)*100)+'%'
pleft='- : '+str(len(left)/len(wheel)*100)+'%'

plt.hist(right,bin,rwidth=0.8,alpha=0.5,color='red',label = pright)
plt.hist(left,bin,rwidth=0.8,alpha=0.5,color='blue',label = pleft)
plt.grid()
plt.xlabel('wheel',fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.show()