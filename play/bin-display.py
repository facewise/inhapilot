import pickle
from random import *
import numpy as np
import cv2
import time


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

wheel = [];pic=[];joy_values=[];screenshot=[];order = []
tmp = [];tmpj=[];tmps=[]
bin=np.arange(-0.5,0.5,0.05)
bin[10] = 0
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


i=3  #bin 설정
img = screenshot[i]
joy = joy_values[i]
for j in range(len(img)):
    curTime = time.time()
    image = img[j]
    cv2.imshow('window', img[j])
    print(str(joy[j]))
    cv2.waitKey(100)