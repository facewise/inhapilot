import pickle
from random import *
import numpy as np
import cv2
import time
import keras
import pandas as pd

def get_model():
    model = keras.models.load_model(modelpath, custom_objects=dependencies)
    return model

def sign_pred(y_true, y_pred):
    mult = y_true * y_pred
    return keras.backend.mean(keras.backend.equal(keras.backend.sign(mult),keras.backend.ones_like(mult)), axis = -1)

modelpath = './mymodel1.h5'
dependencies = {
    'sign_pred': sign_pred
}

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

if __name__ == "__main__":
    i=3  #bin 설정
    img = screenshot[i]
    joy = joy_values[i]
    model = get_model()
    for j in range(len(img)):
        curTime = time.time()
        image = img[j]
        cv2.imshow('window', img[j])
        wheel = model.predict(np.expand_dims(image, axis=0))
        print("original : "+str(joy[j])+" pred : "+str(float(wheel)))
        cv2.waitKey(100)


