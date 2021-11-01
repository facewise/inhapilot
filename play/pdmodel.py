from PIL import ImageGrab
import numpy as np
import cv2
import time
import keras
from scipy.interpolate import interp1d
import pandas as pd

def get_model():
    model = keras.models.load_model(modelpath, custom_objects=dependencies)
    return model

def sign_pred(y_true, y_pred):
    mult = y_true * y_pred
    return keras.backend.mean(keras.backend.equal(keras.backend.sign(mult),keras.backend.ones_like(mult)), axis = -1)

modelpath = './mymodel.h5'
dependencies = {
    'sign_pred': sign_pred
}
##모델을 통해 예측한 휠값 csv파일로 만들기
if __name__ == "__main__":
    model = get_model()
    Loadplace = 'C:/Users/Home/Desktop/1. Data (2)/20200520195533/'
    w = pd.read_csv(Loadplace + '20200520195533.csv') #원본 csv
    wh = []
    im = w['file_name']
    for i in range(len(w)):
        curTime = time.time()
        img = cv2.imread(Loadplace + im[i])
        #img = cv2.resize(img, dsize=(200, 66), interpolation=cv2.INTER_AREA)
        wheel = model.predict(np.expand_dims(img, axis=0))
        wh.append(float(wheel))
    w['wheel'] = wh
    w.to_csv('w5.csv') #모델 csv