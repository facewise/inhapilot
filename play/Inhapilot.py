from PIL import ImageGrab
import pyvxbox as px
import numpy as np
import cv2
import time
import keras
from scipy.interpolate import interp1d

def get_model():
    model = keras.models.load_model(modelpath, custom_objects=dependencies)
    return model

def sign_pred(y_true, y_pred):
    mult = y_true * y_pred
    return keras.backend.mean(keras.backend.equal(keras.backend.sign(mult),keras.backend.ones_like(mult)), axis = -1)

modelpath = './mymodel10.h5'
dependencies = {
    'sign_pred': sign_pred
}

if __name__ == "__main__":
    m = interp1d([-0.5, 0.5], [-32768, 32767])
    model = get_model()
    j = px.XInputDevice(1)
    j.PlugIn()
    while(True):
        curTime = time.time()
        img = ImageGrab.grab(bbox=(0,540,1920,1080))
        img = img.resize((400, 100))
        img=cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)
        img = img / 255.0
        wheel = model.predict(np.expand_dims(img,axis=0))
        wheel = wheel
        print(wheel)
        j.SetAxis('X', int(m(wheel)))


