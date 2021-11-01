from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import pickle
import cv2
import numpy as np
import pandas as pd
import glob

filepath = 'c:/CFile1/datasets.dat'
modelpath = 'c:/CFile1/finalmodel.h5'
loadpath = 'c:/sample1/'
dirs = os.listdir(loadpath)


def collect_data():
    with open(filepath, 'ab') as f:
        data = []
        for filenum in range(0, len(dirs)):
            images = [cv2.imread(pro) for pro in glob.glob(loadpath + dirs[filenum] + "/*.jpg")]
            csv = pd.read_csv(loadpath + dirs[filenum] + '/' + dirs[filenum] + '.csv')
            joy_values = csv['wheel'].values.tolist()

            count = 0

            for i in range(0, len(images)):

                screenshot = images[i]

                if count < len(joy_values):
                    screenshot = np.array(screenshot)

                    data.append([screenshot, joy_values[i]])

                if count == len(images) - 1:
                    count += 1
                    print('Collected data count - {0}.'.format(count))
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                    data = []

                count += 1


def test_data(dirname, csvname):
    x_data = []
    y_data = []
    csv = pd.read_csv(csvname, sep=',')
    joy_values = csv['wheel'].values.tolist()

    images = glob.glob(dirname)

    count = 0

    for img in images:
        screenshot = cv2.imread(img)

        screenshot = np.array(screenshot)
        x_data.append(screenshot)
        y_data.append(joy_values[count])
        count += 1

    return np.array(x_data), np.array(y_data)


def read_data(split=False):
    with open(filepath, 'rb') as f:
        data = []
        while True:
            try:
                temp = pickle.load(f)

                if type(temp) is not list:
                    temp = np.ndarray.tolist(temp)

                data = data + temp
            except EOFError:
                break
        if split:
            x_train = []
            y_train = []

            for i in range(0, len(data)):
                x_train.append(data[i][0])
                y_train.append(data[i][1])

            return np.array(x_train), np.array(y_train)
        else:
            return np.array(data)


def create_model():
    nrows = 66
    ncols = 200
    img_channels = 3
    output_size = 1

    model = Sequential()
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2) ,activation='relu', input_shape=(nrows, ncols, img_channels)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_size))
    model.summary()

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[sign_pred])

    return model


def get_model():
    if os.path.isfile(modelpath):
        dependencies = {
            'sign_pred': sign_pred
            }
        model = keras.models.load_model(modelpath, custom_objects=dependencies)
    else:
        model = create_model()

    return model


def sign_pred(y_true, y_pred):
    mult = y_true * y_pred
    return keras.backend.mean(keras.backend.equal(keras.backend.sign(mult), keras.backend.ones_like(mult)), axis=-1)


def train_model(model):
    x, y = read_data(True)

    # test data set 0.15 : training data set 85%
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.15, random_state=1)

    model = get_model()

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    model.fit(x_train, y_train, epochs=200, batch_size=128, verbose=1,
              validation_data=(x_valid, y_valid), callbacks=[checkpoint, earlystop])

    model.save(modelpath)


def display_data():
    data = read_data()
    data_size = len(data)
    print('Data size  -  ' + str(data_size))

    for i in range(data_size - 1, 0, -1):
        image = data[i]
        cv2.imshow('window', image[0])
        print(str(image[1]))
        cv2.waitKey(50)

