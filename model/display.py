import cnnmodel
import numpy as np
import cv2

model = cnnmodel.get_model()

x_test, y_test = cnnmodel.test_data('c:/CFile1/21/*.jpg',
                                    'c:/CFile1/21/21.csv')

pred = model.predict(x_test)
index = [i for i, v in enumerate(pred) if np.sign(pred[i]) * np.sign(y_test[i]) < 0]
wrong_predicted = [i for i in index]

while True:
    for i in range(len(wrong_predicted) - 1):
        image = x_test[i]
        cv2.imshow('window', image)
        print('prediction : '+str(pred[i])+'    /    '+'original : '+str(y_test[i]))
        cv2.waitKey(1000)


#print(np.sign(pred))
#print(np.sign(y_test))
print(wrong_predicted)
print('wrong count : ' + str(len(wrong_predicted)))
