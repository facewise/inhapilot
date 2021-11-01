import os
import pickle
import cv2
import pandas as pd
import glob
import gzip

def collect_data(dirname, csvname):
    with gzip.open(filepath, 'ab') as f:
      data = []
      csv = pd.read_csv(csvname, sep=',')
      joy_values = csv['wheel'].values.tolist()
      images = glob.glob(dirname)
      count = 0
      for img in images:
          screenshot = cv2.imread(img)
          if count < len(joy_values):
              data.append([screenshot, joy_values[count]])
          if count == len(images) - 1:
              print('Collected data count - {0}.'.format(count))
              pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
              data = []
          count += 1
          print(count)


filepath = './sample.dat'

print(os.listdir('./'))
for i in os.listdir('./'):
    if os.path.isdir('./'+i):
        if i == '__pycache__':
            pass
        else:
            collect_data('./'+i+'/*.jpg','./'+i+'/'+i+'.csv')
