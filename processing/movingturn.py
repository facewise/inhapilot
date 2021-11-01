import numpy as np
import os
import cv2
import pandas as pd
import glob
import random

def transformturn(img,num1):
    imhigh, imwidth,channel = img.shape
    num = abs(num1);
    img4 = img[imhigh-num:imhigh,:]
    for i in range (1,(int)(imhigh/num)+1):
        pts5=np.float32([[imwidth-20, 0], [imwidth-20, num], [imwidth+10, num], [imwidth+10, 0]])
        if(num1>0):
            pts6 = np.float32([[imwidth-20 + (i*i), 0], [imwidth-20+ ((i-1)*(i-1)), num], [imwidth+10+ ((i-1)*(i-1)), num], [imwidth+10 + (i*i), 0]])
        else:
            pts6 = np.float32([[imwidth-20 - (i*i), 0], [imwidth-20- ((i-1)*(i-1)), num], [imwidth+10- ((i-1)*(i-1)), num], [imwidth+10 - (i*i), 0]])
        if imhigh-(i*num)>0 :
            img2 = img[imhigh-((i+1)*num):imhigh - (i * num), :]
        else :img2= img[0:imhigh - (i * num)]
        h, w = img2.shape[:2]
        M = cv2.getPerspectiveTransform(pts5, pts6)
        img3=cv2.warpPerspective(img2,M,(w,h))
        img4 = cv2.vconcat([img3,img4])

    return img4



Loadplace="D:\Smaple 12/"
dir=os.listdir(Loadplace)
Saveplace="D:\Smaple 13/"
makefile=3
for filenum in range(0, len(dir)):


    for makefiles in range(0,makefile):
        k = 0


        os.makedirs(Saveplace + dir[filenum]+str(makefiles) + '/')
        imges = [cv2.imread(pro) for pro in glob.glob(Loadplace + dir[filenum] + "\*.jpg")]
        csv_data = pd.read_csv(Loadplace + dir[filenum] + '/' + dir[filenum] + '.csv')
        csv_svdata = pd.DataFrame()
        add_colN = pd.DataFrame()
        add_colW = csv_data["wheel"]
        for i in range(0, len(imges)):
            randomN = random.randrange(8, 14)
            if random.choice([True, False]):
                randomN = -randomN
            img = transformturn(imges[i],randomN)

            cv2.imwrite(Saveplace + dir[filenum]+str(makefiles) + '/' + str(csv_data.ix[i][1][:-4])+str(makefiles)+'.jpg', img)

            csv_svdata = csv_svdata.append(csv_data.iloc[i][2:4])

            add_colW.loc[i] = add_colW.loc[i] + 2/randomN
        print(add_colW)
        add_colN = csv_data.file_name.apply(lambda x:str(x.split('.')[0])+str(makefiles)+'.jpg')

        print(add_colN)
        csv_svdata["wheel"] = add_colW
        csv_svdata["file_name"] = add_colN
        try:
            csv_svdata = csv_svdata.loc[:, ["file_name", "break", "accel", "wheel"]]
            csv_svdata.to_csv(Saveplace + dir[filenum]+str(makefiles) + '/' + dir[filenum] + str(makefiles)+'.csv')
        except:
            continue

cv2.waitKey(0)
cv2.destroyWindow()
