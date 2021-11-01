import numpy as np
import os
import cv2
import pandas as pd
import glob


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img



kernel_size = 5  # 가우시안

rho = 5
theta = np.pi / 180
threshold = 60
min_line_len = 50
max_line_gap = 50
low_threshold = 10
high_threshold = 20

Loadplace="D:\pro8/"
dir=os.listdir('D:\pro8/')
Saveplace="D:\Smaple 10/"



csv_svdata=pd.DataFrame()
for filenum in range(0,len(dir)):
    k=0
    os.makedirs(Saveplace+dir[filenum]+'/')
    imges = [cv2.imread(pro) for pro in glob.glob(Loadplace+ dir[filenum] + "\*.jpg")]
    csv_data = pd.read_csv(Loadplace+ dir[filenum] +'/' + dir[filenum] + '.csv')
    csv_svdata = pd.DataFrame()
    for i in range(0, len(imges)):

        img = imges[i]
        imhigh,imwidth.channel=img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # gray
        vertices1 = np.array([[(imwidth/2-(22*imwidth/200), imhigh),
                               (imwidth/2-(12*imwidth/200), 0),
                               (imwidth/2, 0),
                               (imwidth/2, imhigh)]], dtype=np.int32)
        vertices2 = np.array([[(imwidth/2, imhigh),
                               (imwidth/2, 0),
                               (imwidth/2+(15*imwidth/200), 0),
                               (imwidth/2+(25*imwidth/200), imhigh)]], dtype=np.int32)
        vertices3 = np.array([[(imwidth/2-(52*imwidth/200), imhigh),
                               (imwidth/2-(42*imwidth/200), 0),
                               (imwidth/2, 0),
                               (imwidth/2, imhigh)]], dtype=np.int32)
        vertices4 = np.array([[(imwidth/2, imhigh),
                               (imwidth/2, 0),
                               (imwidth/2+(45*imwidth/200), 0),
                               (imwidth/2+(55*imwidth/200), imhigh)]], dtype=np.int32)
        verticestest = np.array([[(imwidth/2-(70*imwidth/200), imhigh),
                                  (imwidth/2-(30*imwidth/200), 0),
                                  (imwidth/2+(30*imwidth/200), 0),
                                  (imwidth/2+(70*imwidth/200), imhigh)]], dtype=np.int32)
        for u in range(0, 10):

            blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            for y in range(0, (10 - u)):
                blur_gray = cv2.GaussianBlur(blur_gray, (kernel_size, kernel_size), 10)
            edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
            mask5 = region_of_interest(edges, verticestest)
            try:
                linestest = hough_lines(mask5, rho, theta, threshold, min_line_len, max_line_gap)
                test = True
            except:
                test = False
            if (test == True):

                break


        mask1 = region_of_interest(edges, vertices1)
        mask2 = region_of_interest(edges, vertices2)
        mask3 = region_of_interest(edges, vertices3)
        mask4 = region_of_interest(edges, vertices4)

        try:
            lines1 = hough_lines(mask1, rho, theta, threshold, min_line_len, max_line_gap)
            li1 = True
        except:

            li1 = False
        try:
            lines2 = hough_lines(mask2, rho, theta, threshold, min_line_len, max_line_gap)
            li2 = True
        except:

            li2 = False
        try:
            lines3 = hough_lines(mask3, rho, theta, threshold, min_line_len, max_line_gap)
            li3 = True
        except:

            li3 = False
        try:
            lines4 = hough_lines(mask4, rho, theta, threshold, min_line_len, max_line_gap)
            li4 = True
        except:

            li4 = False

        if (((li1 == True) | (li2 == True))|((csv_data.ix[i][4]<0)&(li3==True))|((csv_data.ix[i][4]>0)&(li4==True))|(csv_data.ix[i][4]>0.1)|(csv_data.ix[i][4]<-0.1)):
            if ((i - k) > 3):
                for t in range((k + 1), (i - 2)):
                    cv2.imwrite(Saveplace + dir[filenum] + '/' + str(csv_data.ix[t - 1][1]), imges[t])
                    csv_svdata = csv_svdata.append(csv_data.iloc[t])
            k = i

        if (i == (len(imges) - 1)):
            for t in range((k + 1), i):
                cv2.imwrite(Saveplace + dir[filenum] + '/' + str(csv_data.ix[t][1]), imges[t])
                csv_svdata = csv_svdata.append(csv_data.iloc[t])
    try:
     csv_svdata = csv_svdata.loc[:, ["file_name", "break", "accel", "wheel"]]
     csv_svdata.to_csv(Saveplace + dir[filenum] + '/' + dir[filenum] + '.csv')
    except:
        continue

print(len(dir))
cv2.waitKey(0)
cv2.destroyWindow()
