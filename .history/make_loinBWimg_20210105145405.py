#%%

import cv2
import os

imgfolder = 'D:/Documents/gifu_data/keras2020/gifu_loincrop_H17-25_normal/'

targetfolder = 'D:/Documents/gifu_data/keras2020/gifu_loincrop_H17-25_normal_BW/'

files = os.listdir(imgfolder)

n = 0

for f in files:
    image = cv2.imread(imgfolder + f, 0)
    ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(targetfolder + str(n) + '_img_' + f, thresh)
    n = n + 1



#%%
