#%%

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


# 基本パス
path = 'D:/LoinDummy/mix/'

# 保存先パス
savepath = 'D:/LoinDummy/mix/predict_gifu/'


# 画像化するnpyファイルののパスを指定してください。
#imglist = np.load(path + 'weights/pred_LoinDummy_mix_predict_201223.npy')
imglist = np.load(path + 'weights/pred_LoinDummy_mix_predict_gifu_test_210104.npy')
#imglist = np.load('D:/Documents/gifu_data/keras2020/gifu_H17-H25_BMS_market1_img_test.npy')

for i in tqdm(range(imglist.shape[0])):
    cv2.imwrite(savepath + str(i) + '_pred.bmp', np.squeeze(imglist[i])*255)


# %%
