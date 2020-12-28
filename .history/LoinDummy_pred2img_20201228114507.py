#%%

import matplotlib.pyplot as plt
import numpy as np


# 基本パス
path = 'D:/LoinDummy/mix/'

# 画像化するnpyファイルののパスを指定してください。
imglist = np.load(path + 'weights/pred_LoinDummy_mix_predict_201223.npy')

print (imglist.shape)