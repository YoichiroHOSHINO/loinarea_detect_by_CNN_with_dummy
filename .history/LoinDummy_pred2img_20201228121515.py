#%%

import matplotlib.pyplot as plt
import numpy as np


# 基本パス
path = 'D:/LoinDummy/mix/'

# 画像化するnpyファイルののパスを指定してください。
imglist = np.load(path + 'weights/pred_LoinDummy_mix_predict_201223.npy')
imglist2 = np.load(path + 'LoinDummy_mix_img_for_train.npy')

plt.imshow(np.squeeze(imglist[0]), cmap='gray')
plt.show()
plt.imshow(np.squeeze(imglist2[0]), cmap='gray')
plt.show()

# %%
