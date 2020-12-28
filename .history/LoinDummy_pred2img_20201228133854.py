#%%

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


# 基本パス
path = 'D:/LoinDummy/mix/'

# 保存先パス
savepath = 'D:/LoinDummy/mix/predict/'


# 画像化するnpyファイルののパスを指定してください。
imglist = np.load(path + 'weights/pred_LoinDummy_mix_predict_201223.npy')

for i in tqdm(range(imglist.shape[0])):
    img = Image.fromarray(np.squeeze(imglist[i]), mode = 'L')
    img.save(savepath + 'pred_' + str(i) + '.bmp')


# %%
