#%%

# ロース芯画像データセットの作成


import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

X = []

path = 'D:/Documents/gifu_data/keras2020/gifu_loincrop_H17-25_normal_BW/'

files = os.listdir(imgfolder)

for i in tqdm(range(10000)):
    dummy = np.asarray(Image.open(path + 'dummy/' + str(i) + '_dummy.bmp'))
    mask = np.asarray(Image.open(path + 'mask/' + str(i) + '_mask.bmp'))
    X.append(dummy)
    Y.append(mask)

X = np.array(X)

X = X.astype('float32')
X = X / 255.0


np.save(path + 'LoinDummy_mix_img_for_train_210104.npy', X) # 学習用画像データファイル名を指定

