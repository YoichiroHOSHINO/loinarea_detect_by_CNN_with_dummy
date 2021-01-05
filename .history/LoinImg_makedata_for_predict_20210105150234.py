#%%

# ロース芯画像データセットの作成

import os
from numpy.lib.type_check import imag
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

X = []

path = 'D:/Documents/gifu_data/keras2020/gifu_loincrop_H17-25_normal_BW/'

files = os.listdir(path)

for f in tqdm(files):
    img = np.asarray(Image.open(path + f))
    X.append(img)

X = np.array(X)

X = X.astype('float32')
X = X / 255.0


np.save(path + 'LoinDummy_mix_img_for_train_210104.npy', X) # 予測用画像データファイル名を指定

