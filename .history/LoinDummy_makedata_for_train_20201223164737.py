#%%

# ロース芯ダミーデータセットの作成

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

X = []
Y = []

path = 'D:/LoinDummy/mix/'

for i in tqdm(range(1000)):
    dummy = np.asarray(Image.open(path + 'dummy/dummy_' + str(i) + '.bmp'))
    mask = np.asarray(Image.open(path + 'mask/mask_' + str(i) + '.bmp'))
    X.append(dummy)
    Y.append(mask)

X = np.array(X)
Y = np.array(Y)

X = X.astype('float32')
X = X / 255.0
Y = Y.astype('float32')
Y = Y / 255.0


np.save(path + 'LoinDummy_mix_img_for_train.npy', X) # 学習用画像データファイル名を指定
np.save(path + 'LoinDummy_mix_label_for_train.npy', Y) # 学習用座標データファイル名を指定


# %%
