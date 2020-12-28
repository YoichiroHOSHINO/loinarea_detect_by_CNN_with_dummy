#%%
# coding:utf-8
 
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Reshape
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image as img
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import plot_model

# 基本パス
path = 'D:/LoinDummy/mix/'

# 予測するロース芯画像データセットのパスを指定してください。
X_test = np.load(path + 'LoinDummy_mix_img_for_train.npy')

# 任意の試験名を指定してください。
testname = 'LoinDummy_mix_predict_201223'

# 学習済みモデルの読み込み。
# .hdf5ファイルのパスを指定してください。 
model = load_model(path + 'weights/model_LoinDummy_mix_train_201223.hdf5')

X_test = np.expand_dims(X_test, axis=-1)

pred = model.predict(X_test, batch_size = 16, verbose=1)

# weightsフォルダに予測結果が保存されます。
np.savetxt(path + 'weights/pred_' + testname + '.csv', pred, delimiter=',')

#%%