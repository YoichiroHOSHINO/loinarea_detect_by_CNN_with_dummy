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
#X_test = np.load(path + 'LoinDummy_mix_img_for_train.npy')
X_test = np.load('D:/Documents/gifu_data/keras2020/gifu_H17-H25_BMS_market1_img_test.npy')

# 任意の試験名を指定してください。
testname = 'LoinDummy_mix_predict_gifu_test_210104'

# 学習済みモデルの読み込み。
# .hdf5ファイルのパスを指定してください。 
model = load_model(path + 'weights/model_LoinDummy_mix_train_201223.hdf5')

X_test = np.expand_dims(X_test, axis=-1)

bat = 16
datagen = img.ImageDataGenerator(samplewise_std_normalization=True)

predict_gen = datagen.flow(X_test, shuffle=None, batch_size=bat)
pred = model.predict_generator(predict_gen, steps=len(predict_gen), verbose=1)

# weightsフォルダに予測結果が保存されます。
np.save(path + 'weights/pred_' + testname + '.npy', pred)

#%%