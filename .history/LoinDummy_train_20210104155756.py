# coding:utf-8
 
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model
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

# 学習用画像データセットのパスを指定してください。
X = np.load(path + 'LoinDummy_mix_img_for_train_210104.npy')  
# 学習用座標データセットのパスを指定してください。
Y = np.load(path + 'LoinDummy_mix_label_for_train210104.npy')

# 任意の試験名を指定してください。出力結果はすべてこの名前＋αが付けられます。
testname = 'LoinDummy_mix_train_210104'

X = X.reshape(X.shape[0], 300, 300,1)
Y = Y.reshape(Y.shape[0], 300, 300,1)

# 学習用データセットの10%を検証群にランダム分割。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)

# バッチサイズ指定。
bat = 16

datagen = img.ImageDataGenerator(samplewise_std_normalization=True)
train_gen = datagen.flow(X_train, Y_train, batch_size=bat)
test_gen = datagen.flow(X_test, Y_test, batch_size=bat)

# CNNを構築
inputs = Input(shape=(300,300,1))

x = BatchNormalization(axis=-1, epsilon=0.001)(inputs)
x = Conv2D(16, (5, 5), padding='same', name='conv_1')(x)
x = Activation('relu', name='relu_1')(x)
x = MaxPooling2D(pool_size=(3, 3), name='pool_1')(x)
x = Dropout(0.05)(x)
x = Conv2D(16, (5, 5), padding='same', name='conv_2')(x)
x = Activation('relu', name='relu_2')(x)
x = MaxPooling2D(pool_size=(3, 3), name='pool_2')(x)
x = Conv2D(32, (5, 5), padding='same', name='conv_3')(x)
x = Activation('relu', name='relu_3')(x)
x = MaxPooling2D(pool_size=(3, 3), name='pool_3')(x)

x = Flatten()(x)
x = Dense(10000, activation='relu', name='dense_relu_1')(x)
#x = Dropout(0.05)(x)
x = Dense(1000, activation='relu', name='dense_relu_2')(x)
#x = Dropout(0.05)(x)
x = Dense(90000, activation='relu', name='dense_relu_3')(x)
predictions = Reshape((300,300,1))(x)

model = Model(inputs=inputs, outputs=predictions)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.001, amsgrad=False) 
model.compile(optimizer=adam, loss='mean_squared_error',  metrics=['mae'])

# モデル図を出力。
plot_model(model, show_shapes=True , to_file=path + 'plotmodel_' + testname + '.png')

# weightsフォルダにモデルを保存する。
model_checkpoint = ModelCheckpoint(filepath=path + 'weights/model_' + testname + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# エポック数など指定。
history = model.fit_generator(train_gen, steps_per_epoch=len(X_train)/bat, initial_epoch=1, epochs=200, verbose=1, validation_data=test_gen, validation_steps=len(X_test)/bat, callbacks=[model_checkpoint])

# weightsフォルダに学習履歴を保存。
pd.DataFrame(history.history).to_csv(path + 'weights/histry_' + testname + '.csv')

# 評価 & 評価結果を逐次出力
print(model.evaluate(X_test, Y_test))

# 予測する画像データセットを指定してください。
#print ('予測用画像を読み込みます')
#P = np.load('./loin_img_test.npy')

#print ('予測用画像を変換します')
#P = np.expand_dims(P, axis=-1)

#print ('予測します')
#predict_gen = datagen.flow(P, shuffle=None, batch_size=bat)
#pred = model.predict_generator(predict_gen, steps=len(predict_gen), verbose=1)

#print ('予測値を保存します')
#np.savetxt('./weights/pred_' + testname + '.csv', pred, delimiter=',')

# グラフ描画
mae = history.history['mean_absolute_error']
val_mae = history.history['val_mean_absolute_error']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

# 平均絶対誤差グラフ
plt.plot(epochs, mae, 'b' ,label = 'training mae')
plt.plot(epochs, val_mae, 'r' , label= 'validation mae')
plt.title('Training and Validation mae')
plt.legend()

plt.figure()

# 損失グラフ
plt.plot(epochs, loss, 'b' ,label = 'training loss')
plt.plot(epochs, val_loss, 'r' , label= 'validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()




