import keras
import pandas as pd
from datetime import datetime
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# yahoo finance からLasertecの2018年1月1日以来の時列データセットを抽出し、データの形式を確認
df = yf.download("6920.T",start='2000-01-01',end = datetime.now(),interval="1d")

#　Closeコラムのみ抽出
data = df["Close"]
dataset = data.values

#　データの正規化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#訓練データの取得
x_train = []
y_train = []
for i in range(60, len(scaled_data)): 
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

# 訓練データのreshape
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

# kerasから必要なライブラリを導入
from keras.models import Sequential
from keras.layers import Dense, LSTM

#LSTMモデル構築
model = Sequential()
model.add(LSTM(128,return_sequences = True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')

#訓練用モデル構築
model.fit(x_train, y_train, epochs=3000, batch_size=32)

# モデルの保存
model.save('lstm_model.h5')

# scalerの保存
import joblib
joblib.dump(scaler, 'scaler.save')

# モデルの読み込み
from keras.models import load_model
model = load_model('lstm_model.h5')

# scalerの読み込み
scaler = joblib.load('scaler.save')