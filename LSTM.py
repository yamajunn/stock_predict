import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf
from datetime import datetime

# データのダウンロード
df1 = yf.download("AAPL", start='2010-01-01', end='2018-01-01', interval="1d")
df2 = yf.download("MSFT", start='2010-01-01', end='2018-01-01', interval="1d")
df3 = yf.download("GOOGL", start='2010-01-01', end='2018-01-01', interval="1d")

# 必要な特徴量を抽出
features = ['Close', 'Volume', 'Open']
train_data1 = df1[features]
train_data2 = df2[features]
train_data3 = df3[features]

# スケーリング
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaled_train_data1 = scaler1.fit_transform(train_data1)

scaler2 = MinMaxScaler(feature_range=(0, 1))
scaled_train_data2 = scaler2.fit_transform(train_data2)

scaler3 = MinMaxScaler(feature_range=(0, 1))
scaled_train_data3 = scaler3.fit_transform(train_data3)

# モデルの訓練データを準備
def create_dataset(data, time_step=60):
    x, y = [], []
    for i in range(time_step, len(data)):
        x.append(data[i-time_step:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

x_train1, y_train1 = create_dataset(scaled_train_data1)
x_train2, y_train2 = create_dataset(scaled_train_data2)
x_train3, y_train3 = create_dataset(scaled_train_data3)

# LSTMモデル構築関数
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# モデルの訓練
model = create_model((x_train1.shape[1], x_train1.shape[2]))
model.fit(x_train1, y_train1, epochs=1, batch_size=128, verbose=1)

model = create_model((x_train2.shape[1], x_train2.shape[2]))
model.fit(x_train2, y_train2, epochs=1, batch_size=128, verbose=1)

model = create_model((x_train3.shape[1], x_train3.shape[2]))
model.fit(x_train3, y_train3, epochs=1, batch_size=128, verbose=1)

# 予測用のデータをダウンロード
test_df = yf.download("AAPL", start='2018-01-01', end=datetime.now(), interval="1d")

# 必要な特徴量を抽出
test_data = test_df[features]

# test_dataをnumpy配列に変換
test_data = np.array(test_data)

# スケーリング
scaled_test_data = scaler1.transform(test_data)

# 検証用データを取得とデータ変換
training_data_len = int(len(scaled_test_data) * 0.7)
test_data = scaled_test_data[training_data_len - 60:, :]
x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, :])
y_test = test_data[60:, 0]

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# 予測値の算出
predictions = model.predict(x_test)

# predictionsの形状を調整
predictions = np.concatenate([predictions, np.zeros((predictions.shape[0], 2))], axis=1)

# 予測値の逆スケーリング
predictions = scaler1.inverse_transform(predictions)[:, 0]

# RMSEを利用して予測精度を確認
test_score = np.sqrt(mean_squared_error(y_test, predictions))
print('Test Score: %.2f RMSE' % (test_score))

# 訓練データと検証データの設定
train = test_df[:training_data_len]
valid = test_df[training_data_len:]
valid['Predictions'] = predictions

# プロット
plt.figure(figsize=(16,6))
plt.title('Original vs Predicted Data')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(train.index, train['Close'], label='Training Data')
plt.plot(valid.index, valid['Close'], label='Original Data')
plt.plot(valid.index, valid['Predictions'], label='Predicted Data')
plt.legend(['Training Data', 'Original Data', 'Predicted Data'], loc='lower right')
plt.show()