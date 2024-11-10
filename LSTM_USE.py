import keras
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
import joblib

# モデルとスケーラーの読み込み
model = keras.models.load_model('lstm_model.h5')
scaler = joblib.load('scaler.save')

# データの読み込みとスケーリング
new_df = yf.download("6919.T", start='2018-01-01', end=datetime.now(), interval="1d")
new_data = new_df["Close"].values.reshape(-1, 1)
scaled_new_data = scaler.transform(new_data)

# データの70%を予測材料、残りの30%を未知の未来と仮定
train_size = int(len(scaled_new_data) * 0.99)
train_data = scaled_new_data[:train_size]
future_days = int(len(scaled_new_data) * 0.01)  # 残り30%の日数を予測範囲に

# 予測の開始に使用する最新60日間
last_60_days = train_data[-60:]
future_predictions = []

# 未来30%の日数分の予測を逐次生成
for _ in range(future_days):
    x_new_test = last_60_days.reshape(1, 60, 1)
    predicted_price = model.predict(x_new_test)
    future_predictions.append(predicted_price[0, 0])
    
    # 予測値を次の入力に追加して最新の60日を更新
    last_60_days = np.append(last_60_days, predicted_price)[-60:].reshape(-1, 1)

# 予測結果をスケールバック
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 予測結果のプロット
future_dates = [new_df.index[train_size - 1] + timedelta(days=i+1) for i in range(future_days)]
future_df = pd.DataFrame({'Date': future_dates, 'Predictions': future_predictions.flatten()})
future_df.set_index('Date', inplace=True)

plt.figure(figsize=(16, 6))
plt.title('LSTM Model Future Predictions for Unseen 30% of Data')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot(new_df['Close'], label='Real Data')
plt.plot(future_df['Predictions'], label='Predicted Future', linestyle='--')
plt.legend(loc='lower right')
plt.show()
