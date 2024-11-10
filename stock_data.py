import yfinance as yf

# 例としてApple(AAPL)のデータを取得
ticker = yf.Ticker("AAPL")

# 過去の株価データ（時系列データ）
historical_data = ticker.history(period="max")

# 企業の静的情報
static_info = ticker.info

# 必要な静的情報を表示
print("Sector:", static_info.get("sector"))
print("Industry:", static_info.get("industry"))
print("Full-time Employees:", static_info.get("fullTimeEmployees"))

# 株価データを表示
print(historical_data)
