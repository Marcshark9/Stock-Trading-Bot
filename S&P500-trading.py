import yfinance as yf
import pandas as pd
from ib_insync import IB, Stock, MarketOrder
import time
from datetime import datetime, time as dt_time
import pytz

# Connect to IBKR API
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=1)  # Default port for IB Gateway


# Function to fetch the list of S&P 500 stocks
def get_sp500_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500 = tables[0]
    return sp500["Symbol"].tolist()


def calculate_indicators(stock_symbol):
    try:
        data = yf.download(stock_symbol, period="2y", interval="1wk")

        # Ensure we have enough data (26 weeks for MACD, 14 weeks for RSI)
        if data.empty or len(data) < 26:  # 26 weeks is the minimum for MACD
            print(f"Insufficient data for {stock_symbol}.")
            return None

        # Calculate 10-week or 20-week SMA (based on available data)
        sma_window = min(20, len(data))  # Use 10 or 20 weeks based on available data
        data["SMA"] = data["Close"].rolling(window=sma_window).mean()

        # Calculate MACD only if we have sufficient data
        if len(data) >= 26:  # Check if data is enough for MACD (12 and 26 periods)
            data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
            data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
            data["MACD"] = data["EMA12"] - data["EMA26"]
            data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
        else:
            print(f"Not enough data for MACD calculation for {stock_symbol}.")
            data["MACD"] = data["Signal"] = None

        # Calculate RSI (requires at least 14 data points)
        if len(data) >= 14:
            delta = data["Close"].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data["RSI"] = 100 - (100 / (1 + rs))
        else:
            print(f"Not enough data for RSI calculation for {stock_symbol}.")
            data["RSI"] = None

        # Calculate volatility (standard deviation of percentage price changes)
        data["Volatility"] = data["Close"].pct_change().rolling(window=10).std()

        # Calculate average trading volume
        data["AvgVolume"] = data["Volume"].rolling(window=10).mean()

        # Replace NaN or 0 values with the previous valid value or a default small value
        data["SMA"] = data["SMA"].fillna(method="ffill")
        data["MACD"] = data["MACD"].fillna(0)
        data["Signal"] = data["Signal"].fillna(0)
        data["RSI"] = data["RSI"].fillna(50)  # Default RSI to 50 if missing
        data["Volatility"] = data["Volatility"].fillna(0)
        data["AvgVolume"] = data["AvgVolume"].fillna(0)

        # Ensure there are no missing values in the required columns for trading signal evaluation
        if any(
            col not in data.columns for col in ["Close", "SMA", "MACD", "Signal", "RSI"]
        ):
            print(f"Missing required columns in {stock_symbol}. Skipping this stock.")
            return None

        return data
    except Exception as e:
        print(f"Error calculating indicators for {stock_symbol}: {e}")
        return None


# Function to filter stocks based on volume and volatility
def filter_stocks(data, min_volume=1_000_000, min_volatility=0.02):
    try:
        avg_volume = data["AvgVolume"].iloc[-1]  # Extract scalar value for AvgVolume
        volatility = data["Volatility"].iloc[-1]  # Extract scalar value for Volatility

        # Ensure avg_volume and volatility are valid numbers (not NaN)
        if pd.isna(avg_volume) or pd.isna(volatility):
            print(
                f"Skipping due to invalid volume/volatility: {avg_volume}, {volatility}"
            )
            return False

        # Check if stock meets volume and volatility thresholds
        return avg_volume > min_volume and volatility > min_volatility
    except Exception as e:
        print(f"Error in filter_stocks: {e}")
        return False


def evaluate_trading_signals(data):
    try:
        # Flatten the columns and remove the multi-index
        data.columns = [col[0] for col in data.columns]

        print("Columns in data:", data.columns)  # Check if the columns are correct

        # Ensure that we have all the required columns before proceeding
        required_columns = ["Close", "SMA", "MACD", "Signal", "RSI"]
        if not all(col in data.columns for col in required_columns):
            print(
                f"Missing required columns: {set(required_columns) - set(data.columns)}"
            )
            return "HOLD"

        # Drop rows where any of the required columns are NaN
        data = data.dropna(subset=required_columns)

        if data.empty:
            print("Insufficient data after dropping NaN values.")
            return "HOLD"

        latest = data.iloc[-1]  # Get the latest row

        # Buy Signal: All indicators align for an uptrend
        if (
            latest["Close"] > latest["SMA"]  # Price above SMA
            and latest["MACD"] > latest["Signal"]  # MACD > Signal Line
            and latest["RSI"] > 50  # RSI > 50
        ):
            return "BUY"

        # Sell Signal: All indicators align for a downtrend
        elif (
            latest["Close"] < latest["SMA"]  # Price below SMA
            and latest["MACD"] < latest["Signal"]  # MACD < Signal Line
            and latest["RSI"] < 50  # RSI < 50
        ):
            return "SELL"

        # Hold if no clear signal
        return "HOLD"
    except Exception as e:
        print(f"Error in evaluate_trading_signals: {e}")
        return "HOLD"


# Function to place buy/sell orders
def place_order(stock_symbol, action):
    try:
        stock = Stock(stock_symbol, "SMART", "USD")
        positions = ib.positions()
        current_position = sum(
            pos.position for pos in positions if pos.contract.symbol == stock_symbol
        )

        if action == "BUY":
            if current_position > 0:
                print(f"Already holding {stock_symbol}. No action taken.")
                return
            order = MarketOrder("BUY", 10)  # Adjust quantity as needed
        elif action == "SELL":
            if current_position <= 0:
                print(f"Not holding {stock_symbol}. No action taken.")
                return
            order = MarketOrder("SELL", current_position)  # Sell all shares

        trade = ib.placeOrder(stock, order)
        print(
            f"{action} order placed for {stock_symbol}. Status: {trade.orderStatus.status}"
        )
    except Exception as e:
        print(f"Error placing {action} order for {stock_symbol}: {e}")


# Function to check if the market is open
def is_market_open():
    est = pytz.timezone("US/Eastern")
    now = datetime.now(est).time()
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    return market_open <= now <= market_close


# Function to monitor and trade stocks in S&P 500
def monitor_and_trade():
    sp500_stocks = get_sp500_stocks()
    print("Monitoring S&P 500 stocks for trading opportunities...")

    for stock_symbol in sp500_stocks:
        print(f"Processing {stock_symbol}...")

        # Fetch historical data and calculate indicators
        data = calculate_indicators(stock_symbol)

        if data is None or len(data) < 10:  # Skip stocks with invalid data
            print(
                f"Skipping {stock_symbol} due to insufficient data or missing indicators."
            )
            continue

        # Filter stocks by volume and volatility
        if not filter_stocks(data):
            print(f"{stock_symbol} does not meet volume/volatility criteria. Skipping.")
            continue

        # Evaluate trading signals
        signal = evaluate_trading_signals(data)
        print(f"{stock_symbol} signal: {signal}")

        if signal == "BUY":
            print(f"{stock_symbol}: Bullish signal detected. Placing buy order...")
            place_order(stock_symbol, "BUY")
        elif signal == "SELL":
            print(f"{stock_symbol}: Bearish signal detected. Placing sell order...")
            place_order(stock_symbol, "SELL")

        # Pause between stock evaluations to avoid rate-limiting
        time.sleep(2)


# Main loop
while True:
    if is_market_open():
        monitor_and_trade()
    else:
        print("Market is closed. Waiting for market hours...")
    time.sleep(3600)  # Run every hour
