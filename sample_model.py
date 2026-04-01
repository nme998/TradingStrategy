import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from edgar import *
import datetime
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

'''
url = "https://data.sec.gov/submissions/CIK0000320193.json"

try:
    with urllib.request.urlopen(url) as response:
        # Read the response and decode the bytes to a string (usually utf-8)
        content_bytes = response.read()
        content_string = content_bytes.decode('utf-8')
        
        # Parse the JSON string into a Python object using json.loads()
        json_data = json.loads(content_string)
        print(json_data)
except urllib.error.URLError as e:
    print(f"An error occurred while trying to reach the URL: {e}")
------------------------------------------------------------------------
set_identity("Nikhil Edwin nikhiledwin89@gmail.com")

company = Company("AAPL")#0000320193
filings = company.get_filings(form="10-K").head(3)
multi_financials = MultiFinancials.extract(filings)
financials = company.get_financials()

# Access individual statements
balance_sheet = financials.balance_sheet
income_statement = financials.income_statement()
cash_flow = financials.cashflow_statement()

print("Balance Sheet:", balance_sheet.head())
'''
MODEL_FILE = "xgb_model.json"

end_date = datetime.datetime.now() 
start_date = end_date - datetime.timedelta(days = 365 * 7)
stock_data = yf.download('AAPL', start=start_date.date(), end=end_date.date(), auto_adjust=False)
stock_data.columns = stock_data.columns.get_level_values(0)
print(stock_data.tail(5))
print(stock_data.shape)

def SMA(data, window_size):
    return data['Close'].rolling(window=window_size).mean()

def EMA(data, window_size):
    return data['Close'].ewm(span=window_size).mean()

def MACD(data, short_window, long_window):
    short_EMA = EMA(data, short_window)
    long_EMA = EMA(data, long_window)
    return short_EMA - long_EMA

def RSI(data, window_size):
    delta = data['Close'].diff()
    delta = delta[1:] 
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=window_size-1 , min_periods=window_size).mean()
    ema_down = down.ewm(com=window_size-1 , min_periods=window_size).mean()
    return ema_up/ema_down

def Bollinger_Bands(data, window_size):
    middle_band = SMA(data, window_size)
    std_dev = data['Close'].rolling(window=window_size).std()
    upper_band = middle_band + (std_dev*2)
    lower_band = middle_band - (std_dev*2)
    return upper_band, lower_band

def check_stationarity(series):

    result = adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

    # Perform differencing if the series is non-stationary
    #if result[1] > 0.05:
    #    print("Differencing the series...")
    #    series = series.diff().dropna()
    #    check_stationarity(series)

def date_features(df):
    # Create time series features based on time series index.
    df.index = pd.to_datetime(df.index)
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

check_stationarity(stock_data['Close'])
check_stationarity(stock_data['Close'].diff(periods=1).dropna())

stock_data["close_diff_1"] = stock_data.Close.diff(periods=1)
#stock_data.index = stock_data["Date"]
#stock_data.drop("Date", axis=1, inplace=True)
stock_data = date_features(stock_data)

plt.rc("figure", figsize=(10,5))
plot_pacf(stock_data['Close'], method='ywm')
plt.show()

stock_data["close(-1)"] = stock_data['Close'].shift(1)
stock_data['SMA'] = SMA(stock_data, 13)
stock_data['EMA'] = EMA(stock_data, 9) 
stock_data['MACD'] = MACD(stock_data, 24, 52)
stock_data['RSI'] = RSI(stock_data, 14)
stock_data['Upper_Band'], stock_data['Lower_Band'] = Bollinger_Bands(stock_data, 10)

stock_data["H_L_diff"] = stock_data["High"] - stock_data["Low"]
stock_data.drop("Adj Close", axis=1, inplace=True)
stock_data.drop("High", axis=1, inplace=True)
stock_data.drop("Low", axis=1, inplace=True)
stock_data["Bands_diff"] = stock_data["Upper_Band"] - stock_data["Lower_Band"]
stock_data.drop("Upper_Band", axis=1, inplace=True)
stock_data.drop("Lower_Band", axis=1, inplace=True)
stock_data["target"] = stock_data["Close"].shift(-1)
print(stock_data.tail(5))

last_row = stock_data.tail(1)
stock_data.drop(stock_data.tail(1).index, inplace=True)
stock_data.dropna(inplace=True)

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

def train_test_split(df, test_size=0.2):
    data = df.values
    
    feature_scaler.fit(data[:, :-1]) 
    target_scaler.fit(data[:, -1:]) 
    scaled_data = feature_scaler.transform(data[:, :-1])
    scaled_target = target_scaler.transform(data[:, -1:])
    data_scaled = np.concatenate((scaled_data, scaled_target), axis=1)
    
    
    n = int(len(data_scaled) * (1 - test_size))
    return data_scaled[:n], data_scaled[n:]

'''
def xgb_prediction(train, value):
    train = np.array(train)
    X, Y = train[:, :-1], train[:, -1]
    global model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)

    model.fit(X, Y)
    val = np.array(value).reshape(1, -1)
    prediction = model.predict(val)
    return prediction[0] 
'''

def xgb_prediction(train, value):
    global model

    if os.path.exists(MODEL_FILE):
        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model.load_model(MODEL_FILE)
        print("Model loaded.")
    else:
        train = np.array(train)
        X, Y = train[:, :-1], train[:, -1]

        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model.fit(X, Y)
        print("Model trained.")

    val = np.array(value).reshape(1, -1)
    prediction = model.predict(val)

    return prediction[0]

def walk_forward_validation(data, percentage=0.2):
    train, test = train_test_split(data, percentage)
    predictions = []
    history = [x for x in train]

    for i in range(len(test)):
        test_X, test_Y = test[i, :-1], test[i, -1] 
        pred = xgb_prediction(history, test_X) 
        predictions.append(pred)
        history.append(test[i])
    
    Y_test = target_scaler.inverse_transform(test[:, -1:].reshape(1, -1))
    Y_pred = target_scaler.inverse_transform(np.array(predictions).reshape(1, -1))
    test_rmse = root_mean_squared_error(Y_test, Y_pred)

    return test_rmse, Y_test, Y_pred

print("---------------------------stock_data---------------------------")
print(stock_data)
test_rmse, Y_test, predictions = walk_forward_validation(stock_data, 0.2)

def plot_values(df, percentage=0.2):

    train, test = train_test_split(df, percentage)
    X, Y = train[:, :-1], train[:, -1]
    train_predictions = model.predict(X)
    
    # unscale the data
    train_predictions = target_scaler.inverse_transform(train_predictions.reshape(-1, 1))
    Y = target_scaler.inverse_transform(Y.reshape(-1, 1))
    train_rmse = root_mean_squared_error(Y, train_predictions)

    print(f"Training RMSE: {train_rmse}")
    print(f"Testing RMSE: {test_rmse}")

    # plot actual vs. predicted values
    # concat Y and Y_test to plot them together
    Y = Y.reshape(-1)
    train_predictions = train_predictions.reshape(-1)
    Y = np.concatenate((Y, Y_test.reshape(-1)), axis=0)
    new_predictions = np.concatenate((train_predictions, predictions.reshape(-1)), axis=0)
    plt.plot(new_predictions, label='Predicted')
    plt.plot(Y, label='Actual')  
    plt.legend()
    plt.show()

plot_values(stock_data, 0.2)

plt.plot(Y_test.reshape(-1), label='Actual', color='purple')
plt.plot(predictions.reshape(-1), label='Predicted', color='gray')
plt.legend()
plt.show()

print(last_row)
prediction = xgb_prediction(stock_data.values, last_row.values[0][:-1])
print("Prediction:", prediction)

#model.save_model(MODEL_FILE)
#print("Model trained and saved.")