import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from edgar import *
import datetime
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

MODEL_FILE = "xgb_model.json"

end_date = datetime.datetime.now() 
start_date = end_date - datetime.timedelta(days = 365 * 5)
stock_data = yf.download('AAPL', start=start_date.date(), end=end_date.date(), auto_adjust=False)
stock_data.columns = stock_data.columns.get_level_values(0)

n_lookback = 30
n_forecast = 3

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

def date_features(df):
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

def Calc_returns(df): 
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    #df = df.dropna()
    print(df.shape)
    return df['log_return'].values.reshape(-1, 1)

def HMM_train(data):
    returns = Calc_returns(data) 
    returns = returns[1:]
    data = data.iloc[1:]

    #Training HMM
    hmm_model = GaussianHMM(
    n_components=2,        # 2 hidden states
    covariance_type="full",
    n_iter=1000,
    random_state=42
)
    hmm_model.fit(returns)

    transition_matrix = hmm_model.transmat_
    state_means = hmm_model.means_.flatten()
    state_vars = np.array([np.diag(cov)[0] for cov in hmm_model.covars_])

    print("Transition Matrix:")
    print(transition_matrix)

    print("\nState Means:")
    print(state_means)

    print("\nState Variances:")
    print(state_vars)

    hidden_states = hmm_model.predict(returns)
    state_probs = hmm_model.predict_proba(returns)

    sorted_states = np.argsort(state_means)

    down_state = sorted_states[0]
    up_state = sorted_states[1]

    print("\nState Mapping:")
    print("Down state:", down_state)
    print("Up state:", up_state)

    data['regime'] = hidden_states
    data['prob_state_0'] = state_probs[:, 0]
    data['prob_state_1'] = state_probs[:, 1]

    print("\nValidation Statistics:")
    for i in range(2):
        state_returns = data[data['regime'] == i]['log_return']
        print(f"State {i}:")
        print("  Mean:", state_returns.mean())
        print("  Std:", state_returns.std())
        print("  Count:", len(state_returns))

    print("\nDiagonal (Persistence) Probabilities:")
    print(np.diag(transition_matrix))

    regime_probs = hmm_model.predict_proba(returns)
    X_with_regime = np.hstack([data, regime_probs])
    print(X_with_regime)
    return(data)

check_stationarity(stock_data['Close'])
check_stationarity(stock_data['Close'].diff(periods=1).dropna())

stock_data["close_diff_1"] = stock_data.Close.diff(periods=1)
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
stock_data = HMM_train(stock_data)
stock_data.drop("log_return", axis=1, inplace=True)
stock_data["target"] = stock_data["Close"].shift(-1)
stock_data["target2"] = stock_data["Close"].shift(-2)
stock_data["target3"] = stock_data["Close"].shift(-3)
print(stock_data[-10:-2])

last_row = stock_data.tail(1)
stock_data.drop(stock_data.tail(1).index, inplace=True)
stock_data.dropna(inplace=True)

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

def train_test_split(df, test_size=0.2):
    data = df.values
    
    feature_scaler.fit(data[:, :-n_forecast]) 
    target_scaler.fit(data[:, -n_forecast:]) 
    scaled_data = feature_scaler.transform(data[:, :-n_forecast])
    scaled_target = target_scaler.transform(data[:, -n_forecast:])
    data_scaled = np.concatenate((scaled_data, scaled_target), axis=1)
    
    
    n = int(len(data_scaled) * (1 - test_size))
    return data_scaled[:n], data_scaled[n:]

def xgb_prediction(train, value):
    global model

    if os.path.exists(MODEL_FILE):
        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model.load_model(MODEL_FILE)
        print("Model loaded.")
    else:
        train = np.array(train)
        X, Y = train[:, :-n_forecast], train[:, -n_forecast:]

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
        test_X, test_Y = test[i, :-n_forecast], test[i, -n_forecast:] 
        pred = xgb_prediction(history, test_X) 
        predictions.append(pred)
        history.append(test[i])
    
    Y_test = target_scaler.inverse_transform(test[:, -n_forecast:])
    Y_pred = target_scaler.inverse_transform(np.array(predictions))
    test_rmse = root_mean_squared_error(Y_test, Y_pred)

    return test_rmse, Y_test, Y_pred

test_rmse, Y_test, predictions = walk_forward_validation(stock_data, 0.2)

def plot_branching_forecast(stock_data, Y_actual, Y_pred, n_forecast=3):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    Y_actual = np.array(Y_actual)
    Y_pred = np.array(Y_pred)
    
    test_start_idx = len(stock_data) - len(Y_actual) - n_forecast + 1

    plt.figure(figsize=(14, 7))

    # plot actual close
    plt.plot(stock_data.index, stock_data['Close'], color='black', label='Actual Close')

    for i in range(len(Y_pred)):
        origin_idx = test_start_idx + i
        future_dates = stock_data.index[origin_idx:origin_idx + n_forecast + 1]
        origin_price = stock_data['Close'].iloc[origin_idx]
        future_prices = [origin_price] + list(Y_pred[i])

        # truncate if needed to match lengths
        if len(future_dates) < len(future_prices):
            future_prices = future_prices[:len(future_dates)]

        plt.plot(future_dates, future_prices, color='red', alpha=0.2)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.title(f"Rolling {n_forecast}-Day Forecast Branching Plot")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_branching_forecast(stock_data, Y_test, predictions, n_forecast=3)

print(last_row)
print("Test RMSE:", test_rmse)
prediction = xgb_prediction(stock_data.values, last_row.values[0][:-n_forecast])
print("Prediction:", prediction)

#model.save_model(MODEL_FILE)
#print("Model trained and saved.")