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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense

MODEL_FILE = "xgb_model.json"

def model_run():
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

    stock_data["return"] = np.log(stock_data["Close"] / stock_data["Close"].shift(1))
    stock_data["volatility_20"] = stock_data["return"].rolling(20).std()
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
    stock_data["return_lag1"] = stock_data["return"].shift(1)
    stock_data["return_lag2"] = stock_data["return"].shift(2)
    stock_data["return_lag3"] = stock_data["return"].shift(3)
    stock_data["return_lag5"] = stock_data["return"].shift(5)
    stock_data["return_lag10"] = stock_data["return"].shift(10)

    lstm_training_df = stock_data.copy()

    #Target Features
    stock_data["target"] = stock_data["return"].shift(-1)
    stock_data["target2"] = stock_data["return"].shift(-2)
    stock_data["target3"] = stock_data["return"].shift(-3)
    stock_data = stock_data.dropna()
    print(stock_data[-10:-2])

    # LSTM target columns = future Close prices
    lstm_training_df["target_close_1"] = lstm_training_df["Close"].shift(-1)
    lstm_training_df["target_close_2"] = lstm_training_df["Close"].shift(-2)
    lstm_training_df["target_close_3"] = lstm_training_df["Close"].shift(-3)

    # Drop rows with NaNs created by shifting
    lstm_training_df = lstm_training_df.dropna()

    last_row = stock_data.tail(1)
    stock_data.drop(stock_data.tail(1).index, inplace=True)
    stock_data.dropna(inplace=True)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    #LSTM Feature Extraction
    def lstm_train_test_split(df, n_lookback, n_forecast, test_size=0.2):

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        # convert dataframe to numpy
        data = df.to_numpy(dtype=np.float32)

        # ---- SCALE DATA ----
        feature_scaler.fit(data[:, :-n_forecast])
        target_scaler.fit(data[:, -n_forecast:])

        scaled_features = feature_scaler.transform(data[:, :-n_forecast])
        scaled_target = target_scaler.transform(data[:, -n_forecast:])

        data_scaled = np.concatenate((scaled_features, scaled_target), axis=1)

        # ---- CREATE SEQUENCES ----
        X, Y = [], []

        for i in range(n_lookback, len(data_scaled) - n_forecast + 1):
            X.append(data_scaled[i-n_lookback:i, :-n_forecast])
            Y.append(data_scaled[i, -n_forecast:])

        X = np.stack(X).astype(np.float32)
        Y = np.stack(Y).astype(np.float32)

        split = int(len(X) * (1 - test_size))

        X_train = X[:split]
        X_test = X[split:]

        Y_train = Y[:split]
        Y_test = Y[split:]

        return X_train, X_test, Y_train, Y_test

    def train_lstm_model(X_train, Y_train, n_forecast, epochs=30):

        model = Sequential()

        model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(32))
        model.add(Dense(16, activation="relu", name="latent_layer"))  # latent features
        model.add(Dense(n_forecast, name="forecast_output"))

        model.compile(
            optimizer="adam",
            loss="mse"
        )

        model.fit(
            X_train.astype("float32"),
            Y_train.astype("float32"),
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )

        

        return model

    def add_lstm_features(lstm_training_df, lstm_model, X_sequences, n_lookback, n_forecast):

        # Extract hidden features
        feature_extractor = Model(
            inputs=lstm_model.inputs,
            outputs=lstm_model.get_layer("latent_layer").output
        )

        latent_features = feature_extractor.predict(X_sequences)

        # Extract LSTM predictions
        lstm_predictions = lstm_model.predict(X_sequences)

        # Create dataframes
        latent_df = pd.DataFrame(
            latent_features,
            columns=[f"lstm_feat_{i}" for i in range(latent_features.shape[1])]
        )

        pred_df = pd.DataFrame(
            lstm_predictions,
            columns=[f"lstm_pred_{i+1}" for i in range(n_forecast)]
        )

        # Align indexes
        start_index = n_lookback
        end_index = n_lookback + len(latent_df)

        latent_df.index = lstm_training_df.index[start_index:end_index]
        pred_df.index = lstm_training_df.index[start_index:end_index]

        # Insert before targets
        insert_loc = len(lstm_training_df.columns) - n_forecast

        for col in latent_df.columns:
            lstm_training_df[col] = np.nan

        for col in pred_df.columns:
            lstm_training_df[col] = np.nan

        lstm_training_df.loc[latent_df.index, latent_df.columns] = latent_df
        lstm_training_df.loc[pred_df.index, pred_df.columns] = pred_df

        return lstm_training_df

    def walk_forward_train(lstm_training_df, train_size=0.8):

        split = int(len(lstm_training_df) * train_size)

        train = lstm_training_df[:split]
        test = lstm_training_df[split:]

        predictions = []

        history = train.copy()
        max_runs = 10
        step = max(1, len(test) // max_runs)

        for i in range(0, len(test), step):

            X_train, X_test, Y_train, Y_test = lstm_train_test_split(
                history, n_lookback, n_forecast
            )

            lstm_model = train_lstm_model(X_train, Y_train, n_forecast, epochs=10)

            pred = lstm_model.predict(X_test[-1].reshape(1, *X_test[-1].shape))

            predictions.append(pred)

            history = pd.concat([history, test.iloc[i:i+step]])

        stock_data = add_lstm_features(lstm_training_df, lstm_model, np.concatenate((X_train, X_test)), n_lookback, n_forecast)

        print(stock_data.filter(like="lstm_feat").describe())
        hidden_model = Model(
            inputs=lstm_model.inputs,
            outputs=lstm_model.get_layer("latent_layer").output
        )
        #DELETE (ONLY FOR DEBUGGING PURPOSES)__________________________________________
        hidden_feats = hidden_model.predict(X_train[:100])
        lstm_outputs = lstm_model.predict(X_train[:100])

        print("Hidden features shape:", hidden_feats.shape)
        print("LSTM outputs shape:", lstm_outputs.shape)

        print("\nHidden feature sample:")
        print(hidden_feats[:5])

        print("\nLSTM forecast sample:")
        print(lstm_outputs[:5])

        print("\nHidden feature std:", hidden_feats.std(axis=0)[:10])
        print("Output std:", lstm_outputs.std(axis=0))
        #________________________________________________________________________
        return predictions

    # LSTM feature generation
    #X_train, X_test, Y_train, Y_test = lstm_train_test_split(lstm_training_df, n_lookback, n_forecast)
    #lstm_model = train_lstm_model(X_train, Y_train, n_forecast)
    #stock_data = add_lstm_features(lstm_training_df, lstm_model, np.concatenate((X_train, X_test)), n_lookback, n_forecast)
    predictions = walk_forward_train(lstm_training_df)

    #XGBoost Prediction
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

        # Convert arrays
        Y_actual = np.array(Y_actual)
        Y_pred = np.array(Y_pred)
        
        test_start_idx = len(stock_data) - len(Y_actual) - n_forecast + 1

        plt.figure(figsize=(14, 7))

        # Plot actual close
        plt.plot(stock_data.index, stock_data['Close'], color='black', label='Actual Close')

        for i in range(len(Y_pred)):
            origin_idx = test_start_idx + i
            future_dates = stock_data.index[origin_idx:origin_idx + n_forecast + 1]
            origin_price = stock_data['Close'].iloc[origin_idx]

            # Convert predicted returns to price for plotting
            future_prices = [origin_price]
            for r in Y_pred[i]:
                future_prices.append(future_prices[-1] * (1 + r))  # cumulative price from returns

            # Truncate if needed
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

    # Plot forecast (predictions are still returns; converted inside the function)
    plot_branching_forecast(stock_data, Y_test, predictions, n_forecast=3)

    last_row_features = stock_data.iloc[-1, :-n_forecast].values
    predicted_returns = xgb_prediction(stock_data.values, last_row_features)

    # Convert predicted returns to prices for human-friendly display
    last_actual_close = stock_data['Close'].iloc[-1]
    predicted_prices = [last_actual_close]  # start from last actual close
    for r in predicted_returns:
        predicted_prices.append(predicted_prices[-1] * (1 + r))  # cumulative price

    # Remove first element (it's just the actual last close)
    predicted_prices = predicted_prices[1:]

    print("Last row features:\n", last_row_features)
    print("Prediction (returns for next 3 days):", predicted_returns)
    print("Prediction converted to closing prices:", predicted_prices)

    return predicted_returns, stock_data
