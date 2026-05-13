import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

LSTM_FEATURES = [
    "return",
    "volatility_20",
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "return_lag5",
    "return_lag10",
    "SMA_10",
    "SMA_20",
    "EMA_10",
    "EMA_20",
    "MACD",
    "RSI",
    "BB_mid",
    "BB_std",
    "BB_upper",
    "BB_lower",
    "dayofweek",
    "month",
    "dayofyear",
    "regime",
    "prob_low",
    "prob_high"
]

def create_reversal_labels(df, horizon=5, move_threshold=0.01, lookback=30):

    df = df.copy()

    returns = df["return"].values
    labels = np.zeros(len(df))

    for t in range(len(df) - horizon - 1):

        past_returns = returns[t-lookback:t]

        future_returns = returns[t+1:t+horizon+1]

        if len(future_returns) < horizon:
            continue

        past_move = np.sum(past_returns)
        future_move = np.sum(future_returns)

        strong_past = abs(past_move) > move_threshold
        strong_future = abs(future_move) > move_threshold

        opposite_direction = (np.sign(past_move) !=  np.sign(future_move))

        if (strong_past and strong_future and opposite_direction) : labels[t] = 1

    df["reversal_label"] = labels
    print(df['reversal_label'].mean())

    return df

def train_lstm(train_df, lookback=30):

    df = train_df.copy()

    df = create_reversal_labels(df)

    missing = [c for c in LSTM_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing LSTM features: {missing}")
    
    # -----------------------------
    # scale features
    # -----------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[LSTM_FEATURES].values)

    y = df["reversal_label"].values

    # -----------------------------
    # build sequences
    # -----------------------------
    X, Y = [], []

    for i in range(lookback, len(df)):
        X.append(X_scaled[i-lookback:i])
        Y.append(y[i])

    X = np.array(X)
    Y = np.array(Y)

    # -----------------------------
    # model
    # -----------------------------
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")  
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy"
    )

    model.fit(
        X, Y,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    return model, scaler

def detect_reversal(lstm_model, scaler, lookback_window):

    X = lookback_window[LSTM_FEATURES].values

    X = scaler.transform(X)
    X = np.expand_dims(X, axis=0)

    pred = lstm_model.predict(X, verbose=0)[0][0]

    return pred