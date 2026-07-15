import numpy as np
import pandas as pd

def realised_volatility(returns, window=20):
    return (returns.rolling(window).std(ddof=1) * np.sqrt(252))

def parkinson_volatility(high, low, window=20):
    log_hl = np.log(high / low)
    variance = ((log_hl ** 2).rolling(window).mean() / (4 * np.log(2)))

    return np.sqrt(variance * 252)

def garman_klass_volatility(open_, high, low, close, window=20):

    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    variance = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).rolling(window).mean()

    return np.sqrt(variance * 252)


def volatility_of_volatility(realised_vol, window=20):
    return realised_vol.rolling(window).std(ddof=1)


def rolling_skew(returns, window=20):
    return returns.rolling(window).skew()

def rolling_kurtosis(returns, window=20):
    return returns.rolling(window).kurt()

def normalized_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window).mean()

    return atr / close

def add_volatility_features(df):
    df = df.copy()

    df["rv20"] = realised_volatility(df["return"], 20)
    df["parkinson20"] = parkinson_volatility(df["High"], df["Low"], 20)
    df["gk20"] = garman_klass_volatility(df["Open"], df["High"], df["Low"], df["Close"], 20)
    df["vol_of_vol"] = volatility_of_volatility(df["rv20"], 20)
    df["skew20"] = rolling_skew(df["return"], 20)
    df["kurt20"] = rolling_kurtosis(df["return"], 20)
    df["atr_norm"] = normalized_atr(df)

    return df

def future_realised_volatility(returns, window):

    return (returns.shift(-1).rolling(window=window).std(ddof=1).shift(-(window - 1)) * np.sqrt(252))


def add_volatility_targets(df):
    df = df.copy()

    df["target_1"] = future_realised_volatility(df["return"], 5)
    df["target_2"] = future_realised_volatility(df["return"], 10)
    df["target_3"] = future_realised_volatility(df["return"], 20)

    return df