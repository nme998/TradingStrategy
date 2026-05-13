import numpy as np
import pandas as pd
import yfinance as yf
import datetime

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# FEATURES
# =========================================================
def build_base_features(df):

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # -----------------------
    # RETURNS
    # -----------------------
    df["return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_20"] = df["return"].rolling(20).std()

    # -----------------------
    # LAG RETURNS (XGB FEATURES)
    # -----------------------
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag{lag}"] = df["return"].shift(lag)

    # -----------------------
    # TECHNICAL INDICATORS
    # -----------------------
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()

    df["MACD"] = df["EMA_10"] - df["EMA_20"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=14).mean()
    avg_loss = loss.ewm(span=14).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["BB_mid"] = df["Close"].rolling(20).mean()
    df["BB_std"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    # -----------------------
    # DATE FEATURES
    # -----------------------
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear

    df = df.dropna()

    return df


# =========================================================
# HMM
# =========================================================
def fit_hmm(train_df):

    returns = train_df["return"].values.reshape(-1, 1)

    hmm = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=500,
        random_state=42
    )

    hmm.fit(returns)

    hidden = hmm.predict(returns)
    means = [returns[hidden == i].mean() for i in range(2)]

    down_state = int(np.argmin(means))
    up_state = int(np.argmax(means))

    return hmm, down_state, up_state


# =========================================================
# APPLY HMM
# =========================================================
def apply_hmm(df, hmm):

    states = hmm.predict(df["return"].values.reshape(-1, 1))
    probs = hmm.predict_proba(df["return"].values.reshape(-1, 1))

    df = df.copy()
    df["regime"] = states
    df["prob_low"] = probs[:, 0]
    df["prob_high"] = probs[:, 1]

    return df


# =========================================================
# CROSS SECTIONAL FEATURES
# =========================================================
def add_cross_sectional_features(df):

    df = df.copy()

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    # Sort BEFORE grouping (important)
    df = df.sort_values(["Ticker", df.index.name])

    print("\n[CROSS-SECTIONAL] feature columns:")
    print(df.drop(columns=["Close"]).columns.tolist())

    # -----------------------
    # GROUP BY DATE (INDEX LEVEL)
    # -----------------------
    grouped = df.groupby(df.index)

    # Z-score across tickers per day
    rank_vals = grouped["return"].rank()
    mean = grouped["return"].transform("mean")
    std = grouped["return"].transform("std")
    zscore_vals = (df["return"] - mean) / (std + 1e-8)

    insert_loc = df.columns.get_loc("return") + 1

    df.insert(insert_loc, "rank_return", rank_vals)
    df.insert(insert_loc + 1, "zscore_return", zscore_vals)


    # -----------------------
    # FINAL SORT (date → ticker)
    # -----------------------
    df = df.sort_values([df.index.name, "Ticker"])

    # -----------------------
    # REMOVE TICKER (NOT FOR MODEL)
    # -----------------------
    #df = df.drop(columns=["Ticker"])

    return df


# =========================================================
# PIPELINE
# =========================================================
def get_feature_data(ticker):

    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=365 * 10)

    df = yf.download(ticker, start=start, end=end)
    print("LAST ROW OF YF PULL: ", df.tail(3))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = build_base_features(df)

    return df