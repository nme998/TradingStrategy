import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta

from statsmodels.tsa.stattools import coint, adfuller

class OUModel:

    def fit(self, spread):

        spread = spread.dropna()
        x = spread[:-1].values
        y = spread[1:].values

        a, b = np.polyfit(x, y, 1)
        theta = -np.log(b)
        mu = a / (1 - b)
        residuals = y - (a + b * x)
        sigma = np.std(residuals)

        return {"theta": theta, "mu": mu, "sigma": sigma}
    
    def ou_zscore(self, spread, params):
        return (spread.iloc[-1] - params["mu"] ) / params["sigma"]


class CointegrationEngine:

    def __init__(self, tickers, date):
        self.tickers = tickers
        self.date = date
        self.lookback = 500
        self.prices = self._download_prices()

    # -----------------------------------------
    # Download price data
    # -----------------------------------------
    def _download_prices(self):
        end_date = pd.Timestamp(self.date)
        start_date = end_date - pd.DateOffset(years=5)
        data = yf.download(tickers=self.tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        prices = data["Close"]
        prices = prices.tail(self.lookback)

        return prices.dropna()

    # -----------------------------------------
    # Find cointegrated pairs
    # -----------------------------------------
    def find_cointegrated_pairs(self, coint_threshold=0.05, adf_threshold=0.05):
        pairs = []
        for asset_1, asset_2 in itertools.combinations(self.prices.columns, 2):

            series1 = self.prices[asset_1]
            series2 = self.prices[asset_2]
            coint_score, coint_pvalue, _ = coint(series1, series2)

            if coint_pvalue > coint_threshold:
                continue

            beta = self._estimate_beta( series1, series2)
            spread = self.build_spread(asset_1, asset_2, beta)
            adf_pvalue = adfuller(spread.dropna())[1]

            if adf_pvalue > adf_threshold:
                continue

            pairs.append({
                "asset_1": asset_1,
                "asset_2": asset_2,
                "beta": beta,
                "coint_score": coint_score,
                "coint_pvalue": coint_pvalue,
                "adf_pvalue": adf_pvalue
            })

        if not pairs:
            return pd.DataFrame()

        return (pd.DataFrame(pairs).sort_values(["coint_pvalue", "adf_pvalue"]).reset_index(drop=True))

    # -----------------------------------------
    # Estimate hedge ratio
    # -----------------------------------------
    def _estimate_beta(self, y, x):
        x = np.asarray(x)
        y = np.asarray(y)
        beta = np.cov(y, x)[0, 1] / np.var(x)

        return beta

    # -----------------------------------------
    # Build spread
    # -----------------------------------------
    def build_spread(self, asset_1, asset_2, beta):
        return (self.prices[asset_1] - beta * self.prices[asset_2])

    # -----------------------------------------
    # Rolling spread z-score
    # -----------------------------------------
    def zscore(self, series, window=50):
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()

        return (series - mean) / std