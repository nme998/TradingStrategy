import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import itertools


class CointegrationEngine:

    def __init__(self, price_df):
        """
        price_df:
            index = dates
            columns = tickers
            values = adjusted close prices
        """
        self.prices = price_df.dropna()

    # -----------------------------------------
    # 1. Find cointegrated pairs
    # -----------------------------------------
    def find_cointegrated_pairs(self, p_threshold=0.05):

        tickers = self.prices.columns
        pairs = []

        for t1, t2 in itertools.combinations(tickers, 2):

            series1 = self.prices[t1]
            series2 = self.prices[t2]

            score, pvalue, _ = coint(series1, series2)

            if pvalue < p_threshold:

                beta = self._estimate_beta(series1, series2)

                pairs.append({
                    "asset_1": t1,
                    "asset_2": t2,
                    "p_value": pvalue,
                    "coint_score": score,
                    "beta": beta
                })

        return pd.DataFrame(pairs).sort_values("p_value")

    # -----------------------------------------
    # 2. Estimate hedge ratio (beta)
    # -----------------------------------------
    def _estimate_beta(self, y, x):
        """
        OLS regression: y = beta * x
        """
        x = np.asarray(x)
        y = np.asarray(y)

        beta = np.cov(y, x)[0, 1] / np.var(x)

        return beta

    # -----------------------------------------
    # 3. Build spread series
    # -----------------------------------------
    def build_spread(self, asset_1, asset_2, beta):

        spread = self.prices[asset_1] - beta * self.prices[asset_2]

        return spread

    # -----------------------------------------
    # 4. Z-score of spread (for later strategy)
    # -----------------------------------------
    def zscore(self, series, window=50):

        mean = series.rolling(window).mean()
        std = series.rolling(window).std()

        z = (series - mean) / std

        return z