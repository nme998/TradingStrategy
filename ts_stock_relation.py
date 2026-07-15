import itertools
import numpy as np
import pandas as pd
import yfinance as yf

from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm


class OUModel:
    def fit(self, spread):
        spread = spread.dropna()
        if len(spread) < 50:
            print(f"Spread too short: {len(spread)} | Skipping OU fit")
            return None

        x = spread[:-1].values
        y = spread[1:].values
        phi, intercept = np.polyfit(x, y, 1)

        if phi <= 0 or phi >= 1:
            print(f"Invalid phi: {phi:.4f} | Skipping OU fit")
            return None

        residuals = y - (intercept + phi * x)
        sigma_eps = np.std(residuals)

        if sigma_eps <= 1e-8:
            print(f"Sigma_eps too small: {sigma_eps:.8f} | Skipping OU fit")
            return None

        theta = -np.log(phi)
        mu = intercept / (1 - phi)
        sigma_eq = sigma_eps / np.sqrt(1 - phi**2)
        half_life = np.log(2) / theta

        return {
            "phi": phi,
            "theta": theta,
            "mu": mu,
            "sigma_eq": sigma_eq,
            "half_life": half_life
        }

    def ou_zscore(self, current_spread, params):
        if params is None:
            return None
        
        return (current_spread - params["mu"]) / params["sigma_eq"]

    def build_spread(self, asset_1_lookback, asset_2_lookback, beta, intercept):
        df = pd.concat([
            np.log(asset_1_lookback["Close"]),
            np.log(asset_2_lookback["Close"])
        ], axis=1).dropna()

        if len(df) == 0:
            return pd.Series(dtype=float)

        spread = (
            df.iloc[:, 0]
            - (intercept + beta * df.iloc[:, 1])
        )

        return spread


class CointegrationEngine:
    def __init__(self, tickers, date, prices = None):
        self.tickers = tickers
        self.date = pd.Timestamp(date)
        self.lookback = 500
        self.prices = prices.tail(500).dropna()

    # -----------------------------------------
    # Find cointegrated pairs
    # -----------------------------------------
    def find_cointegrated_pairs(self, coint_threshold=0.05):
        pairs = []

        tested = 0
        passed_coint = 0
        passed_ou = 0

        for asset_1, asset_2 in itertools.combinations(self.prices.columns, 2):

            tested += 1

            series1 = np.log(self.prices[asset_1])
            series2 = np.log(self.prices[asset_2])

            try:
                coint_score, coint_pvalue, _ = coint(series1, series2)

                print(
                    f"{asset_1}-{asset_2} | "
                    f"Coint p={coint_pvalue:.4f}"
                )

                if coint_pvalue > coint_threshold:
                    continue

                passed_coint += 1

                beta, intercept = self.estimate_hedge_ratio(
                    series1,
                    series2
                )

                spread = (
                    series1 - (intercept + beta * series2)
                ).dropna()

                if len(spread) < 100:
                    continue


                params = OUModel().fit(spread)

                if params is None:
                    print("    OU FAILED")
                    continue

                print(
                    f"    HalfLife={params['half_life']:.2f}"
                )

                if params["half_life"] < 2:
                    continue

                if params["half_life"] > 50:
                    continue

                passed_ou += 1

                pairs.append({
                    "asset_1": asset_1,
                    "asset_2": asset_2,
                    "intercept": intercept,
                    "beta": beta,

                    "ou_mu": params["mu"],
                    "ou_sigma": params["sigma_eq"],
                    "ou_theta": params["theta"],

                    "coint_score": coint_score,
                    "coint_pvalue": coint_pvalue,
                    "half_life": params["half_life"]
                })

            except Exception as e:
                print(asset_1, asset_2, e)

        print(
            f"\nTESTED={tested} "
            f"COINT={passed_coint} "
            f"OU={passed_ou}\n"
        )

        return pd.DataFrame(pairs)

    # -----------------------------------------
    # Hedge ratio
    # -----------------------------------------
    def estimate_hedge_ratio(self, y, x):

        X = sm.add_constant(x)

        model = sm.OLS(y, X).fit()

        intercept = model.params.iloc[0]
        beta = model.params.iloc[1]

        return beta, intercept

    # -----------------------------------------
    # Historical spread
    # -----------------------------------------
    def build_spread(self, asset_1, asset_2, beta, intercept):
        spread = (self.prices[asset_1] - (intercept + beta * self.prices[asset_2]))
        return spread.dropna()

    # -----------------------------------------
    # Classical rolling z-score
    # -----------------------------------------
    def zscore(self, series, window=50):
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()

        return (series - mean) / std