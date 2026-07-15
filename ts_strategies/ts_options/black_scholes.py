import numpy as np
from scipy.stats import norm


def compute_d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):

    if S <= 0:
        raise ValueError("Underlying price S must be positive.")

    if K <= 0:
        raise ValueError("Strike price K must be positive.")

    if T <= 0:
        raise ValueError("Time to expiry T must be positive.")

    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:

    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    call = (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    return float(call)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:

    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    put = (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))

    return float(put)