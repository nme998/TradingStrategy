import numpy as np
from scipy.stats import norm

from ts_strategies.ts_options.black_scholes import compute_d1_d2


def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)

    option_type = option_type.lower()

    if option_type == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))

    elif option_type == "put":
        return float(np.exp(-q * T) * (norm.cdf(d1) - 1))

    else:
        raise ValueError("option_type must be 'call' or 'put'")



def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:

    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)

    gamma = (np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))

    return float(gamma)



def calculate_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)

    vega = (S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))

    return float(vega)


def calculate_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, per_day: bool = True) -> float:

    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    first_term = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))

    option_type = option_type.lower()

    if option_type == "call":
        theta = (first_term - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1))

    elif option_type == "put":
        theta = (first_term + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1))

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if per_day:
        theta /= 365.0

    return float(theta)


def calculate_rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:

    _, d2 = compute_d1_d2(S, K, T, r, sigma, q)

    option_type = option_type.lower()

    if option_type == "call":
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2))

    elif option_type == "put":
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2))

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(rho)