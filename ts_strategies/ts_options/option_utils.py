from datetime import datetime

import numpy as np

def business_days_to_expiry(current_date, expiry_date):
    return np.busday_count(np.datetime64(current_date), np.datetime64(expiry_date))


def time_to_expiry(current_date, expiry_date):
    days = business_days_to_expiry(current_date, expiry_date)

    return max(days / 252.0, 0.0)

def moneyness(S, K):
    return S / K

def intrinsic_value(S, K, option_type):

    option_type = option_type.lower()

    if option_type == "call":
        return max(S - K, 0.0)

    elif option_type == "put":
        return max(K - S, 0.0)

    raise ValueError("option_type must be 'call' or 'put'")


def extrinsic_value(option_price, S, K, option_type):
    return option_price - intrinsic_value(S, K, option_type)

def payoff_call(ST, K):
    return max(ST - K, 0.0)


def payoff_put(ST, K):
    return max(K - ST, 0.0)

def realised_volatility(returns, annualisation_factor=252):
    returns = np.asarray(returns)

    return (np.std(returns, ddof=1) * np.sqrt(annualisation_factor))

def annualise_volatility(volatility, periods_per_year=252):
    return volatility * np.sqrt(periods_per_year)