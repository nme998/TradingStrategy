from ts_strategies.ts_options.black_scholes import (black_scholes_call, black_scholes_put)

from ts_strategies.ts_options.greeks import calculate_vega


def _implied_volatility(market_price: float, S: float, K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
    initial_guess: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Generic implied volatility solver.

    Parameters
    ----------
    market_price : float
        Observed market option price.

    initial_guess : float
        Initial volatility estimate.

    tolerance : float
        Stop once pricing error is below this threshold.

    max_iterations : int
        Maximum Newton iterations.

    Returns
    -------
    float
        Implied volatility.
    """

    sigma = initial_guess

    option_type = option_type.lower()

    for _ in range(max_iterations):

        if option_type == "call":
            model_price = black_scholes_call(
                S, K, T, r, sigma, q
            )

        elif option_type == "put":
            model_price = black_scholes_put(
                S, K, T, r, sigma, q
            )

        else:
            raise ValueError("option_type must be 'call' or 'put'")

        price_error = model_price - market_price

        # Converged
        if abs(price_error) < tolerance:
            return float(sigma)

        option_vega = calculate_vega(S, K, T, r, sigma, q)

        # Avoid divide-by-zero
        if option_vega < 1e-10:
            break

        # Newton-Raphson update
        sigma -= price_error / option_vega

        # Keep volatility positive
        sigma = max(sigma, 1e-6)

    raise RuntimeError("Implied volatility failed to converge.")


def implied_vol_call(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    initial_guess: float = 0.20,
) -> float:

    return _implied_volatility(market_price, S, K, T, r, option_type="call", q=q, initial_guess=initial_guess)


def implied_vol_put(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    initial_guess: float = 0.20,
) -> float:

    return _implied_volatility(market_price, S, K, T, r, option_type="put", q=q, initial_guess=initial_guess)