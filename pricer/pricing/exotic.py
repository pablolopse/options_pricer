"""
Monte Carlo pricing with variance reduction, and path-dependent (Asian) options.
"""

import numpy as np


def _gbm_paths(S, T, r, sigma, n_paths, n_steps, antithetic=True):
    """
    Simulate GBM price paths, shape (n_paths, n_steps + 1), paths[:, 0] == S.
    With antithetic=True, paths come in Z / -Z mirrored pairs (n_paths must be even).
    """
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z_half = np.random.standard_normal((half, n_steps))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = np.random.standard_normal((n_paths, n_steps))

    log_returns = drift + vol * Z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = np.empty((Z.shape[0], n_steps + 1))
    paths[:, 0] = S
    paths[:, 1:] = S * np.exp(log_paths)
    return paths


def _price_and_se(discounted_payoffs, antithetic):
    """
    Mean price and Monte Carlo standard error. When antithetic=True,
    discounted_payoffs is assumed arranged as [Z-half; -Z-half] (see _gbm_paths),
    so the standard error is computed on the per-pair averages — this is what
    actually captures the variance reduction from the negative correlation
    between paired draws. Averaging plain individual payoffs (ignoring the
    pairing) would report the same price but hide the variance reduction.
    """
    if antithetic:
        half = len(discounted_payoffs) // 2
        pair_means = (discounted_payoffs[:half] + discounted_payoffs[half:]) / 2
        price = pair_means.mean()
        std_error = pair_means.std(ddof=1) / np.sqrt(half)
    else:
        price = discounted_payoffs.mean()
        std_error = discounted_payoffs.std(ddof=1) / np.sqrt(len(discounted_payoffs))
    return price, std_error


def european_monte_carlo(S, K, T, r, sigma, n_paths=10000, n_steps=1, option_type='call',
                          antithetic=True, seed=None):
    """
    Price a vanilla European option by Monte Carlo simulation, with optional
    antithetic-variates variance reduction.

    Args:
        S, K, T, r, sigma: standard Black-Scholes inputs.
        n_paths (int): number of simulated paths (rounded down to even if antithetic).
        n_steps (int): time steps used to simulate the path to maturity (1 is enough
            for a European payoff, since only the terminal price matters).
        option_type (str): 'call' or 'put'.
        antithetic (bool): use antithetic variates (Z and -Z paired draws) to reduce
            estimator variance versus plain Monte Carlo.
        seed (int or None): if given, seeds the RNG for reproducibility.

    Returns:
        (price, standard_error): the discounted mean payoff and its Monte Carlo
        standard error, so variance-reduction benefit can be measured directly.
    """
    if seed is not None:
        np.random.seed(seed)
    if antithetic and n_paths % 2 != 0:
        n_paths += 1

    paths = _gbm_paths(S, T, r, sigma, n_paths, n_steps, antithetic=antithetic)
    terminal = paths[:, -1]
    if option_type == 'call':
        payoffs = np.maximum(terminal - K, 0)
    else:
        payoffs = np.maximum(K - terminal, 0)

    discounted = np.exp(-r * T) * payoffs
    return _price_and_se(discounted, antithetic)


def asian_option_monte_carlo(S, K, T, r, sigma, n_paths=10000, n_steps=50, option_type='call',
                              antithetic=True, seed=None):
    """
    Price an arithmetic-average Asian option by Monte Carlo simulation: the payoff
    depends on the average simulated price over the path, not just the terminal
    price, so this is a genuinely path-dependent derivative (unlike the vanilla
    European/American pricers elsewhere in this package).

    Args:
        S, K, T, r, sigma: standard Black-Scholes inputs.
        n_paths (int): number of simulated paths (rounded down to even if antithetic).
        n_steps (int): number of averaging points along the path to maturity.
        option_type (str): 'call' or 'put'.
        antithetic (bool): use antithetic variates (Z and -Z paired draws).
        seed (int or None): if given, seeds the RNG for reproducibility.

    Returns:
        (price, standard_error): the discounted mean payoff and its Monte Carlo
        standard error.
    """
    if seed is not None:
        np.random.seed(seed)
    if antithetic and n_paths % 2 != 0:
        n_paths += 1

    paths = _gbm_paths(S, T, r, sigma, n_paths, n_steps, antithetic=antithetic)
    average_price = paths[:, 1:].mean(axis=1)
    if option_type == 'call':
        payoffs = np.maximum(average_price - K, 0)
    else:
        payoffs = np.maximum(K - average_price, 0)

    discounted = np.exp(-r * T) * payoffs
    return _price_and_se(discounted, antithetic)
