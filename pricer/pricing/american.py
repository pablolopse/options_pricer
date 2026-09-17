"""
American Option Pricing Models
Implements Binomial Tree, Finite Difference, and Longstaff-Schwartz Monte Carlo methods.
"""

import numpy as np
from scipy.stats import norm

# Binomial Tree pricing (American option)
def american_binomial(S, K, T, r, sigma, n_steps, option_type='call'):
    """
    Price American option using binomial tree.
    Args:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiration (years)
        r (float): Risk-free rate
        sigma (float): Volatility
        n_steps (int): Number of time steps in the tree
        option_type (str): 'call' or 'put'
    Returns:
        float: Option price
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    asset_prices = np.zeros(n_steps + 1)
    for i in range(n_steps + 1):
        asset_prices[i] = S * (u ** (n_steps - i)) * (d ** i)
    option_values = np.zeros(n_steps + 1)
    if option_type == 'call':
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)
    for step in range(n_steps - 1, -1, -1):
        for i in range(step + 1):
            asset_price = S * (u ** (step - i)) * (d ** i)
            option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            if option_type == 'call':
                exercise_value = max(asset_price - K, 0)
            else:
                exercise_value = max(K - asset_price, 0)
            option_values[i] = max(option_values[i], exercise_value)
    return option_values[0]

# Finite Difference pricing (American option)
def american_finite_difference(S, K, T, r, sigma, option_type='put', n_price=100, n_time=1000):
    """
    Price American option using explicit finite difference method.
    Args:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiration (years)
        r (float): Risk-free rate
        sigma (float): Volatility
        option_type (str): 'call' or 'put'
        n_price (int): Number of price grid points
        n_time (int): Number of time steps
    Returns:
        float: Option price
    """
    S_max = 2 * K
    dS = S_max / n_price
    dt = T / n_time
    if dt > 0.5 * dS**2 / sigma**2 / S_max**2:
        n_time = int(T / (0.4 * dS**2 / sigma**2 / S_max**2))
        dt = T / n_time
    prices = np.linspace(0, S_max, n_price + 1)
    values = np.zeros((n_price + 1, n_time + 1))
    if option_type == 'call':
        values[:, -1] = np.maximum(prices - K, 0)
    else:
        values[:, -1] = np.maximum(K - prices, 0)
    if option_type == 'call':
        values[-1, :] = S_max - K
        values[0, :] = 0
    else:
        values[0, :] = K
        values[-1, :] = 0
    for j in range(n_time - 1, -1, -1):
        for i in range(1, n_price):
            S_i = i * dS
            delta = (values[i+1, j+1] - values[i-1, j+1]) / (2 * dS)
            gamma = (values[i+1, j+1] - 2*values[i, j+1] + values[i-1, j+1]) / dS**2
            theta = -0.5 * sigma**2 * S_i**2 * gamma - r * S_i * delta + r * values[i, j+1]
            values[i, j] = values[i, j+1] - dt * theta
            if option_type == 'call':
                exercise_value = max(S_i - K, 0)
            else:
                exercise_value = max(K - S_i, 0)
            values[i, j] = max(values[i, j], exercise_value)
    return np.interp(S, prices, values[:, 0])

# Longstaff-Schwartz Monte Carlo pricing (American option)
def american_lsm(S, K, T, r, sigma, n_paths=10000, n_steps=50, option_type='put'):
    """
    Price American option using Longstaff-Schwartz method (Monte Carlo).
    Args:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiration (years)
        r (float): Risk-free rate
        sigma (float): Volatility
        n_paths (int): Number of Monte Carlo paths
        n_steps (int): Number of time steps
        option_type (str): 'call' or 'put'
    Returns:
        float: Option price
    """
    np.random.seed(42)
    dt = T / n_steps
    discount = np.exp(-r * dt)
    Z = np.random.standard_normal((n_paths, n_steps))
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    if option_type == 'call':
        payoffs = np.maximum(paths - K, 0)
    else:
        payoffs = np.maximum(K - paths, 0)
    exercise = np.zeros_like(payoffs)
    exercise[:, -1] = payoffs[:, -1] > 0
    for t in range(n_steps - 1, 0, -1):
        itm = payoffs[:, t] > 0
        if np.sum(itm) > 0:
            X = paths[itm, t]
            X_matrix = np.column_stack([np.ones_like(X), X, X**2])
            Y = np.zeros(np.sum(itm))
            for i, path_idx in enumerate(np.where(itm)[0]):
                future_exercise = np.where(exercise[path_idx, t+1:])[0]
                if len(future_exercise) > 0:
                    exercise_time = future_exercise[0] + t + 1
                    Y[i] = payoffs[path_idx, exercise_time] * np.exp(-r * (exercise_time - t) * dt)
            if len(Y) > 3:
                coeffs = np.linalg.lstsq(X_matrix, Y, rcond=None)[0]
                continuation_values = X_matrix @ coeffs
                exercise_now = payoffs[itm, t] > continuation_values
                exercise_indices = np.where(itm)[0][exercise_now]
                exercise[exercise_indices, t] = 1
                exercise[exercise_indices, t+1:] = 0
    option_values = []
    for i in range(n_paths):
        exercise_time = np.where(exercise[i, :])[0]
        if len(exercise_time) > 0:
            t = exercise_time[0]
            option_values.append(payoffs[i, t] * np.exp(-r * t * dt))
        else:
            option_values.append(0)
    return np.mean(option_values)
