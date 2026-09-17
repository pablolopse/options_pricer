"""
Greeks calculation for options.
Provides sensitivities (Delta, Gamma, Theta, Vega, Rho) for Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm

# Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks for a European option using Black-Scholes.
    Args:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiration (years)
        r (float): Risk-free rate
        sigma (float): Volatility
        option_type (str): 'call' or 'put'
    Returns:
        dict: Greeks (delta, gamma, theta, vega, rho)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    if option_type == 'call':
        delta = cdf_d1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2) / 365
    else:
        delta = cdf_d1 - 1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100
    rho = K * T * np.exp(-r * T) * (cdf_d2 if option_type == 'call' else -norm.cdf(-d2)) / 100
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
