"""
European Option Pricing Models
Implements Black-Scholes and Merton jump-diffusion models.
"""

import numpy as np
import math
from scipy.stats import norm
from ..utils.helpers import get_risk_free_rate

# Black-Scholes pricing (European option)
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes price for a European call or put option.
    Args:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to expiration (years)
        r (float): Risk-free rate
        sigma (float): Volatility
        option_type (str): 'call' or 'put'
    Returns:
        float: Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Merton jump-diffusion pricing (European option)
def merton_jump(S, K, T, r, sigma, m, v, lam, t='call'):
    """
    Price European options under the Merton jump-diffusion model.
    Args:
        S (float): Spot price
        K (float or array): Strike price(s)
        T (float): Time to expiration (years)
        r (float): Risk-free rate
        sigma (float): Volatility
        m (float): Jump mean
        v (float): Jump volatility
        lam (float): Jump intensity
        t (str): 'call' or 'put'
    Returns:
        float or np.ndarray: Option price(s)
    """
    if np.isscalar(K):
        Ks = [K]
        scalar_out = True
    else:
        Ks = np.array(K)
        scalar_out = False
    results = []
    for K_val in Ks:
        price = 0
        for k in range(40):
            r_k = r - lam*(m-1) + (k*np.log(m)) / T
            sigma_k = np.sqrt(sigma**2 + (k * v**2) / T)
            k_fact = math.factorial(k)
            weight = np.exp(-m*lam*T) * (m*lam*T)**k / k_fact
            if t == 'call':
                price += weight * black_scholes(S, K_val, T, r_k, sigma_k, 'call')
            else:
                price += weight * black_scholes(S, K_val, T, r_k, sigma_k, 'put')
        results.append(price)
    return results[0] if scalar_out else np.array(results)
