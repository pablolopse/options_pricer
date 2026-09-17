# Example: Compare option pricing models and Greeks
# -----------------------------------------------
# This script demonstrates how to use the pricer package to price a European and American option
# using several models, and prints the Greeks for the same option.

from pricer.pricing import black_scholes, merton_jump, american_binomial, american_finite_difference, american_lsm
from pricer.utils.helpers import get_risk_free_rate
from pricer.utils.greeks import calculate_greeks

if __name__ == "__main__":
    # --- Option parameters ---
    S = 100      # Spot price of the underlying asset
    K = 100      # Strike price of the option
    T = 1        # Time to expiration (in years)
    sigma = 0.2  # Volatility (annualized)
    n_steps = 100  # Steps for binomial/finite difference models
    n_paths = 1000 # Paths for LSM Monte Carlo
    option_type = 'call'  # 'call' or 'put'

    # --- Get risk-free rate (with fallback if API fails) ---
    try:
        r = get_risk_free_rate()
    except Exception:
        r = 0.03  # Fallback value if Yahoo Finance API fails

    print("\n--- Option Pricing Comparison ---")
    print(f"Spot: {S}, Strike: {K}, T: {T}, r: {r:.4f}, sigma: {sigma}, Type: {option_type}\n")

    # --- European Models ---
    # Black-Scholes (analytical, no jumps)
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    print(f"Black-Scholes: {bs_price:.4f}  # European {option_type}")

    # Merton Jump-Diffusion (European, includes jumps)
    merton_price = merton_jump(S, K, T, r, sigma, m=1, v=0.1, lam=1, t=option_type)
    print(f"Merton Jump-Diffusion: {merton_price:.4f}  # European {option_type} with jumps")

    # --- American Models ---
    # Binomial Tree (American)
    am_binomial = american_binomial(S, K, T, r, sigma, n_steps, option_type)
    print(f"American Binomial: {am_binomial:.4f}  # American {option_type}")

    # Finite Difference (American)
    am_fd = american_finite_difference(S, K, T, r, sigma, option_type, n_price=100, n_time=100)
    print(f"American Finite Difference: {am_fd:.4f}  # American {option_type}")

    # Longstaff-Schwartz Monte Carlo (American)
    am_lsm = american_lsm(S, K, T, r, sigma, n_paths=n_paths, n_steps=50, option_type=option_type)
    print(f"American LSM: {am_lsm:.4f}  # American {option_type} (Monte Carlo)")

    # --- Greeks (sensitivities) ---
    greeks = calculate_greeks(S, K, T, r, sigma, option_type)
    print("\nGreeks:")
    for k, v in greeks.items():
        print(f"  {k:6}: {v:.4f}")

    # --- Notes ---
    # - European models do not account for early exercise (use for stocks without dividends)
    # - American models allow early exercise (important for American-style options)
    # - Greeks help understand risk and sensitivity to market parameters