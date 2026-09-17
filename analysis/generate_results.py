"""
Analysis / results-generation script for the options_pricer repository.

This script is NOT part of the original `pricer` package. It was added purely
to exercise the existing pricing functions (black_scholes, merton_jump,
american_binomial, american_finite_difference, american_lsm, calculate_greeks)
and produce plots for the project README / portfolio writeup.

It does not modify any code inside `pricer/`.

Notes on scope (read before extending):
- `american_lsm` hard-codes `np.random.seed(42)` internally, so every call is
  deterministic given (S, K, T, r, sigma, n_paths, n_steps). This is good for
  reproducibility but means successive points in a "vs n_paths" convergence
  sweep are NOT independent Monte Carlo replications of each other (numpy's
  seeded draw for a larger `n_paths` is not a superset of the draw for a
  smaller one, and different `n_paths` produce differently-shaped draws from
  the same seed). Convergence is still meaningfully illustrated because the
  estimator's dispersion around the true value shrinks as n_paths grows.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pricer.pricing import (
    black_scholes,
    merton_jump,
    american_binomial,
    american_finite_difference,
    american_lsm,
    european_monte_carlo,
    asian_option_monte_carlo,
)
from pricer.utils.helpers import get_risk_free_rate
from pricer.utils.greeks import calculate_greeks

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Common parameters ---
S = 100.0
K = 100.0
T = 1.0
sigma = 0.20
option_type = "call"

try:
    r = get_risk_free_rate()
except Exception:
    r = 0.04

print(f"Using r = {r:.4f} (live 10Y Treasury proxy via yfinance, or 0.04 fallback)")

bs_price = black_scholes(S, K, T, r, sigma, option_type)
print(f"Black-Scholes reference price: {bs_price:.4f}")


# =========================================================================
# 1. Model comparison bar chart (reproduces example.py, plotted)
# =========================================================================
def plot_model_comparison():
    merton_price = merton_jump(S, K, T, r, sigma, m=1, v=0.1, lam=1, t=option_type)
    am_binomial = american_binomial(S, K, T, r, sigma, n_steps=200, option_type=option_type)
    am_fd = american_finite_difference(S, K, T, r, sigma, option_type, n_price=100, n_time=100)
    am_lsm = american_lsm(S, K, T, r, sigma, n_paths=20000, n_steps=50, option_type=option_type)

    labels = [
        "Black-Scholes\n(European)",
        "Merton Jump-Diff.\n(European)",
        "Binomial Tree\n(American)",
        "Finite Diff.\n(American)",
        "LSM Monte Carlo\n(American)",
    ]
    values = [bs_price, merton_price, am_binomial, am_fd, am_lsm]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Option price ($)")
    ax.set_title(f"Call option price by model  (S={S:.0f}, K={K:.0f}, T={T:.0f}y, "
                 f"$\\sigma$={sigma}, r={r:.3f})")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return dict(zip(["black_scholes", "merton", "binomial", "finite_diff", "lsm"], values))


# =========================================================================
# 2. LSM Monte Carlo convergence to Black-Scholes as n_paths grows
#    (valid check: for a non-dividend-paying call, early exercise is never
#    optimal, so the American price must equal the European/BS price.
#    This lets the existing American-only LSM engine be validated directly
#    against the closed-form Black-Scholes formula.)
# =========================================================================
def plot_lsm_convergence():
    path_counts = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 80000]
    prices = []
    for n in path_counts:
        p = american_lsm(S, K, T, r, sigma, n_paths=n, n_steps=50, option_type="call")
        prices.append(p)
        print(f"  n_paths={n:>6d}  LSM price={p:.4f}  abs.err={abs(p - bs_price):.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(path_counts, prices, "o-", color="#C44E52", label="LSM Monte Carlo price (call)")
    ax.axhline(bs_price, color="#4C72B0", linestyle="--", label=f"Black-Scholes = {bs_price:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Option price ($)")
    ax.set_title("LSM Monte Carlo convergence toward Black-Scholes\n"
                 "(non-dividend call: American = European, so this is a valid check)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "lsm_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return path_counts, prices


# =========================================================================
# 3. Binomial tree & finite-difference convergence vs number of steps
# =========================================================================
def plot_step_convergence():
    steps_list = [2, 5, 10, 20, 50, 100, 200, 400, 800]
    binom_prices = [american_binomial(S, K, T, r, sigma, n, "call") for n in steps_list]

    fd_steps = [10, 20, 50, 100, 200]
    fd_prices = [
        american_finite_difference(S, K, T, r, sigma, "call", n_price=n, n_time=n)
        for n in fd_steps
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_list, binom_prices, "o-", color="#55A868", label="Binomial tree (call)")
    ax.plot(fd_steps, fd_prices, "s-", color="#8172B2", label="Finite difference (call)")
    ax.axhline(bs_price, color="#4C72B0", linestyle="--", label=f"Black-Scholes = {bs_price:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of steps / grid points (log scale)")
    ax.set_ylabel("Option price ($)")
    ax.set_title("Binomial tree & finite-difference convergence\n(non-dividend call, so both should approach Black-Scholes)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "step_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# 4. Greeks vs spot price
# =========================================================================
def plot_greeks():
    spots = np.linspace(60, 140, 81)
    greek_names = ["delta", "gamma", "theta", "vega", "rho"]
    series = {g: [] for g in greek_names}
    for s in spots:
        g = calculate_greeks(s, K, T, r, sigma, option_type)
        for name in greek_names:
            series[name].append(g[name])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for ax, name, c in zip(axes, greek_names, colors):
        ax.plot(spots, series[name], color=c)
        ax.axvline(K, color="gray", linestyle=":", linewidth=1)
        ax.set_title(name.capitalize())
        ax.set_xlabel("Spot price")
        ax.grid(alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle(f"Black-Scholes Greeks vs spot price (K={K:.0f}, T={T:.0f}y, "
                 f"$\\sigma$={sigma}, r={r:.3f}, call)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "greeks_vs_spot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# 5. Antithetic-variates variance reduction: standard error vs. n_paths,
#    plain Monte Carlo vs. antithetic, for a vanilla European call.
# =========================================================================
def plot_variance_reduction():
    path_counts = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    plain_se, anti_se = [], []
    for n in path_counts:
        _, se_plain = european_monte_carlo(S, K, T, r, sigma, n_paths=n, antithetic=False, seed=1)
        _, se_anti = european_monte_carlo(S, K, T, r, sigma, n_paths=n, antithetic=True, seed=1)
        plain_se.append(se_plain)
        anti_se.append(se_anti)
        reduction = 100 * (1 - se_anti / se_plain)
        print(f"  n_paths={n:>6d}  plain SE={se_plain:.4f}  antithetic SE={se_anti:.4f}  reduction={reduction:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(path_counts, plain_se, "o-", color="#C44E52", label="Plain Monte Carlo")
    ax.plot(path_counts, anti_se, "s-", color="#55A868", label="Antithetic variates")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Standard error of price estimate (log scale)")
    ax.set_title("Variance reduction: antithetic variates vs. plain Monte Carlo\n(European call)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "variance_reduction.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# 6. Path-dependent option: arithmetic-average Asian call vs. vanilla
#    European call, and Asian price convergence with n_paths.
# =========================================================================
def plot_asian_option():
    path_counts = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    asian_prices, asian_se = [], []
    for n in path_counts:
        p, se = asian_option_monte_carlo(S, K, T, r, sigma, n_paths=n, n_steps=50, antithetic=True, seed=1)
        asian_prices.append(p)
        asian_se.append(se)
        print(f"  n_paths={n:>6d}  Asian call price={p:.4f}  se={se:.4f}")

    final_asian_price = asian_prices[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].errorbar(path_counts, asian_prices, yerr=asian_se, fmt="o-", color="#8172B2",
                      capsize=3, label="Asian call (arithmetic average)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of simulated paths (log scale)")
    axes[0].set_ylabel("Option price ($)")
    axes[0].set_title("Asian option price convergence\n(error bars = Monte Carlo standard error)")
    axes[0].legend()

    labels = ["European call\n(Black-Scholes)", "Asian call\n(arithmetic average, MC)"]
    values = [bs_price, final_asian_price]
    axes[1].bar(labels, values, color=["#4C72B0", "#8172B2"], edgecolor="black", linewidth=0.6)
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.1, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Option price ($)")
    axes[1].set_title("Asian vs. European call\n(same S, K, T, r, sigma)")

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "asian_option.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    return final_asian_price


if __name__ == "__main__":
    comparison = plot_model_comparison()
    plot_lsm_convergence()
    plot_step_convergence()
    plot_greeks()
    plot_variance_reduction()
    plot_asian_option()
    print("\nAll plots written to:", RESULTS_DIR)
