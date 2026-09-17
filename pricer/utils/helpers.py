"""
Utility functions for option pricer package.
Provides financial data helpers such as risk-free rate retrieval.
"""

import yfinance as yf

# Get the current 10 year Treasury yield as a proxy for the risk-free rate
# Returns the yield as a decimal (e.g., 0.03 for 3%)
def get_risk_free_rate():
    """
    Fetches the current 10-year US Treasury yield from Yahoo Finance.
    Returns:
        float: Risk-free rate as a decimal (e.g., 0.03 for 3%)
    Raises:
        Exception: If the Yahoo Finance API call fails or rate is unavailable.
    """
    try:
        return yf.Ticker('^TNX').info['regularMarketPrice'] / 100
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.04
