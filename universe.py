"""Universe definition — small sample list used by back-tests + signal regression.

GENERIC FRAMEWORK CODE. User-specific watchlist / portfolio tickers / sector
overrides live in `user_config.py`.

For the scanner's wider universe (~540 names from S&P 500 + Nasdaq 100), see
`wide_universe.py`.
"""

from __future__ import annotations

try:
    import user_config
    _user_watchlist = list(user_config.WATCHLIST)
    _user_portfolio = list(user_config.PERSONAL_PORTFOLIO_TICKERS)
    _user_sector_overrides = dict(user_config.TICKER_TO_SECTOR_OVERRIDES)
except ImportError:
    _user_watchlist = []
    _user_portfolio = []
    _user_sector_overrides = {}


# Names actively watched for breakout signals — used by the regression test.
# Combines a tiny default set (for first-time users) with the user's watchlist.
TARGETS = sorted(set(["MU", "SNDK", "NBIS", "ALAB"] + _user_watchlist))


# Small Nasdaq starter set — used by the small-universe scanner (`scan.py`),
# back-test (`backtest.py`), and dashboard's default-load list. The wider
# scanner pulls from `wide_universe.py`.
NASDAQ_SAMPLE = [
    "NVDA", "AVGO", "AMD", "MRVL",         # semis
    "META", "GOOGL", "MSFT", "AMZN",       # mega-cap tech
    "AAPL", "NFLX", "TSLA",                # other mega-cap
    "PEP", "COST",                          # boring consumer
]


# User's personally-held names that aren't in the major indices.
PORTFOLIO_HOLDINGS = list(_user_portfolio)


BENCHMARK = "QQQ"

# Sector ETFs — fetched alongside the universe so we can compute per-name
# sector-relative strength.
SECTOR_ETFS = ["SOXX", "XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLE", "XLI", "XLU", "XLB", "XLRE"]

# Human-readable sector names for dashboard / output displays
SECTOR_ETF_TO_NAME = {
    "SOXX": "Semiconductors",
    "XLK":  "Technology",
    "XLC":  "Communications",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLF":  "Financials",
    "XLV":  "Healthcare",
    "XLI":  "Industrials",
    "XLE":  "Energy",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
}


def sector_name(etf_ticker: str | None) -> str | None:
    """Return the human-readable sector name for a given sector-ETF ticker."""
    if etf_ticker is None:
        return None
    return SECTOR_ETF_TO_NAME.get(etf_ticker, etf_ticker)


# Sector mapping. Standard S&P 500 / Nasdaq 100 names get their sector from
# the framework defaults; user-specific names get tagged via overrides in
# user_config.py.
_DEFAULT_TICKER_TO_SECTOR = {
    # Semis / AI infra
    "NVDA": "SOXX", "AMD": "SOXX", "MU": "SOXX", "MRVL": "SOXX",
    "AVGO": "SOXX", "SNDK": "SOXX", "ALAB": "SOXX", "NBIS": "SOXX",
    "INTC": "SOXX", "TSM": "SOXX", "ASML": "SOXX", "ARM": "SOXX",
    "QCOM": "SOXX", "LRCX": "SOXX", "AMAT": "SOXX", "KLAC": "SOXX",
    "ON": "SOXX", "MCHP": "SOXX", "ADI": "SOXX", "TXN": "SOXX",
    # Mega-cap tech / software
    "AAPL": "XLK", "MSFT": "XLK",
    # Communication services
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC",
    # Consumer discretionary
    "AMZN": "XLY", "TSLA": "XLY",
    # Consumer staples
    "COST": "XLP", "PEP": "XLP",
}

# Composite: defaults + user overrides (user wins on conflict)
TICKER_TO_SECTOR = {**_DEFAULT_TICKER_TO_SECTOR, **_user_sector_overrides}


ALL_TICKERS = sorted(set(TARGETS + NASDAQ_SAMPLE + PORTFOLIO_HOLDINGS + [BENCHMARK] + SECTOR_ETFS))
