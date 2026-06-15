"""User-specific configuration template — copy to user_config.py and fill in.

This file is the template anyone forking the project can use. The real
`user_config.py` is gitignored and holds your personal financial data.

Setup:
  cp user_config.example.py user_config.py
  # edit user_config.py with your own holdings, account IDs, and rules
"""

from __future__ import annotations


# ============================================================
# ACCOUNTS — your broker account IDs and what to call them
# ============================================================
# tax_status: one of "tax-free", "tax-deferred", "tax-advantaged", "taxable"

ACCOUNT_INFO = {
    # "<your_account_id>": {"label": "401k", "tax_status": "tax-deferred"},
    # "<your_roth_id>":    {"label": "Roth IRA", "tax_status": "tax-free"},
}


# ============================================================
# RULES — your portfolio constraints
# ============================================================

# Accounts where the agent may recommend trades. Typically tax-advantaged
# accounts (Roth, 401k, HSA) since trades have no tax cost there. Add taxable
# accounts if you accept the cap-gains/loss tradeoff.
TRADE_ELIGIBLE_ACCOUNTS: set[str] = set()

# Accounts that are taxable — agent surfaces cap-gains/loss impact on sells.
TAXABLE_ACCOUNTS: set[str] = set()

# Tickers the agent must never recommend selling. Common cases:
#   - Employer stock you've decided to hold (RSU concentration)
#   - Sentimental positions
#   - Tax-locked positions (large embedded LT gains)
LOCKED_POSITIONS: set[str] = set()

# Max position size as % of total household value. LOCKED_POSITIONS are exempt.
MAX_POSITION_PCT = 10.0


# ============================================================
# MARGIN STATUS — set to None if no margin account
# ============================================================
# If you don't use margin, leave this as None and the dashboard skips the
# margin-status panel.

INDIVIDUAL_MARGIN_SNAPSHOT: dict | None = None
# Example shape:
# INDIVIDUAL_MARGIN_SNAPSHOT = {
#     "as_of": "2026-06-14T16:48:00-04:00",
#     "account_id": "<your_margin_account>",
#     "account_equity": 400_000.00,           # net equity after margin debt
#     "equity_pct": 60.0,                     # equity / gross holdings %
#     "margin_buying_power": 300_000.00,
#     "non_margin_buying_power": 100_000.00,
#     "settled_cash": 0.0,
#     "available_without_margin_impact": 0.0,
#     "reserved_for_options": 0.0,
#     "cash_market_value": 20_000.00,
#     "margin_market_value": 600_000.00,      # gross stock value
#     "option_market_value": 0.0,             # short option position, negative if owed
#     "net_debit": -250_000.00,               # margin debt (negative)
#     "margin_interest_accrued_month": 1_000.00,
#     "margin_interest_accrued_daily": 35.00,
#     "margin_interest_rate_pct": 5.0,
# }


# ============================================================
# HOLDINGS — your positions
# Format: (account_id, ticker, quantity, price, value, cost_basis_or_None)
# ============================================================
# Use HOLDINGS_CURRENT for live snapshots (no reconciliation needed).
# Use HOLDINGS_MAY31 for past statement positions to be reconciled with a
# transaction CSV (see portfolio.py reconcile() logic).

HOLDINGS_CURRENT: list[tuple] = [
    # Example:
    # ("<account_id>", "AAPL", 50.0, 180.00, 9000.00, 5500.00),
]

HOLDINGS_MAY31: list[tuple] = []


# ============================================================
# UNIVERSE EXTENSIONS — names you personally care about
# ============================================================

# Tickers you actively watch for breakout signals.
WATCHLIST: list[str] = []

# Additional names beyond S&P 500 / Nasdaq 100 to include in the scanner.
PERSONAL_PORTFOLIO_TICKERS: list[str] = []

# Sector ETF assignments for tickers not in the standard sector map.
# Sectors: SOXX (semis), XLK (tech), XLC (communications), XLY (consumer disc),
# XLP (staples), XLF (financials), XLV (healthcare), XLI (industrials),
# XLE (energy), XLU (utilities), XLB (materials), XLRE (real estate)
TICKER_TO_SECTOR_OVERRIDES: dict[str, str] = {
    # "MYTICKER": "XLK",
}
