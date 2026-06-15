"""Options chain fetch + Black-Scholes Greeks + position pricing.

Provides three layers:
  1. fetch_options_chain(ticker, expiry) — pull live chain from yfinance
  2. compute_greeks() — Black-Scholes pricing + delta/gamma/theta/vega
  3. price_user_options() — read OPTIONS_POSITIONS from user_config, return
     current marks, Greeks, and P&L for each open contract

All Greeks are computed locally because Yahoo Finance doesn't expose them.
Theta is returned per CALENDAR DAY (most useful for tracking decay).
"""

from __future__ import annotations
from datetime import datetime, date
from pathlib import Path
import math
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from data_fetch import DATA_DIR

OPTIONS_CACHE_DIR = DATA_DIR / "options"
OPTIONS_CACHE_DIR.mkdir(exist_ok=True)

# Risk-free rate proxy — could pull live from FRED, this is a fine approximation
RISK_FREE_RATE = 0.045


# ============================================================
# Black-Scholes Greeks
# ============================================================

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Standard Black-Scholes d1 / d2."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, days_to_expiry: int, sigma: float,
             opt_type: str = "call", r: float = RISK_FREE_RATE) -> float:
    """Black-Scholes theoretical price."""
    if days_to_expiry <= 0:
        # At expiry: payoff = max(S-K, 0) for call, max(K-S, 0) for put
        return max(S - K, 0.0) if opt_type == "call" else max(K - S, 0.0)
    T = days_to_expiry / 365.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, days_to_expiry: int, sigma: float,
              opt_type: str = "call", r: float = RISK_FREE_RATE) -> dict:
    """Returns delta, gamma, theta (per day), vega (per 1% IV)."""
    if days_to_expiry <= 0 or sigma <= 0:
        # No time value left
        if opt_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    T = days_to_expiry / 365.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T) * 0.01  # per 1% IV move

    if opt_type == "call":
        delta = norm.cdf(d1)
        theta_annual = (-S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta_annual = (-S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_annual / 365.0,   # per calendar day
        "vega": vega,
    }


# ============================================================
# Chain fetching (yfinance)
# ============================================================

def fetch_expiries(ticker: str) -> list[str]:
    """Return all available expiration dates for a ticker."""
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


def fetch_options_chain(ticker: str, expiry: str | None = None) -> dict:
    """Fetch the calls+puts chain for one expiry. If expiry is None, uses nearest.

    Returns dict with: spot, expiry, calls (DataFrame), puts (DataFrame).
    """
    t = yf.Ticker(ticker)
    expiries = list(t.options)
    if not expiries:
        return {"spot": None, "expiry": None, "calls": None, "puts": None}
    if expiry is None:
        expiry = expiries[0]
    elif expiry not in expiries:
        # Find closest available expiry
        target = pd.Timestamp(expiry)
        deltas = [(abs((pd.Timestamp(e) - target).days), e) for e in expiries]
        expiry = min(deltas)[1]

    chain = t.option_chain(expiry)
    info = t.history(period="1d")
    spot = float(info["Close"].iloc[-1]) if not info.empty else None

    return {
        "ticker": ticker,
        "spot": spot,
        "expiry": expiry,
        "calls": chain.calls,
        "puts": chain.puts,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def lookup_contract(chain: dict, strike: float, opt_type: str) -> dict | None:
    """Find a specific strike's row in a chain dict."""
    if chain.get("calls") is None:
        return None
    df = chain["calls"] if opt_type == "call" else chain["puts"]
    match = df[df["strike"] == strike]
    if match.empty:
        # Try nearest
        match = df.iloc[(df["strike"] - strike).abs().argsort()[:1]]
        if match.empty:
            return None
    row = match.iloc[0]
    return {
        "strike": float(row["strike"]),
        "bid": float(row.get("bid", 0)),
        "ask": float(row.get("ask", 0)),
        "last": float(row.get("lastPrice", 0)),
        "iv": float(row.get("impliedVolatility", 0)),
        "volume": int(row.get("volume", 0) or 0),
        "open_interest": int(row.get("openInterest", 0) or 0),
    }


# ============================================================
# Price user's open positions
# ============================================================

def _shares_held_across_accounts(ticker: str) -> float:
    """Sum of shares held across all user accounts (for covered/naked detection)."""
    try:
        import user_config
        return sum(
            qty for (acct, t, qty, *_) in user_config.HOLDINGS_CURRENT
            if t == ticker
        ) + sum(
            qty for (acct, t, qty, *_) in user_config.HOLDINGS_MAY31
            if t == ticker
        )
    except Exception:
        return 0.0


def price_user_options() -> pd.DataFrame:
    """Read OPTIONS_POSITIONS from user_config, return current state per position.

    OPTIONS_POSITIONS schema (each entry is a tuple):
      (account_id, underlying, strike, expiry_YYYY-MM-DD, opt_type, contracts, premium_received_per_contract)

      contracts: positive = long, negative = short (covered/naked call seller)
    """
    try:
        import user_config
        positions = getattr(user_config, "OPTIONS_POSITIONS", [])
    except (ImportError, AttributeError):
        positions = []

    if not positions:
        return pd.DataFrame()

    # Compute total contracts written per ticker (for covered/naked detection)
    contracts_by_ticker: dict[str, int] = {}
    for pos in positions:
        if len(pos) == 7 and pos[5] < 0:  # short call/put
            contracts_by_ticker[pos[1]] = contracts_by_ticker.get(pos[1], 0) + abs(pos[5])

    today = pd.Timestamp(date.today())
    rows = []

    for pos in positions:
        if len(pos) == 7:
            acct, ticker, strike, expiry, opt_type, contracts, premium_per_contract = pos
        else:
            continue

        expiry_ts = pd.Timestamp(expiry)
        dte = max(0, (expiry_ts - today).days)

        # Pull live chain for this expiry
        chain = fetch_options_chain(ticker, str(expiry))
        spot = chain["spot"]
        contract = lookup_contract(chain, strike, opt_type) if spot else None

        if contract is None or spot is None:
            rows.append({
                "account": acct, "ticker": ticker, "strike": strike,
                "expiry": expiry, "type": opt_type, "contracts": contracts,
                "error": "no_data",
            })
            continue

        # Use mid of bid/ask if available, else last
        if contract["bid"] > 0 and contract["ask"] > 0:
            current_mid = (contract["bid"] + contract["ask"]) / 2
        else:
            current_mid = contract["last"]

        iv = contract["iv"] or 0.30  # default to 30% if missing
        greeks = bs_greeks(spot, strike, dte, iv, opt_type)

        # P&L: for short positions, P&L = (premium received - current cost to close) * 100 * |contracts|
        # contracts is negative for short, so we use abs() and adjust sign
        is_short = contracts < 0
        n_abs = abs(contracts)
        if is_short:
            pnl_per_contract = (premium_per_contract - current_mid) * 100
        else:
            pnl_per_contract = (current_mid - premium_per_contract) * 100
        total_pnl = pnl_per_contract * n_abs

        # Covered/naked: for short calls, check if shares held >= contracts × 100
        coverage = "n/a"
        if is_short and opt_type == "call":
            shares_held = _shares_held_across_accounts(ticker)
            total_contracts_written = contracts_by_ticker.get(ticker, 0)
            shares_needed = total_contracts_written * 100
            if shares_held >= shares_needed:
                coverage = "COVERED"
            else:
                shortage = shares_needed - shares_held
                coverage = f"NAKED ({int(shortage)} sh short)"

        rows.append({
            "account": acct,
            "ticker": ticker,
            "strike": strike,
            "expiry": expiry,
            "dte": dte,
            "type": opt_type.upper(),
            "side": "SHORT" if is_short else "LONG",
            "coverage": coverage,
            "contracts": int(n_abs),
            "spot": round(spot, 2),
            "moneyness_pct": round((spot / strike - 1) * 100, 1),
            "current_mid": round(current_mid, 2),
            "iv": round(iv * 100, 1),
            "premium_per_contract": round(premium_per_contract, 2),
            "total_pnl_usd": round(total_pnl, 0),
            "pnl_pct_of_premium": round(pnl_per_contract / premium_per_contract * 100, 0)
                if premium_per_contract else None,
            "delta": round(greeks["delta"], 3),
            "gamma": round(greeks["gamma"], 4),
            "theta_per_day": round(greeks["theta"] * 100 * n_abs, 0),  # $ per day total
            "vega": round(greeks["vega"] * 100 * n_abs, 0),            # $ per 1% IV move
        })

    return pd.DataFrame(rows)


# ============================================================
# CLI: smoke test
# ============================================================

if __name__ == "__main__":
    print("Testing Black-Scholes Greeks...")
    g = bs_greeks(S=100, K=100, days_to_expiry=30, sigma=0.30, opt_type="call")
    print(f"  ATM call, 30d, 30% IV: {g}")
    print(f"  Price: ${bs_price(100, 100, 30, 0.30, 'call'):.2f}")
    print()

    print("Pricing your open options positions...")
    df = price_user_options()
    if df.empty:
        print("  (No OPTIONS_POSITIONS defined in user_config.py)")
    else:
        print(df.to_string(index=False))
        print()
        total_pnl = df["total_pnl_usd"].sum()
        total_theta = df["theta_per_day"].sum()
        print(f"Total open P&L: ${total_pnl:,.0f}")
        print(f"Total theta (daily decay you collect on shorts): ${-total_theta:,.0f}/day"
              if total_theta < 0 else f"Total theta: ${total_theta:,.0f}/day")
