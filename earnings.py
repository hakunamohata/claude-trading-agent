"""Earnings proximity tag — context for the judgment layer.

Pulls next earnings date per ticker via yfinance. Cached to a parquet so the
slow per-ticker .calendar lookups only happen weekly.

NOTE: yfinance earnings data is best-effort — some tickers return empty
calendars. We surface "unknown" rather than blocking the pipeline.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from data_fetch import DATA_DIR


EARNINGS_CACHE = DATA_DIR / "earnings.parquet"
STALE_AFTER_DAYS = 3  # earnings dates don't change daily — weekly refresh is fine


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=STALE_AFTER_DAYS)


def _fetch_one(ticker: str) -> dict:
    """Best-effort fetch of next earnings date for a single ticker.

    Per user feedback: earnings are quarterly (every 2.5-3.5 months). If the
    confirmed next date is unavailable, estimate it as last reported + 90 days.
    """
    try:
        t = yf.Ticker(ticker)
        # 1) Try the confirmed next date from calendar
        cal = t.calendar
        next_date = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                next_date = ed[0]
            elif ed is not None:
                next_date = ed
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.columns:
                next_date = cal.iloc[0]["Earnings Date"]
        if next_date is not None:
            return {"ticker": ticker, "next_earnings": pd.Timestamp(next_date), "source": "confirmed"}

        # 2) Fall back to last reported + 90d estimate
        try:
            hist = t.earnings_dates
            if hist is not None and not hist.empty:
                # earnings_dates includes both past and projected; filter to past
                past = hist[hist.index < pd.Timestamp.now(tz=hist.index.tz)]
                if not past.empty:
                    last = past.index.max()
                    est = pd.Timestamp(last).tz_localize(None) + pd.Timedelta(days=90)
                    return {"ticker": ticker, "next_earnings": est, "source": "estimated_+90d"}
        except Exception:
            pass

        return {"ticker": ticker, "next_earnings": None, "source": "unknown"}
    except Exception:
        return {"ticker": ticker, "next_earnings": None, "source": "unknown"}


def build_earnings_cache(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """Build/refresh per-ticker next-earnings-date cache."""
    if not force and _is_fresh(EARNINGS_CACHE):
        return pd.read_parquet(EARNINGS_CACHE)

    rows = [_fetch_one(t) for t in tickers]
    df = pd.DataFrame(rows).set_index("ticker")
    df.to_parquet(EARNINGS_CACHE)
    return df


def days_to_earnings(earnings_df: pd.DataFrame, ticker: str,
                     as_of: pd.Timestamp | None = None) -> int | None:
    """Days until next earnings for a ticker. None if unknown."""
    if ticker not in earnings_df.index:
        return None
    next_date = earnings_df.loc[ticker, "next_earnings"]
    if pd.isna(next_date):
        return None
    as_of = as_of or pd.Timestamp.now().normalize()
    return int((pd.Timestamp(next_date).normalize() - as_of).days)


def earnings_proximity_label(days: int | None) -> str:
    """Categorical label for display: imminent / soon / far / not-imminent.

    Per user feedback: earnings come quarterly (every 2.5-3.5 months). When the
    next date is unknown, treat it as 'not-imminent' — never as a risk.
    """
    if days is None:
        return "not-imminent (quarterly cycle, estimate far)"
    if days < 0:
        return "passed (recent)"
    if days <= 7:
        return "imminent (<=7d)"
    if days <= 21:
        return "soon (8-21d)"
    return f"far ({days}d)"


if __name__ == "__main__":
    from universe import ALL_TICKERS
    df = build_earnings_cache(ALL_TICKERS, force=True)
    print(df.to_string())
