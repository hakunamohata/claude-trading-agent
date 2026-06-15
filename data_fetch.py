"""Fetch and cache daily OHLCV via yfinance.

Cache strategy: one parquet per ticker under data/. Re-fetch if the file is
missing or older than one trading day. `auto_adjust=True` so splits don't fake
breakouts.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_PERIOD = "2y"
STALE_AFTER_HOURS = 18  # re-fetch if cache older than this


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=STALE_AFTER_HOURS)


def fetch_one(ticker: str, period: str = DEFAULT_PERIOD, force: bool = False) -> pd.DataFrame:
    path = _cache_path(ticker)
    if not force and _is_fresh(path):
        return pd.read_parquet(path)

    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    # Normalize column names to lowercase, keep tz-naive index
    df.index = df.index.tz_localize(None)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.to_parquet(path)
    return df


def fetch_many(tickers: list[str], period: str = DEFAULT_PERIOD, force: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        try:
            out[t] = fetch_one(t, period=period, force=force)
        except Exception as e:
            print(f"  ! {t}: {e}")
    return out


if __name__ == "__main__":
    from universe import ALL_TICKERS
    data = fetch_many(ALL_TICKERS)
    for t, df in data.items():
        print(f"{t:6s} rows={len(df):4d}  {df.index[0].date()} -> {df.index[-1].date()}")
