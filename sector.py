"""Sector-relative strength tags — context for theme detection.

For each ticker mapped to a sector ETF, compute:
  - sector_rs_60: sector ETF's 60-day return minus SPY's 60-day return
  - sector_score: 1-99 ranking of sector ETFs by composite RS

A name in a leading sector deserves more conviction than the same name in a
lagging sector — even with identical individual setups.
"""

from __future__ import annotations
import pandas as pd

from universe import SECTOR_ETFS, TICKER_TO_SECTOR


def compute_sector_strength(raw: dict[str, pd.DataFrame],
                            broad_benchmark: str = "QQQ") -> pd.DataFrame:
    """Returns DataFrame indexed by date, columns = sector ETFs, values =
    (sector_60d_ret - benchmark_60d_ret) * 100. Larger = leading sector."""
    if broad_benchmark not in raw:
        raise RuntimeError(f"Broad benchmark {broad_benchmark} missing")
    bench_close = raw[broad_benchmark]["close"]
    bench_ret_60 = bench_close.pct_change(60)

    cols = {}
    for etf in SECTOR_ETFS:
        if etf not in raw:
            continue
        etf_close = raw[etf]["close"]
        etf_ret_60 = etf_close.pct_change(60).reindex(bench_close.index).ffill()
        cols[etf] = (etf_ret_60 - bench_ret_60.reindex(bench_close.index)) * 100

    return pd.DataFrame(cols).reindex(bench_close.index)


def sector_for_ticker(ticker: str) -> str | None:
    return TICKER_TO_SECTOR.get(ticker)


def sector_strength_label(rs_excess_pct: float | None) -> str:
    """Categorical: leading / inline / lagging."""
    if rs_excess_pct is None or pd.isna(rs_excess_pct):
        return "unknown"
    if rs_excess_pct >= 10:
        return "leading"
    if rs_excess_pct >= 0:
        return "inline (above market)"
    if rs_excess_pct >= -10:
        return "inline (below market)"
    return "lagging"
