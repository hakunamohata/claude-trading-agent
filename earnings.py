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


# ---------------------------------------------------------------------------
# Pre-earnings runup stats — historical pattern of how a stock behaves
# in the N days leading into its earnings print. Useful as a risk overlay
# for covered-call writers: if a stock spans an earnings runup window, the
# strike is more likely to be tested.
# ---------------------------------------------------------------------------

RUNUP_CACHE = DATA_DIR / "earnings_runup.parquet"
RUNUP_WINDOWS_DEFAULT = (7, 14, 21, 30)


def historical_runup_stats(
    ticker: str,
    windows: tuple[int, ...] = RUNUP_WINDOWS_DEFAULT,
    n_earnings: int = 4,
) -> dict:
    """Per-window pre-earnings runup stats over the last N earnings cycles.

    For each (window_days, earnings_event), measures the percentage change from
    `window_days` calendar days before the earnings date through the earnings
    day's close. Returns avg, win-rate (% positive), and avg peak (max high in
    window) per window.

    Falls back to empty dict on any data error — caller decides display.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_dates
        if hist is None or hist.empty:
            return {}
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        past = hist[hist["Reported EPS"].notna()].sort_index(ascending=False)
        if past.empty:
            return {}
        last_n = past.index[:n_earnings]

        ohlcv = t.history(period="2y", interval="1d")
        if ohlcv.empty:
            return {}
        ohlcv.index = pd.to_datetime(ohlcv.index).tz_localize(None).normalize()
        closes = ohlcv["Close"]
        highs = ohlcv["High"]

        out: dict[str, list[float]] = {f"runup_{w}d": [] for w in windows}
        out.update({f"peak_{w}d": [] for w in windows})
        for er in last_n:
            if er not in closes.index:
                prior = closes.index[closes.index < er]
                if len(prior) == 0:
                    continue
                er_close_idx = prior[-1]
            else:
                er_close_idx = er
            er_close = float(closes.loc[er_close_idx])
            for win in windows:
                start_target = er - pd.Timedelta(days=win)
                prior_idx = closes.index[closes.index <= start_target]
                if len(prior_idx) == 0:
                    continue
                start_close = float(closes.loc[prior_idx[-1]])
                runup_pct = (er_close - start_close) / start_close * 100
                window_highs = highs.loc[(highs.index >= prior_idx[-1]) & (highs.index <= er_close_idx)]
                peak_pct = (float(window_highs.max()) - start_close) / start_close * 100 if not window_highs.empty else None
                out[f"runup_{win}d"].append(runup_pct)
                if peak_pct is not None:
                    out[f"peak_{win}d"].append(peak_pct)

        summary = {"ticker": ticker, "n_events": len(last_n)}
        for win in windows:
            runups = out[f"runup_{win}d"]
            peaks = out[f"peak_{win}d"]
            if not runups:
                continue
            summary[f"avg_runup_{win}d"] = sum(runups) / len(runups)
            summary[f"pct_positive_{win}d"] = sum(1 for r in runups if r > 0) / len(runups) * 100
            summary[f"avg_peak_{win}d"] = (sum(peaks) / len(peaks)) if peaks else None
        return summary
    except Exception:
        return {}


def build_runup_cache(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """Build/refresh per-ticker historical runup cache. Weekly is plenty."""
    if not force and _is_fresh(RUNUP_CACHE):
        try:
            return pd.read_parquet(RUNUP_CACHE)
        except Exception:
            pass
    rows = []
    for t in tickers:
        stats = historical_runup_stats(t)
        if stats:
            rows.append(stats)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ticker")
    df.to_parquet(RUNUP_CACHE)
    return df


def runup_risk_pct(
    runup_df: pd.DataFrame,
    ticker: str,
    days_to_earnings: int | None,
    expiry_dte: int,
) -> tuple[float | None, str]:
    """Risk metric: how much does the option's lifetime overlap with the
    historical pre-earnings runup window?

    Returns (peak_runup_pct, label). peak_runup_pct is the average historical
    peak runup magnitude over the most relevant window if the option spans
    that window, else None.

    Window selection: matches the option's DTE to the historical window.
    If the option expires AFTER earnings (spans the event), uses the longest
    window. If it expires N days BEFORE earnings, only the windows that overlap
    [earnings-N, earnings] count.
    """
    if days_to_earnings is None or ticker not in runup_df.index:
        return None, "no-data"
    row = runup_df.loc[ticker]
    # Option spans earnings → use the longest available window
    if expiry_dte > days_to_earnings:
        win = 30 if not pd.isna(row.get("avg_peak_30d")) else (
              21 if not pd.isna(row.get("avg_peak_21d")) else 14)
        peak = row.get(f"avg_peak_{win}d")
        pos = row.get(f"pct_positive_{win}d")
        return (float(peak) if pd.notna(peak) else None,
                f"spans-er ({win}d hist peak +{peak:.1f}% / {pos:.0f}% pos)" if pd.notna(peak) else "spans-er")
    # Option expires before earnings — how much of the runup window does it cover?
    # If option expires N days before earnings and the relevant window is W days
    # before earnings, the overlap is from (earnings - W) to expiry, i.e. W - N days.
    days_before_er_at_expiry = days_to_earnings - expiry_dte
    if days_before_er_at_expiry >= 30:
        return None, "well-before"
    # Find the smallest window whose start is BEFORE expiry → expiry overlaps that window
    for win in (7, 14, 21, 30):
        if win > days_before_er_at_expiry:
            peak = row.get(f"avg_peak_{win}d")
            pos = row.get(f"pct_positive_{win}d")
            if pd.isna(peak):
                continue
            return (float(peak),
                    f"in-{win}d window (+{peak:.1f}% avg peak / {pos:.0f}% pos)")
    return None, "well-before"


if __name__ == "__main__":
    from universe import ALL_TICKERS
    df = build_earnings_cache(ALL_TICKERS, force=True)
    print(df.to_string())
