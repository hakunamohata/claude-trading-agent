"""Backtest the XTREND signal over held tickers.

For each ticker, finds every bar where XTREND fires (long-term > 0 AND
T3 of short-term flipped up). Measures forward returns at 5d / 10d / 20d
/ 40d horizons. Per-ticker stats: win rate, mean return, max drawdown.
Aggregates across the held basket. Writes markdown to
data/snapshots/<today>/xtrender_backtest.md.

Skipped: cash sleeves, mutual funds, and ETFs we don't trade actively.

CLI:
    python xtrender_backtest.py                  # all held equities, 3y
    python xtrender_backtest.py --years 5        # longer window
    python xtrender_backtest.py --tickers MSFT,NVDA,MU
    python xtrender_backtest.py --hold-bars 20   # only the 20d horizon
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# UTF-8 console on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from data_fetch import fetch_one, DATA_DIR
from xtrender import compute_xtrender
from user_config import HOLDINGS_CURRENT


# Cash sleeves + mutual fund / non-tradable equivalents — skip these
NON_EQUITY = {
    "CASH_TOD", "CASH_ROTH", "CASH_HSA", "CASH_ROLLOVER",
    "FDRXX", "SPAXX", "NHFSMKX98", "SMID_CAP_GROWTH", "STABLE_VALUE_ACCT",
    "VANG_500_INDEX", "BTC_LPATH_2040", "FID_GR_CO_POOL_S", "DRAM",
}

DEFAULT_HORIZONS = [5, 10, 20, 40]


# ============================================================
# Per-ticker backtest
# ============================================================

def backtest_ticker(ticker: str, years: int, horizons: list[int]) -> dict:
    """Pull `years` of history; mark every XTREND fire; measure forward returns.

    Returns a dict with per-horizon stats and the list of entry rows.
    """
    period = f"{years}y"
    try:
        df = fetch_one(ticker, period=period)
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

    if df is None or df.empty or len(df) < 100:
        return {"ticker": ticker, "error": f"insufficient history ({len(df) if df is not None else 0} bars)"}

    close = df["close"]
    x = compute_xtrender(close)
    fires_mask = x["xtr_bull_regime"] & x["xtr_bull_flip"]

    entries = []
    for d in close.index[fires_mask.fillna(False)]:
        entry_price = float(close.loc[d])
        row = {"date": d, "entry_price": entry_price}
        # Compute each horizon's forward return
        for h in horizons:
            future_idx = close.index.get_loc(d) + h
            if future_idx < len(close):
                exit_price = float(close.iloc[future_idx])
                row[f"ret_{h}d"] = (exit_price / entry_price - 1) * 100
                # Best / worst on the path
                window = close.iloc[close.index.get_loc(d): future_idx + 1]
                row[f"max_{h}d"] = (window.max() / entry_price - 1) * 100
                row[f"min_{h}d"] = (window.min() / entry_price - 1) * 100
            else:
                row[f"ret_{h}d"] = np.nan
                row[f"max_{h}d"] = np.nan
                row[f"min_{h}d"] = np.nan
        entries.append(row)

    if not entries:
        return {"ticker": ticker, "n_signals": 0, "entries": [], "horizons": {}}

    entries_df = pd.DataFrame(entries)
    horizon_stats = {}
    for h in horizons:
        col = f"ret_{h}d"
        valid = entries_df[col].dropna()
        if valid.empty:
            horizon_stats[h] = {"n": 0}
            continue
        max_col = entries_df[f"max_{h}d"].dropna()
        min_col = entries_df[f"min_{h}d"].dropna()
        horizon_stats[h] = {
            "n": len(valid),
            "win_rate_pct": round((valid > 0).mean() * 100, 1),
            "mean_pct": round(valid.mean(), 2),
            "median_pct": round(valid.median(), 2),
            "best_pct": round(valid.max(), 2),
            "worst_pct": round(valid.min(), 2),
            "mean_max_pct": round(max_col.mean(), 2),
            "mean_min_pct": round(min_col.mean(), 2),
        }

    return {
        "ticker": ticker,
        "n_signals": len(entries),
        "first_date": entries_df["date"].min().date(),
        "last_date": entries_df["date"].max().date(),
        "horizons": horizon_stats,
        "entries": entries,
    }


# ============================================================
# Aggregation
# ============================================================

def aggregate(results: list[dict], horizons: list[int]) -> dict:
    """Cross-ticker aggregate stats."""
    all_returns = {h: [] for h in horizons}
    for r in results:
        if "error" in r or r.get("n_signals", 0) == 0:
            continue
        for entry in r["entries"]:
            for h in horizons:
                v = entry.get(f"ret_{h}d")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    all_returns[h].append(v)

    agg = {}
    for h in horizons:
        rets = all_returns[h]
        if not rets:
            agg[h] = {"n": 0}
            continue
        rets = pd.Series(rets)
        agg[h] = {
            "n": len(rets),
            "win_rate_pct": round((rets > 0).mean() * 100, 1),
            "mean_pct": round(rets.mean(), 2),
            "median_pct": round(rets.median(), 2),
            "best_pct": round(rets.max(), 2),
            "worst_pct": round(rets.min(), 2),
        }
    return agg


# ============================================================
# Render markdown
# ============================================================

def render_markdown(results: list[dict], agg: dict, args) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    horizons = args.horizons
    lines = []
    lines.append(f"# XTREND backtest — {today}")
    lines.append("")
    lines.append(f"**Universe**: {len([r for r in results if 'error' not in r])} held equities")
    lines.append(f"**Window**: {args.years} years of daily bars")
    lines.append(f"**Horizons**: {', '.join(f'{h}d' for h in horizons)} forward returns")
    lines.append(f"**Signal**: XTREND mode = long-term Xtrender > 0 AND T3 of short-term Xtrender flipped UP (lime circle)")
    lines.append("")

    # Aggregate header
    lines.append("## Aggregate across all held tickers")
    lines.append("")
    lines.append("| Horizon | N signals | Win rate | Mean return | Median | Best | Worst |")
    lines.append("|---|---|---|---|---|---|---|")
    for h in horizons:
        s = agg.get(h, {})
        if s.get("n", 0) == 0:
            lines.append(f"| {h}d | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {h}d | {s['n']} | {s['win_rate_pct']:.1f}% | "
            f"{s['mean_pct']:+.2f}% | {s['median_pct']:+.2f}% | "
            f"{s['best_pct']:+.2f}% | {s['worst_pct']:+.2f}% |"
        )
    lines.append("")

    # Per-ticker tables
    lines.append("## Per-ticker breakdown")
    lines.append("")
    valid = [r for r in results if "error" not in r and r.get("n_signals", 0) > 0]
    no_signals = [r["ticker"] for r in results if "error" not in r and r.get("n_signals", 0) == 0]
    errored = [(r["ticker"], r["error"]) for r in results if "error" in r]

    # Sort by 20d mean return (descending) — fall back to first horizon
    sort_h = 20 if 20 in horizons else horizons[0]
    valid.sort(key=lambda r: r["horizons"].get(sort_h, {}).get("mean_pct", -999), reverse=True)

    for r in valid:
        lines.append(f"### {r['ticker']}")
        lines.append(f"_{r['n_signals']} signals between {r['first_date']} and {r['last_date']}_")
        lines.append("")
        lines.append("| Horizon | N | Win rate | Mean return | Median | Best | Worst | Mean peak | Mean trough |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for h in horizons:
            s = r["horizons"].get(h, {})
            if s.get("n", 0) == 0:
                continue
            lines.append(
                f"| {h}d | {s['n']} | {s['win_rate_pct']:.1f}% | "
                f"{s['mean_pct']:+.2f}% | {s['median_pct']:+.2f}% | "
                f"{s['best_pct']:+.2f}% | {s['worst_pct']:+.2f}% | "
                f"{s['mean_max_pct']:+.2f}% | {s['mean_min_pct']:+.2f}% |"
            )
        lines.append("")

        # Most recent 3 entries (visual sanity check)
        lines.append("Most recent signals:")
        for e in r["entries"][-3:]:
            ret_strs = []
            for h in horizons:
                v = e.get(f"ret_{h}d")
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    ret_strs.append(f"{h}d=pending")
                else:
                    ret_strs.append(f"{h}d={v:+.2f}%")
            lines.append(f"- {e['date'].date()} @ ${e['entry_price']:.2f}  ({', '.join(ret_strs)})")
        lines.append("")

    if no_signals:
        lines.append(f"## No signals fired")
        lines.append(f"_{len(no_signals)} tickers: {', '.join(no_signals)}_")
        lines.append("")
    if errored:
        lines.append("## Errors")
        for t, e in errored:
            lines.append(f"- **{t}**: {e}")
        lines.append("")

    lines.append("---")
    lines.append("> Win rate = % of signals where forward return was positive at the horizon.")
    lines.append("> Mean peak / trough = mean of best / worst close on the path from entry to horizon (not at the exact horizon).")
    lines.append("> Term definitions on the dashboard's **Glossary** page.")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3, help="Years of history to backtest")
    parser.add_argument("--tickers", type=str, default="",
                        help="Comma-separated ticker override (default: all held equities)")
    parser.add_argument("--horizons", type=str, default="5,10,20,40",
                        help="Comma-separated horizon list in days")
    parser.add_argument("--hold-bars", type=int, default=None,
                        help="Convenience — restrict horizons to this single value")
    args = parser.parse_args()

    if args.hold_bars is not None:
        args.horizons = [args.hold_bars]
    else:
        args.horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = sorted({r[1] for r in HOLDINGS_CURRENT if r[1] not in NON_EQUITY})

    print(f"Backtesting XTREND on {len(tickers)} tickers, {args.years}y window, horizons {args.horizons}")
    print()

    results = []
    for t in tickers:
        print(f"  {t}...", end=" ", flush=True)
        r = backtest_ticker(t, args.years, args.horizons)
        if "error" in r:
            print(f"ERR {r['error']}")
        else:
            print(f"{r['n_signals']} signals")
        results.append(r)

    print()
    agg = aggregate(results, args.horizons)
    print(f"Aggregate stats:")
    for h in args.horizons:
        s = agg.get(h, {})
        if s.get("n", 0) > 0:
            print(f"  {h}d: n={s['n']}, win_rate={s['win_rate_pct']:.1f}%, "
                  f"mean={s['mean_pct']:+.2f}%, best={s['best_pct']:+.2f}%, worst={s['worst_pct']:+.2f}%")
        else:
            print(f"  {h}d: no signals")

    md = render_markdown(results, agg, args)
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = DATA_DIR / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "xtrender_backtest.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
