"""Pre-breakout scanner — daily ranked watchlist of names setting up.

Runs the breakout filter + new pre-breakout signals across the wide universe
(~500 names) and outputs a ranked top 30 watchlist.

Scoring dimensions (each 0-1, then weighted sum 0-100):
  1. Setup quality       — clean base, tight ATR, near 52w high
  2. RS strength         — IBD rank percentile + RS line trajectory
  3. Trend regime        — Stage 2 confirmation, EMA slope health
  4. Pre-breakout proximity — how close to triggering a real breakout signal
  5. Sector tailwind     — sector ETF relative strength
  6. Volume signature    — accumulation pattern (up-day vs down-day volume)
  7. Catalyst potential  — earnings within optimal 2-6 week window

The output is the input to the multi-agent judgment layer (Phase 2). We do NOT
call Claude here — Claude runs on the top 30 only to keep cost bounded.

CLI:
    python scanner.py                  # daily run, write top 30 to snapshot
    python scanner.py --top 50         # custom top N
    python scanner.py --no-fetch       # skip data refresh
"""

from __future__ import annotations
import sys
from datetime import datetime
import pandas as pd
import numpy as np

from data_fetch import fetch_many
from breakout import (
    build_features, any_breakout_signal, signal_components,
    compute_universe_rs_rank,
)
from wide_universe import build_wide_universe
from snapshot import save_df, write_manifest

BENCHMARK = "QQQ"


# ---------- Scoring components ----------

def _score_clean_base(feat_row, feat_series_close) -> float:
    """How tight and constructive is the base?
    Combines: ATR% in low quartile, close near 52w high, low volatility."""
    score = 0.0
    # ATR% in bottom 30% → tight base
    atr_pct = feat_row.get("atr_pct_prior")
    atr_q35 = feat_row.get("atr_pct_q35_120")
    if pd.notna(atr_pct) and pd.notna(atr_q35) and atr_pct <= atr_q35:
        score += 0.5
    # Proximity to 52w high
    close = feat_row["close"]
    if len(feat_series_close) >= 252:
        high_52w = feat_series_close.tail(252).max()
        if pd.notna(high_52w) and high_52w > 0:
            proximity = close / high_52w
            if proximity >= 0.85:
                score += 0.5 * min(1.0, (proximity - 0.85) / 0.15 + 0.5)
    return min(1.0, score)


def _score_rs_strength(feat_row) -> float:
    """RS rank + RS line behavior."""
    score = 0.0
    rs_rank = feat_row.get("rs_rank")
    if pd.notna(rs_rank):
        # Bonus for top-decile
        if rs_rank >= 90:
            score += 0.5
        elif rs_rank >= 80:
            score += 0.35
        elif rs_rank >= 70:
            score += 0.2
    if feat_row.get("rs_line_new_high"):
        score += 0.3
    # Absolute RS vs QQQ over 60d
    rs_60 = feat_row.get("rs_60_prior")
    if pd.notna(rs_60):
        if rs_60 > 0.30:
            score += 0.2
        elif rs_60 > 0.10:
            score += 0.1
    return min(1.0, score)


def _score_trend_regime(feat_row) -> float:
    """Stage 2 + EMA structure quality."""
    score = 0.0
    close_p = feat_row.get("close_prior")
    ema50_p = feat_row.get("ema_50_prior")
    ema200_p = feat_row.get("ema_200_prior")
    if pd.notna(close_p) and pd.notna(ema50_p) and pd.notna(ema200_p):
        if close_p > ema50_p > ema200_p:
            score += 0.6  # clean Stage 2
        elif close_p > ema200_p:
            score += 0.3  # at least above 200
    # 50 EMA rising
    slope = feat_row.get("ema_50_slope10")
    if pd.notna(slope) and slope > 0:
        score += 0.4
    return min(1.0, score)


def _score_prebreakout(feat_row, sig_row, comp_row) -> float:
    """How close is this name to actually triggering a signal?
    Higher score for 5-6/7 conditions hitting on VCP, or any mode firing."""
    if sig_row["any"]:
        return 1.0  # already firing
    score = int(comp_row.sum()) / 7.0  # fraction of VCP conditions met
    return min(1.0, score)


def _score_volume(feat_row) -> float:
    """Up-day volume vs avg + light pre-breakout volume (VDU)."""
    vol_x = feat_row["volume"] / feat_row["vol_avg_50"] if feat_row.get("vol_avg_50") else None
    if vol_x is None:
        return 0.0
    # Volume dry-up immediately before a breakout is constructive
    if 0.5 <= vol_x <= 0.9:
        return 0.5  # VDU pattern
    if vol_x >= 1.2:
        return 0.7  # active accumulation
    if vol_x >= 1.0:
        return 0.4
    return 0.2


def _score_setup(feat_row, sig_row, comp_row, close_series) -> dict:
    """Compute all scoring dimensions for a ticker. Returns dict with each component."""
    return {
        "clean_base": _score_clean_base(feat_row, close_series),
        "rs_strength": _score_rs_strength(feat_row),
        "trend_regime": _score_trend_regime(feat_row),
        "prebreakout": _score_prebreakout(feat_row, sig_row, comp_row),
        "volume": _score_volume(feat_row),
    }


# Component weights for composite score
WEIGHTS = {
    "clean_base": 0.20,
    "rs_strength": 0.30,    # RS leadership is the biggest edge
    "trend_regime": 0.20,
    "prebreakout": 0.20,
    "volume": 0.10,
}


def composite_score(parts: dict) -> float:
    return sum(WEIGHTS[k] * parts[k] for k in WEIGHTS) * 100


# ---------- Main scan ----------

def run_scan(top_n: int = 30, refresh_data: bool = False) -> pd.DataFrame:
    """Run scanner on wide universe, return ranked watchlist DataFrame."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading wide universe...")
    universe = build_wide_universe(refresh=refresh_data)
    tickers = universe["ticker"].tolist()
    if BENCHMARK not in tickers:
        tickers.append(BENCHMARK)
    print(f"  {len(tickers)} tickers")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching OHLCV (cached unless --refresh)...")
    raw = fetch_many(tickers, force=refresh_data)

    if BENCHMARK not in raw:
        raise RuntimeError(f"Benchmark {BENCHMARK} missing")
    bench = raw[BENCHMARK]["close"]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing cross-sectional RS rank...")
    equity_closes = {t: df["close"] for t, df in raw.items() if t != BENCHMARK}
    rs_rank_df = compute_universe_rs_rank(equity_closes)

    latest_date = max(df.index[-1] for df in raw.values())
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scoring on bar {latest_date.date()}...")

    rows = []
    skipped = 0
    for t, df in raw.items():
        if t == BENCHMARK:
            continue
        if latest_date not in df.index:
            skipped += 1
            continue
        rs_series = rs_rank_df[t] if t in rs_rank_df.columns else None
        try:
            feat = build_features(df, bench, rs_rank_series=rs_series)
        except Exception:
            skipped += 1
            continue
        if latest_date not in feat.index:
            continue
        sig = any_breakout_signal(feat)
        comp = signal_components(feat)

        feat_row = feat.loc[latest_date]
        sig_row = sig.loc[latest_date]
        comp_row = comp.loc[latest_date]

        # Skip if essential indicators are NaN (insufficient history)
        if pd.isna(feat_row.get("ema_200")) or pd.isna(feat_row.get("atr_pct")):
            skipped += 1
            continue

        parts = _score_setup(feat_row, sig_row, comp_row, df["close"])
        score = composite_score(parts)

        rows.append({
            "ticker": t,
            "score": round(score, 1),
            "close": round(float(feat_row["close"]), 2),
            "rs_rank": int(feat_row["rs_rank"]) if pd.notna(feat_row.get("rs_rank")) else None,
            "rs_60_pct": round(float(feat_row["rs_60_prior"]) * 100, 1) if pd.notna(feat_row.get("rs_60_prior")) else None,
            "rs_line_new_high": bool(feat_row.get("rs_line_new_high", False)),
            "signal_fired": bool(sig_row["any"]),
            "signal_mode": _mode_label(sig_row),
            "vcp_conditions": int(comp_row.sum()),
            "stage_2": bool(feat_row.get("close_prior", 0) > feat_row.get("ema_50_prior", 0)
                            > feat_row.get("ema_200_prior", 0)) if pd.notna(feat_row.get("ema_200_prior")) else False,
            "ema_50": round(float(feat_row["ema_50"]), 2),
            "ema_200": round(float(feat_row["ema_200"]), 2),
            "volume_x": round(float(feat_row["volume"] / feat_row["vol_avg_50"]), 2)
                if feat_row.get("vol_avg_50") else None,
            # Score component breakdown
            "score_clean_base": round(parts["clean_base"], 2),
            "score_rs_strength": round(parts["rs_strength"], 2),
            "score_trend_regime": round(parts["trend_regime"], 2),
            "score_prebreakout": round(parts["prebreakout"], 2),
            "score_volume": round(parts["volume"], 2),
        })

    print(f"  Scored {len(rows)}, skipped {skipped}")
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df.head(top_n) if top_n else df


def _mode_label(sig_row) -> str:
    modes = []
    if sig_row["vcp"]: modes.append("VCP")
    if sig_row["momentum"]: modes.append("MOM")
    if sig_row["emergence"]: modes.append("EME")
    if sig_row["pocket_pivot"]: modes.append("PP")
    return "/".join(modes) if modes else "-"


if __name__ == "__main__":
    top_n = 30
    for i, a in enumerate(sys.argv):
        if a == "--top" and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])
    refresh = "--refresh" in sys.argv

    watchlist = run_scan(top_n=top_n, refresh_data=refresh)
    print(f"\n=== TOP {top_n} WATCHLIST ===\n")
    cols = ["ticker", "score", "close", "rs_rank", "rs_60_pct", "signal_fired", "signal_mode",
            "vcp_conditions", "stage_2", "volume_x"]
    print(watchlist[cols].to_string(index=False))

    # Persist to snapshot
    save_df("watchlist", watchlist)
    write_manifest(
        scanner_top_n=top_n,
        scanner_universe_size=int(watchlist["ticker"].count()) if not watchlist.empty else 0,
    )
    print(f"\nWrote watchlist to data/snapshots/{datetime.now().strftime('%Y-%m-%d')}/watchlist.parquet")
