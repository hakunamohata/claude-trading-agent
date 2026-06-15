"""Daily scan — runs the breakout filter on the most recent bar and prints
both active signals and "setting up" near-misses.

Writes results to `data/scan_<date>.parquet` so the UI (future) can read them.

Usage:
    python scan.py             # use cached data if fresh (<18h)
    python scan.py --refresh   # force-refresh data first
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import zoneinfo
import pandas as pd

from universe import ALL_TICKERS, BENCHMARK, SECTOR_ETFS
from data_fetch import fetch_many, DATA_DIR
from breakout import build_features, any_breakout_signal, signal_components

_NON_TRADEABLE = set(SECTOR_ETFS) | {BENCHMARK}


ET = zoneinfo.ZoneInfo("America/New_York")


def _mode_label(sig_row) -> str:
    if sig_row["vcp"]:
        return "VCP"
    if sig_row["emergence"]:
        return "EME"
    if sig_row["pocket_pivot"]:
        return "PP"
    return "MOM"


def run(force_refresh: bool = False) -> None:
    now_et = datetime.now(ET)
    print(f"Scan run: {now_et.strftime('%Y-%m-%d %H:%M %Z')}")
    if force_refresh:
        print("Forcing data refresh...")

    raw = fetch_many(ALL_TICKERS, force=force_refresh)
    if BENCHMARK not in raw:
        raise RuntimeError(f"Benchmark {BENCHMARK} missing from fetched data")
    bench = raw[BENCHMARK]["close"]

    latest_date = max(df.index[-1] for df in raw.values())
    print(f"Most-recent bar across universe: {latest_date.date()}\n")

    active_rows: list[dict] = []
    setup_rows: list[dict] = []

    for t, df in raw.items():
        if t in _NON_TRADEABLE:
            continue
        if latest_date not in df.index:
            continue  # ticker missing today's bar (recently halted, etc.)

        feat = build_features(df, bench)
        sig = any_breakout_signal(feat)
        comps = signal_components(feat)

        sig_row = sig.loc[latest_date]
        feat_row = feat.loc[latest_date]
        comp_row = comps.loc[latest_date]

        vol_x = feat_row["volume"] / feat_row["vol_avg_50"] if feat_row["vol_avg_50"] else None
        rs_60 = feat_row["rs_60_prior"] * 100 if pd.notna(feat_row["rs_60_prior"]) else None

        if sig_row["any"]:
            active_rows.append({
                "ticker": t,
                "mode": _mode_label(sig_row),
                "close": round(feat_row["close"], 2),
                "vol_x": round(vol_x, 2) if vol_x is not None else None,
                "rs_60_%": round(rs_60, 1) if rs_60 is not None else None,
                "ema_50": round(feat_row["ema_50"], 2),
                "ema_200": round(feat_row["ema_200"], 2),
            })
        else:
            score = int(comp_row.sum())
            if 5 <= score <= 6:
                missing = [c for c, v in comp_row.items() if not v]
                setup_rows.append({
                    "ticker": t,
                    "score": f"{score}/7",
                    "close": round(feat_row["close"], 2),
                    "vol_x": round(vol_x, 2) if vol_x is not None else None,
                    "rs_60_%": round(rs_60, 1) if rs_60 is not None else None,
                    "missing": ", ".join(missing),
                })

    print("=== Active signals (filter fired on most recent bar) ===")
    if not active_rows:
        print("  (none)\n")
    else:
        df_a = pd.DataFrame(active_rows).sort_values("ticker")
        print(df_a.to_string(index=False))
        print()

    print("=== Setting up (5-6 of 7 conditions on most recent bar) ===")
    if not setup_rows:
        print("  (none)\n")
    else:
        df_s = pd.DataFrame(setup_rows).sort_values("score", ascending=False)
        print(df_s.to_string(index=False))
        print()

    # Persist for future UI consumption
    out_dir = Path(DATA_DIR) / "scans"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"scan_{latest_date.date()}.parquet"
    payload = pd.DataFrame(
        [{**r, "category": "active"} for r in active_rows]
        + [{**r, "category": "setup"} for r in setup_rows]
    )
    if not payload.empty:
        payload["scan_run_at"] = now_et.isoformat()
        payload["bar_date"] = latest_date.date().isoformat()
        payload.to_parquet(out_path)
        print(f"Wrote {len(payload)} rows -> {out_path}")
    else:
        print("(no rows to write)")


if __name__ == "__main__":
    force = "--refresh" in sys.argv or "--force" in sys.argv
    run(force_refresh=force)
