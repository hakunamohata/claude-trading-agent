"""Regression test for the breakout filter.

Pins the known winning signals as expected fires. Run this after any change to
breakout.py — it will FAIL loudly if a previously-caught winner stops firing.

Usage:
    python regression_test.py
"""

from __future__ import annotations
import sys
import pandas as pd

from universe import ALL_TICKERS, BENCHMARK
from data_fetch import fetch_many
from breakout import build_features, any_breakout_signal


# Pinned winning signals — (date, expected_mode). `expected_mode` may be
# "VCP", "MOM", "EME", or "ANY" (we'll accept any mode firing on that date).
# These all had strong forward 20d returns when the filter previously caught them.
EXPECTED_WINS: dict[str, list[tuple[str, str]]] = {
    "AMD": [
        ("2026-04-16", "VCP"),  # +61.6% in 20d
        ("2026-04-24", "VCP"),  # +34.4% in 20d
    ],
    "MU": [
        ("2025-12-19", "MOM"),  # +46.4% in 20d
        ("2025-12-22", "MOM"),  # +43.8%
        ("2025-12-29", "MOM"),  # +47.9%
        ("2026-01-02", "MOM"),  # +38.8%
        ("2026-05-04", "MOM"),  # +84.6%
        ("2026-05-05", "MOM"),  # +68.6%
    ],
    "SNDK": [
        ("2026-01-06", "MOM"),  # +67.2% in 20d
        ("2026-01-07", "MOM"),  # +63.0%
        ("2026-01-09", "MOM"),  # +54.6%
        ("2026-05-01", "MOM"),  # +48.4%
        ("2026-05-04", "MOM"),  # +36.7%
    ],
    "NBIS": [
        ("2026-02-06", "EME"),  # +10.3% in 20d
        ("2026-02-12", "EME"),  # +25.9%
        ("2026-02-13", "EME"),  # +32.5%
        ("2026-02-19", "EME"),  # +12.9%
        ("2026-03-16", "VCP"),  # +24.7%
        ("2026-04-10", "VCP"),  # +22.1%
        ("2026-05-04", "VCP"),  # +47.7%
    ],
    "ALAB": [
        ("2026-04-24", "EME"),  # +44.2% in 20d
        ("2026-05-05", "EME"),  # +68.5%
    ],
    "MRVL": [
        ("2026-04-08", "MOM"),  # +50.5% in 20d
        ("2026-04-20", "VCP"),  # +14.3%
        ("2026-04-23", "VCP"),  # +15.2%
    ],
    "AVGO": [
        ("2026-05-29", "VCP"),
    ],
}

# Pocket Pivot wins — pinned independently so PP regressions are caught.
# These are PP-mode catches with strong forward 20d returns.
EXPECTED_PP_WINS: dict[str, list[str]] = {
    "MU": [
        "2025-12-18",  # +46.9% in 20d — earliest catch, PP only
        "2026-04-22",  # +50.2%
        "2026-04-27",  # +70.8%
        "2026-05-04",  # +84.6%
    ],
}


def run() -> int:
    print("Regression test — pinned winning signals")
    print("=" * 60)

    raw = fetch_many(ALL_TICKERS)
    bench = raw[BENCHMARK]["close"]

    passed = 0
    failed = 0
    fails: list[str] = []

    for ticker in sorted(EXPECTED_WINS.keys()):
        if ticker not in raw:
            for date, _ in EXPECTED_WINS[ticker]:
                fails.append(f"  FAIL  {ticker} {date} — ticker not in universe")
                failed += 1
            continue

        feat = build_features(raw[ticker], bench)
        sig = any_breakout_signal(feat)

        for date_str, expected_mode in EXPECTED_WINS[ticker]:
            date = pd.Timestamp(date_str)
            if date not in sig.index:
                fails.append(f"  FAIL  {ticker} {date_str} — date not in data")
                failed += 1
                continue

            sig_row = sig.loc[date]
            if not sig_row["any"]:
                fails.append(f"  FAIL  {ticker} {date_str} — no signal fired "
                             f"(expected {expected_mode})")
                failed += 1
                continue

            if expected_mode != "ANY":
                mode_key = expected_mode.lower()
                if mode_key == "eme":
                    mode_key = "emergence"
                elif mode_key == "mom":
                    mode_key = "momentum"
                if not sig_row[mode_key]:
                    # Some signal fired but not the expected one — log as warning,
                    # not failure, since the win is preserved (the user might accept this).
                    fired = [m.upper() for m in ["vcp", "momentum", "emergence"]
                             if sig_row.get(m, False)]
                    fails.append(f"  WARN  {ticker} {date_str} — fired as "
                                 f"{','.join(fired)} not {expected_mode} (win preserved)")
                    passed += 1  # still count as pass since signal fired
                    continue

            passed += 1

    # ---- PP-specific regression ----
    print("\nPocket Pivot pinned wins")
    print("-" * 40)
    for ticker, dates in EXPECTED_PP_WINS.items():
        if ticker not in raw:
            for d in dates:
                fails.append(f"  FAIL  {ticker} {d} PP — ticker missing")
                failed += 1
            continue
        feat = build_features(raw[ticker], bench)
        sig = any_breakout_signal(feat)
        for date_str in dates:
            date = pd.Timestamp(date_str)
            if date not in sig.index:
                fails.append(f"  FAIL  {ticker} {date_str} PP — date missing")
                failed += 1
                continue
            if not sig.loc[date, "pocket_pivot"]:
                fails.append(f"  FAIL  {ticker} {date_str} PP — did not fire")
                failed += 1
            else:
                passed += 1

    total = passed + failed
    print(f"\nPassed: {passed}/{total}    Failed: {failed}/{total}")
    print("=" * 60)

    if fails:
        for f in fails:
            print(f)
    else:
        print("All pinned wins still fire. PASS")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
