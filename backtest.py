"""Run the breakout filter over the small universe and print diagnostics.

Outputs:
  1. Every signal day per target (MU, SNDK) over the last 6 months, with forward
     5d / 20d returns so we can eyeball whether the trigger actually marked a move.
  2. All signal days across the small universe in the same window — gut-check
     of how trigger-happy the filter is.
  3. For the targets, near-miss days (6 of 7 conditions True) — shows what the
     filter almost caught and which condition vetoed it.
"""

from __future__ import annotations
import pandas as pd

from universe import ALL_TICKERS, TARGETS, BENCHMARK, SECTOR_ETFS
from data_fetch import fetch_many
from breakout import build_features, any_breakout_signal, signal_components

# Set of tickers to exclude from the per-name signal loop. ETFs are fetched
# alongside the universe so we can compute sector strength, but they should
# never be treated as tradeable signals.
_NON_TRADEABLE = set(SECTOR_ETFS) | {BENCHMARK}


LOOKBACK_MONTHS = 6


def forward_returns(close: pd.Series, days: int) -> pd.Series:
    return close.shift(-days) / close - 1.0


def run() -> None:
    print("Fetching data...")
    raw = fetch_many(ALL_TICKERS)
    bench = raw[BENCHMARK]["close"]

    end_date = max(df.index[-1] for df in raw.values())
    window_start = end_date - pd.DateOffset(months=LOOKBACK_MONTHS)

    # Build features per ticker
    feats = {}
    sigs = {}  # DataFrame per ticker: vcp / momentum / any
    for t, df in raw.items():
        if t in _NON_TRADEABLE:
            continue
        f = build_features(df, bench)
        feats[t] = f
        sigs[t] = any_breakout_signal(f)

    print(f"\nBack-test window: {window_start.date()} -> {end_date.date()}\n")

    # ---- 1. Target signals with forward returns ----
    for t in TARGETS:
        print(f"=== {t} — signal days ===")
        f = feats[t]
        sig_df = sigs[t]
        in_window = sig_df.index >= window_start
        sig_days = sig_df[in_window & sig_df["any"]]
        if sig_days.empty:
            print("  (no signals in window)\n")
            continue
        fwd5 = forward_returns(f["close"], 5)
        fwd20 = forward_returns(f["close"], 20)
        for d in sig_days.index:
            mode = ("VCP" if sig_df.loc[d, "vcp"]
                    else "EME" if sig_df.loc[d, "emergence"]
                    else "PP" if sig_df.loc[d, "pocket_pivot"]
                    else "MOM")
            print(f"  {d.date()}  [{mode}]  close={f.loc[d, 'close']:.2f}  "
                  f"vol_x={f.loc[d, 'volume']/f.loc[d, 'vol_avg_50']:.1f}  "
                  f"+5d={fwd5.loc[d]*100:+.1f}%  +20d={fwd20.loc[d]*100:+.1f}%")
        print()

    # ---- 2. Universe-wide signal density ----
    print("=== Universe-wide signals in window ===")
    rows = []
    for t, sig_df in sigs.items():
        sd = sig_df[(sig_df.index >= window_start) & sig_df["any"]]
        for d in sd.index:
            f = feats[t]
            mode = ("VCP" if sig_df.loc[d, "vcp"]
                    else "EME" if sig_df.loc[d, "emergence"]
                    else "PP" if sig_df.loc[d, "pocket_pivot"]
                    else "MOM")
            fwd20 = forward_returns(f["close"], 20).loc[d]
            rows.append((d.date(), t, mode, f.loc[d, "close"],
                         f.loc[d, "volume"] / f.loc[d, "vol_avg_50"],
                         fwd20 * 100 if pd.notna(fwd20) else None))
    if not rows:
        print("  (no signals across universe)\n")
    else:
        df_all = pd.DataFrame(rows, columns=["date", "ticker", "mode", "close", "vol_x", "+20d_%"])
        df_all = df_all.sort_values("date").reset_index(drop=True)
        print(df_all.to_string(index=False))
        print(f"\n  Total signals: {len(df_all)} across {df_all['ticker'].nunique()} names")
        print(f"  VCP: {(df_all['mode']=='VCP').sum()}  |  MOM: {(df_all['mode']=='MOM').sum()}  |  EME: {(df_all['mode']=='EME').sum()}")
        print(f"  Trading days in window: {sum((bench.index >= window_start) & (bench.index <= end_date))}")
        signals_per_day = len(df_all) / sum((bench.index >= window_start) & (bench.index <= end_date))
        print(f"  Signals/day: {signals_per_day:.2f}")
    print()

    # ---- 3. Near-misses on target names ----
    for t in TARGETS:
        comps = signal_components(feats[t])
        in_window = comps.index >= window_start
        scores = comps[in_window].sum(axis=1)
        near_miss = comps[in_window][scores == 6]
        if near_miss.empty:
            continue
        print(f"=== {t} — near-misses (6 of 7 conditions) ===")
        for d, row in near_miss.iterrows():
            missing = [c for c, v in row.items() if not v]
            print(f"  {d.date()}  missing: {', '.join(missing)}")
        print()


if __name__ == "__main__":
    run()
