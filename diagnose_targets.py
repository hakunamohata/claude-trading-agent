"""Diagnose what MU and SNDK actually did in the window, and inspect the
filter components on their biggest move days."""

from __future__ import annotations
import pandas as pd

from universe import TARGETS, BENCHMARK
from data_fetch import fetch_many
from breakout import build_features, signal_components

LOOKBACK_MONTHS = 6


def run():
    raw = fetch_many(TARGETS + [BENCHMARK])
    bench = raw[BENCHMARK]["close"]

    end = max(df.index[-1] for df in raw.values())
    window_start = end - pd.DateOffset(months=LOOKBACK_MONTHS)

    for t in TARGETS:
        df = raw[t]
        df_w = df[df.index >= window_start].copy()

        # Forward 20-day return on every day in window
        df_w["fwd20"] = df_w["close"].shift(-20) / df_w["close"] - 1.0

        print(f"\n=== {t} — top 8 forward-20d moves in window ===")
        top = df_w.nlargest(8, "fwd20")[["close", "volume", "fwd20"]]
        for d, row in top.iterrows():
            print(f"  {d.date()}  close={row['close']:.2f}  fwd20={row['fwd20']*100:+.1f}%")

        print(f"\n=== {t} — overall window range ===")
        print(f"  start close: {df_w['close'].iloc[0]:.2f} on {df_w.index[0].date()}")
        print(f"  end   close: {df_w['close'].iloc[-1]:.2f} on {df_w.index[-1].date()}")
        print(f"  window return: {(df_w['close'].iloc[-1]/df_w['close'].iloc[0]-1)*100:+.1f}%")
        print(f"  window max close: {df_w['close'].max():.2f} on {df_w['close'].idxmax().date()}")

        # Show filter components on the single best move day
        best_day = df_w["fwd20"].idxmax()
        if pd.notna(best_day):
            print(f"\n=== {t} — filter components on best move day ({best_day.date()}) ===")
            feat = build_features(df, bench)
            comps = signal_components(feat)
            row = comps.loc[best_day]
            for k, v in row.items():
                print(f"  {k:15s} {'YES' if v else 'no '}")
            f = feat.loc[best_day]
            print(f"  -- values --")
            print(f"  close                {f['close']:.2f}")
            print(f"  high_50_prior        {f['high_50_prior']:.2f}")
            print(f"  volume / vol_avg_50  {f['volume']/f['vol_avg_50']:.2f}x")
            print(f"  atr_pct_prior        {f['atr_pct_prior']*100:.2f}%")
            print(f"  atr_pct_q35_120      {f['atr_pct_q35_120']*100:.2f}%")
            print(f"  ema_50 prior         {f['ema_50_prior']:.2f}")
            print(f"  ema_200 prior        {f['ema_200_prior']:.2f}")
            print(f"  rs_60_prior          {f['rs_60_prior']*100:+.1f}%")


if __name__ == "__main__":
    run()
