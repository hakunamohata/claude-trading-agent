"""B-Xtrender — momentum-of-momentum oscillator (Bharat Jhunjhunwala).

Port of the Pine Script v4 indicator by Puppytherapy. Two zero-centered
oscillators built from RSI applied to smoothed-price inputs, plus a T3
(Tillson's 6-stage EMA cascade) smoothed signal line on the short-term.

  shortTerm   = RSI( EMA(close, 5) - EMA(close, 20), 15 ) - 50
  longTerm    = RSI( EMA(close, 20), 15 ) - 50
  t3_short    = T3( shortTerm, length=5, b=0.7 )

Trading interpretation
  - longTerm > 0 → bull regime (longs allowed)
  - longTerm < 0 → bear regime (skip longs)
  - lime circle  → t3_short turns UP after being down 2 bars → entry signal
  - red circle   → t3_short turns DOWN after being up 2 bars → exit signal

Mode used by the breakout scanner:
  XTREND = bull regime AND lime-circle entry on this bar.

CLI:
    python xtrender.py MSFT          # show today's values + last 30 bars
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from breakout import ema, compute_rsi


# ============================================================
# Indicator math
# ============================================================

def t3(series: pd.Series, length: int = 5, b: float = 0.7) -> pd.Series:
    """Tillson's T3 — 6-stage EMA cascade weighted by volume factor b.

    With b=0.7 the cascade reduces lag aggressively while keeping a smooth
    curve. Default `b` matches the Pine Script indicator.
    """
    e1 = ema(series, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    c1 = -(b ** 3)
    c2 = 3 * b * b + 3 * (b ** 3)
    c3 = -6 * b * b - 3 * b - 3 * (b ** 3)
    c4 = 1 + 3 * b + (b ** 3) + 3 * b * b
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def compute_xtrender(
    close: pd.Series,
    short_l1: int = 5,
    short_l2: int = 20,
    short_l3: int = 15,
    long_l1: int = 20,
    long_l2: int = 15,
    t3_length: int = 5,
) -> pd.DataFrame:
    """Returns DataFrame indexed like `close` with columns:

      - xtr_short:        short-term oscillator (RSI of MACD-diff), centered at 0
      - xtr_long:         long-term oscillator (RSI of EMA), centered at 0
      - xtr_t3_short:     T3-smoothed short-term
      - xtr_bull_regime:  bool — long-term > 0
      - xtr_bull_flip:    bool — T3 turned UP after a 2-bar down (lime circle)
      - xtr_bear_flip:    bool — T3 turned DOWN after a 2-bar up (red circle)
    """
    ema_short_l1 = ema(close, short_l1)
    ema_short_l2 = ema(close, short_l2)
    short_input = ema_short_l1 - ema_short_l2
    xtr_short = compute_rsi(short_input, n=short_l3) - 50

    ema_long_l1 = ema(close, long_l1)
    xtr_long = compute_rsi(ema_long_l1, n=long_l2) - 50

    xtr_t3_short = t3(xtr_short, length=t3_length, b=0.7)

    # Bottom flip — equivalent to the Pine plotshape:
    #   t3[0] > t3[-1] AND t3[-1] < t3[-2]
    bull_flip = (xtr_t3_short > xtr_t3_short.shift(1)) & (
        xtr_t3_short.shift(1) < xtr_t3_short.shift(2)
    )
    # Top flip
    bear_flip = (xtr_t3_short < xtr_t3_short.shift(1)) & (
        xtr_t3_short.shift(1) > xtr_t3_short.shift(2)
    )

    return pd.DataFrame({
        "xtr_short": xtr_short,
        "xtr_long": xtr_long,
        "xtr_t3_short": xtr_t3_short,
        "xtr_bull_regime": (xtr_long > 0).fillna(False),
        "xtr_bull_flip": bull_flip.fillna(False),
        "xtr_bear_flip": bear_flip.fillna(False),
    }, index=close.index)


# ============================================================
# Signal — XTREND mode for the breakout scanner
# ============================================================

def xtrender_signal(feat: pd.DataFrame) -> pd.Series:
    """XTREND mode: lime entry circle while long-term regime is bullish.

    Reads the xtr_* columns added by build_features. If they're missing,
    computes them from feat['close'] on the fly.
    """
    if "xtr_t3_short" not in feat.columns:
        x = compute_xtrender(feat["close"])
        bull_regime = x["xtr_bull_regime"]
        bull_flip = x["xtr_bull_flip"]
    else:
        bull_regime = feat["xtr_bull_regime"]
        bull_flip = feat["xtr_bull_flip"]
    return (bull_regime & bull_flip).fillna(False)


# ============================================================
# CLI smoke test
# ============================================================

if __name__ == "__main__":
    import sys
    from data_fetch import fetch_one

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    print(f"Pulling {ticker}...")
    df = fetch_one(ticker)
    x = compute_xtrender(df["close"])
    last = x.tail(30).copy()
    last["close"] = df["close"].tail(30)
    last["signal_XTREND"] = (x["xtr_bull_regime"] & x["xtr_bull_flip"]).tail(30)

    print()
    print(f"{ticker} — last 30 bars")
    print(last[["close", "xtr_short", "xtr_long", "xtr_t3_short",
                "xtr_bull_regime", "xtr_bull_flip", "xtr_bear_flip",
                "signal_XTREND"]].round(2).to_string())

    fires = last[last["signal_XTREND"]]
    print()
    print(f"XTREND entries in last 30 bars: {len(fires)}")
    if not fires.empty:
        for d in fires.index:
            print(f"  {d.date()} @ ${last.loc[d, 'close']:.2f}")
