"""Indicators and breakout filter — hand-rolled (no pandas-ta).

All signals are designed so that a True value on row t depends only on data
available by the close of bar t. Pre-breakout conditions (vol contraction,
trend, RS) use `.shift(1)` so the breakout bar's own action doesn't
self-validate the setup.

Composite filter (all must hold on bar t):
  1. Close >= prior 50-day high  (new high — the breakout itself)
  2. Volume >= 1.5 * 50-day avg volume on bar t
  3. Bar t closes in the top third of its range
  4. ATR% as of bar t-1 was in the bottom 35% of the prior 120-day distribution
     (quiet base — volatility contraction before the move)
  5. Stage-2 trend as of bar t-1: close > 50 EMA > 200 EMA
  6. 50 EMA rising as of bar t-1 (positive 10-day diff)
  7. Relative strength vs benchmark over prior 60 days > 0 (as of bar t-1)
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# ---------- Indicators ----------

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ATR."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_close = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_close).abs(), (l - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ---------- Feature engineering ----------

def compute_ad_score(df: pd.DataFrame, lookback: int = 65) -> pd.Series:
    """IBD-style Accumulation/Distribution score.

    For each bar t, computes the rolling N-day A/D score (default 65 = ~13 weeks):
        buying_pressure  = sum of (close - prior_close) * volume on up days
        selling_pressure = sum of |close - prior_close| * volume on down days
        ad_score        = (buying - selling) / (buying + selling)

    Range: [-1, +1]. Letter-grade thresholds (see `ad_label`):
        >= +0.30  "A"  heavy accumulation
        >= +0.10  "B"  mild accumulation
        >= -0.10  "C"  neutral
        >= -0.30  "D"  mild distribution
        <  -0.30  "E"  heavy distribution

    Weights each day's contribution by |price change| * volume — so a tiny
    drift up on huge volume contributes less than a strong up close on equal
    volume. This is the standard Wyckoff/IBD effort-vs-result formulation.

    Computed using only data through bar t (no look-ahead).
    """
    c = df["close"]
    v = df["volume"]
    change = c - c.shift(1)
    dollar_vol = change * v
    buy = dollar_vol.where(change > 0, 0).rolling(lookback).sum()
    sell = (-dollar_vol).where(change < 0, 0).rolling(lookback).sum()
    denom = (buy + sell).replace(0, np.nan)
    return (buy - sell) / denom


def ad_label(score) -> str:
    """Map A/D score to IBD-style letter grade. Returns '?' for NaN."""
    if pd.isna(score):
        return "?"
    if score >= 0.30: return "A"
    if score >= 0.10: return "B"
    if score >= -0.10: return "C"
    if score >= -0.30: return "D"
    return "E"


def compute_universe_rs_rank(
    closes: dict[str, pd.Series],
    weights: dict[int, float] | None = None,
) -> pd.DataFrame:
    """IBD-style cross-sectional RS rank (1-99) per-day across all tickers.

    Composite return = weighted sum of [1m, 3m, 6m, 12m] returns.
    Default weights mirror IBD: 40% 1m, 20% each 3m/6m/12m.

    With a small universe (< 50 names) the percentile is coarse — useful as a
    context feature, not as a precise filter threshold. Replace the universe
    with Russell 1000+ to get true IBD-style precision.
    """
    weights = weights or {21: 0.4, 63: 0.2, 126: 0.2, 252: 0.2}
    close_df = pd.DataFrame(closes)
    composite = sum(w * close_df.pct_change(p) for p, w in weights.items())
    rank_df = composite.rank(axis=1, pct=True) * 99
    return rank_df.round()


# ---------- Tier 1: external TA overlays ----------

def compute_anchored_vwap(df: pd.DataFrame, anchor_window: int = 60) -> pd.Series:
    """Anchored VWAP from the most recent N-day swing low.

    For each bar t, finds the index of the lowest low in (t - anchor_window, t]
    and computes the cumulative volume-weighted typical price from that anchor
    forward through t. As the anchor shifts (new swing low), AVWAP re-anchors.

    Per external TA playbook: "uncanny" for support/resistance from key
    inflection points. Use as a context feature and as the basis for the
    `avwap_reclaim_signal` Tier 2 mode below.

    No look-ahead: each row's AVWAP uses only bars from its anchor through t.
    """
    h, l, c = df["high"], df["low"], df["close"]
    v = df["volume"]
    typical = (h + l + c) / 3.0
    tpv = typical * v

    anchor_iloc = l.rolling(anchor_window, min_periods=1).apply(
        lambda x: int(np.argmin(x.values)) + (len(x) - len(x.values)), raw=False
    )

    out = pd.Series(index=df.index, dtype="float64")
    cum_tpv = tpv.cumsum().values
    cum_v = v.cumsum().values
    for i in range(len(df)):
        win_start = max(0, i - anchor_window + 1)
        anchor_local = int(np.argmin(l.iloc[win_start:i + 1].values))
        anchor_i = win_start + anchor_local
        if anchor_i == 0:
            num = cum_tpv[i]
            den = cum_v[i]
        else:
            num = cum_tpv[i] - cum_tpv[anchor_i - 1]
            den = cum_v[i] - cum_v[anchor_i - 1]
        out.iloc[i] = num / den if den > 0 else np.nan
    return out


def compute_demark_setup(c: pd.Series) -> pd.Series:
    """DeMark 9 Sequential Setup count.

    Positive count [+1..+9] = bullish setup (consecutive closes > close 4 bars ago).
    Negative count [-1..-9] = bearish setup (consecutive closes < close 4 bars ago).
    A printed +9 or -9 signals potential exhaustion — used for EXIT/trim timing,
    not entry. Reset to 0 when the streak breaks.

    Per external TA playbook: a DeMark 9 is worth monitoring, while a 13 is very rare.
    """
    diff = c - c.shift(4)
    out = np.zeros(len(c), dtype=int)
    for i in range(4, len(c)):
        d = diff.iloc[i]
        if pd.isna(d):
            out[i] = 0
        elif d > 0:
            out[i] = (out[i - 1] + 1) if out[i - 1] > 0 else 1
        elif d < 0:
            out[i] = (out[i - 1] - 1) if out[i - 1] < 0 else -1
        else:
            out[i] = 0
        if out[i] > 9:
            out[i] = 9
        if out[i] < -9:
            out[i] = -9
    return pd.Series(out, index=c.index)


def compute_rsi(c: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (smoothed). Range [0, 100].

    Per external TA playbook: in uptrends, **RSI 40 acts as support** during corrections;
    in downtrends, **RSI 60 acts as resistance** during bounces. Watch reversals
    around those lines, not the textbook 30/70 overbought/oversold.
    """
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_fib_levels(df: pd.DataFrame, window: int = 60) -> dict[str, pd.Series]:
    """Fibonacci retracement levels of the rolling N-day swing.

    Computes the highest high and lowest low over the prior `window` bars (as of
    bar t, using shift(1) to avoid look-ahead). Returns the 38.2 / 50 / 61.8
    retracement LEVELS of that swing.

    Interpretation:
        If price is in an uptrend (swing low first, then swing high), the
        retracement levels are PULLBACK supports. If price holds 38.2% on a
        pullback and reclaims, that's a Fib-bounce buy.
    """
    high_window = df["high"].rolling(window).max().shift(1)
    low_window = df["low"].rolling(window).min().shift(1)
    swing_range = high_window - low_window
    return {
        "fib_high_60": high_window,
        "fib_low_60": low_window,
        "fib_382": high_window - 0.382 * swing_range,
        "fib_500": high_window - 0.500 * swing_range,
        "fib_618": high_window - 0.618 * swing_range,
    }


def build_features(
    df: pd.DataFrame,
    benchmark_close: pd.Series,
    rs_rank_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Add all indicator columns. Returns a copy."""
    out = df.copy()
    c = out["close"]

    out["ema_50"] = ema(c, 50)
    out["ema_200"] = ema(c, 200)

    # 9/21 SMA cloud (Ravish-style swing trading band).
    # Green cloud (9>21) = bullish bias; red cloud (9<21) = bearish.
    # Used in dashboard charts and as additional context for swing entries.
    out["sma_9"] = c.rolling(9).mean()
    out["sma_21"] = c.rolling(21).mean()
    out["cloud_bullish"] = out["sma_9"] > out["sma_21"]

    # Short-term trader EMA stack (8 / 20 / 55) — display-only context.
    # The source playbook uses 200 SMA, but we already track 200 EMA which is close enough.
    out["ema_8"] = ema(c, 8)
    out["ema_20"] = ema(c, 20)
    out["ema_55"] = ema(c, 55)

    # Wilder's RSI for the 40/60 reversal levels.
    out["rsi_14"] = compute_rsi(c, n=14)
    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = out["atr_14"] / c

    out["vol_avg_50"] = out["volume"].rolling(50).mean()

    # IBD-style A/D score (13-week accumulation/distribution). Distinguishes
    # quiet drift-up (low A/D) from real institutional accumulation (high A/D).
    out["ad_score_65"] = compute_ad_score(out, lookback=65)

    # Prior N-day high — shift(1) so today's high doesn't count against itself.
    out["high_50_prior"] = out["high"].rolling(50).max().shift(1)

    # ATR% as of yesterday, vs 120-day distribution of yesterday-or-older ATR%.
    atr_pct_prior = out["atr_pct"].shift(1)
    out["atr_pct_prior"] = atr_pct_prior
    out["atr_pct_q35_120"] = atr_pct_prior.rolling(120).quantile(0.35)

    # Trend / structure as of yesterday so today's breakout bar isn't the only
    # thing pulling the EMA above its prior level.
    out["close_prior"] = c.shift(1)
    out["ema_50_prior"] = out["ema_50"].shift(1)
    out["ema_200_prior"] = out["ema_200"].shift(1)
    out["ema_50_slope10"] = out["ema_50"].shift(1) - out["ema_50"].shift(11)

    # Top-third close
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_in_top_third"] = (out["close"] - out["low"]) / rng >= (2 / 3)

    # Relative strength vs benchmark over prior 60 trading days, computed as of
    # yesterday so it's purely pre-breakout information.
    bench_aligned = benchmark_close.reindex(out.index).ffill()
    stock_ret_60 = c.pct_change(60)
    bench_ret_60 = bench_aligned.pct_change(60)
    out["rs_60_prior"] = (stock_ret_60 - bench_ret_60).shift(1)

    # ---- RS line ----
    # Stock-vs-benchmark price ratio. New highs in this line often *lead*
    # price breakouts — when RS line breaks before price does, that's emerging
    # leadership invisible in absolute price action.
    out["rs_line"] = c / bench_aligned
    out["rs_line_50d_high_prior"] = out["rs_line"].rolling(50).max().shift(1)
    out["rs_line_new_high"] = out["rs_line"] >= out["rs_line_50d_high_prior"]

    # ---- IBD-style RS rank (cross-sectional, plugged in from outside) ----
    # NOTE: as-of column — value at index t is the rank computed from today's
    # close. If you later add a filter condition that uses rs_rank, shift(1)
    # it first to avoid look-ahead. Currently used for display only.
    if rs_rank_series is not None:
        out["rs_rank"] = rs_rank_series.reindex(out.index)
    else:
        out["rs_rank"] = pd.Series(index=out.index, dtype="float64")

    # ---- Tier 1: external TA overlays (additive context) ----
    out["avwap_swinglow_60"] = compute_anchored_vwap(out, anchor_window=60)
    out["avwap_swinglow_60_prior"] = out["avwap_swinglow_60"].shift(1)
    out["above_avwap"] = c > out["avwap_swinglow_60"]
    # AVWAP reclaim today = was below AVWAP yesterday, above today
    out["avwap_reclaim_today"] = (
        (out["close_prior"] < out["avwap_swinglow_60_prior"])
        & (c >= out["avwap_swinglow_60"])
    )

    out["demark_setup"] = compute_demark_setup(c)
    out["demark_9_buy"] = out["demark_setup"] == 9   # bullish exhaustion (potential top)
    out["demark_9_sell"] = out["demark_setup"] == -9  # bearish exhaustion (potential bottom)

    fib = compute_fib_levels(out, window=60)
    for k, v in fib.items():
        out[k] = v
    fib_tol = 0.02
    out["near_fib_382"] = ((c - out["fib_382"]).abs() / c) <= fib_tol
    out["near_fib_500"] = ((c - out["fib_500"]).abs() / c) <= fib_tol
    out["near_fib_618"] = ((c - out["fib_618"]).abs() / c) <= fib_tol
    out["above_fib_382"] = c > out["fib_382"]

    return out


# ---------- Composite filter ----------

def breakout_signal(feat: pd.DataFrame) -> pd.Series:
    """Return a boolean Series — True where all breakout conditions hold."""
    cond_breakout_high = feat["close"] >= feat["high_50_prior"]
    cond_volume_surge = feat["volume"] >= 1.5 * feat["vol_avg_50"]
    cond_top_third = feat["close_in_top_third"]

    cond_quiet_base = feat["atr_pct_prior"] <= feat["atr_pct_q35_120"]

    cond_stage2 = (
        (feat["close_prior"] > feat["ema_50_prior"])
        & (feat["ema_50_prior"] > feat["ema_200_prior"])
    )
    cond_ema_rising = feat["ema_50_slope10"] > 0
    cond_rs_positive = feat["rs_60_prior"] > 0

    return (
        cond_breakout_high
        & cond_volume_surge
        & cond_top_third
        & cond_quiet_base
        & cond_stage2
        & cond_ema_rising
        & cond_rs_positive
    )


def momentum_continuation_signal(feat: pd.DataFrame) -> pd.Series:
    """Second mode: trend continuation in established Stage-2 uptrends.

    Empirically, AI-memory cycle names (MU, SNDK 2025-26) and similar mega-momentum
    moves don't form quiet bases — they trend on persistent strength. The VCP
    filter rejects them on `quiet_base` and `volume_surge`. This mode requires
    *much* stronger RS (>+20% vs QQQ over 60d) and accepts a shorter (20-day)
    breakout window with merely-average volume.
    """
    high_20_prior = feat["high"].rolling(20).max().shift(1)
    rng = (feat["high"] - feat["low"]).replace(0, np.nan)

    cond_stage2 = (
        (feat["close_prior"] > feat["ema_50_prior"])
        & (feat["ema_50_prior"] > feat["ema_200_prior"])
    )
    cond_elite_rs = feat["rs_60_prior"] > 0.20  # +20pp vs QQQ over 60d
    cond_ema_rising = feat["ema_50_slope10"] > 0
    cond_breakout_20 = feat["close"] >= high_20_prior
    cond_volume_normal = feat["volume"] >= feat["vol_avg_50"]
    cond_top_half = (feat["close"] - feat["low"]) / rng >= 0.5

    return (
        cond_stage2
        & cond_elite_rs
        & cond_ema_rising
        & cond_breakout_20
        & cond_volume_normal
        & cond_top_half
    )


def stage_emergence_signal(feat: pd.DataFrame) -> pd.Series:
    """Third mode: Stage 2 emergence — catches the *birth* of an uptrend.

    Empirically: ALAB on 2026-03-26 had its single best forward 20d move (+87%)
    but our trend-following filter was blind to it because the stock was 21%
    below its 200 EMA with RS at -22% — a classic post-base reversal. By the
    time Stage 2 was established (May), the easy gains were gone.

    Trigger conditions:
      - Close reclaims the 200 EMA today (was below it any time in prior 20d)
      - 200 EMA slope flat or turning up (no longer declining)
      - Structure: 50 EMA > 200 EMA OR close > prior 50-day high
      - Volume >= 1.2x 50-day avg (confirming participation)
      - Top half of day's range
    """
    close = feat["close"]
    ema_50_prior = feat["ema_50_prior"]
    ema_200 = feat["ema_200"]
    ema_200_prior = feat["ema_200_prior"]

    cond_reclaim = close > ema_200
    was_below_200 = ((feat["close_prior"] < ema_200_prior).rolling(20).max() == 1)
    cond_200_turning = ema_200_prior >= ema_200.shift(21)

    # Golden cross requirement — empirically required to filter false positives
    # during deep drawdowns. ALAB's 3 Dec-Feb 2026 false positives (lost 28-38%
    # in 20d) all had 50 EMA < 200 EMA at signal time; ALAB's true wins on
    # 4/24 + 5/5 had 50 EMA > 200 EMA. NBIS Feb wins also pass golden cross.
    cond_golden_cross = ema_50_prior > ema_200_prior

    cond_volume = feat["volume"] >= 1.2 * feat["vol_avg_50"]
    rng = (feat["high"] - feat["low"]).replace(0, np.nan)
    cond_top_half = (close - feat["low"]) / rng >= 0.5

    return (
        cond_reclaim
        & was_below_200
        & cond_200_turning
        & cond_golden_cross
        & cond_volume
        & cond_top_half
    )


def pocket_pivot_signal(feat: pd.DataFrame) -> pd.Series:
    """Fourth mode: Minervini's pocket pivot — institutional buying *inside* a base.

    Empirically, names like ALAB had pocket-pivot-style up days during their
    bottoming phases that the trend-following modes can't see (since they
    require established Stage 2). This catches accumulation inside a sideways
    base or shallow pullback.

    Trigger:
      - Today is an up day (close > yesterday's close)
      - Today's volume > max volume of any DOWN day in the prior 10 days
        (Minervini's specific institutional-print test)
      - Close >= 50 EMA (in or above its base)
      - Close within 15% of the 50-day high (not extended)
      - 200 EMA flat or rising (not catching falling knives)
    """
    close = feat["close"]
    close_prior = feat["close_prior"]

    cond_up_day = close > close_prior

    # Volume of any down day in prior 10 days
    is_down = feat["close"] < feat["close"].shift(1)
    down_vol = feat["volume"].where(is_down, other=0)
    max_down_vol_10d_prior = down_vol.shift(1).rolling(10).max()
    cond_pp_volume = feat["volume"] > max_down_vol_10d_prior

    cond_above_50_ema = close >= feat["ema_50"]
    high_50_prior = feat["high"].rolling(50).max().shift(1)
    cond_not_extended = close >= 0.85 * high_50_prior

    cond_200_not_falling = feat["ema_200_prior"] >= feat["ema_200"].shift(21)

    return (
        cond_up_day
        & cond_pp_volume
        & cond_above_50_ema
        & cond_not_extended
        & cond_200_not_falling
    )


def avwap_reclaim_signal(feat: pd.DataFrame) -> pd.Series:
    """Fifth mode (Tier 2 external TA overlay): close reclaims AVWAP from swing low.

    Per external TA playbook: AVWAP from a key swing point is "uncanny" for S/R.
    When price has been below the AVWAP and reclaims it on volume in a Stage-2
    trend, that's institutional re-engagement.

    Trigger:
      - AVWAP reclaim today (close below AVWAP yesterday, above today)
      - Stage 2 trend on prior bar (close > 50 EMA > 200 EMA)
      - Volume >= 1.2x 50-day avg
      - Top half of day's range
      - 50 EMA rising (no failing trend)

    Catches mean-reversion in Stage-2 trenders pulling back to AVWAP and
    bouncing — a setup the VCP and PP modes both miss.
    """
    cond_reclaim = feat["avwap_reclaim_today"].fillna(False)
    cond_stage2 = (
        (feat["close_prior"] > feat["ema_50_prior"])
        & (feat["ema_50_prior"] > feat["ema_200_prior"])
    )
    cond_volume = feat["volume"] >= 1.2 * feat["vol_avg_50"]
    rng = (feat["high"] - feat["low"]).replace(0, np.nan)
    cond_top_half = (feat["close"] - feat["low"]) / rng >= 0.5
    cond_ema_rising = feat["ema_50_slope10"] > 0
    return (
        cond_reclaim
        & cond_stage2
        & cond_volume
        & cond_top_half
        & cond_ema_rising
    )


def fib_bounce_signal(feat: pd.DataFrame) -> pd.Series:
    """Sixth mode (Tier 2 external TA overlay): bounce off a Fibonacci retracement level.

    Per external TA playbook: "When different Fibonacci sub-divisions line up, you have
    a very accurate target." Pullbacks that hold a Fib level and reclaim are
    high-conviction entries because the level is a known institutional zone.

    Trigger:
      - Close was below the 38.2% retracement level yesterday, above today
        (the bounce/reclaim event)
      - 60-day swing range is meaningful (high > 1.05x low — avoid sideways grinds)
      - Stage 2 trend on prior bar (uptrend context — don't catch falling knives)
      - Volume >= 1.2x 50-day avg
      - Top half close
      - Close still well below the recent high (room to run; not extended)
    """
    c = feat["close"]
    c_prior = feat["close_prior"]
    fib_382 = feat["fib_382"]
    fib_382_prior = fib_382.shift(1)

    cond_reclaim_382 = (c_prior < fib_382_prior) & (c >= fib_382)
    cond_meaningful_swing = feat["fib_high_60"] > 1.05 * feat["fib_low_60"]
    cond_stage2 = (
        (c_prior > feat["ema_50_prior"])
        & (feat["ema_50_prior"] > feat["ema_200_prior"])
    )
    cond_volume = feat["volume"] >= 1.2 * feat["vol_avg_50"]
    rng = (feat["high"] - feat["low"]).replace(0, np.nan)
    cond_top_half = (c - feat["low"]) / rng >= 0.5
    cond_not_extended = c <= 0.95 * feat["fib_high_60"]

    return (
        cond_reclaim_382
        & cond_meaningful_swing
        & cond_stage2
        & cond_volume
        & cond_top_half
        & cond_not_extended
    )


def any_breakout_signal(feat: pd.DataFrame) -> pd.DataFrame:
    """Combined: returns DataFrame with all modes + `any` column."""
    vcp = breakout_signal(feat)
    mom = momentum_continuation_signal(feat)
    eme = stage_emergence_signal(feat)
    pp = pocket_pivot_signal(feat)
    avwap = avwap_reclaim_signal(feat)
    fib = fib_bounce_signal(feat)
    return pd.DataFrame(
        {
            "vcp": vcp, "momentum": mom, "emergence": eme, "pocket_pivot": pp,
            "avwap_reclaim": avwap, "fib_bounce": fib,
            "any": vcp | mom | eme | pp | avwap | fib,
        },
        index=feat.index,
    )


def signal_components(feat: pd.DataFrame) -> pd.DataFrame:
    """Per-condition booleans, useful for diagnosing why a near-miss didn't fire."""
    return pd.DataFrame({
        "breakout_high": feat["close"] >= feat["high_50_prior"],
        "volume_surge": feat["volume"] >= 1.5 * feat["vol_avg_50"],
        "top_third": feat["close_in_top_third"],
        "quiet_base": feat["atr_pct_prior"] <= feat["atr_pct_q35_120"],
        "stage2": (feat["close_prior"] > feat["ema_50_prior"])
                  & (feat["ema_50_prior"] > feat["ema_200_prior"]),
        "ema_rising": feat["ema_50_slope10"] > 0,
        "rs_positive": feat["rs_60_prior"] > 0,
    }, index=feat.index)
