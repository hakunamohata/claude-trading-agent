"""Macro regime score — VIX + market breadth + credit-risk appetite.

Combines three independent macro indicators into a single 0-100 score:
  - VIX level: 0-100 where lower VIX = higher score (risk-on)
  - Market breadth: % of S&P 500 above 50 EMA + % above 200 EMA
  - Credit risk: HYG (high-yield ETF) above its 200 EMA → credit risk-on

Pure framework. Plugs into multi-agent prompts as additional context.

CLI:
    python macro_gate.py        # current regime + per-component breakdown
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from data_fetch import fetch_one, fetch_many
from breakout import ema
from futures import snapshot as futures_snapshot

VIX_TICKER = "^VIX"
CREDIT_TICKER = "HYG"      # iShares high-yield corporate bond
TREASURY_TICKER = "IEF"    # 7-10 year treasuries (for credit-spread proxy)


# ============================================================
# Component scores (0-100, higher = risk-on)
# ============================================================

def _score_vix(vix: float) -> float:
    """VIX < 15 = strongly risk-on, > 30 = strongly risk-off."""
    if vix <= 12: return 95
    if vix <= 15: return 85
    if vix <= 18: return 70
    if vix <= 22: return 55
    if vix <= 28: return 35
    if vix <= 35: return 20
    return 10


def _score_breadth(pct_above_50ema: float, pct_above_200ema: float) -> float:
    """Average % above MAs — typically 30-80% normal range."""
    avg = (pct_above_50ema + pct_above_200ema) / 2
    if avg >= 80: return 90
    if avg >= 70: return 80
    if avg >= 60: return 70
    if avg >= 50: return 60
    if avg >= 40: return 45
    if avg >= 30: return 30
    if avg >= 20: return 20
    return 10


def _score_credit(hyg_close: float, hyg_ema200: float, ratio_trend_pct: float) -> float:
    """HYG above 200 EMA + rising vs treasuries = credit risk-on."""
    above_200 = hyg_close > hyg_ema200
    base = 70 if above_200 else 30
    # Trend adjustment: HYG/IEF ratio change over 20 days
    if ratio_trend_pct > 2: base += 15
    elif ratio_trend_pct > 0: base += 5
    elif ratio_trend_pct > -2: base -= 5
    else: base -= 15
    return max(5, min(95, base))


def _regime_label(score: float) -> str:
    if score >= 75: return "risk-on (favorable for breakouts)"
    if score >= 60: return "neutral-bullish"
    if score >= 45: return "neutral"
    if score >= 30: return "neutral-bearish (caution)"
    return "risk-off (avoid new positions)"


# ============================================================
# Market breadth via wide universe
# ============================================================

def _compute_breadth() -> dict:
    """% of S&P 500 currently above their own 50 and 200 EMAs."""
    try:
        from wide_universe import build_wide_universe
    except Exception:
        return {"pct_above_50ema": None, "pct_above_200ema": None, "n_evaluated": 0}

    universe = build_wide_universe()
    sp500 = universe[universe["source"] == "S&P 500"]["ticker"].tolist()
    # Pull cached prices; skip names without sufficient history
    above_50 = 0
    above_200 = 0
    evaluated = 0

    for t in sp500:
        path = Path(__file__).parent / "data" / f"{t}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if len(df) < 200:
                continue
            close = df["close"]
            e50 = ema(close, 50).iloc[-1]
            e200 = ema(close, 200).iloc[-1]
            last = close.iloc[-1]
            if last > e50: above_50 += 1
            if last > e200: above_200 += 1
            evaluated += 1
        except Exception:
            continue

    if evaluated == 0:
        return {"pct_above_50ema": None, "pct_above_200ema": None, "n_evaluated": 0}

    return {
        "pct_above_50ema": round(above_50 / evaluated * 100, 1),
        "pct_above_200ema": round(above_200 / evaluated * 100, 1),
        "n_evaluated": evaluated,
    }


# ============================================================
# Main
# ============================================================

def compute_regime() -> dict:
    """Returns full regime breakdown + composite score 0-100."""
    # VIX
    try:
        vix_df = fetch_one(VIX_TICKER, period="3mo")
        vix_now = float(vix_df["close"].iloc[-1])
    except Exception:
        vix_now = None
    vix_score = _score_vix(vix_now) if vix_now is not None else None

    # Breadth
    breadth = _compute_breadth()
    breadth_score = (
        _score_breadth(breadth["pct_above_50ema"], breadth["pct_above_200ema"])
        if breadth["pct_above_50ema"] is not None else None
    )

    # Credit
    try:
        raw = fetch_many([CREDIT_TICKER, TREASURY_TICKER])
        hyg = raw[CREDIT_TICKER]["close"]
        ief = raw[TREASURY_TICKER]["close"]
        hyg_now = float(hyg.iloc[-1])
        hyg_ema200 = float(ema(hyg, 200).iloc[-1])
        ratio = hyg / ief.reindex(hyg.index).ffill()
        ratio_change = (ratio.iloc[-1] / ratio.iloc[-21] - 1) * 100
        credit_score = _score_credit(hyg_now, hyg_ema200, ratio_change)
    except Exception:
        hyg_now = hyg_ema200 = ratio_change = credit_score = None

    # Composite
    valid = [s for s in (vix_score, breadth_score, credit_score) if s is not None]
    composite = sum(valid) / len(valid) if valid else None

    # Futures overnight (not scored into composite — purely informational for
    # next-session positioning).
    try:
        fut = futures_snapshot()
    except Exception:
        fut = None

    return {
        "composite_score": round(composite, 1) if composite is not None else None,
        "regime_label": _regime_label(composite) if composite is not None else "unknown",
        "futures": fut,
        "vix": {
            "value": vix_now,
            "score": vix_score,
        },
        "breadth": {
            "pct_above_50_ema": breadth["pct_above_50ema"],
            "pct_above_200_ema": breadth["pct_above_200ema"],
            "n_evaluated": breadth["n_evaluated"],
            "score": breadth_score,
        },
        "credit": {
            "hyg_close": hyg_now,
            "hyg_200_ema": hyg_ema200,
            "hyg_above_200_ema": hyg_now > hyg_ema200 if hyg_now and hyg_ema200 else None,
            "hyg_ief_ratio_20d_change_pct": round(ratio_change, 2) if ratio_change is not None else None,
            "score": credit_score,
        },
    }


if __name__ == "__main__":
    import json
    r = compute_regime()
    print(json.dumps(r, indent=2, default=str))
    print()
    print(f"REGIME: {r['regime_label']}  (score {r['composite_score']}/100)")
