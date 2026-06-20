"""QQQ LEAPS dip-buy engine — long-dated call buying on QQQ pullbacks.

Strategy (from the video the user shared):
  Entry signal (both must be true):
    1. QQQ gaps down ≥ 1% (today's open at least 1% below yesterday's close)
    2. QQQ is above its 100-day simple moving average (bull-market regime)
  Position:
    - Buy to open ~60-delta call at ~12 months expiry (LEAPS)
  Exit:
    - Take profit at +50% premium gain (e.g. paid $88 → close at $132)
    - No stop loss (max loss = premium paid; held to expiry if needed)

Rationale: only takes the trade during bull-market pullbacks (≥1% gap-down filters
chop; >100-day SMA filters bear regimes). LEAPS minimize theta drag and give a
year for the thesis to play out. 60 delta gives meaningful directional exposure
(~0.6 shares of QQQ per contract) at much lower capital than buying 100 shares.

Daily output: data/snapshots/<today>/qqq_leaps_dipbuy.md

CLI:
    python qqq_leaps_dipbuy.py                  # check today's signal + open positions
    python qqq_leaps_dipbuy.py --backtest       # backtest the rule on 5y of history
    python qqq_leaps_dipbuy.py --backtest --years 10
"""

from __future__ import annotations
import io
import sys
from datetime import datetime, date
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from data_fetch import DATA_DIR, fetch_many
from options import fetch_expiries, fetch_options_chain, bs_greeks


TICKER = "QQQ"

DEFAULTS = {
    "gap_down_pct_threshold": -1.0,      # today's open vs yesterday's close
    "sma_window":              100,
    "target_delta":            0.60,
    "target_dte_min":          330,      # ~11 months
    "target_dte_max":          400,      # ~13 months
    "take_profit_pct":         50.0,     # close when LEAPS gains 50%
    # Backtest proxy: a 60-delta 12-month LEAPS at-the-money gains ~50% in premium
    # if the underlying rises ~15% (rough rule: 60% delta × 15% × leverage factor).
    # Tuneable knob below.
    "backtest_qqq_win_threshold_pct": 15.0,
    "backtest_horizon_days":   365,
}


# ============================================================
# Entry signal detection
# ============================================================

def gap_down_pct(df: pd.DataFrame, idx: int) -> float | None:
    """Today's open vs yesterday's close, as a percentage."""
    if idx <= 0 or idx >= len(df):
        return None
    today_open = float(df["open"].iloc[idx])
    yesterday_close = float(df["close"].iloc[idx - 1])
    if pd.isna(today_open) or pd.isna(yesterday_close) or yesterday_close <= 0:
        return None
    return (today_open / yesterday_close - 1) * 100


def sma(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].rolling(window=window, min_periods=window).mean()


def is_above_sma(df: pd.DataFrame, idx: int, window: int) -> bool | None:
    sma_series = sma(df, window)
    close = df["close"].iloc[idx]
    sma_val = sma_series.iloc[idx]
    if pd.isna(close) or pd.isna(sma_val):
        return None
    return bool(close > sma_val)


def check_entry_today(df: pd.DataFrame) -> dict:
    """Evaluate today's bar against the entry rules. Returns a dict explaining
    each condition and whether the trade fires today."""
    df = df.dropna(subset=["close"])
    if df.empty:
        return {"signal": False, "reason": "no data"}
    idx = len(df) - 1
    g = gap_down_pct(df, idx)
    above = is_above_sma(df, idx, DEFAULTS["sma_window"])
    sma_val = float(sma(df, DEFAULTS["sma_window"]).iloc[idx]) if not pd.isna(sma(df, DEFAULTS["sma_window"]).iloc[idx]) else None
    close = float(df["close"].iloc[idx])
    cond1 = g is not None and g <= DEFAULTS["gap_down_pct_threshold"]
    cond2 = bool(above)
    return {
        "signal":           cond1 and cond2,
        "date":             df.index[idx].date(),
        "today_open":       float(df["open"].iloc[idx]) if not pd.isna(df["open"].iloc[idx]) else None,
        "yesterday_close":  float(df["close"].iloc[idx - 1]) if idx > 0 else None,
        "today_close":      close,
        "gap_down_pct":     round(g, 2) if g is not None else None,
        "sma_100":          round(sma_val, 2) if sma_val is not None else None,
        "above_sma":        cond2,
        "cond_gap_down":    cond1,
        "cond_above_sma":   cond2,
    }


# ============================================================
# Today's LEAPS pick (when signal fires)
# ============================================================

def find_60d_leap(spot: float) -> dict | None:
    """Find the best ~60-delta call near 12-month expiry. Falls back to nearest
    expiry in the (DTE_min, DTE_max) window."""
    today = pd.Timestamp(date.today())
    try:
        expiries = fetch_expiries(TICKER)
    except Exception:
        return None
    candidates: list[tuple[int, str, int]] = []  # (dte, expiry, distance_from_target_window_mid)
    target_mid = (DEFAULTS["target_dte_min"] + DEFAULTS["target_dte_max"]) // 2
    for e in expiries:
        try:
            dte = (pd.Timestamp(e) - today).days
        except Exception:
            continue
        if DEFAULTS["target_dte_min"] <= dte <= DEFAULTS["target_dte_max"]:
            candidates.append((dte, e, abs(dte - target_mid)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    dte, exp, _ = candidates[0]
    chain = fetch_options_chain(TICKER, exp)
    calls = chain.get("calls")
    if calls is None or calls.empty:
        return None

    best = None
    best_distance = 1e9
    for _, row in calls.iterrows():
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        iv = float(row.get("impliedVolatility", 0) or 0) or 0.20
        oi = int(row.get("openInterest", 0) or 0)
        if bid <= 0 or ask <= 0 or oi < 50:
            continue
        strike = float(row["strike"])
        greeks = bs_greeks(spot, strike, dte, iv, "call")
        distance = abs(greeks["delta"] - DEFAULTS["target_delta"])
        if distance < best_distance:
            best_distance = distance
            mid = (bid + ask) / 2
            best = {
                "strike":  strike,
                "expiry":  exp,
                "dte":     dte,
                "bid":     round(bid, 2),
                "ask":     round(ask, 2),
                "mid":     round(mid, 2),
                "iv":      round(iv * 100, 1),
                "delta":   round(greeks["delta"], 2),
                "theta":   round(greeks["theta"], 3),
                "open_interest": oi,
                "cost_per_contract":     round(mid * 100, 2),
                "take_profit_premium":   round(mid * (1 + DEFAULTS["take_profit_pct"] / 100), 2),
                "take_profit_dollars":   round(mid * 100 * DEFAULTS["take_profit_pct"] / 100, 0),
                "breakeven_at_expiry":   round(strike + mid, 2),
                "max_loss":              round(mid * 100, 2),
            }
    return best


# ============================================================
# Open position tracking
# ============================================================

def open_qqq_leaps() -> list[dict]:
    """Return user's open QQQ LEAPS (long calls only), with status."""
    try:
        import user_config
    except Exception:
        return []
    rows = []
    today = pd.Timestamp(date.today())
    for pos in getattr(user_config, "OPTIONS_POSITIONS", []):
        if isinstance(pos, dict):
            p = pos
        elif isinstance(pos, (list, tuple)) and len(pos) >= 7:
            p = {"account_id": pos[0], "ticker": pos[1], "strike": pos[2],
                 "expiry": pos[3], "opt_type": pos[4], "contracts": pos[5],
                 "premium_per_share_avg": pos[6]}
        else:
            continue
        if p["ticker"] != TICKER or p["opt_type"] != "call" or p["contracts"] <= 0:
            continue
        rows.append(p)
    if not rows:
        return []

    # Pull current chain to mark each position
    out = []
    for r in rows:
        try:
            chain = fetch_options_chain(TICKER, str(r["expiry"]))
        except Exception:
            out.append({**r, "error": "no chain"})
            continue
        spot = chain.get("spot")
        calls = chain.get("calls")
        if calls is None or spot is None:
            out.append({**r, "error": "no data"})
            continue
        match = calls[calls["strike"] == r["strike"]]
        if match.empty:
            match = calls.iloc[(calls["strike"] - r["strike"]).abs().argsort()[:1]]
        row = match.iloc[0]
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(row.get("lastPrice", 0))
        iv = float(row.get("impliedVolatility", 0) or 0) or 0.20
        dte = max(0, (pd.Timestamp(r["expiry"]) - today).days)
        greeks = bs_greeks(spot, r["strike"], dte, iv, "call")

        cost_per_contract = float(r["premium_per_share_avg"]) * 100
        current_value     = mid * 100
        pnl               = (current_value - cost_per_contract) * int(r["contracts"])
        pct_gain_premium  = (mid / float(r["premium_per_share_avg"]) - 1) * 100 if r["premium_per_share_avg"] else None
        tp_target         = float(r["premium_per_share_avg"]) * (1 + DEFAULTS["take_profit_pct"] / 100)
        distance_to_tp_pct = (tp_target / mid - 1) * 100 if mid > 0 else None
        out.append({
            "account_id":       r["account_id"],
            "ticker":           TICKER,
            "strike":           r["strike"],
            "expiry":           r["expiry"],
            "contracts":        int(r["contracts"]),
            "premium_paid_avg": float(r["premium_per_share_avg"]),
            "cost_per_contract":  round(cost_per_contract, 2),
            "spot":               round(spot, 2),
            "current_mid":        round(mid, 2),
            "current_value":      round(current_value, 2),
            "pnl_usd":            round(pnl, 0),
            "pct_gain_premium":   round(pct_gain_premium, 1) if pct_gain_premium is not None else None,
            "dte":                dte,
            "delta":              round(greeks["delta"], 2),
            "theta_per_day":      round(greeks["theta"] * 100 * int(r["contracts"]), 2),
            "iv":                 round(iv * 100, 1),
            "tp_target_premium":  round(tp_target, 2),
            "distance_to_tp_pct": round(distance_to_tp_pct, 1) if distance_to_tp_pct is not None else None,
            "tp_hit":             bool(pct_gain_premium is not None and pct_gain_premium >= DEFAULTS["take_profit_pct"]),
            "moneyness_pct":      round((spot / r["strike"] - 1) * 100, 2),
        })
    return out


# ============================================================
# Backtest
# ============================================================

def backtest(df: pd.DataFrame, years: int = 5,
             win_threshold_pct: float | None = None,
             horizon_days: int | None = None) -> dict:
    """Walk historical bars, find signal days, compute forward QQQ returns.

    Proxy for LEAPS take-profit: count a trade as a 'win' if QQQ rises by
    win_threshold_pct (default 15%) within horizon_days (default 365). A 60-delta
    12-month LEAPS on QQQ rising 15% roughly hits the 50% premium gain target.
    Tunable via DEFAULTS or the args.
    """
    df = df.dropna(subset=["open", "close"]).copy()
    win = win_threshold_pct or DEFAULTS["backtest_qqq_win_threshold_pct"]
    horizon = horizon_days or DEFAULTS["backtest_horizon_days"]
    if df.empty:
        return {"signals": [], "n_signals": 0}

    cutoff = df.index[-1] - pd.DateOffset(years=years)
    sma_series = sma(df, DEFAULTS["sma_window"])
    df["gap_pct"] = (df["open"] / df["close"].shift(1) - 1) * 100
    df["above_sma"] = df["close"] > sma_series

    signals = []
    for i in range(1, len(df)):
        if df.index[i] < cutoff:
            continue
        gap = df["gap_pct"].iloc[i]
        above = df["above_sma"].iloc[i]
        if pd.isna(gap) or pd.isna(above):
            continue
        if gap > DEFAULTS["gap_down_pct_threshold"]:
            continue
        if not above:
            continue
        entry_close = float(df["close"].iloc[i])
        # Walk forward up to horizon_days; record outcome
        end_idx = min(i + horizon, len(df) - 1)
        forward = df["close"].iloc[i + 1 : end_idx + 1]
        if forward.empty:
            continue
        max_gain_pct = (forward.max() / entry_close - 1) * 100
        hit_idx = (forward / entry_close - 1) * 100 >= win
        days_to_hit = None
        if hit_idx.any():
            days_to_hit = int((forward.index[hit_idx.argmax()] - df.index[i]).days)
        final_gain_pct = (forward.iloc[-1] / entry_close - 1) * 100
        signals.append({
            "entry_date":    df.index[i].date(),
            "entry_close":   round(entry_close, 2),
            "gap_pct":       round(float(gap), 2),
            "max_gain_pct":  round(float(max_gain_pct), 2),
            "final_gain_pct": round(float(final_gain_pct), 2),
            "win":           bool(hit_idx.any()),
            "days_to_hit":   days_to_hit,
            "days_held":     min(horizon, len(forward)),
        })
    n = len(signals)
    wins = sum(1 for s in signals if s["win"])
    return {
        "signals": signals,
        "n_signals": n,
        "n_wins": wins,
        "hit_rate_pct": round(wins / n * 100, 1) if n else 0.0,
        "mean_max_gain_pct":  round(sum(s["max_gain_pct"] for s in signals) / n, 2) if n else 0.0,
        "mean_final_gain_pct": round(sum(s["final_gain_pct"] for s in signals) / n, 2) if n else 0.0,
        "median_days_to_hit": round(pd.Series([s["days_to_hit"] for s in signals if s["days_to_hit"] is not None]).median(), 0)
                              if any(s["days_to_hit"] for s in signals) else None,
        "win_threshold_pct": win,
        "horizon_days":      horizon,
        "lookback_years":    years,
    }


# ============================================================
# Reporting
# ============================================================

def format_report(signal_info: dict, leap_pick: dict | None,
                  open_positions: list[dict], backtest_result: dict | None) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# QQQ LEAPS Dip-Buy — {today}")
    lines.append("")
    lines.append("Strategy: buy 12-month QQQ calls (~60 delta) when QQQ gaps down ≥ 1% AND price > 100-day average. Take profit at 50% premium gain. No stop loss.")
    lines.append("")

    # ---- Today's signal status ---------------------------------------------
    lines.append("## Today's signal")
    lines.append("")
    if signal_info.get("signal"):
        lines.append("🟢 **SIGNAL: FIRED — enter a new LEAPS today.**")
    else:
        lines.append("⚪ **No signal today.** Conditions:")
    lines.append("")
    lines.append(f"- Date: {signal_info.get('date')}")
    lines.append(f"- QQQ close: ${signal_info.get('today_close', 0):.2f}")
    lines.append(f"- 100-day average: ${signal_info.get('sma_100') or 0:.2f}")
    g = signal_info.get('gap_down_pct')
    if g is not None:
        lines.append(f"- Today's gap: {g:+.2f}% — {'✅ ≤ -1%' if signal_info.get('cond_gap_down') else '❌ above -1% (no gap)'}")
    else:
        lines.append(f"- Today's gap: n/a")
    lines.append(f"- Above 100-day average: {'✅ yes (bull regime)' if signal_info.get('cond_above_sma') else '❌ no (bear regime — skip)'}")
    lines.append("")

    # ---- Recommended trade card when signal fires --------------------------
    if signal_info.get("signal") and leap_pick:
        lines.append("## Recommended trade")
        lines.append("")
        lines.append(f"**Buy 1× QQQ {int(leap_pick['strike'])}C expiring {leap_pick['expiry']}** 🟢")
        lines.append("")
        lines.append(f"- Delta: {leap_pick['delta']:.2f}")
        lines.append(f"- Days to expiration: {leap_pick['dte']}")
        lines.append(f"- Premium (mid): ${leap_pick['mid']:.2f}/share")
        lines.append(f"- Capital required (per contract): ${leap_pick['cost_per_contract']:,.0f}")
        lines.append(f"- Max loss: ${leap_pick['max_loss']:,.0f} (the premium paid)")
        lines.append(f"- Take-profit target: ${leap_pick['take_profit_premium']:.2f}/share = "
                     f"+${leap_pick['take_profit_dollars']:,.0f} per contract")
        lines.append(f"- Breakeven at expiry: ${leap_pick['breakeven_at_expiry']:.2f} (QQQ must close above this)")
        lines.append(f"- Today's option volume liquidity: {leap_pick['open_interest']:,} contracts open interest")
        lines.append(f"- Implied volatility: {leap_pick['iv']:.0f}%")
        lines.append("")

    # ---- Open positions ----------------------------------------------------
    if open_positions:
        lines.append("## Open QQQ LEAPS positions")
        lines.append("")
        lines.append("| Strike | Expiry | Contracts | Premium paid | Spot | Current mid | P&L | Premium gain % | Delta | Days to expiry | Take-profit target | Distance to take-profit | Verdict |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in open_positions:
            if p.get("error"):
                lines.append(f"| ${int(p['strike'])} | {p['expiry']} | {p.get('contracts', '?')} | — | — | — | — | — | — | — | — | — | ERROR {p['error']} |")
                continue
            verdict = "🟢 TAKE PROFIT" if p["tp_hit"] else \
                      ("📅 NEAR EXPIRY" if p["dte"] <= 30 else "⏳ HOLD")
            lines.append(
                f"| ${int(p['strike'])} | {p['expiry']} | {p['contracts']} | "
                f"${p['premium_paid_avg']:.2f} | ${p['spot']:.2f} | ${p['current_mid']:.2f} | "
                f"${p['pnl_usd']:+,.0f} | {p['pct_gain_premium']:+.1f}% | "
                f"{p['delta']:.2f} | {p['dte']} | "
                f"${p['tp_target_premium']:.2f} | {p['distance_to_tp_pct']:+.1f}% | {verdict} |"
            )
        lines.append("")
        # Highlight any TP-eligible
        for p in open_positions:
            if not p.get("error") and p["tp_hit"]:
                lines.append(f"> 🟢 **TAKE-PROFIT TRIGGERED on the {int(p['strike'])}C {p['expiry']}** — "
                             f"premium is up {p['pct_gain_premium']:.1f}% (target was {DEFAULTS['take_profit_pct']}%). "
                             f"Sell to close to lock in ${p['pnl_usd']:+,.0f}.")
                lines.append("")
            elif not p.get("error") and p["dte"] <= 60:
                lines.append(f"> 📅 Position on {int(p['strike'])}C expires in {p['dte']} days. "
                             "If still far from take-profit, decide between rolling forward 12 months "
                             "or letting time decay continue. Theta accelerates inside 60 days.")
                lines.append("")

    # ---- Backtest summary --------------------------------------------------
    if backtest_result and backtest_result.get("n_signals"):
        b = backtest_result
        lines.append(f"## Backtest summary — last {b.get('lookback_years', 5)} years of QQQ history")
        lines.append("")
        lines.append(f"- Signals fired: **{b['n_signals']}**")
        lines.append(f"- Wins (QQQ rose ≥{b['win_threshold_pct']:.0f}% within {b['horizon_days']} days): "
                     f"**{b['n_wins']}** ({b['hit_rate_pct']:.0f}% hit rate)")
        lines.append(f"- Mean maximum gain during horizon: **+{b['mean_max_gain_pct']:.1f}%**")
        lines.append(f"- Mean final gain at horizon end: **{b['mean_final_gain_pct']:+.1f}%**")
        if b.get("median_days_to_hit") is not None:
            lines.append(f"- Median days to take-profit when hit: **{b['median_days_to_hit']:.0f} days**")
        lines.append("")
        lines.append("> Note: backtest uses a QQQ-price proxy (≥15% underlying rise) for LEAPS take-profit. "
                     "Actual LEAPS premium gain depends on implied volatility and time decay; "
                     "the proxy is a reasonable approximation for a 60-delta 12-month call.")
        lines.append("")

    lines.append("> _Term definitions on the dashboard's **Glossary** page._")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def run(do_backtest: bool = False, years: int = 5) -> dict:
    print(f"Loading QQQ history...")
    raw = fetch_many([TICKER], force=False)
    df = raw.get(TICKER)
    if df is None or df.empty:
        raise SystemExit("Could not fetch QQQ history.")

    signal = check_entry_today(df)
    print(f"  Today: gap={signal.get('gap_down_pct')}, above 100-SMA={signal.get('above_sma')}, "
          f"signal={signal.get('signal')}")

    leap = None
    if signal.get("signal"):
        print("  Signal fired — finding ~60-delta 12-month LEAPS...")
        try:
            chain = fetch_options_chain(TICKER)
            spot = chain.get("spot")
            if spot is not None:
                leap = find_60d_leap(spot)
        except Exception as e:
            print(f"  ! chain fetch failed: {e}")

    print("Pricing open QQQ LEAPS positions...")
    open_positions = open_qqq_leaps()
    for p in open_positions:
        if p.get("error"):
            print(f"  ! {p['strike']}C {p['expiry']}: {p['error']}")
        else:
            print(f"  {int(p['strike'])}C {p['expiry']}: mid ${p['current_mid']:.2f} "
                  f"({p['pct_gain_premium']:+.1f}%) P&L ${p['pnl_usd']:+,.0f}")

    bt = None
    if do_backtest:
        print(f"Running backtest on {years}y of QQQ data...")
        bt = backtest(df, years=years)
        print(f"  {bt['n_signals']} signals; {bt['n_wins']} wins ({bt['hit_rate_pct']:.0f}%)")

    report = format_report(signal, leap, open_positions, bt)
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = DATA_DIR / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "qqq_leaps_dipbuy.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_md}\n")
    print(report)
    return {
        "signal": signal, "leap_pick": leap, "open_positions": open_positions,
        "backtest": bt, "report": report, "out_md": str(out_md),
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    do_backtest = "--backtest" in args
    years = 5
    for i, a in enumerate(args):
        if a == "--years" and i + 1 < len(args):
            years = int(args[i + 1])
    run(do_backtest=do_backtest, years=years)
