"""Multi-name covered-call income engine — extension of msft_income.py to
every eligible position across every trade-eligible account.

Tier 2A v2. Goal is the same: cover the user's $69K/yr passive-income target
(margin interest + nanny + school) using covered-call premium. MSFT alone
maxes out at ~$40K/yr under the 50% capacity cap, so this engine adds the
other names the user holds enough of to write covered calls on:

  401k BrokerageLink  — NVDA, PLTR, UPST, SHOP, IREN, DOCN, ARM, ...
  Roth IRA            — PLTR, NBIS, META, TSLA, IONQ, BLSH, AXP, ...
  HSA                 — ANET, PANW, ...
  Individual margin   — MSFT (locked underlying), SPCX
  Individual TOD      — GOOGL, ALAB, AMZN, ...

Eligibility rules:
  1. Account in TRADE_ELIGIBLE_ACCOUNTS (user_config)
  2. Holding >= 100 shares (one-contract minimum)
  3. After 50% capacity cap, remaining contracts >= 1
  4. LB latest action not in {BUY, ADD} — writing a CC there would cap the
     upside we are trying to capture. EXIT/TRIM names are kept as "graceful
     exit" candidates (the CC effectively sets your exit price at strike + premium).
  5. MSFT is special: shares LOCKED, so capacity cap is enforced even harder
     (no assignment allowed — roll required when delta drifts).

Selection algorithm (greedy):
  - Score every OTM strike across every ticker (same scoring as msft_income.py)
  - Sort all GREEN candidates by annualized $/contract descending
  - Walk the list; for each candidate:
      take min(remaining_target / per_contract_annual, ticker_capacity_remaining) contracts
  - Stop when target hit or no more candidates

Output:
  data/snapshots/<today>/cc_income.md  — full report
  Prints to stdout (so daily_run.py can capture it).

CLI:
    python cc_income_engine.py                        # default $69K, 0.15-0.30 delta, 25-50 DTE
    python cc_income_engine.py --target 50000
    python cc_income_engine.py --delta-max 0.35
    python cc_income_engine.py --tickers MSFT,NVDA,PLTR  # restrict universe
"""

from __future__ import annotations
import sys
import io
import json
from datetime import datetime, date
from pathlib import Path
from typing import Iterable
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from data_fetch import DATA_DIR, fetch_many
from options import fetch_expiries, fetch_options_chain, bs_greeks
from breakout import build_features, ad_label
from earnings import build_earnings_cache
from msft_income import (
    EXPENSE_COMPONENTS,
    DEFAULTS as MSFT_DEFAULTS,
    _score_candidate,
)


BENCHMARK = "QQQ"


# ---------- Configuration ---------------------------------------------------

DEFAULTS = {
    "target_annual_usd":  sum(EXPENSE_COMPONENTS.values()),  # $69K
    "delta_min":          0.15,
    "delta_max":          0.30,
    "dte_min":            25,
    "dte_max":            50,
    "min_open_interest":  100,
    "max_spread_pct":     20.0,
    "capacity_max_pct":   50.0,         # hard cap per ticker
    "min_shares_per_100": 1,            # floor: need at least 1 contract of capacity after cap
    "max_per_ticker_share_pct": 40.0,   # no single ticker funds more than 40% of target (diversification)
}

NON_EQUITY = {"FDRXX", "SPAXX", "CASH_ROTH", "CASH_TOD", "CASH_HSA"}


# ---------- User state -------------------------------------------------------

def _consolidated_holdings() -> list[dict]:
    """Return list of {account_id, ticker, shares} aggregated across HOLDINGS_CURRENT."""
    import user_config
    agg: dict[tuple[str, str], float] = {}
    for (acct, t, qty, *_) in getattr(user_config, "HOLDINGS_CURRENT", []):
        if t in NON_EQUITY:
            continue
        key = (acct, t)
        agg[key] = agg.get(key, 0) + float(qty)
    return [{"account_id": k[0], "ticker": k[1], "shares": v} for k, v in agg.items()]


def _existing_short_calls_by_ticker() -> dict[str, int]:
    """Count short-call contracts already written per ticker."""
    import user_config
    out: dict[str, int] = {}
    for pos in getattr(user_config, "OPTIONS_POSITIONS", []):
        if isinstance(pos, dict):
            p = pos
        elif isinstance(pos, (list, tuple)) and len(pos) >= 6:
            p = {"ticker": pos[1], "opt_type": pos[4], "contracts": pos[5]}
        else:
            continue
        if p["opt_type"] == "call" and p["contracts"] < 0:
            out[p["ticker"]] = out.get(p["ticker"], 0) + abs(p["contracts"])
    return out


def _trade_eligible_accounts() -> set[str]:
    import user_config
    return set(getattr(user_config, "TRADE_ELIGIBLE_ACCOUNTS", set()))


def _locked() -> set[str]:
    import user_config
    return set(getattr(user_config, "LOCKED_POSITIONS", set()))


def _account_label(acct_id: str) -> str:
    import user_config
    info = getattr(user_config, "ACCOUNT_INFO", {}).get(acct_id, {})
    return info.get("label", acct_id)


# ---------- Latest LB judgments ---------------------------------------------

def _latest_lb_actions() -> dict[tuple[str, str], dict]:
    """Most-recent judgments_portfolio.jsonl, keyed by (ticker, account_label_short)."""
    snaps = sorted((DATA_DIR / "snapshots").glob("*"))
    for snap in reversed(snaps):
        f = snap / "judgments_portfolio.jsonl"
        if not f.exists():
            continue
        out: dict[tuple[str, str], dict] = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pm = rec.get("result", {}).get("pm", {})
            if not pm.get("action"):
                continue
            account_short = rec.get("account", "?").split()[0] if rec.get("account") else "?"
            out[(rec["ticker"], account_short)] = {
                "action":     pm["action"],
                "score":      pm.get("final_score"),
                "confidence": pm.get("confidence"),
                "as_of":      snap.name,
            }
        if out:
            return out
    return {}


# ---------- Eligibility -----------------------------------------------------

def build_inventory(only_tickers: set[str] | None = None) -> list[dict]:
    """One row per (ticker, account) with capacity + LB-action context."""
    holdings = _consolidated_holdings()
    eligible_accts = _trade_eligible_accounts()
    existing = _existing_short_calls_by_ticker()
    lb_actions = _latest_lb_actions()
    locked = _locked()

    # Aggregate existing CCs by ticker (they could span multiple accounts)
    out: list[dict] = []
    for h in holdings:
        if h["account_id"] not in eligible_accts:
            continue
        if only_tickers and h["ticker"] not in only_tickers:
            continue
        shares = int(h["shares"])
        cap_total = shares // 100
        if cap_total < DEFAULTS["min_shares_per_100"]:
            continue
        # Existing CC consumption is portfolio-wide on a ticker — but the contract
        # is held in ONE account. For simplicity we treat existing CCs as
        # subtracting from THIS account's capacity proportional to shares. The
        # user almost always writes CCs in the same account as the shares.
        already = existing.get(h["ticker"], 0)
        # If the user has 1775 MSFT shares in Individual and 3 contracts written
        # there, that consumes 3 contracts of THAT account's capacity. We assume
        # CCs were written in the same account that has the most shares.
        cap_remaining_raw = max(0, cap_total - already)
        cap_after_50 = int(cap_remaining_raw * DEFAULTS["capacity_max_pct"] / 100)
        if cap_after_50 < DEFAULTS["min_shares_per_100"]:
            # Note it but skip recommendations
            out.append({
                "ticker":          h["ticker"],
                "account_id":      h["account_id"],
                "account_label":   _account_label(h["account_id"]),
                "shares":          shares,
                "contracts_total":     cap_total,
                "contracts_existing":  min(already, cap_total),
                "contracts_available": 0,
                "lb_action":       lb_actions.get((h["ticker"], _account_label(h["account_id"]).split()[0]), {}).get("action"),
                "skipped_reason":  f"no capacity after 50% cap ({cap_remaining_raw} remaining)",
                "locked":          h["ticker"] in locked,
            })
            continue
        lb = lb_actions.get((h["ticker"], _account_label(h["account_id"]).split()[0]), {})
        out.append({
            "ticker":          h["ticker"],
            "account_id":      h["account_id"],
            "account_label":   _account_label(h["account_id"]),
            "shares":          shares,
            "contracts_total":     cap_total,
            "contracts_existing":  min(already, cap_total),
            "contracts_available": cap_after_50,
            "lb_action":       lb.get("action"),
            "lb_score":        lb.get("score"),
            "lb_as_of":        lb.get("as_of"),
            "skipped_reason":  None,
            "locked":          h["ticker"] in locked,
        })
    return out


# ---------- Candidate gathering ---------------------------------------------

def _to_float(v, default=0.0) -> float:
    try:
        f = float(v)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


def _to_int(v, default=0) -> int:
    return int(_to_float(v, default))


# ---------- Risk overlay: technicals + earnings -----------------------------

def compute_technical_features(ticker: str, raw: dict, bench: pd.Series) -> dict:
    """One-shot features for the risk overlay. Quiet on missing data.

    Anchors to the latest VALID (non-NaN close) bar — handles the same stale
    intraday-NaN-row issue that originally broke scanner.py.
    """
    df = raw.get(ticker)
    if df is None or df.empty:
        return {}
    try:
        feat = build_features(df, bench)
    except Exception:
        return {}
    feat = feat.dropna(subset=["close"])
    if feat.empty:
        return {}
    last = feat.iloc[-1]
    close = float(last["close"])
    ema50 = float(last["ema_50"]) if not pd.isna(last.get("ema_50")) else close
    ret = lambda n: float((df["close"].iloc[-1] / df["close"].iloc[-1 - n] - 1) * 100) \
                       if len(df) >= n + 1 else None
    out = {
        "spot":            close,
        "pct_above_50ema": (close - ema50) / ema50 * 100 if ema50 > 0 else 0.0,
        "ret_5d":          ret(5),
        "ret_20d":         ret(20),
        "ret_60d":         ret(60),
        "ad_grade":        ad_label(last.get("ad_score_65")),
        "ad_score":        float(last["ad_score_65"]) if not pd.isna(last.get("ad_score_65")) else None,
        "atr_pct":         float(last["atr_pct"]) * 100 if not pd.isna(last.get("atr_pct")) else None,
    }
    # Earnings-history features: return since last print + prior-year same-quarter move
    try:
        out.update(_earnings_history_features(ticker, df))
    except Exception:
        pass
    return out


def _earnings_history_features(ticker: str, df: pd.DataFrame) -> dict:
    """Pull prior earnings dates and compute:
       - since_last_earnings_pct: today's close vs close on most recent past earnings
       - prior_year_earnings_move_pct: close-to-close move around earnings ~365d ago
                                       (proxy for typical event vol on same fiscal quarter)
    """
    import yfinance as yf
    try:
        eh = yf.Ticker(ticker).earnings_dates
    except Exception:
        return {}
    if eh is None or eh.empty:
        return {}
    tz = eh.index.tz
    now = pd.Timestamp.now(tz=tz)
    past = eh[eh.index < now]
    if past.empty:
        return {}

    out: dict = {}

    # Return since last earnings: from close at-or-just-before that earnings day to today
    last_e = past.index.max().tz_localize(None)
    try:
        idx = df.index.searchsorted(last_e, side="right") - 1
        if 0 <= idx < len(df) - 1:
            base = float(df["close"].iloc[idx])
            today_close = float(df["close"].iloc[-1])
            out["since_last_earnings_pct"]  = (today_close / base - 1) * 100
            out["last_earnings_date"]       = str(last_e.date())
    except Exception:
        pass

    # Prior-year same-quarter earnings move (close-to-close around that day)
    target = last_e - pd.Timedelta(days=365)
    # Find the earnings date closest to (target ± 45d)
    window = past[(past.index >= pd.Timestamp(target - pd.Timedelta(days=45), tz=tz)) &
                  (past.index <= pd.Timestamp(target + pd.Timedelta(days=45), tz=tz))]
    if not window.empty:
        py_e = window.index[0].tz_localize(None)
        try:
            idx = df.index.searchsorted(py_e, side="right") - 1
            if 0 <= idx and idx + 1 < len(df):
                pre  = float(df["close"].iloc[max(0, idx - 1)])
                post = float(df["close"].iloc[idx + 1])
                out["prior_year_earnings_move_pct"] = (post / pre - 1) * 100
                out["prior_year_earnings_date"]     = str(py_e.date())
        except Exception:
            pass
    return out


def compute_risk_adjustment(tech: dict, spans_earnings: bool) -> tuple[float, list[str]]:
    """Return (pop_adjustment_pp, reasons).

    Positive adjustment = SAFER for the call seller (lower assignment risk than BS says).
    Negative adjustment = MORE DANGEROUS than BS says.

    Rules (calibrated against the ALAB-vs-PLTR mismatch observed today):
      pct_above_50ema:  <0 → +5 (downtrend), >25 → -15 (extended), >10 → -5
      5-day return:     >+10% → -10 (momentum extension), <-3% → +5 (weakness)
      A/D grade:        A/B → -5 (accumulation continuing), D/E → +5 (distribution = topping)
      spans_earnings:   -10 (binary event risk through expiry)
    """
    adj = 0.0
    reasons: list[str] = []
    pa50 = tech.get("pct_above_50ema")
    if pa50 is not None:
        if pa50 < 0:
            adj += 5;  reasons.append(f"+5 (downtrend, {pa50:.1f}% vs 50EMA)")
        elif pa50 > 25:
            adj -= 15; reasons.append(f"-15 (extended {pa50:.1f}% above 50EMA)")
        elif pa50 > 10:
            adj -= 5;  reasons.append(f"-5 ({pa50:.1f}% above 50EMA)")
    r5 = tech.get("ret_5d")
    if r5 is not None:
        if r5 > 10:
            adj -= 10; reasons.append(f"-10 (+{r5:.1f}% in 5d — momentum)")
        elif r5 < -3:
            adj += 5;  reasons.append(f"+5 ({r5:.1f}% 5d weakness)")
    grade = tech.get("ad_grade")
    if grade in ("A", "B"):
        adj -= 5; reasons.append(f"-5 (A/D {grade} accumulation)")
    elif grade in ("D", "E"):
        adj += 5; reasons.append(f"+5 (A/D {grade} distribution)")
    if spans_earnings:
        adj -= 10; reasons.append("-10 (expiry spans earnings)")
        # Sharpen the earnings penalty using historical context
        since_last = tech.get("since_last_earnings_pct")
        if since_last is not None:
            if since_last > 25:
                adj -= 5; reasons.append(f"-5 (+{since_last:.0f}% since last earnings — sell-the-news risk)")
            elif since_last < -15:
                adj += 5; reasons.append(f"+5 ({since_last:.0f}% since last earnings — expectations reset)")
        py_move = tech.get("prior_year_earnings_move_pct")
        if py_move is not None and abs(py_move) >= 8:
            # Prior-year same-quarter print moved >=8% — expect a similar event-vol surprise
            adj -= 5; reasons.append(f"-5 (prior-year same-Q earnings moved {py_move:+.1f}% — high event vol)")
    return adj, reasons


def _color_adjusted(cand: dict) -> str:
    """Re-color GREEN/YELLOW/RED on ADJUSTED POP, not raw delta."""
    pop = cand["adjusted_pop"]
    rr = cand["mid"] / (cand["upside_capped_at_pct"]) if cand["upside_capped_at_pct"] > 0 else 0
    if pop >= 65 and rr >= 0.20:
        return "GREEN"
    if pop >= 55 and rr >= 0.10:
        return "YELLOW"
    return "RED"


def _risk_verdict(adj: float, spans_earnings: bool) -> str:
    """Human-readable verdict on the per-trade risk."""
    if spans_earnings:
        return "EARNINGS-RISK"
    if adj >= 5:
        return "SAFE"
    if adj >= 0:
        return "MODERATE"
    if adj >= -10:
        return "AGGRESSIVE"
    return "DANGEROUS"


def gather_for_ticker(ticker: str, spot: float, dte_min: int, dte_max: int,
                      delta_min: float, delta_max: float,
                      technicals: dict | None = None,
                      next_earnings: pd.Timestamp | None = None) -> list[dict]:
    today = pd.Timestamp(date.today())
    try:
        expiries = fetch_expiries(ticker)
    except Exception:
        return []
    in_window = []
    for e in expiries:
        try:
            d = (pd.Timestamp(e) - today).days
        except Exception:
            continue
        if dte_min <= d <= dte_max:
            in_window.append((d, e))
    if not in_window:
        return []

    tech = technicals or {}
    candidates: list[dict] = []
    for dte, exp in in_window:
        try:
            chain = fetch_options_chain(ticker, exp)
        except Exception:
            continue
        calls = chain.get("calls")
        if calls is None or calls.empty:
            continue
        # Does this expiry span the next earnings event?
        spans_earnings = False
        if next_earnings is not None and not pd.isna(next_earnings):
            try:
                spans_earnings = (today <= next_earnings <= pd.Timestamp(exp))
            except Exception:
                spans_earnings = False
        otm = calls[calls["strike"] >= spot]
        for _, row in otm.iterrows():
            try:
                c = _score_candidate(
                    spot=spot, strike=_to_float(row["strike"]), dte=dte,
                    bid=_to_float(row.get("bid")),
                    ask=_to_float(row.get("ask")),
                    iv=_to_float(row.get("impliedVolatility")),
                    oi=_to_int(row.get("openInterest")),
                    vol=_to_int(row.get("volume")),
                )
            except Exception:
                continue
            if c is None:
                continue
            if not (delta_min <= c["delta"] <= delta_max):
                continue

            # Risk overlay: adjust POP for technicals + earnings event risk
            adj, reasons = compute_risk_adjustment(tech, spans_earnings)
            adjusted_pop = max(0.0, min(100.0, c["pop_pct"] + adj))

            c.update({
                "ticker":           ticker,
                "expiry":           exp,
                "spans_earnings":   spans_earnings,
                "next_earnings":    str(next_earnings.date()) if next_earnings is not None and not pd.isna(next_earnings) else None,
                "pop_adjustment":   round(adj, 1),
                "adjusted_pop":     round(adjusted_pop, 1),
                "risk_reasons":     reasons,
                "risk_verdict":     _risk_verdict(adj, spans_earnings),
            })
            c["color"] = _color_adjusted(c)
            candidates.append(c)
    return candidates


# ---------- Greedy selection ------------------------------------------------

def select_mix(per_ticker_candidates: dict[str, list[dict]],
               inventory: list[dict],
               target_annual_usd: float) -> dict:
    """Greedy mix selection.

    1. For each ticker keep only its BEST candidate (highest annualized).
       Rationale: writing two different strikes on the same name same cycle
       is awkward operationally; pick one strike per ticker per cycle.
    2. Sort all best-of-ticker candidates by (color rank, annualized DESC).
    3. Walk in order, allocate contracts until target hit or capacity exhausted.
       Cap each ticker's allocation at max_per_ticker_share_pct of target.
    """
    color_rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    inv_by_ticker = {row["ticker"]: row for row in inventory}

    bests: list[dict] = []
    for ticker, cands in per_ticker_candidates.items():
        if not cands:
            continue
        # Only GREEN by default (50% cap is risk discipline — don't relax it via color)
        green = [c for c in cands if c["color"] == "GREEN"]
        pool = green or cands  # if no green, take best yellow as fallback
        best = sorted(pool, key=lambda c: (color_rank[c["color"]], -c["annualized_per_contract"]))[0]
        bests.append(best)

    bests.sort(key=lambda c: (color_rank[c["color"]], -c["annualized_per_contract"]))

    selected: list[dict] = []
    remaining = target_annual_usd
    max_per_ticker = target_annual_usd * DEFAULTS["max_per_ticker_share_pct"] / 100

    for cand in bests:
        if remaining <= 0:
            break
        inv = inv_by_ticker.get(cand["ticker"])
        if not inv or inv.get("contracts_available", 0) <= 0:
            continue
        cap = inv["contracts_available"]
        per_yr = cand["annualized_per_contract"]
        if per_yr <= 0:
            continue
        # Diversification cap per ticker
        max_contracts_this_ticker = min(cap, int(max_per_ticker / per_yr) or 1)
        # Contracts needed for remaining target
        n_needed = max(1, int(round(remaining / per_yr)))
        n = min(n_needed, max_contracts_this_ticker)
        if n <= 0:
            continue
        contribution = n * per_yr
        selected.append({
            **cand,
            "account_label":     inv["account_label"],
            "account_id":        inv["account_id"],
            "contracts_chosen":  n,
            "contracts_available": cap,
            "annual_contribution": round(contribution, 0),
            "premium_this_cycle":  round(n * cand["premium_per_contract"], 0),
            "lb_action":         inv.get("lb_action"),
            "locked":            inv.get("locked", False),
        })
        remaining -= contribution

    total_annual = sum(s["annual_contribution"] for s in selected)
    picked_tickers = {s["ticker"] for s in selected}
    return {
        "selected":         selected,
        "skipped_no_green": [t for t, c in per_ticker_candidates.items()
                             if c and all(x["color"] != "GREEN" for x in c)
                             and t not in picked_tickers],
        "no_candidates":    [t for t, c in per_ticker_candidates.items() if not c],
        "total_annual":     total_annual,
        "target":           target_annual_usd,
        "shortfall":        max(0, target_annual_usd - total_annual),
        "coverage_pct":     total_annual / target_annual_usd * 100 if target_annual_usd else 0,
    }


# ---------- Reporting -------------------------------------------------------

def format_report(inventory: list[dict], mix: dict, spots: dict[str, float],
                  args: dict, lb_actions: dict) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# Covered-Call Income Engine — {today}")
    lines.append("")
    lines.append(f"**Target annual income**: ${args['target_annual_usd']:,.0f}")
    lines.append(f"**Strategy**: sell Δ{args['delta_min']:.2f}-{args['delta_max']:.2f} calls "
                 f"{args['dte_min']}-{args['dte_max']} DTE per ticker, recycle each cycle")
    lines.append(f"**Per-ticker capacity cap**: {DEFAULTS['capacity_max_pct']:.0f}% of available contracts")
    lines.append(f"**Per-ticker income cap**: {DEFAULTS['max_per_ticker_share_pct']:.0f}% of target (diversification)")
    lines.append("")
    lines.append("Sized to cover annual obligations:")
    for label, amount in EXPENSE_COMPONENTS.items():
        lines.append(f"  - {label}: ${amount:,}")
    lines.append(f"  - **Total: ${sum(EXPENSE_COMPONENTS.values()):,}**")
    lines.append("")
    lines.append("> _Term definitions (delta, probability of profit, risk verdicts, etc.) on the dashboard's **Glossary** page._")
    lines.append("")

    # --- Selected trades -----------------------------------------------------
    sel = mix["selected"]
    lines.append("## Today's recommended trades")
    lines.append("")
    if not sel:
        lines.append("_No GREEN candidates met criteria. Try widening `--delta-max` or `--dte-max`._")
    else:
        lines.append("| Ticker | Account | Contracts | Strike | Expiry | Days to expiry | Delta | Probability of profit (Black-Scholes) | Probability of profit (adjusted) | Risk verdict | Premium per share | Cycle premium | Annualized premium | LB action |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in sel:
            lb_tag = s.get("lb_action") or "—"
            if s["locked"]:
                lb_tag += "🔒"
            verdict_tag = s.get("risk_verdict", "—")
            if s.get("spans_earnings"):
                verdict_tag += " 📅"
            lines.append(
                f"| {s['ticker']} | {s['account_label'].split()[0]} | {s['contracts_chosen']} | "
                f"${int(s['strike'])} | {s['expiry']} | {s['dte']} | "
                f"{s['delta']:.2f} | {s['pop_pct']:.0f}% | "
                f"{s.get('adjusted_pop', s['pop_pct']):.0f}% | "
                f"{verdict_tag} | "
                f"${s['mid']:.2f} | ${s['premium_this_cycle']:,.0f} | "
                f"${s['annual_contribution']:,.0f} | {lb_tag} |"
            )
        lines.append("")
        # Per-trade risk reasoning
        lines.append("### Risk overlay detail")
        lines.append("")
        for s in sel:
            line = f"- **{s['ticker']} ${int(s['strike'])}C {s['expiry']}**: "
            line += f"BS POP {s['pop_pct']:.0f}% → adj {s.get('adjusted_pop', s['pop_pct']):.0f}%"
            reasons = s.get("risk_reasons") or []
            if reasons:
                line += " (" + "; ".join(reasons) + ")"
            if s.get("spans_earnings"):
                line += f" — 📅 earnings {s.get('next_earnings')} inside contract life"
            lines.append(line)
        lines.append("")

        # TRIM-as-CC / EXIT-as-CC explicit cash math
        trim_exits = [s for s in sel if s.get("lb_action") in ("TRIM", "EXIT")]
        if trim_exits:
            lines.append("### TRIM/EXIT-as-CC — cash math vs spot sell")
            lines.append("")
            lines.append("When LB rates a position TRIM or EXIT, writing a covered call is an alternative to a spot sale. ")
            lines.append("The CC retains shares + collects premium up front. If assigned, you exit at strike + premium (typically better than spot). ")
            lines.append("If not assigned, the premium effectively averages down your basis.")
            lines.append("")
            for s in trim_exits:
                shares = s["contracts_chosen"] * 100
                spot = spots.get(s["ticker"])
                if spot is None:
                    continue
                strike = s["strike"]
                premium_per_sh = s["mid"]
                adj_pop = s.get("adjusted_pop", s["pop_pct"]) / 100  # probability of NOT being assigned
                p_assigned = 1 - adj_pop
                spot_sell_value = shares * spot
                if_assigned_value = shares * strike + shares * premium_per_sh
                assigned_delta = if_assigned_value - spot_sell_value
                if_not_assigned_premium = shares * premium_per_sh
                expected_value = adj_pop * (spot_sell_value + if_not_assigned_premium) + p_assigned * if_assigned_value

                tag = "TRIM-as-CC" if s["lb_action"] == "TRIM" else "EXIT-as-CC"
                lines.append(f"- **{tag} {s['ticker']}** ({shares} shares):")
                lines.append(f"    - Spot sell today: ${spot_sell_value:,.0f} ({shares} × ${spot:.2f})")
                lines.append(f"    - Write {s['contracts_chosen']}× ${int(strike)}C @ ${premium_per_sh:.2f} → collect **${if_not_assigned_premium:,.0f}** today")
                lines.append(f"    - If assigned ({p_assigned*100:.0f}% prob): sell at ${if_assigned_value:,.0f} = "
                             f"**${assigned_delta:+,.0f} vs spot sell**")
                lines.append(f"    - If not assigned ({adj_pop*100:.0f}% prob): keep shares + ${if_not_assigned_premium:,.0f} premium → "
                             f"basis on these {shares} shares drops by ${premium_per_sh:.2f}/sh")
                lines.append(f"    - Expected value: **${expected_value:,.0f}** (vs spot sell ${spot_sell_value:,.0f}) — "
                             f"**+${expected_value - spot_sell_value:,.0f} edge**")
                lines.append("")

    # --- Coverage summary ---------------------------------------------------
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Projected annual income: **${mix['total_annual']:,.0f}** of "
                 f"${args['target_annual_usd']:,.0f} target (**{mix['coverage_pct']:.0f}%**)")
    if mix["shortfall"] > 0:
        lines.append(f"- **Short ${mix['shortfall']:,.0f}/yr** — try `--delta-max 0.35` or `--dte-min 14 --dte-max 30`")
    else:
        lines.append(f"- **Target HIT** with ${-mix['shortfall']:,.0f} of headroom")
    lines.append("")

    # --- Per-account breakdown ---------------------------------------------
    if sel:
        by_acct: dict[str, list[dict]] = {}
        for s in sel:
            by_acct.setdefault(s["account_label"], []).append(s)
        lines.append("## Per-account income")
        lines.append("")
        for acct, items in sorted(by_acct.items()):
            total = sum(x["annual_contribution"] for x in items)
            cycle = sum(x["premium_this_cycle"] for x in items)
            tickers = ", ".join(f"{x['ticker']}×{x['contracts_chosen']}" for x in items)
            lines.append(f"- **{acct}** — ${total:,.0f}/yr (${cycle:,.0f} this cycle) — {tickers}")
        lines.append("")

    # --- Inventory ----------------------------------------------------------
    lines.append("## Eligibility inventory")
    lines.append("")
    lines.append("| Ticker | Account | Shares | Cap | Existing | Available | LB | Status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    inv_sorted = sorted(inventory, key=lambda r: (-r.get("contracts_available", 0), r["ticker"]))
    for row in inv_sorted:
        status = "PICKED" if any(s["ticker"] == row["ticker"] and s["account_id"] == row["account_id"] for s in sel) else \
                 ("LOCKED" if row.get("locked") else (row.get("skipped_reason") or "—"))
        lb = row.get("lb_action") or "—"
        lines.append(
            f"| {row['ticker']} | {row['account_label'].split()[0]} | "
            f"{row['shares']:,} | {row['contracts_total']} | "
            f"{row['contracts_existing']} | {row['contracts_available']} | "
            f"{lb} | {status} |"
        )
    lines.append("")

    # --- Skipped ----------------------------------------------------------
    if mix["no_candidates"]:
        lines.append("## Tickers without chain candidates in window")
        lines.append("- " + ", ".join(mix["no_candidates"]))
        lines.append("")
    if mix["skipped_no_green"]:
        lines.append("## Tickers with only YELLOW/RED candidates (skipped)")
        lines.append("- " + ", ".join(mix["skipped_no_green"]))
        lines.append("")

    # --- Risk discipline reminder -----------------------------------------
    lines.append("> **Risk discipline**: 50% per-ticker capacity cap. "
                 "Existing OPTIONS_POSITIONS in `user_config.py` are subtracted before applying the cap. "
                 "Locked positions (🔒) cannot be assigned — roll out and up when delta drifts above 0.40 with under 14 days to expiry.")
    return "\n".join(lines)


# ---------- Main ------------------------------------------------------------

def run(target_annual_usd: float = DEFAULTS["target_annual_usd"],
        delta_min: float = DEFAULTS["delta_min"],
        delta_max: float = DEFAULTS["delta_max"],
        dte_min: int = DEFAULTS["dte_min"],
        dte_max: int = DEFAULTS["dte_max"],
        only_tickers: Iterable[str] | None = None) -> dict:
    args = {
        "target_annual_usd": target_annual_usd,
        "delta_min": delta_min, "delta_max": delta_max,
        "dte_min": dte_min,     "dte_max": dte_max,
    }

    only = set(only_tickers) if only_tickers else None
    inventory = build_inventory(only_tickers=only)
    if not inventory:
        print("No eligible holdings found.")
        return {}

    eligible = [r for r in inventory if r["contracts_available"] > 0 and r.get("lb_action") not in ("BUY", "ADD")]
    print(f"Eligible tickers for new CCs: {len(eligible)} "
          f"(of {len(inventory)} held in trade-eligible accounts)")
    for r in eligible:
        flag = " [LOCKED]" if r["locked"] else ""
        lb = r.get("lb_action") or "—"
        print(f"  {r['ticker']:6s} {r['account_label'].split()[0]:<12} "
              f"shares={r['shares']:>6,} cap={r['contracts_available']} (after 50% + existing) "
              f"LB={lb}{flag}")

    # Pre-load OHLCV and earnings cache for all eligible tickers in ONE batch.
    # fetch_many is cached, so this is free if already pulled today.
    tickers = sorted({r["ticker"] for r in eligible} | {BENCHMARK})
    print(f"Loading OHLCV + technicals for {len(tickers)} tickers...")
    raw = fetch_many(tickers, force=False)
    bench = raw[BENCHMARK]["close"] if BENCHMARK in raw else None
    print("Building earnings cache...")
    earn_df = build_earnings_cache([r["ticker"] for r in eligible])

    # Per-ticker technicals + earnings date (computed once, reused per candidate)
    tech_by_ticker: dict[str, dict] = {}
    next_earn_by_ticker: dict[str, pd.Timestamp | None] = {}
    for r in eligible:
        t = r["ticker"]
        if t in tech_by_ticker:
            continue
        tech_by_ticker[t] = compute_technical_features(t, raw, bench) if bench is not None else {}
        ne = earn_df.loc[t, "next_earnings"] if t in earn_df.index else None
        next_earn_by_ticker[t] = pd.Timestamp(ne) if ne is not None and not pd.isna(ne) else None

    per_ticker_candidates: dict[str, list[dict]] = {}
    spots: dict[str, float] = {}
    for r in eligible:
        t = r["ticker"]
        if t in per_ticker_candidates:
            continue
        print(f"  fetching {t} chains...", flush=True)
        try:
            chain = fetch_options_chain(t)
            spot = chain.get("spot")
        except Exception as e:
            print(f"    ! {t}: {e}")
            per_ticker_candidates[t] = []
            continue
        if spot is None:
            per_ticker_candidates[t] = []
            continue
        spots[t] = spot
        try:
            per_ticker_candidates[t] = gather_for_ticker(
                t, spot, dte_min, dte_max, delta_min, delta_max,
                technicals=tech_by_ticker.get(t),
                next_earnings=next_earn_by_ticker.get(t),
            )
        except Exception as e:
            print(f"    ! {t} gather failed: {e}")
            per_ticker_candidates[t] = []

    mix = select_mix(per_ticker_candidates, eligible, target_annual_usd)
    report = format_report(inventory, mix, spots, args, _latest_lb_actions())

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = DATA_DIR / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "cc_income.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_md}\n")
    print(report)
    return {"report": report, "out_md": str(out_md), "mix": mix, "inventory": inventory}


if __name__ == "__main__":
    args = sys.argv[1:]
    kwargs: dict = {}
    for i, a in enumerate(args):
        if a == "--target" and i + 1 < len(args):
            kwargs["target_annual_usd"] = float(args[i + 1])
        elif a == "--delta-max" and i + 1 < len(args):
            kwargs["delta_max"] = float(args[i + 1])
        elif a == "--delta-min" and i + 1 < len(args):
            kwargs["delta_min"] = float(args[i + 1])
        elif a == "--dte-min" and i + 1 < len(args):
            kwargs["dte_min"] = int(args[i + 1])
        elif a == "--dte-max" and i + 1 < len(args):
            kwargs["dte_max"] = int(args[i + 1])
        elif a == "--tickers" and i + 1 < len(args):
            kwargs["only_tickers"] = [t.strip().upper() for t in args[i + 1].split(",")]
    run(**kwargs)
