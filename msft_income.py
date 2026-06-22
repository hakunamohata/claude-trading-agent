"""MSFT covered-call income engine — wheel strategy designed to recycle monthly.

Tier 2A. The goal is PASSIVE INCOME, not optimal one-shot trading. Every month
(or whenever the active call expires / gets bought back), repeat the same rule:
sell N out-of-the-money calls 30-45 DTE at a target delta band that produces
predictable annualized yield.

Target by default: $69K/yr (sum of recurring household obligations the user
wants this income stream to cover):
   $15K margin interest + $24K nanny + $30K school = $69K
Annualized over the ~$673K MSFT share base, that's ~10.3% APY — right in the
classic covered-call wheel range. Override with --target.

Inputs:
  - Current MSFT spot + live option chain (yfinance via options.fetch_options_chain)
  - User's MSFT shares & already-written calls (user_config.OPTIONS_POSITIONS)
  - User's locked-position rule (MSFT shares cannot be sold — only options written)

Outputs (printed + written to data/snapshots/<today>/msft_income.md):
  - Today's recommended trade: N contracts @ strike X, expiry Y, expected premium Z
  - Top 5 alternatives with scoring breakdown
  - Annualized projection: "If you sell this each month, expected annual income ≈ $X,
    assignment probability ≈ Y%"
  - Open MSFT short calls + their current Greeks / capacity used
  - Assignment scenario: what happens if MSFT closes above strike at expiry

CLI:
    python msft_income.py                       # default $15K/yr target
    python msft_income.py --target 30000        # push toward 10% APY
    python msft_income.py --delta-max 0.20      # tighter assignment risk
    python msft_income.py --dte-min 20 --dte-max 60
"""

from __future__ import annotations
import sys
import io
from datetime import datetime, date
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from data_fetch import DATA_DIR
from options import (
    fetch_expiries, fetch_options_chain, bs_greeks, bs_price, RISK_FREE_RATE,
)

TICKER = "MSFT"


# ---------- Configuration knobs (defaults tuned for "passive income") -------

# Recurring household obligations this stream is sized to cover.
# Drives both the target default AND the breakdown shown in the report.
EXPENSE_COMPONENTS = {
    "Margin interest": 15_000,
    "Nanny":           24_000,
    "School":          30_000,
}

DEFAULTS = {
    "target_annual_usd":  sum(EXPENSE_COMPONENTS.values()),  # $69K
    "delta_min":          0.15,
    "delta_max":          0.30,     # 0.15-0.30 = classic wheel range
    "dte_min":            25,
    "dte_max":            50,       # ~monthly cycle
    "min_open_interest":  100,      # filter illiquid strikes
    "max_spread_pct":     20.0,     # ask must be within 20% of mid
    "capacity_max_pct":   70.0,     # HARD CAP — raised 50->70 on 2026-06-21 (covers rising margin interest)
}


# ---------- User position state ---------------------------------------------

def _msft_shares_and_existing_calls() -> dict:
    """Read MSFT shares + existing short calls from user_config."""
    import user_config
    shares = sum(
        qty for (acct, t, qty, *_) in user_config.HOLDINGS_CURRENT if t == TICKER
    )
    existing_calls: list[dict] = []
    for pos in getattr(user_config, "OPTIONS_POSITIONS", []):
        if isinstance(pos, dict):
            p = pos
        elif isinstance(pos, (list, tuple)) and len(pos) == 7:
            p = {"account_id": pos[0], "ticker": pos[1], "strike": pos[2],
                 "expiry": pos[3], "opt_type": pos[4], "contracts": pos[5],
                 "premium_per_share_avg": pos[6]}
        else:
            continue
        if p["ticker"] != TICKER or p["opt_type"] != "call" or p["contracts"] >= 0:
            continue
        existing_calls.append(p)
    contracts_committed = sum(abs(p["contracts"]) for p in existing_calls)
    return {
        "shares": int(shares),
        "existing_calls": existing_calls,
        "contracts_committed": contracts_committed,
        "contracts_available": int(shares // 100) - contracts_committed,
    }


# ---------- Candidate scoring -----------------------------------------------

def _score_candidate(spot: float, strike: float, dte: int, bid: float, ask: float,
                     iv: float, oi: int, vol: int, shares_per_contract: int = 100) -> dict | None:
    """Score one (strike, expiry) candidate. Returns None if it fails liquidity filters."""
    if bid <= 0 or ask <= 0 or oi < DEFAULTS["min_open_interest"]:
        return None
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid * 100 if mid > 0 else 999
    if spread_pct > DEFAULTS["max_spread_pct"]:
        return None
    iv_use = iv if iv > 0 else 0.30
    greeks = bs_greeks(spot, strike, dte, iv_use, "call")
    if greeks["delta"] <= 0 or greeks["delta"] >= 1:
        return None

    # POP for a short call (approx) = 1 - delta
    pop_pct = (1 - greeks["delta"]) * 100

    # Premium per contract (100 shares)
    premium_per_contract = mid * shares_per_contract

    # Income metrics
    premium_per_day = premium_per_contract / dte
    # Annualized yield ON THE SHARE BASE (not on premium): % of spot, annualized
    annualized_yield_pct = (mid / spot) * (365 / dte) * 100
    # Annualized $ income per contract
    annualized_per_contract = premium_per_contract * (365 / dte)

    # Assignment opportunity cost: if assigned, you give up (spot moves above strike)
    breakeven_at_assignment = strike + mid  # effective sell price including premium
    upside_capped_at_pct = (breakeven_at_assignment / spot - 1) * 100

    return {
        "strike": float(strike),
        "dte":    int(dte),
        "bid":    round(bid, 2),
        "ask":    round(ask, 2),
        "mid":    round(mid, 2),
        "iv":     round(iv_use * 100, 1),
        "spread_pct": round(spread_pct, 1),
        "open_interest": int(oi),
        "volume": int(vol or 0),
        "delta":  round(greeks["delta"], 3),
        "theta_per_day": round(greeks["theta"], 3),
        "pop_pct": round(pop_pct, 1),
        "premium_per_contract":  round(premium_per_contract, 0),
        "premium_per_day":       round(premium_per_day, 2),
        "annualized_yield_pct":  round(annualized_yield_pct, 2),
        "annualized_per_contract": round(annualized_per_contract, 0),
        "breakeven_at_assignment": round(breakeven_at_assignment, 2),
        "upside_capped_at_pct":  round(upside_capped_at_pct, 2),
    }


def _color(cand: dict) -> str:
    """POP × R/R rubric per medloh/stockpile SPREADS.md.
    For covered calls, R/R = premium / (spot * upside_capped_at_pct / 100)."""
    pop = cand["pop_pct"]
    capped_dollars = cand["upside_capped_at_pct"] / 100 * 100
    rr = cand["mid"] / capped_dollars if capped_dollars > 0 else 0
    if pop >= 65 and rr >= 0.20:
        return "GREEN"
    if pop >= 55 and rr >= 0.10:
        return "YELLOW"
    return "RED"


def gather_candidates(spot: float,
                      dte_min: int, dte_max: int,
                      delta_min: float, delta_max: float) -> list[dict]:
    """Pull all live MSFT call chains in the DTE window, score every strike,
    return only those whose Greeks fall inside the delta band.
    """
    today = pd.Timestamp(date.today())
    expiries = fetch_expiries(TICKER)
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

    candidates: list[dict] = []
    for dte, exp in in_window:
        chain = fetch_options_chain(TICKER, exp)
        calls = chain.get("calls")
        if calls is None:
            continue
        # Only OTM strikes (strike >= spot)
        otm = calls[calls["strike"] >= spot]
        for _, row in otm.iterrows():
            c = _score_candidate(
                spot=spot, strike=float(row["strike"]), dte=dte,
                bid=float(row.get("bid", 0) or 0),
                ask=float(row.get("ask", 0) or 0),
                iv=float(row.get("impliedVolatility", 0) or 0),
                oi=int(row.get("openInterest", 0) or 0),
                vol=int(row.get("volume", 0) or 0),
            )
            if c is None:
                continue
            if not (delta_min <= c["delta"] <= delta_max):
                continue
            c["expiry"] = exp
            c["color"] = _color(c)
            candidates.append(c)
    return candidates


def recommend(candidates: list[dict], target_annual_usd: float,
              contracts_available: int) -> dict:
    """Pick today's trade + project annual income at that pacing.

    Strategy: among GREEN/YELLOW candidates, take the highest annualized_per_contract
    and figure out how many contracts to sell each cycle to hit the target.
    """
    if not candidates:
        return {"pick": None, "alternatives": [], "monthly_n": 0, "projected_annual": 0}

    # Sort by annualized $/contract, but prefer GREEN over YELLOW
    color_rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    ranked = sorted(
        candidates,
        key=lambda c: (color_rank[c["color"]], -c["annualized_per_contract"]),
    )
    pick = ranked[0]

    # How many contracts per cycle to hit target?
    per_contract_per_year = pick["annualized_per_contract"]
    n_needed = max(1, int(round(target_annual_usd / per_contract_per_year)))
    # HARD CAP on capacity utilization. User's risk discipline overrides target.
    cap_max = max(1, int(contracts_available * DEFAULTS["capacity_max_pct"] / 100))
    n_to_sell = min(n_needed, cap_max)
    capped_by_capacity = (n_needed > cap_max)
    projected_annual = n_to_sell * per_contract_per_year

    return {
        "pick":               pick,
        "alternatives":       ranked[1:6],
        "monthly_n":          n_to_sell,
        "cycles_per_year":    365 / pick["dte"],
        "projected_annual":   projected_annual,
        "target":             target_annual_usd,
        "capacity_used_pct":  n_to_sell / contracts_available * 100 if contracts_available else 0,
        "capacity_cap_pct":   DEFAULTS["capacity_max_pct"],
        "capped_by_capacity": capped_by_capacity,
        "n_needed_unconstrained": n_needed,
    }


# ---------- Reporting --------------------------------------------------------

def format_report(spot: float, position: dict, rec: dict, args: dict) -> str:
    lines = []
    today = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"# MSFT Covered-Call Wheel — {today}")
    lines.append("")
    lines.append(f"**Spot**: ${spot:.2f}   **Target annual income**: ${args['target_annual_usd']:,.0f}")
    lines.append(f"**Strategy**: sell Δ{args['delta_min']:.2f}-{args['delta_max']:.2f} calls "
                 f"{args['dte_min']}-{args['dte_max']} DTE, recycle each cycle")
    lines.append("")

    # Expense breakdown — what this income stream is sized to cover
    target = args["target_annual_usd"]
    default_target = sum(EXPENSE_COMPONENTS.values())
    if abs(target - default_target) < 1:
        lines.append("Sized to cover annual obligations:")
        for label, amount in EXPENSE_COMPONENTS.items():
            lines.append(f"  - {label}: ${amount:,}")
        lines.append(f"  - **Total: ${default_target:,}**")
        lines.append("")

    # Position summary
    lines.append("## Position state")
    lines.append("")
    lines.append(f"- Shares held: **{position['shares']:,}** "
                 f"(${position['shares'] * spot:,.0f})")
    lines.append(f"- Contracts already written: {position['contracts_committed']} "
                 f"({position['contracts_committed'] * 100} sh covered)")
    lines.append(f"- Contracts of remaining capacity: **{position['contracts_available']}**")
    if position["existing_calls"]:
        lines.append("")
        lines.append("Open short calls:")
        for c in position["existing_calls"]:
            lines.append(f"  - {abs(c['contracts'])}× MSFT {int(c['strike'])}C "
                         f"exp {c['expiry']}  (avg premium ${c['premium_per_share_avg']:.2f}/sh)")
    lines.append("")

    if rec["pick"] is None:
        lines.append("## No candidates pass filters today")
        lines.append("")
        lines.append("Try widening --delta-max, --dte-max, or --max-spread-pct.")
        return "\n".join(lines)

    pick = rec["pick"]
    color_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(pick['color'], "")
    lines.append("## Today's recommended trade")
    lines.append("")
    lines.append(f"**Sell {rec['monthly_n']}× MSFT {int(pick['strike'])}C expiring {pick['expiry']}** {color_emoji}")
    lines.append("")
    lines.append("- **Delta**: " f"{pick['delta']:.2f}")
    lines.append("- **Probability of profit**: " f"{pick['pop_pct']:.0f}%")
    lines.append("- **Days to expiration**: " f"{pick['dte']}")
    lines.append("")
    lines.append("> _Term definitions are on the dashboard's **Glossary** page._")
    lines.append("")
    lines.append(f"- Premium: **${pick['mid']:.2f}/sh** "
                 f"(bid ${pick['bid']:.2f} / ask ${pick['ask']:.2f}, spread {pick['spread_pct']:.0f}%)")
    lines.append(f"- Premium captured this cycle: **${pick['premium_per_contract'] * rec['monthly_n']:,.0f}** "
                 f"({rec['monthly_n']} × ${pick['premium_per_contract']:,.0f})")
    lines.append(f"- Annualized yield on shares: **{pick['annualized_yield_pct']:.2f}%**")
    lines.append(f"- Assignment breakeven (effective sell price if called away): "
                 f"${pick['breakeven_at_assignment']:.2f} "
                 f"(+{pick['upside_capped_at_pct']:.1f}% vs spot)")
    lines.append(f"- Theta per day: ${pick['theta_per_day'] * 100 * rec['monthly_n']:.0f}/day collected")
    lines.append(f"- **Open interest**: {pick['open_interest']:,} contracts")
    lines.append(f"- **Today's option volume**: {pick['volume']:,} contracts")
    lines.append(f"- **Implied volatility**: {pick['iv']:.0f}%")
    lines.append("")

    # Annualized projection
    lines.append("## If you recycle this every cycle")
    lines.append("")
    lines.append(f"- Cycles per year (at {pick['dte']} DTE): **{rec['cycles_per_year']:.1f}**")
    lines.append(f"- Projected annual income: **${rec['projected_annual']:,.0f}** "
                 f"(target ${rec['target']:,})")
    lines.append(f"- Capacity used: **{rec['capacity_used_pct']:.0f}%** of "
                 f"{position['contracts_available']} available contracts")
    if rec["capped_by_capacity"]:
        shortfall = rec["target"] - rec["projected_annual"]
        coverage_pct = rec["projected_annual"] / rec["target"] * 100
        lines.append(f"- **Capacity-capped at {rec['capacity_cap_pct']:.0f}%** — would need "
                     f"{rec['n_needed_unconstrained']} contracts to hit target but limited to {rec['monthly_n']}.")
        lines.append(f"- **Covers {coverage_pct:.0f}% of target** "
                     f"(${rec['projected_annual']:,.0f} of ${rec['target']:,.0f}) — "
                     f"**short ${shortfall:,.0f}/yr**.")
        lines.append(f"- Ways to close the gap without breaching the {rec['capacity_cap_pct']:.0f}% cap:")
        lines.append(f"    1. Push to higher delta (try `--delta-max 0.40`) — more premium per contract")
        lines.append(f"    2. Shorten DTE (try `--dte-min 14 --dte-max 30`) — more cycles per year")
        lines.append(f"    3. Cover the remainder with income from other names (see Tier 2B Time Flies)")
    elif rec["projected_annual"] >= rec["target"]:
        delta = rec["projected_annual"] - rec["target"]
        lines.append(f"- **Target HIT** with ${delta:,.0f} of headroom. "
                     f"Could lower delta or sell fewer contracts to reduce assignment risk.")
    else:
        shortfall = rec["target"] - rec["projected_annual"]
        lines.append(f"- **Short ${shortfall:,.0f}** of target — push delta higher or shorten DTE.")
    lines.append("")

    # Capacity discipline reminder (always shown)
    n = rec["monthly_n"]
    expected_itm = round(n * (1 - pick["pop_pct"] / 100), 1)
    buyback_est = expected_itm * spot * 0.05 * 100
    lines.append(f"> **Risk discipline**: hard cap is {rec['capacity_cap_pct']:.0f}% of capacity. "
                 f"At this trade ({rec['monthly_n']} contracts), expect ~{expected_itm:g} "
                 f"contract(s) per cycle to drift ITM. Stress test: a 5% rip in MSFT would cost "
                 f"~${buyback_est:,.0f} to close ITM contracts pre-expiry.")
    lines.append("")

    # Assignment scenario
    lines.append("## Assignment scenario (per contract)")
    lines.append("")
    lines.append(f"If MSFT closes ≥ ${int(pick['strike'])} at expiry ({pick['expiry']}):")
    lines.append(f"- 100 shares get called away at ${int(pick['strike'])} = ${int(pick['strike'])*100:,}")
    lines.append(f"- Plus retained premium ${pick['premium_per_contract']:,.0f}")
    lines.append(f"- Effective sell price ${pick['breakeven_at_assignment']:.2f}/sh "
                 f"({pick['upside_capped_at_pct']:+.1f}% vs spot)")
    lines.append(f"- Probability (model est.): ~{100 - pick['pop_pct']:.0f}%")
    lines.append("")
    lines.append("> Reminder: MSFT shares are LOCKED. If assigned, you'd need to buy back the "
                 "call before expiry. Practical rule: roll out and up when delta drifts above 0.40 "
                 "with under 14 days to expiry.")
    lines.append("")

    # Alternatives
    if rec["alternatives"]:
        lines.append("## Top 5 alternatives")
        lines.append("")
        lines.append("| Strike | Expiry | Days to expiry | Delta | Probability of profit | Premium per share | Annualized yield | Color |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for a in rec["alternatives"]:
            lines.append(
                f"| ${int(a['strike'])} | {a['expiry']} | {a['dte']} | "
                f"{a['delta']:.2f} | {a['pop_pct']:.0f}% | "
                f"${a['mid']:.2f} | {a['annualized_yield_pct']:.1f}% | {a['color']} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------- Main ------------------------------------------------------------

def run(target_annual_usd: float = DEFAULTS["target_annual_usd"],
        delta_min: float = DEFAULTS["delta_min"],
        delta_max: float = DEFAULTS["delta_max"],
        dte_min: int = DEFAULTS["dte_min"],
        dte_max: int = DEFAULTS["dte_max"]) -> dict:
    args = {
        "target_annual_usd": target_annual_usd,
        "delta_min": delta_min, "delta_max": delta_max,
        "dte_min": dte_min,     "dte_max": dte_max,
    }

    print(f"Fetching MSFT chains in {dte_min}-{dte_max} DTE window...")
    position = _msft_shares_and_existing_calls()
    # Use a single chain call to get spot rather than re-fetching
    bootstrap = fetch_options_chain(TICKER)
    spot = bootstrap.get("spot")
    if spot is None:
        raise SystemExit("Could not fetch MSFT spot price.")
    print(f"  spot ${spot:.2f}, shares held {position['shares']:,}, "
          f"contracts available {position['contracts_available']}")

    candidates = gather_candidates(spot, dte_min, dte_max, delta_min, delta_max)
    print(f"  {len(candidates)} candidates passed filters")
    if not candidates:
        print("  Try widening --delta-max or --dte-max.")
        rec = {"pick": None, "alternatives": [], "monthly_n": 0,
               "cycles_per_year": 0, "projected_annual": 0,
               "target": target_annual_usd, "capacity_used_pct": 0}
    else:
        rec = recommend(candidates, target_annual_usd, position["contracts_available"])

    report = format_report(spot, position, rec, args)
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = DATA_DIR / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "msft_income.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_md}\n")
    print(report)
    return {"report": report, "out_md": str(out_md), "recommendation": rec, "position": position, "spot": spot}


if __name__ == "__main__":
    args = sys.argv[1:]
    kwargs = {}
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
    run(**kwargs)
