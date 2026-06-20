"""Buy-write screener — identify NEW names worth buying 100+ shares and
immediately writing a covered call on.

This is the inverse of cc_income_engine.py:
  - cc_income_engine.py  → CC on shares you ALREADY OWN
  - cc_buywrite.py       → CC on shares you would BUY to write CC on

The two are designed to be complementary:
  - cc_income_engine SKIPS LB BUY/ADD names (don't cap upside on positions you
    expect to grow).
  - cc_buywrite USES LB BUY/ADD names (capping upside is fine when the cap is
    your entry strategy — buy-write enters with built-in downside protection).

Strategy parameters (defaults):
  - Universe: scanner top 30 + LB BUY/ADD picks from latest portfolio_judge
  - Capital per buy-write: configurable (default $25K max per name)
  - Delta band: 0.20-0.30 (slightly tighter than wheel — buy-write is an entry,
    so we want upside if the stock runs, but real downside protection)
  - DTE: 25-50 days
  - Filters: GREEN-rated only after risk overlay (adj POP >= 65, R/R >= 0.20)
  - Min open interest 100, max spread 20% of mid
  - Min liquidity floor on shares: 100 shares (1 contract minimum)

Scoring (each candidate gets):
  - Annualized cash yield   = premium / cost_basis × (365/dte) × 100
  - Downside protection %   = premium / spot × 100  (how much spot can drop before underwater)
  - Adjusted POP            = full risk overlay (same as cc_income_engine)
  - Capital required        = 100 × spot - premium  (you collect premium up front)

Output:
  data/snapshots/<today>/cc_buywrite.md

CLI:
    python cc_buywrite.py                     # default $25K max per name, full universe
    python cc_buywrite.py --max-cost 15000    # cap capital per trade
    python cc_buywrite.py --top 10            # top N candidates
    python cc_buywrite.py --tickers WDC,STX   # restrict universe
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
from options import fetch_expiries, fetch_options_chain
from breakout import build_features, ad_label
from earnings import build_earnings_cache
from snapshot import load_df
from cc_income_engine import (
    _to_float, _to_int,
    compute_technical_features, compute_risk_adjustment, _color_adjusted, _risk_verdict,
    EXPENSE_COMPONENTS,
)
from msft_income import _score_candidate


BENCHMARK = "QQQ"

DEFAULTS = {
    "delta_min":          0.20,
    "delta_max":          0.30,
    "dte_min":            25,
    "dte_max":            50,
    "min_open_interest":  100,
    "max_spread_pct":     20.0,
    "max_cost_per_name":  25_000,   # cap capital per buy-write
    "top_n":              10,
    "min_adj_pop":        65,       # GREEN floor on the risk overlay
    "min_annualized_yield_pct": 8,  # don't bother below this — easier passive vehicles exist
    "min_downside_pct":   2.0,      # premium must give at least this much spot-drop cushion
}


# ---------- Universe construction -------------------------------------------

def _scanner_top(top_n: int = 30) -> list[str]:
    """Pull today's scanner watchlist tickers."""
    today = datetime.now().strftime("%Y-%m-%d")
    wl = load_df("watchlist", today)
    if wl is None or wl.empty:
        return []
    return wl.head(top_n)["ticker"].tolist()


def _lb_buy_add_names() -> list[str]:
    """Pull tickers LB rated BUY or ADD in the most recent portfolio_judge."""
    snaps = sorted((DATA_DIR / "snapshots").glob("*"))
    for snap in reversed(snaps):
        f = snap / "judgments_portfolio.jsonl"
        if not f.exists():
            continue
        names = []
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pm = rec.get("result", {}).get("pm", {})
            if pm.get("action") in ("BUY", "ADD"):
                names.append(rec["ticker"])
        if names:
            return names
    return []


def _already_owned() -> set[str]:
    """Tickers already in HOLDINGS_CURRENT — exclude from buy-write screen."""
    import user_config
    return {t for (_, t, *_) in getattr(user_config, "HOLDINGS_CURRENT", [])}


def build_universe(only_tickers: set[str] | None = None,
                   include_scanner_top: int = 30) -> list[str]:
    if only_tickers:
        return sorted(only_tickers)
    owned = _already_owned()
    universe = set(_scanner_top(include_scanner_top))
    universe |= set(_lb_buy_add_names())
    return sorted(universe - owned)


# ---------- Per-candidate scoring -------------------------------------------

def score_buywrite_for_ticker(ticker: str, spot: float,
                              dte_min: int, dte_max: int,
                              delta_min: float, delta_max: float,
                              tech: dict, next_earnings: pd.Timestamp | None,
                              max_cost: float) -> list[dict]:
    """Score every viable buy-write candidate (one per strike/expiry).

    Filters:
      - cost (100 * spot - premium) <= max_cost
      - adjusted POP >= min_adj_pop
      - annualized cash yield >= min_yield
      - downside protection % >= min_downside
    Returns sorted by (color rank, annualized yield desc).
    """
    today = pd.Timestamp(date.today())
    try:
        expiries = fetch_expiries(ticker)
    except Exception:
        return []
    in_window = [(d, e) for e in expiries
                 if dte_min <= (d := (pd.Timestamp(e) - today).days) <= dte_max]
    if not in_window:
        return []

    candidates: list[dict] = []
    for dte, exp in in_window:
        try:
            chain = fetch_options_chain(ticker, exp)
        except Exception:
            continue
        calls = chain.get("calls")
        if calls is None or calls.empty:
            continue

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

            adj, reasons = compute_risk_adjustment(tech, spans_earnings)
            adjusted_pop = max(0.0, min(100.0, c["pop_pct"] + adj))

            # Buy-write-specific metrics
            cost_per_100 = 100 * spot - c["premium_per_contract"]
            if cost_per_100 > max_cost:
                continue
            annualized_cash_yield = (c["premium_per_contract"] / cost_per_100) * (365 / dte) * 100
            downside_protection_pct = (c["mid"] / spot) * 100
            breakeven_price = spot - c["mid"]
            # If assigned, total realized $ on the 100-share lot
            if_assigned_total = c["premium_per_contract"] + 100 * (c["strike"] - spot)

            c.update({
                "ticker":               ticker,
                "spot":                 spot,
                "expiry":               exp,
                "spans_earnings":       spans_earnings,
                "next_earnings":        str(next_earnings.date()) if next_earnings is not None and not pd.isna(next_earnings) else None,
                "pop_adjustment":       round(adj, 1),
                "adjusted_pop":         round(adjusted_pop, 1),
                "risk_reasons":         reasons,
                "risk_verdict":         _risk_verdict(adj, spans_earnings),
                "cost_per_contract":    round(cost_per_100, 0),
                "annualized_cash_yield_pct": round(annualized_cash_yield, 1),
                "downside_protection_pct":   round(downside_protection_pct, 2),
                "breakeven_price":      round(breakeven_price, 2),
                "if_assigned_pnl":      round(if_assigned_total, 0),
                "if_assigned_pnl_pct":  round(if_assigned_total / cost_per_100 * 100, 2),
            })
            c["color"] = _color_adjusted(c)

            # Final filters: GREEN-only by adjusted POP, plus min yield + cushion
            if adjusted_pop < DEFAULTS["min_adj_pop"]:
                continue
            if annualized_cash_yield < DEFAULTS["min_annualized_yield_pct"]:
                continue
            if downside_protection_pct < DEFAULTS["min_downside_pct"]:
                continue
            candidates.append(c)

    # Best per (ticker, expiry) — highest annualized yield
    if not candidates:
        return []
    return sorted(candidates, key=lambda x: -x["annualized_cash_yield_pct"])[:3]


# ---------- Reporting -------------------------------------------------------

def format_report(ranked: list[dict], args: dict, skipped: dict) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# Buy-Write Screener — {today}")
    lines.append("")
    lines.append("Names you do NOT own that screen as good buy-write entries: buy 100 shares + sell a delta-0.20 to 0.30 out-of-the-money call same day. Captures premium up front, gives downside protection, caps upside at strike + premium.")
    lines.append("")
    lines.append(f"**Filters**: Δ{args['delta_min']:.2f}-{args['delta_max']:.2f}, {args['dte_min']}-{args['dte_max']} DTE, "
                 f"adj POP ≥{DEFAULTS['min_adj_pop']}%, annualized yield ≥{DEFAULTS['min_annualized_yield_pct']}%, "
                 f"max cost ${args['max_cost_per_name']:,}/name")
    lines.append("")

    if not ranked:
        lines.append("_No candidates pass filters. Try widening `--delta-max` or `--max-cost`._")
        return "\n".join(lines)

    lines.append("## Top buy-write candidates")
    lines.append("")
    lines.append("| Rank | Ticker | Spot | Strike | Expiry | Days to expiry | Delta | Probability of profit (adjusted) | Risk verdict | Premium per share | Capital required | Annualized yield | Downside cushion | If assigned |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(ranked, 1):
        lines.append(
            f"| {i} | **{c['ticker']}** | ${c['spot']:.2f} | ${int(c['strike'])} | {c['expiry']} | {c['dte']} | "
            f"{c['delta']:.2f} | {c['adjusted_pop']:.0f}% | {c['risk_verdict']} | "
            f"${c['mid']:.2f} | ${c['cost_per_contract']:,.0f} | "
            f"**{c['annualized_cash_yield_pct']:.1f}%** | "
            f"{c['downside_protection_pct']:.1f}% | ${c['if_assigned_pnl']:+,.0f} ({c['if_assigned_pnl_pct']:+.1f}%) |"
        )
    lines.append("")

    # Per-candidate detail
    lines.append("## Per-candidate detail")
    lines.append("")
    for i, c in enumerate(ranked, 1):
        lines.append(f"### {i}. {c['ticker']} — buy 100 shares @ ${c['spot']:.2f} + sell {int(c['strike'])}C exp {c['expiry']}")
        lines.append("")
        lines.append(f"- Capital required: **${c['cost_per_contract']:,.0f}** "
                     f"(100 shares at ${c['spot']:.2f} = ${100*c['spot']:,.0f}, minus ${c['premium_per_contract']:,.0f} premium)")
        lines.append(f"- Premium captured today: **${c['premium_per_contract']:,.0f}**")
        lines.append(f"- Annualized cash yield on capital: **{c['annualized_cash_yield_pct']:.1f}%** "
                     f"({c['premium_per_contract'] / c['cost_per_contract'] * 100:.2f}% per cycle × {365/c['dte']:.1f} cycles/yr)")
        lines.append(f"- Downside protection: ${c['mid']:.2f}/sh = "
                     f"**{c['downside_protection_pct']:.1f}% cushion** before the trade is underwater")
        lines.append(f"- Breakeven price at expiry: ${c['breakeven_price']:.2f}")
        lines.append(f"- If assigned: net P&L per contract = **${c['if_assigned_pnl']:+,.0f}** "
                     f"({c['if_assigned_pnl_pct']:+.1f}% on capital)")
        lines.append(f"- BS POP {c['pop_pct']:.0f}% → adj POP **{c['adjusted_pop']:.0f}%** (verdict: {c['risk_verdict']})")
        if c.get("risk_reasons"):
            lines.append(f"- Risk overlay: {'; '.join(c['risk_reasons'])}")
        if c.get("spans_earnings"):
            lines.append(f"- 📅 Earnings {c.get('next_earnings')} falls inside the contract life")
        lines.append("")

    # Skipped notes
    if skipped:
        lines.append("## Universe coverage")
        lines.append("")
        lines.append(f"- Tickers screened: {sum(len(v) for v in skipped.values()) + len(ranked)}")
        for reason, names in skipped.items():
            if names:
                lines.append(f"- Skipped — {reason}: {', '.join(sorted(names)[:20])}"
                             f"{' …' if len(names) > 20 else ''}")
        lines.append("")

    lines.append("> **Operational notes**: limit at mid, work it down. Submit the BUY SHARES order first, "
                 "then the SELL TO OPEN call order — Fidelity will recognize the cover and not require margin. "
                 "After fills, update HOLDINGS_CURRENT and OPTIONS_POSITIONS in user_config.py.")
    return "\n".join(lines)


# ---------- Main ------------------------------------------------------------

def run(top_n: int = DEFAULTS["top_n"],
        delta_min: float = DEFAULTS["delta_min"],
        delta_max: float = DEFAULTS["delta_max"],
        dte_min: int = DEFAULTS["dte_min"],
        dte_max: int = DEFAULTS["dte_max"],
        max_cost_per_name: float = DEFAULTS["max_cost_per_name"],
        only_tickers: Iterable[str] | None = None,
        include_scanner_top: int = 30) -> dict:
    args = {
        "top_n": top_n,
        "delta_min": delta_min, "delta_max": delta_max,
        "dte_min": dte_min,     "dte_max": dte_max,
        "max_cost_per_name": max_cost_per_name,
    }

    only = set(only_tickers) if only_tickers else None
    universe = build_universe(only_tickers=only, include_scanner_top=include_scanner_top)
    print(f"Universe: {len(universe)} unowned tickers (LB BUY/ADD + scanner top {include_scanner_top})")
    if not universe:
        print("No candidates.")
        return {}

    # Pre-load OHLCV + benchmark + earnings
    tickers = sorted(set(universe) | {BENCHMARK})
    print(f"Loading OHLCV for {len(tickers)} tickers...")
    raw = fetch_many(tickers, force=False)
    bench = raw[BENCHMARK]["close"] if BENCHMARK in raw else None
    print("Building earnings cache...")
    earn_df = build_earnings_cache(universe)

    # Per-ticker tech + earnings
    tech_by_ticker = {t: compute_technical_features(t, raw, bench) if bench is not None else {}
                      for t in universe}
    next_earn_by_ticker = {
        t: (pd.Timestamp(earn_df.loc[t, "next_earnings"])
            if t in earn_df.index and earn_df.loc[t, "next_earnings"] is not None
            and not pd.isna(earn_df.loc[t, "next_earnings"]) else None)
        for t in universe
    }

    all_candidates: list[dict] = []
    skipped: dict[str, list[str]] = {
        "no chain data":           [],
        "no candidates passed filters": [],
    }
    for t in universe:
        spot = tech_by_ticker.get(t, {}).get("spot")
        if spot is None:
            skipped["no chain data"].append(t)
            continue
        print(f"  scoring {t} @ ${spot:.2f}...", flush=True)
        cands = score_buywrite_for_ticker(
            t, spot, dte_min, dte_max, delta_min, delta_max,
            tech=tech_by_ticker.get(t, {}),
            next_earnings=next_earn_by_ticker.get(t),
            max_cost=max_cost_per_name,
        )
        if not cands:
            skipped["no candidates passed filters"].append(t)
            continue
        all_candidates.extend(cands)

    # Final ranking: one entry per ticker (best yield), top N overall
    by_ticker: dict[str, dict] = {}
    for c in all_candidates:
        cur = by_ticker.get(c["ticker"])
        if cur is None or c["annualized_cash_yield_pct"] > cur["annualized_cash_yield_pct"]:
            by_ticker[c["ticker"]] = c
    ranked = sorted(by_ticker.values(), key=lambda x: -x["annualized_cash_yield_pct"])[:top_n]

    report = format_report(ranked, args, skipped)
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = DATA_DIR / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "cc_buywrite.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_md}\n")
    print(report)
    return {"report": report, "out_md": str(out_md), "ranked": ranked}


if __name__ == "__main__":
    args = sys.argv[1:]
    kwargs: dict = {}
    for i, a in enumerate(args):
        if a == "--top" and i + 1 < len(args):
            kwargs["top_n"] = int(args[i + 1])
        elif a == "--delta-max" and i + 1 < len(args):
            kwargs["delta_max"] = float(args[i + 1])
        elif a == "--delta-min" and i + 1 < len(args):
            kwargs["delta_min"] = float(args[i + 1])
        elif a == "--dte-min" and i + 1 < len(args):
            kwargs["dte_min"] = int(args[i + 1])
        elif a == "--dte-max" and i + 1 < len(args):
            kwargs["dte_max"] = int(args[i + 1])
        elif a == "--max-cost" and i + 1 < len(args):
            kwargs["max_cost_per_name"] = float(args[i + 1])
        elif a == "--tickers" and i + 1 < len(args):
            kwargs["only_tickers"] = [t.strip().upper() for t in args[i + 1].split(",")]
    run(**kwargs)
