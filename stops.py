"""Per-position stop suggestions for the held portfolio.

Implements the "every position needs a sell plan" discipline from external TA
references without prescribing one approach — surfaces all valid stop
candidates so the user can pick the one that fits their conviction on each name.

Methodologies (highest level = tightest stop = strictest discipline):
  1. 50-EMA breach       — preferred for quality compounders (META/GOOGL/AMZN)
  2. AVWAP from 60d swing low — "uncanny" S/R per external TA reference
  3. Fib 38.2 breakdown  — pattern failure line
  4. 20% below rolling 60d peak — default trailing fallback

Recommended stop = the HIGHEST valid level (closest to current price = tightest
discipline). Override per ticker as you read the chart.

Outputs:
  - data/snapshots/<today>/stops.md — full table + per-position detail
  - Dashboard sidebar can call compute_all_stops() and render the table inline.

CLI:
  python stops.py             # compute + write snapshot, print summary
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd

from user_config import HOLDINGS_CURRENT, LOCKED_POSITIONS, OPTIONS_POSITIONS
from universe import ALL_TICKERS, BENCHMARK
from data_fetch import fetch_many
from breakout import build_features


NON_STOCK_MARKERS = (
    "FDRXX", "SPAXX", "FZFXX", "CASH",
    "STABLE", "VANG", "SMID", "NHFSMKX", "FID_GR", "DRAM", "BTC_LPATH", "SPCX",
)


def is_stock(ticker: str) -> bool:
    return not any(m in ticker for m in NON_STOCK_MARKERS)


def aggregate_holdings() -> dict[str, dict]:
    """Aggregate HOLDINGS_CURRENT tuples to {ticker -> {shares, value, cost_basis}}."""
    agg: dict[str, dict] = {}
    for row in HOLDINGS_CURRENT:
        _, ticker, shares, _, value, cost = row
        if not is_stock(ticker):
            continue
        if ticker not in agg:
            agg[ticker] = {"shares": 0.0, "value": 0.0, "cost_basis": 0.0}
        agg[ticker]["shares"] += shares
        agg[ticker]["value"] += value
        if cost is not None:
            agg[ticker]["cost_basis"] += cost
    return agg


def has_active_short_call(ticker: str) -> bool:
    """True if there's an open short call on this ticker."""
    return any(
        o.get("ticker") == ticker
        and o.get("contracts", 0) < 0
        and o.get("opt_type", "call") == "call"
        for o in OPTIONS_POSITIONS
    )


def compute_stops_for_ticker(feat: pd.DataFrame) -> dict:
    """Compute candidate stop levels for one ticker.

    Returns dict with current, peak_60d, candidates (dict of {key: {level, method, pct_below}}),
    and recommended_key (the tightest valid stop).
    """
    last = feat.iloc[-1]
    close = float(last["close"])
    peak_60 = float(feat["high"].tail(60).max())

    raw_candidates = {
        "ema_50": {
            "level": last.get("ema_50"),
            "method": "50 EMA breach (quality compounder discipline)",
        },
        "avwap": {
            "level": last.get("avwap_swinglow_60"),
            "method": "AVWAP from 60d swing low (uncanny S/R)",
        },
        "fib_382": {
            "level": last.get("fib_382"),
            "method": "Fib 38.2 breakdown (pattern failure)",
        },
        "trailing_20": {
            "level": peak_60 * 0.80,
            "method": "20% below 60d peak (default trailing)",
        },
    }

    valid = {}
    above_levels = {}
    for key, c in raw_candidates.items():
        level = c["level"]
        if level is None or pd.isna(level):
            continue
        level = float(level)
        if level >= close:
            # Level above current price = this reference level has already been breached
            above_levels[key] = {
                "level": level,
                "method": c["method"],
                "pct_above": (level - close) / close * 100,
            }
            continue
        valid[key] = {
            "level": level,
            "method": c["method"],
            "pct_below": (close - level) / close * 100,
        }

    if not valid:
        # All reference levels are ABOVE current price — stock is already below all stops
        return {
            "current": close,
            "peak_60d": peak_60,
            "candidates": {},
            "recommended_key": None,
            "all_above_warning": True,
            "above_levels": above_levels,
        }

    # Recommended = tightest = highest level (closest to current price)
    recommended_key = max(valid, key=lambda k: valid[k]["level"])
    return {
        "current": close,
        "peak_60d": peak_60,
        "candidates": valid,
        "recommended_key": recommended_key,
        "above_levels": above_levels,
    }


def compute_all_stops(feats: dict[str, pd.DataFrame] | None = None) -> dict[str, dict]:
    """Compute stop suggestions for every held stock."""
    agg = aggregate_holdings()

    if feats is None:
        held = [t for t in agg if t in ALL_TICKERS]
        raw = fetch_many(held + [BENCHMARK])
        bench = raw[BENCHMARK]["close"]
        feats = {t: build_features(raw[t], bench) for t in held if t in raw}

    results: dict[str, dict] = {}
    for ticker, pos in agg.items():
        entry = {
            "ticker": ticker,
            "shares": pos["shares"],
            "value": pos["value"],
            "cost_basis": pos["cost_basis"],
            "has_short_call": has_active_short_call(ticker),
        }
        if ticker in LOCKED_POSITIONS:
            entry["locked"] = True
            results[ticker] = entry
            continue
        if ticker not in feats:
            entry["no_data"] = True
            results[ticker] = entry
            continue
        entry.update(compute_stops_for_ticker(feats[ticker]))
        results[ticker] = entry
    return results


def render_stops_markdown(stops_by_ticker: dict[str, dict]) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# Per-Position Stop Suggestions — {today}",
        "",
        "Stops computed per external TA methodology. Recommended stop = tightest valid level.",
        "Methodologies (highest = tightest = strictest discipline):",
        "  1. 50 EMA breach (preferred for quality compounders)",
        "  2. AVWAP from 60d swing low (uncanny S/R)",
        "  3. Fib 38.2 breakdown (pattern failure)",
        "  4. 20% below rolling 60d peak (default trailing)",
        "",
        "**Rule (per external TA discipline)**: when CLOSE breaches the stop level, sell at market the next day. No questions.",
        "**Sequencing (per memory)**: if a position has an active short call, close the option FIRST, then sell shares.",
        "",
        "## Summary table",
        "",
        "| Ticker | Shares | Value | Current | Avg Cost | Gain % | CC? | Recommended Stop | Method | % Below |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|:---|---:|",
    ]
    for ticker in sorted(stops_by_ticker.keys()):
        r = stops_by_ticker[ticker]
        sh = r["shares"]
        val = r["value"]
        cc_tag = "Yes" if r.get("has_short_call") else ""

        if r.get("locked"):
            lines.append(
                f"| **{ticker}** ⚠ | {sh:.1f} | ${val:,.0f} | locked | | | | LOCKED — no stop | per rules | n/a |"
            )
            continue
        if r.get("no_data"):
            lines.append(
                f"| {ticker} | {sh:.1f} | ${val:,.0f} | n/a | | | {cc_tag} | (no price data) | | |"
            )
            continue
        if not r.get("candidates"):
            warn = "⚠ **ALREADY BELOW ALL STOPS — review thesis**" if r.get("all_above_warning") else "(no valid stop)"
            lines.append(
                f"| {ticker} | {sh:.1f} | ${val:,.0f} | ${r['current']:.2f} | | | {cc_tag} | {warn} | | |"
            )
            continue

        cb = r["cost_basis"] / sh if sh > 0 and r["cost_basis"] > 0 else 0
        gain_pct = ((r["current"] - cb) / cb * 100) if cb > 0 else 0
        rec = r["candidates"][r["recommended_key"]]
        cost_str = f"${cb:.2f}" if cb > 0 else ""
        gain_str = f"{gain_pct:+.1f}%" if cb > 0 else ""
        lines.append(
            f"| {ticker} | {sh:.1f} | ${val:,.0f} | ${r['current']:.2f} | {cost_str} | {gain_str} | "
            f"{cc_tag} | **${rec['level']:.2f}** | {rec['method']} | {rec['pct_below']:.1f}% |"
        )

    lines.extend(["", "---", "", "## Per-position detail (all candidates ranked)", ""])
    for ticker in sorted(stops_by_ticker.keys()):
        r = stops_by_ticker[ticker]
        if r.get("locked") or r.get("no_data"):
            continue
        lines.append(f"### {ticker} @ ${r['current']:.2f}  (60d peak ${r['peak_60d']:.2f})")
        if r.get("has_short_call"):
            lines.append("> ⚠ Active short call — close the option BEFORE selling shares.")
        if r.get("all_above_warning"):
            lines.append("")
            lines.append("> ⚠ **REVIEW: stock is below ALL reference levels.** No valid stop "
                         "below price — every methodology says it's already broken. Either the "
                         "thesis is invalidated, or this is the wash-out before a bounce. "
                         "Either way, deliberate review required.")
            lines.append("")
            for key, c in sorted(r.get("above_levels", {}).items(), key=lambda kv: kv[1]["level"]):
                lines.append(f"- ${c['level']:.2f} ({c['pct_above']:.1f}% ABOVE current) — {c['method']}")
            lines.append("")
            continue
        ranked = sorted(r["candidates"].items(), key=lambda kv: -kv[1]["level"])
        for key, c in ranked:
            marker = " ⭐ recommended" if key == r["recommended_key"] else ""
            lines.append(
                f"- **${c['level']:.2f}** ({c['pct_below']:.1f}% below) — {c['method']}{marker}"
            )
        if r.get("above_levels"):
            lines.append("")
            lines.append("_Reference levels already breached (above current price):_")
            for key, c in sorted(r["above_levels"].items(), key=lambda kv: kv[1]["level"]):
                lines.append(f"- ${c['level']:.2f} ({c['pct_above']:.1f}% above) — {c['method']}")
        lines.append("")
    return "\n".join(lines)


def save_stops_snapshot(stops_by_ticker: dict[str, dict], root: Path | None = None) -> Path:
    root = root or Path(__file__).parent
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = root / "data" / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stops.md"
    out_path.write_text(render_stops_markdown(stops_by_ticker), encoding="utf-8")
    return out_path


def main() -> None:
    print("Computing per-position stops...")
    stops = compute_all_stops()
    path = save_stops_snapshot(stops)
    print(f"Wrote {path}")
    print()
    print("Summary:")
    for ticker in sorted(stops.keys()):
        r = stops[ticker]
        if r.get("locked"):
            print(f"  {ticker:>6}  LOCKED")
            continue
        if r.get("no_data"):
            print(f"  {ticker:>6}  (no data)")
            continue
        if not r.get("candidates"):
            print(f"  {ticker:>6}  ${r['current']:.2f}  no valid stop below price")
            continue
        rec = r["candidates"][r["recommended_key"]]
        cc = " [CC]" if r.get("has_short_call") else ""
        print(
            f"  {ticker:>6}  ${r['current']:7.2f}  stop ${rec['level']:7.2f} "
            f"({rec['pct_below']:5.1f}% below)  {rec['method']}{cc}"
        )


if __name__ == "__main__":
    main()
