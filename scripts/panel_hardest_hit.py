"""Run the 7-agent panel across the day's hardest-hit names in parallel.

Used 2026-06-23 after AI-semis dump + BofA report. Each ticker gets the full
multi_agent.evaluate_full(include_investor_agents=True). Then synthesize each
PM verdict against the BofA stance for that name.

Output: data/snapshots/<today>/hardest_hit_panel.md
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from universe import ALL_TICKERS, BENCHMARK, TICKER_TO_SECTOR
from data_fetch import fetch_many
from breakout import build_features, any_breakout_signal, signal_components
from macro_gate import compute_regime
from earnings import build_earnings_cache, days_to_earnings, earnings_proximity_label, historical_runup_stats
from user_config import HOLDINGS_CURRENT, LOCKED_POSITIONS
from multi_agent import evaluate_full


TICKERS = ["MU", "SNDK", "ALAB", "ARM", "AMD"]

# BofA Jun-23 thesis points per ticker — fed to PM as context.
BOFA_NOTES = {
    "MU":   "BofA Jun-23: PO raised $950 -> $1,500, BUY. SOTP CY28E (2.5x P/B trad mem + 31x PE for HBM). DRAM +309% CY26, NAND +295%. MU/Anthropic LTA cited as key visibility.",
    "ALAB": "BofA Jun-23: PO raised $240 -> $450, NEUTRAL. 77x CY28E PE (~2.0x PEG). EPS power $9+ by CY30E.",
    "ARM":  "BofA Jun-23: PO raised $335 -> $460, NEUTRAL. SOTP CY30E discounted back 2y (2.5x PEG IP + 31x PE Chip). Agentic CPU $170bn TAM by CY30.",
    "AMD":  "Not in the Jun-23 PO update list. Still benefits from AI accelerator theme: BofA CY30 server silicon $615bn (vs $209bn CY25, +24% CAGR). Logic +39% CY26.",
    "SNDK": "Not in the Jun-23 PO update directly, but BofA NAND +295% CY26E forecast and structural memory supply/pricing thesis directly applies. SNDK is the pure NAND play in the held book.",
}

# Theme tags per ticker
THEME_TAGS = {
    "MU":   "AI memory / HBM",
    "SNDK": "NAND / AI memory",
    "ALAB": "AI infra / connectivity",
    "ARM":  "Agentic CPU / mobile",
    "AMD":  "AI accelerator",
}


def run_one(ticker: str, raw: dict, bench, earn_df, regime, total_portfolio: float) -> dict:
    """Run full 7-agent panel on one ticker. Returns dict for table render."""
    feat = build_features(raw[ticker], bench)
    sig = any_breakout_signal(feat)
    comp = signal_components(feat)
    latest = feat.index[-1]
    feat_row = feat.loc[latest]
    sig_row = sig.loc[latest]
    comp_row = comp.loc[latest]

    days = days_to_earnings(earn_df, ticker)
    earn_label = earnings_proximity_label(days)

    sector_etf = TICKER_TO_SECTOR.get(ticker)
    sector_rs = None
    if sector_etf and sector_etf in raw:
        sec_close = raw[sector_etf]["close"]
        sec_60 = sec_close.pct_change(60).iloc[-1]
        bench_60 = bench.pct_change(60).iloc[-1]
        sector_rs = float(sec_60 - bench_60)

    macro_score = regime.get("composite_score") or regime.get("score")
    theme_tag = THEME_TAGS.get(ticker, "neutral")

    pos_value = sum(h[4] for h in HOLDINGS_CURRENT if h[1] == ticker)
    in_locked = ticker in LOCKED_POSITIONS

    research = {
        "catalyst_summary": BOFA_NOTES.get(ticker, ""),
        "source_url": "holdings/BofA Report.pdf",
        "asof": "2026-06-23",
    }

    print(f"  [{ticker}] running 7-agent panel...", flush=True)
    try:
        result = evaluate_full(
            ticker=ticker,
            feat_row=feat_row, sig_row=sig_row, comp_row=comp_row,
            earnings_label=earn_label,
            sector_rs=sector_rs,
            position_value_usd=pos_value,
            total_portfolio_usd=total_portfolio,
            account_type="Mixed",
            trade_eligible=True,
            ticker_in_locked=in_locked,
            include_investor_agents=True,
            close_series=raw[ticker]["close"],
            macro_score=macro_score,
            theme_tag=theme_tag,
            research_report=research,
        )
    except Exception as e:
        print(f"  [{ticker}] FAILED: {e}", flush=True)
        return {"ticker": ticker, "error": str(e)}

    return {
        "ticker": ticker,
        "spot": float(feat_row["close"]),
        "pos_value": pos_value,
        "pm_action": result.pm.action,
        "pm_confidence": result.pm.confidence,
        "pm_thesis": result.pm.thesis,
        "technical_score": getattr(result.technical, "score", None),
        "fundamental_score": getattr(result.fundamental, "score", None),
        "sentiment_score": getattr(result.sentiment, "score", None),
        "risk_score": getattr(result.risk, "score", None),
        "minervini_score": getattr(result.minervini, "score", None) if result.minervini else None,
        "druckenmiller_score": getattr(result.druckenmiller, "score", None) if result.druckenmiller else None,
        "burry_score": getattr(result.burry, "score", None) if result.burry else None,
        "druckenmiller_thesis": getattr(result.druckenmiller, "thesis", None) if result.druckenmiller else None,
        "bofa_note": BOFA_NOTES.get(ticker, ""),
    }


def main():
    print("Loading universe OHLCV + macro...")
    raw = fetch_many(ALL_TICKERS)
    bench = raw[BENCHMARK]["close"]
    earn_df = build_earnings_cache(TICKERS)
    regime = compute_regime()
    total_portfolio = sum(h[4] for h in HOLDINGS_CURRENT)

    print(f"\nRunning 7-agent panel across {len(TICKERS)} hardest-hit names (parallel)...")
    print(f"Macro: {regime.get('composite_score') or regime.get('score')}/100\n")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(run_one, t, raw, bench, earn_df, regime, total_portfolio): t for t in TICKERS}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"ticker": futures[fut], "error": str(e)})

    rows.sort(key=lambda r: TICKERS.index(r["ticker"]))

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(__file__).parent.parent / "data" / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hardest_hit_panel.md"

    lines = [
        f"# Hardest-Hit AI Semis — 7-Agent Panel + BofA Synthesis — {today}",
        "",
        "Day-of-dump triangulation: the AI semis cohort dropped 6-14% intraday while BofA",
        "published bullish PO raises this morning. Each ticker gets the full functional",
        "panel (Technical / Fundamental / Sentiment / Risk) plus the three legendary-investor",
        "lenses (Minervini / Druckenmiller / Burry), with the BofA PO context fed to the PM",
        "synthesizer.",
        "",
        f"Macro regime: {regime.get('composite_score') or regime.get('score'):.0f}/100",
        "",
        "## Per-ticker verdicts (PM action)",
        "",
        "| Ticker | Spot | Position | PM | Conf | Tech | Fund | Sent | Risk | Minervini | Drucken | Burry | BofA stance |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['ticker']} | — | — | ERROR | | | | | | | | | {r.get('error','')} |")
            continue
        bofa_short = "BUY +42%" if r["ticker"] == "MU" else (
                     "NEUTRAL +24%" if r["ticker"] == "ARM" else (
                     "NEUTRAL +13%" if r["ticker"] == "ALAB" else (
                     "n/a — derivative" if r["ticker"] in ("SNDK", "AMD") else "n/a")))
        lines.append(
            f"| **{r['ticker']}** | ${r['spot']:.2f} | ${r['pos_value']:,.0f} | "
            f"**{r['pm_action']}** | {r['pm_confidence']}/10 | "
            f"{r['technical_score'] or '—'} | {r['fundamental_score'] or '—'} | "
            f"{r['sentiment_score'] or '—'} | {r['risk_score'] or '—'} | "
            f"{r['minervini_score'] or '—'} | {r['druckenmiller_score'] or '—'} | "
            f"{r['burry_score'] or '—'} | {bofa_short} |"
        )

    lines.extend(["", "## Per-ticker detail", ""])
    for r in rows:
        if "error" in r:
            lines.append(f"### {r['ticker']} — error")
            lines.append(f"`{r.get('error','')}`")
            continue
        lines.append(f"### {r['ticker']} — PM **{r['pm_action']}** (confidence {r['pm_confidence']}/10)")
        lines.append("")
        lines.append(f"- Spot: ${r['spot']:.2f}")
        lines.append(f"- Position value: ${r['pos_value']:,.0f}")
        lines.append(f"- BofA stance (Jun-23): {r['bofa_note']}")
        lines.append("")
        lines.append("**PM thesis**")
        lines.append("")
        lines.append(f"> {r['pm_thesis']}")
        lines.append("")
        if r.get("druckenmiller_thesis"):
            lines.append("**Druckenmiller lens**")
            lines.append("")
            lines.append(f"> {r['druckenmiller_thesis']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten: {out_path}")
    print(f"\n=== Verdict summary ===")
    for r in rows:
        if "error" in r:
            print(f"  {r['ticker']:>6}: ERROR")
            continue
        print(f"  {r['ticker']:>6}: PM={r['pm_action']:<8} conf={r['pm_confidence']}/10  spot=${r['spot']:.2f}")


if __name__ == "__main__":
    main()
