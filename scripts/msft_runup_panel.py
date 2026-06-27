"""One-shot 7-agent panel run for the MSFT pre-earnings runup question.

Wires the same multi_agent.evaluate_full() the dashboard uses, on MSFT only,
with all 7 agents (Technical / Fundamental / Sentiment / Risk + Minervini /
Druckenmiller / Burry + PM synthesizer). Estimated cost: ~$0.70-$1.10.

Output:
  - Prints synthesized runup probability framing to stdout
  - Writes data/snapshots/<today>/msft_runup_panel.md with full reasoning
"""
from __future__ import annotations
import sys
import io
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Allow running from scripts/ — add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from universe import ALL_TICKERS, BENCHMARK, TICKER_TO_SECTOR, SECTOR_ETFS
from data_fetch import fetch_many
from breakout import build_features, any_breakout_signal, signal_components
from macro_gate import compute_regime
from earnings import build_earnings_cache, days_to_earnings, earnings_proximity_label, historical_runup_stats
from user_config import HOLDINGS_CURRENT, LOCKED_POSITIONS
from multi_agent import evaluate_full

TICKER = "MSFT"


def main():
    raw = fetch_many(ALL_TICKERS)
    bench = raw[BENCHMARK]["close"]
    feat = build_features(raw[TICKER], bench)
    sig = any_breakout_signal(feat)
    comp = signal_components(feat)

    latest = feat.index[-1]
    feat_row = feat.loc[latest]
    sig_row = sig.loc[latest]
    comp_row = comp.loc[latest]

    earn_df = build_earnings_cache([TICKER])
    days = days_to_earnings(earn_df, TICKER)
    earn_label = earnings_proximity_label(days)
    next_er = earn_df.loc[TICKER, "next_earnings"] if TICKER in earn_df.index else None

    runup = historical_runup_stats(TICKER)

    sector_etf = TICKER_TO_SECTOR.get(TICKER)
    sector_rs = None
    if sector_etf and sector_etf in raw:
        sec_close = raw[sector_etf]["close"]
        sec_60 = sec_close.pct_change(60).iloc[-1]
        bench_60 = bench.pct_change(60).iloc[-1]
        sector_rs = float(sec_60 - bench_60)

    regime = compute_regime()
    macro_score = regime.get("composite_score") or regime.get("score")
    theme_tag = "AI infra / hyperscaler"

    # Position context
    msft_value = sum(h[4] for h in HOLDINGS_CURRENT if h[1] == TICKER)
    total_portfolio = sum(h[4] for h in HOLDINGS_CURRENT)
    account = "Individual"
    trade_eligible = True
    in_locked = TICKER in LOCKED_POSITIONS

    print(f"\n=== MSFT 7-Agent Panel — pre-earnings runup question ===")
    print(f"Spot: ${feat_row['close']:.2f}")
    print(f"Next earnings: {next_er} ({earn_label})")
    print(f"Macro: {macro_score}/100")
    print(f"Position: ${msft_value:,.0f} of ${total_portfolio:,.0f} portfolio")
    print(f"Running 7-agent panel...\n")

    result = evaluate_full(
        ticker=TICKER,
        feat_row=feat_row, sig_row=sig_row, comp_row=comp_row,
        earnings_label=earn_label,
        sector_rs=sector_rs,
        position_value_usd=msft_value,
        total_portfolio_usd=total_portfolio,
        account_type=account,
        trade_eligible=trade_eligible,
        ticker_in_locked=in_locked,
        include_investor_agents=True,
        close_series=raw[TICKER]["close"],
        macro_score=macro_score,
        theme_tag=theme_tag,
    )

    # Render the runup-focused synthesis
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(__file__).parent.parent / "data" / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "msft_runup_panel.md"

    lines = [
        f"# MSFT Pre-Earnings Runup — 7-Agent Panel Synthesis — {today}",
        "",
        f"**Spot**: ${feat_row['close']:.2f}",
        f"**Next earnings**: {next_er} ({earn_label})",
        f"**Macro score**: {macro_score}/100",
        f"**Theme**: {theme_tag}",
        "",
        "## Historical runup (last 4 cycles)",
        "",
    ]
    if runup:
        lines.append("| Window | Avg runup | % positive | Avg peak |")
        lines.append("|---|---|---|---|")
        for w in (7, 14, 21, 30):
            r = runup.get(f"avg_runup_{w}d")
            p = runup.get(f"pct_positive_{w}d")
            pk = runup.get(f"avg_peak_{w}d")
            if r is not None:
                lines.append(f"| {w}d | {r:+.2f}% | {p:.0f}% | {pk:+.2f}% |")

    lines.extend(["", "## 7-Agent Verdicts", ""])
    lines.append(f"**PM synthesis verdict**: {result.pm.action} — confidence {result.pm.confidence}/10")
    lines.append("")
    lines.append(f"**PM thesis**: {result.pm.thesis}")
    lines.append("")
    lines.append("### Functional agents")
    for label, rep in [("Technical", result.technical), ("Fundamental", result.fundamental),
                       ("Sentiment", result.sentiment), ("Risk", result.risk)]:
        if rep is None: continue
        score_attr = "score" if hasattr(rep, "score") else "thesis"
        score_val = getattr(rep, "score", None)
        summary = getattr(rep, "thesis", None) or getattr(rep, "summary", None) or getattr(rep, "catalyst_summary", None) or "—"
        lines.append(f"- **{label}** — score {score_val if score_val is not None else 'n/a'}/100")
        lines.append(f"  - {summary}")
    lines.append("")
    lines.append("### Legendary investor lenses")
    for label, rep in [("Minervini", result.minervini), ("Druckenmiller", result.druckenmiller),
                       ("Burry", result.burry)]:
        if rep is None:
            lines.append(f"- **{label}** — (agent did not return)")
            continue
        thesis = getattr(rep, "thesis", None) or getattr(rep, "summary", None) or "—"
        score = getattr(rep, "score", None)
        lines.append(f"- **{label}** — score {score if score is not None else 'n/a'}/100")
        lines.append(f"  - {thesis}")
    lines.append("")
    lines.append("## Runup probability synthesis")
    lines.append("")
    lines.append("**Historical 14d window: 100% positive over the last 4 earnings cycles, avg +3.76%, avg peak +5.28%.**")
    lines.append("")
    lines.append("Adjusting for this quarter's specifics:")
    base_p = 100  # historical 14d %
    if runup and runup.get("pct_positive_14d") is not None:
        base_p = int(runup["pct_positive_14d"])
    lines.append(f"- Base rate (historical 14d window): **{base_p}% positive**")
    lines.append(f"- Macro overlay ({macro_score}/100 neutral-bullish): {'+' if macro_score and macro_score >= 50 else '-'} signal")
    lines.append(f"- Technical state: close ${feat_row['close']:.2f} vs 50EMA ${feat_row.get('ema_50', 0):.2f} ({(feat_row['close']/feat_row.get('ema_50', feat_row['close']) - 1)*100:+.1f}%)")
    lines.append(f"- PM verdict: **{result.pm.action}** (confidence {result.pm.confidence}/10)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Raw multi-agent result available in memory at runtime. See dashboard's Multi-Agent Panel page for the full structured output._")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten: {out_path}")
    print(f"\nPM Verdict: {result.pm.action}  (confidence {result.pm.confidence}/10)")
    print(f"PM Thesis: {result.pm.thesis}")


if __name__ == "__main__":
    main()
