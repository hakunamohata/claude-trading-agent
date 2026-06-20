"""Run multi-agent judgment on every non-locked portfolio position.

Default: 4 functional agents (~$0.025/ticker, ~$0.70 per full run).
With --investor-agents: adds Minervini/Druckenmiller/Burry (~$0.04/ticker, ~$1.10 run).
With --with-research: adds web-search research per ticker (~$0.10/ticker added).

Output: data/snapshots/<today>/judgments_portfolio.jsonl with structured results.

CLI:
    python portfolio_judge.py                            # 4 agents only
    python portfolio_judge.py --investor-agents          # 7 agents
    python portfolio_judge.py --investor-agents --with-research   # 7 + live research
    python portfolio_judge.py --refresh                  # bypass research cache
"""

from __future__ import annotations
import os
import sys
import io
from datetime import datetime
import pandas as pd

# Force UTF-8 on Windows console so we can print unicode chars like −, →, etc.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from data_fetch import fetch_many
from breakout import (
    build_features, any_breakout_signal, signal_components, compute_universe_rs_rank,
)
from earnings import build_earnings_cache, days_to_earnings, earnings_proximity_label
from sector import compute_sector_strength
from snapshot import save_jsonl
from multi_agent import evaluate_full
from universe import ALL_TICKERS, BENCHMARK, SECTOR_ETFS, TICKER_TO_SECTOR
from portfolio import (
    HOLDINGS_DIR, ACCOUNT_LABEL, TRADE_ELIGIBLE_ACCOUNTS, LOCKED_POSITIONS,
)
import scratchpad


NON_EQUITY = {"FDRXX", "SPAXX", "NHFSMKX98", "CASH_ROTH", "CASH_TOD", "CASH_HSA"}


# Theme tag inference — used by Druckenmiller agent for "is this a megatrend name?"
# Simple sector-based mapping; refine over time.
THEME_BY_SECTOR_ETF = {
    "SOXX": "AI infra / semis super-cycle",
    "XLK": "AI / cloud / software platform",
    "XLC": "communications / digital advertising",
    "XLY": "consumer discretionary / EVs",
    "XLP": "defensive consumer staples",
    "XLF": "financials / fintech",
    "XLV": "healthcare / biotech",
    "XLI": "industrials / re-industrialization",
    "XLE": "energy",
    "XLU": "utilities / AI power",
    "XLB": "materials",
    "XLRE": "real estate / data centers",
}


def _load_existing_judgments() -> dict[tuple, dict]:
    """Return existing judgments_portfolio.jsonl entries keyed by (ticker, account)."""
    from data_fetch import DATA_DIR
    p = DATA_DIR / "snapshots" / datetime.now().strftime("%Y-%m-%d") / "judgments_portfolio.jsonl"
    if not p.exists():
        return {}
    out = {}
    import json as _j
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = _j.loads(line)
            out[(r["ticker"], r["account"])] = r
        except Exception:
            continue
    return out


def run(include_investor_agents: bool = False,
        with_research: bool = False,
        research_refresh: bool = False,
        missing_only: bool = False) -> list[dict]:
    run_id = scratchpad.start_run(
        kind="portfolio_judge",
        args={
            "investor_agents": include_investor_agents,
            "with_research": with_research,
            "research_refresh": research_refresh,
            "missing_only": missing_only,
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7"),
        },
    )
    print(f"Scratchpad run: {run_id}")

    pf = pd.read_parquet(HOLDINGS_DIR / "positions_current.parquet")
    pf_total = float(pf["value"].sum())

    # Filter to scoreable positions
    work = pf[
        ~pf["ticker"].isin(LOCKED_POSITIONS | NON_EQUITY)
        & pf["ticker"].notna()
    ].copy()
    flags = []
    if include_investor_agents: flags.append("+investor agents")
    if with_research: flags.append("+live research")
    if missing_only: flags.append("missing-only mode")
    flag_str = f"  ({', '.join(flags)})" if flags else ""
    print(f"Multi-agent on {len(work)} positions (total household ${pf_total:,.0f}){flag_str}")

    # Preserve existing entries if missing_only
    existing = _load_existing_judgments() if missing_only else {}
    if existing:
        print(f"  Found {len(existing)} existing judgments — will preserve them and only score new tickers.")

    # Macro regime is needed when investor agents are on (Druckenmiller uses it)
    macro_score = None
    if include_investor_agents:
        print("Computing macro regime...")
        try:
            from macro_gate import compute_regime
            regime = compute_regime()
            macro_score = regime.get("composite_score")
            print(f"  Macro: {macro_score:.0f}/100 — {regime.get('regime_label', '?')}")
        except Exception as e:
            print(f"  ! macro regime failed: {e}")

    # Load market context once
    tickers_needed = work["ticker"].tolist()
    print("Loading market data...")
    raw = fetch_many(list(set(tickers_needed) | set(ALL_TICKERS)))
    bench = raw[BENCHMARK]["close"]

    equity_closes = {t: df["close"] for t, df in raw.items()
                     if t not in SECTOR_ETFS and t != BENCHMARK}
    rs_df = compute_universe_rs_rank(equity_closes)
    sector_strength = compute_sector_strength(raw, broad_benchmark=BENCHMARK)
    earnings_df = build_earnings_cache([t for t in tickers_needed if t not in SECTOR_ETFS])
    latest = max(df.index[-1] for df in raw.values())

    print(f"Scoring (date: {latest.date()})\n")
    results = []
    skipped_existing = 0
    for _, row in work.iterrows():
        t = row["ticker"]
        acct = row["account_id"]
        acct_label_short = ACCOUNT_LABEL.get(acct, acct)

        # Skip if already done (missing_only mode)
        if existing and (t, acct_label_short) in existing:
            results.append(existing[(t, acct_label_short)])
            skipped_existing += 1
            continue

        if t not in raw or latest not in raw[t].index:
            print(f"  ! {t}: no data")
            continue
        try:
            rs_series = rs_df[t] if t in rs_df.columns else None
            feat = build_features(raw[t], bench, rs_rank_series=rs_series)
            sig = any_breakout_signal(feat)
            comp = signal_components(feat)
            if latest not in feat.index:
                continue
            feat_row = feat.loc[latest]
            sig_row = sig.loc[latest]
            comp_row = comp.loc[latest]
            sec_id = TICKER_TO_SECTOR.get(t)
            sec_str = (float(sector_strength.loc[latest, sec_id])
                       if sec_id and sec_id in sector_strength.columns
                       and latest in sector_strength.index
                       and pd.notna(sector_strength.loc[latest, sec_id])
                       else None)
            days_e = days_to_earnings(earnings_df, t, as_of=latest)
            earnings_lbl = earnings_proximity_label(days_e)

            value = float(row["value"])

            # Optionally pull live research summary for this ticker (web search)
            research_dict = None
            if with_research:
                try:
                    from research import research_ticker
                    rep = research_ticker(t, use_cache=not research_refresh)
                    research_dict = rep.model_dump()
                except Exception as e:
                    print(f"    ! research failed for {t}: {e}")

            theme_tag = THEME_BY_SECTOR_ETF.get(sec_id, "neutral")

            print(f"  Scoring {t} ({ACCOUNT_LABEL.get(acct, acct).split()[0]})  ${value:,.0f}...")
            result = evaluate_full(
                ticker=t,
                feat_row=feat_row, sig_row=sig_row, comp_row=comp_row,
                earnings_label=earnings_lbl, sector_rs=sec_str,
                position_value_usd=value, total_portfolio_usd=pf_total,
                account_type=ACCOUNT_LABEL.get(acct, acct),
                trade_eligible=(acct in TRADE_ELIGIBLE_ACCOUNTS),
                ticker_in_locked=(t in LOCKED_POSITIONS),
                include_investor_agents=include_investor_agents,
                close_series=raw[t]["close"] if include_investor_agents else None,
                macro_score=macro_score,
                theme_tag=theme_tag,
                research_report=research_dict,
            )
            results.append({
                "ticker": t,
                "account": ACCOUNT_LABEL.get(acct, acct),
                "value": value,
                "pct_portfolio": round(value / pf_total * 100, 2),
                "cost_basis": float(row["cost_basis"]) if pd.notna(row.get("cost_basis")) else None,
                "result": result.model_dump(),
            })
        except Exception as e:
            print(f"  ! {t}: {e}")

    save_jsonl("judgments_portfolio", results)
    fresh = len(results) - skipped_existing
    print(f"\nWrote {len(results)} judgments ({fresh} freshly scored, {skipped_existing} preserved from prior run) "
          f"to data/snapshots/{datetime.now().strftime('%Y-%m-%d')}/judgments_portfolio.jsonl")

    # Summary by LB action
    print("\n=== SUMMARY BY ACTION ===")
    from collections import defaultdict
    by_action = defaultdict(list)
    for r in results:
        by_action[r["result"]["pm"]["action"]].append(r)

    for action in ["BUY", "ADD", "HOLD", "TRIM", "EXIT", "AVOID"]:
        rs = by_action.get(action, [])
        if not rs:
            continue
        total = sum(r["value"] for r in rs)
        print(f"\n{action}  ({len(rs)} positions, ${total:,.0f}):")
        for r in sorted(rs, key=lambda x: -x["value"]):
            pm = r["result"]["pm"]
            print(f"  {r['ticker']:6s} ({r['account'].split()[0]:<12}) "
                  f"${r['value']:>8,.0f} ({r['pct_portfolio']:>4.1f}%)  "
                  f"LB {pm['final_score']:>3d}/{pm['confidence']}/10")
            print(f"    {pm['thesis']}")
            print(f"    SIZING: {pm['sizing_note']}")

    manifest = scratchpad.end_run()
    if manifest:
        print(f"\nScratchpad: {manifest['calls']} calls, "
              f"{manifest['input_tokens']:,} in + {manifest['output_tokens']:,} out tokens, "
              f"est ${manifest['est_cost_usd']:.3f} -> {manifest['jsonl_path']}")

    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    include_investor = "--investor-agents" in args
    with_research = "--with-research" in args
    research_refresh = "--refresh" in args
    missing_only = "--missing-only" in args
    run(
        include_investor_agents=include_investor,
        with_research=with_research,
        research_refresh=research_refresh,
        missing_only=missing_only,
    )
