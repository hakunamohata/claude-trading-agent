"""Run multi-agent judgment on every non-locked portfolio position.

Costs ~$0.025/ticker × 27 non-MSFT positions = ~$0.70 per full run. Cached so
re-runs hit disk.

Output: data/snapshots/<today>/judgments.jsonl with full multi-agent results
"""

from __future__ import annotations
import sys
from datetime import datetime
import pandas as pd

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


NON_EQUITY = {"FDRXX", "SPAXX", "NHFSMKX98", "CASH_ROTH", "CASH_TOD", "CASH_HSA"}


def run() -> list[dict]:
    pf = pd.read_parquet(HOLDINGS_DIR / "positions_current.parquet")
    pf_total = float(pf["value"].sum())

    # Filter to scoreable positions
    work = pf[
        ~pf["ticker"].isin(LOCKED_POSITIONS | NON_EQUITY)
        & pf["ticker"].notna()
    ].copy()
    print(f"Multi-agent on {len(work)} positions (total household ${pf_total:,.0f})")

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
    for _, row in work.iterrows():
        t = row["ticker"]
        acct = row["account_id"]
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
            print(f"  Scoring {t} ({ACCOUNT_LABEL.get(acct, acct).split()[0]})  ${value:,.0f}...")
            result = evaluate_full(
                ticker=t,
                feat_row=feat_row, sig_row=sig_row, comp_row=comp_row,
                earnings_label=earnings_lbl, sector_rs=sec_str,
                position_value_usd=value, total_portfolio_usd=pf_total,
                account_type=ACCOUNT_LABEL.get(acct, acct),
                trade_eligible=(acct in TRADE_ELIGIBLE_ACCOUNTS),
                ticker_in_locked=(t in LOCKED_POSITIONS),
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
    print(f"\nWrote {len(results)} judgments to data/snapshots/{datetime.now().strftime('%Y-%m-%d')}/judgments_portfolio.jsonl")

    # Summary by PM action
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
                  f"PM {pm['final_score']:>3d}/{pm['confidence']}/10")
            print(f"    {pm['thesis']}")
            print(f"    SIZING: {pm['sizing_note']}")

    return results


if __name__ == "__main__":
    run()
