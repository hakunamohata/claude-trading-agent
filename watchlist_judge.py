"""Run multi-agent judgment on the top N from today's scanner watchlist.

Pulls watchlist from data/snapshots/<today>/watchlist.parquet, scores each name
through the 4-agent + PM pipeline, writes results to judgments.jsonl in the same
snapshot dir.

CLI:
    python watchlist_judge.py              # top 10
    python watchlist_judge.py --top 30     # all 30
"""

from __future__ import annotations
import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from data_fetch import fetch_many
from breakout import (
    build_features, any_breakout_signal, signal_components, compute_universe_rs_rank,
)
from earnings import build_earnings_cache, days_to_earnings, earnings_proximity_label
from sector import compute_sector_strength
from snapshot import load_df, save_jsonl, SNAPSHOTS_DIR
from multi_agent import evaluate_full
from wide_universe import build_wide_universe
from universe import TICKER_TO_SECTOR
from portfolio import HOLDINGS_DIR, ACCOUNT_LABEL, TRADE_ELIGIBLE_ACCOUNTS, LOCKED_POSITIONS


def run(top_n: int = 10) -> list[dict]:
    today = datetime.now().strftime("%Y-%m-%d")
    wl = load_df("watchlist", today)
    if wl is None or wl.empty:
        print("No watchlist found. Run scanner.py first.")
        return []

    top = wl.head(top_n).copy()
    print(f"Loaded watchlist top {len(top)} from snapshot {today}")
    tickers = top["ticker"].tolist()

    print("Loading market context...")
    raw = fetch_many(tickers + ["QQQ"] + ["SOXX", "XLK", "XLC", "XLY", "XLP", "XLF", "XLV", "XLI", "XLE", "XLU", "XLB", "XLRE"])
    bench = raw["QQQ"]["close"]

    equity_closes = {t: df["close"] for t, df in raw.items()
                     if t.startswith(("XL", "SOXX", "QQQ")) is False}
    rs_df = compute_universe_rs_rank(equity_closes)
    sector_strength = compute_sector_strength(raw, broad_benchmark="QQQ")
    earnings_df = build_earnings_cache(tickers)

    # Load current portfolio for risk context
    pf = pd.read_parquet(HOLDINGS_DIR / "positions_current.parquet")
    pf_total = float(pf["value"].sum())
    pf_lookup = pf.set_index(["ticker", "account_id"])["value"].to_dict()

    latest = max(df.index[-1] for df in raw.values())
    results = []
    for _, row in top.iterrows():
        t = row["ticker"]
        if t not in raw or latest not in raw[t].index:
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

            # Check if user holds this name and where
            held_positions = [(k, v) for k, v in pf_lookup.items() if k[0] == t]
            position_value = sum(v for _, v in held_positions)
            held_accounts = [k[1] for k, _ in held_positions]
            account_type = ACCOUNT_LABEL.get(held_accounts[0], "not held") if held_accounts else "not held"
            trade_eligible = any(a in TRADE_ELIGIBLE_ACCOUNTS for a in held_accounts) or not held_accounts
            ticker_locked = t in LOCKED_POSITIONS

            print(f"  Scoring {t} (score {row['score']})...")
            result = evaluate_full(
                ticker=t,
                feat_row=feat_row, sig_row=sig_row, comp_row=comp_row,
                earnings_label=earnings_lbl, sector_rs=sec_str,
                position_value_usd=position_value, total_portfolio_usd=pf_total,
                account_type=account_type, trade_eligible=trade_eligible,
                ticker_in_locked=ticker_locked,
            )
            results.append({
                "ticker": t,
                "scanner_score": float(row["score"]),
                "position_held_usd": position_value,
                "account": account_type,
                "result": result.model_dump(),
            })
        except Exception as e:
            print(f"  ! {t}: {e}")

    if results:
        save_jsonl("judgments", results)
        print(f"\nWrote {len(results)} judgments to data/snapshots/{today}/judgments.jsonl")

    # Print summary table
    print("\n=== MULTI-AGENT VERDICTS ===")
    print(f"{'TICKER':6} {'SCAN':5} {'TECH':5} {'FUND':5} {'SENT':5} {'RISK':5} {'PM':5} {'BIAS':12} {'ACTION':7}")
    for r in results:
        pm = r["result"]["pm"]
        print(f"{r['ticker']:6} {r['scanner_score']:>5.1f} "
              f"{r['result']['technical']['score']:>5d} {r['result']['fundamental']['score']:>5d} "
              f"{r['result']['sentiment']['score']:>5d} {r['result']['risk']['score']:>5d} "
              f"{pm['final_score']:>5d} {pm['bias']:>12} {pm['action']:>7}")
    print()
    for r in results:
        pm = r["result"]["pm"]
        print(f"{r['ticker']}  ({pm['action']}, conviction {pm['confidence']}/10)")
        print(f"  THESIS: {pm['thesis']}")
        print(f"  RISK:   {pm['key_risk']}")
        print(f"  SIZING: {pm['sizing_note']}")
        print()
    return results


if __name__ == "__main__":
    top_n = 10
    for i, a in enumerate(sys.argv):
        if a == "--top" and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])
    run(top_n=top_n)
