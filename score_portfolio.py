"""Score all non-MSFT portfolio positions with Claude and print actions.

Use this CLI when you want recommendations without the dashboard. Same scoring
engine the dashboard's "Score" button uses — results cache to data/judgments/.

Usage:
    python score_portfolio.py
    python score_portfolio.py --refresh   # bypass cache, force fresh scores
"""

from __future__ import annotations
import sys
import pandas as pd

from universe import ALL_TICKERS, BENCHMARK, SECTOR_ETFS, TICKER_TO_SECTOR
from data_fetch import fetch_many
from breakout import build_features, any_breakout_signal, signal_components, compute_universe_rs_rank
from earnings import build_earnings_cache, days_to_earnings, earnings_proximity_label
from sector import compute_sector_strength
from judgment import build_payload, evaluate_ticker
from portfolio import (
    HOLDINGS_DIR, ACCOUNT_LABEL, LOCKED_POSITIONS, TRADE_ELIGIBLE_ACCOUNTS,
    MAX_POSITION_PCT, position_action, margin_annual_cost,
)


def run(force_refresh: bool = False) -> None:
    pf = pd.read_parquet(HOLDINGS_DIR / "positions_current.parquet")
    total = float(pf["value"].sum())
    print(f"Total mapped portfolio: ${total:,.0f}")
    print(f"Excluding LOCKED ({LOCKED_POSITIONS}): "
          f"${pf[~pf['ticker'].isin(LOCKED_POSITIONS)]['value'].sum():,.0f}")
    print(f"Trade-eligible accounts: {[ACCOUNT_LABEL[a] for a in TRADE_ELIGIBLE_ACCOUNTS]}")
    print(f"Margin annual cost: ${margin_annual_cost():,.0f}/year (MSFT account, separate strategy)")
    print()

    print("Loading market data + features...")
    raw = fetch_many(ALL_TICKERS)
    bench = raw[BENCHMARK]["close"]

    equity_closes = {t: df["close"] for t, df in raw.items()
                     if t not in SECTOR_ETFS and t != BENCHMARK}
    rs_rank_df = compute_universe_rs_rank(equity_closes)
    sector_strength = compute_sector_strength(raw, broad_benchmark=BENCHMARK)
    earnings_df = build_earnings_cache([t for t in ALL_TICKERS if t not in SECTOR_ETFS and t != BENCHMARK])

    latest_date = max(df.index[-1] for df in raw.values())

    print(f"Scoring with Claude (date: {latest_date.date()})...\n")
    results = []
    for _, row in pf.iterrows():
        t = row["ticker"]
        acct = row["account_id"]
        if t in LOCKED_POSITIONS:
            continue
        if t not in raw or t in SECTOR_ETFS or t == BENCHMARK:
            continue  # cash / non-equity skipped
        if latest_date not in raw[t].index:
            continue

        rs_series = rs_rank_df[t] if t in rs_rank_df.columns else None
        feat = build_features(raw[t], bench, rs_rank_series=rs_series)
        sig = any_breakout_signal(feat)
        comp = signal_components(feat)
        if latest_date not in feat.index:
            continue

        feat_row = feat.loc[latest_date]
        sig_row = sig.loc[latest_date]
        comp_row = comp.loc[latest_date]

        rs_rank_val = feat_row.get("rs_rank")
        rs_rank_v = int(rs_rank_val) if pd.notna(rs_rank_val) else None
        sec_id = TICKER_TO_SECTOR.get(t)
        sec_str = float(sector_strength.loc[latest_date, sec_id]) if (
            sec_id and sec_id in sector_strength.columns
            and latest_date in sector_strength.index
            and pd.notna(sector_strength.loc[latest_date, sec_id])
        ) else None
        days_e = days_to_earnings(earnings_df, t, as_of=latest_date)
        earnings_lbl = earnings_proximity_label(days_e)
        rs_line_nh = bool(feat_row.get("rs_line_new_high", False))

        payload = build_payload(t, feat_row, sig_row, comp_row,
                                rs_rank=rs_rank_v, sector_rs=sec_str,
                                earnings_label=earnings_lbl, rs_line_new_high=rs_line_nh)
        value = float(row["value"])
        pct = value / total * 100
        payload["current_position_value_usd"] = round(value, 2)
        payload["current_position_pct_of_portfolio"] = round(pct, 2)
        payload["account_type"] = row["account_label"]
        if pd.notna(row.get("cost_basis")):
            payload["unrealized_pnl_pct"] = round((value - row["cost_basis"]) / row["cost_basis"] * 100, 1)

        try:
            j = evaluate_ticker(str(latest_date.date()), f"{t}__{acct}", payload, use_cache=not force_refresh)
        except Exception as e:
            print(f"  ! {t}: {e}")
            continue

        action = position_action(t, acct, value, total, j.score)
        results.append({
            "ticker": t,
            "account": ACCOUNT_LABEL[acct].split()[0] if acct in ACCOUNT_LABEL else acct,
            "value": value,
            "pct": pct,
            "score": j.score,
            "bias": j.bias,
            "action": action,
            "thesis": j.thesis,
            "risks": j.risks,
            "pnl_pct": (value / row["cost_basis"] - 1) * 100 if pd.notna(row.get("cost_basis")) and row["cost_basis"] else None,
        })

    # ---- Print summary ----
    df = pd.DataFrame(results).sort_values(
        by="action",
        key=lambda s: s.map({"TRIM": 0, "AVOID": 1, "BUY": 2, "HOLD": 3, "NOT-ACTIONABLE": 4})
    )

    print("=" * 88)
    print(f"{'TICKER':6s} {'ACCOUNT':14s} {'VALUE':>10s}  {'%':>5s} {'SCORE':>5s} {'BIAS':6s} {'ACTION':>15s}")
    print("=" * 88)
    for _, r in df.iterrows():
        print(f"{r['ticker']:6s} {r['account']:14s} ${r['value']:>8,.0f}  {r['pct']:>4.1f}% {r['score']:>5d} {r['bias']:6s} {r['action']:>15s}")
    print()

    # Action lists
    print("=" * 88)
    print("ACTION BREAKDOWN")
    print("=" * 88)
    for action_type in ["TRIM", "AVOID", "BUY", "HOLD", "NOT-ACTIONABLE"]:
        sub = df[df["action"] == action_type]
        if sub.empty:
            continue
        print(f"\n{action_type}  ({len(sub)} positions, ${sub['value'].sum():,.0f}):")
        for _, r in sub.iterrows():
            print(f"  {r['ticker']:6s} ({r['account']})  score {r['score']}")
            print(f"    THESIS: {r['thesis']}")
            print(f"    RISKS:  {r['risks']}")


if __name__ == "__main__":
    force = "--refresh" in sys.argv or "--force" in sys.argv
    run(force_refresh=force)
