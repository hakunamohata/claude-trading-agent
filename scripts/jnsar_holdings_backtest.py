"""Run JNSAR long-only backtest on every held name and rank by suitability.

For each held ticker:
  - Pull cached OHLCV from Jan 1, 2026 (already on disk from refresh.py)
  - Seed the EMAs from real Jan-2 H/L/C (no template contamination)
  - Compute 5HEMA, 5LEMA, 5CEMA with alpha = 2/6 (matches Testing.xlsx)
  - JNSAR = AVERAGE of last 5 rows of HEMA/LEMA/CEMA combined (15 values)
  - Detect crosses: CLOSE > JNSAR (enter long), CLOSE < JNSAR (exit)
  - Realistic execution: entry at NEXT day's open, exit at exit-day close
  - Stats: win rate, expectancy, sum of returns, vs buy-and-hold

Output:
  data/snapshots/<today>/jnsar_backtest.md

Cost: $0. No LLM calls. Pure mechanical backtest.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_fetch import DATA_DIR, fetch_many

NON_EQUITY = {"FDRXX", "CASH_ROTH", "CASH_TOD", "CASH_HSA"}
START_DATE = "2026-01-01"

# alpha for 5-period EMA — matches Testing.xlsx formulas exactly: 2/(N+1) = 2/6
ALPHA = 2.0 / 6.0


def compute_jnsar(df):
    """Given a DataFrame with open/high/low/close index by date, return df with
    hema, lema, cema, jnsar columns. Seeds the EMAs from the first row's H/L/C."""
    df = df.copy()
    hema = [df["high"].iloc[0]]
    lema = [df["low"].iloc[0]]
    cema = [df["close"].iloc[0]]
    for i in range(1, len(df)):
        hema.append(hema[-1] + (df["high"].iloc[i]  - hema[-1]) * ALPHA)
        lema.append(lema[-1] + (df["low"].iloc[i]   - lema[-1]) * ALPHA)
        cema.append(cema[-1] + (df["close"].iloc[i] - cema[-1]) * ALPHA)
    df["hema"] = hema
    df["lema"] = lema
    df["cema"] = cema
    jnsar = []
    for i in range(len(df)):
        lo = max(0, i - 4)
        block = df.iloc[lo:i+1][["hema", "lema", "cema"]].values.flatten()
        jnsar.append(block.mean())
    df["jnsar"] = jnsar
    df["above"] = df["close"] > df["jnsar"]
    return df


def walk_signals(df):
    """Return list of trade dicts (entry_date, entry_px, exit_date, exit_px, return_pct, win, days)."""
    trades = []
    in_pos = False
    entry_date = entry_price = None
    for i in range(1, len(df)):
        prev_above = df["above"].iloc[i-1]
        curr_above = df["above"].iloc[i]
        today = df.index[i]
        if not in_pos and curr_above and not prev_above:
            if i + 1 < len(df):
                entry_date = df.index[i+1]
                entry_price = df["open"].iloc[i+1]
                in_pos = True
        elif in_pos and not curr_above and prev_above:
            exit_date = today
            exit_price = df["close"].iloc[i]
            ret = (exit_price / entry_price - 1) * 100
            trades.append({
                "entry_date": entry_date.date(),
                "entry_px": round(entry_price, 2),
                "exit_date": exit_date.date(),
                "exit_px": round(exit_price, 2),
                "return_pct": round(ret, 2),
                "win": ret > 0,
                "days": (exit_date - entry_date).days,
            })
            in_pos = False
    if in_pos:
        exit_date = df.index[-1]
        exit_price = df["close"].iloc[-1]
        ret = (exit_price / entry_price - 1) * 100
        trades.append({
            "entry_date": entry_date.date(),
            "entry_px": round(entry_price, 2),
            "exit_date": f"{exit_date.date()} (open)",
            "exit_px": round(exit_price, 2),
            "return_pct": round(ret, 2),
            "win": ret > 0,
            "days": (exit_date - entry_date).days,
        })
    return trades


def main():
    import user_config as uc

    # All held tickers ex cash sleeves
    tickers = sorted({t for _, t, *_ in uc.HOLDINGS_CURRENT if t not in NON_EQUITY})
    print(f"Backtesting JNSAR on {len(tickers)} held names: {tickers}")

    # Bulk-pull cached data (uses parquet on disk, no network if already fetched)
    raw = fetch_many(tickers, force=False)
    rows = []
    failed = []

    for t in tickers:
        try:
            df = raw[t].dropna(subset=["close"]).copy()
            df = df.loc[df.index >= pd.Timestamp(START_DATE)]
            if len(df) < 20:
                failed.append((t, f"only {len(df)} bars available"))
                continue
            df = compute_jnsar(df)
            trades = walk_signals(df)

            if not trades:
                rows.append({
                    "ticker": t, "n_trades": 0, "wins": 0, "losses": 0,
                    "win_rate": 0.0, "avg_ret": 0.0, "sum_ret": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0,
                    "buy_hold": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2),
                    "edge": None, "verdict": "no signals",
                })
                continue

            wins = sum(1 for tr in trades if tr["win"])
            losses_l = [tr["return_pct"] for tr in trades if not tr["win"]]
            wins_l   = [tr["return_pct"] for tr in trades if tr["win"]]
            n = len(trades)
            win_rate = wins / n * 100
            avg_ret  = sum(tr["return_pct"] for tr in trades) / n
            sum_ret  = sum(tr["return_pct"] for tr in trades)
            avg_win  = (sum(wins_l) / wins) if wins else 0.0
            avg_loss = (sum(losses_l) / len(losses_l)) if losses_l else 0.0
            bh       = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
            edge     = sum_ret - bh    # if positive, JNSAR beat buy-and-hold

            # Verdict heuristic:
            #   win_rate >= 50% AND positive expectancy AND beats B&H  → GOOD FIT
            #   beats B&H but win rate < 50%                            → DEFENSIVE FIT (good in downtrends)
            #   loses vs B&H and win rate < 40%                         → POOR FIT
            #   else                                                    → NEUTRAL
            if win_rate >= 50 and avg_ret > 0 and edge > 0:
                verdict = "GOOD FIT"
            elif edge > 5 and bh < 0:
                verdict = "DEFENSIVE FIT"
            elif edge < -5 or (win_rate < 40 and avg_ret < 0):
                verdict = "POOR FIT"
            else:
                verdict = "NEUTRAL"

            rows.append({
                "ticker":   t,
                "n_trades": n,
                "wins":     wins,
                "losses":   n - wins,
                "win_rate": round(win_rate, 1),
                "avg_ret":  round(avg_ret, 2),
                "sum_ret":  round(sum_ret, 2),
                "avg_win":  round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "buy_hold": round(bh, 2),
                "edge":     round(edge, 2),
                "verdict":  verdict,
                "trades":   trades,
            })
        except Exception as e:
            failed.append((t, str(e)[:80]))

    # Sort by edge descending (JNSAR's value-add vs buy-and-hold)
    rows.sort(key=lambda r: -(r.get("edge") or -999))

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = DATA_DIR / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "jnsar_backtest.md"

    # ----- Build markdown -----
    md = [
        f"# JNSAR Long-Only Backtest — All Held Names",
        f"",
        f"**As of**: {today}",
        f"**Window**: 2026-01-01 to most recent cached close",
        f"**Names tested**: {len(rows)} (held positions ex cash sleeves)",
        f"**Method**: 5-period EMA on H/L/C (α = 2/6) → JNSAR = avg of last 5×3 EMA matrix.",
        f"Long-only: enter next open after CLOSE crosses ABOVE JNSAR; exit at close",
        f"when CLOSE crosses BELOW. No filters, no LLM, pure mechanical.",
        f"",
        f"## Verdict legend",
        f"",
        f"| Verdict | Color | Meaning |",
        f"|---|---|---|",
        f"| GOOD FIT | GREEN | Win rate ≥ 50%, positive expectancy, beats buy-and-hold |",
        f"| DEFENSIVE FIT | YELLOW | Beats buy-and-hold by >5% in a downtrending name (system limits losses) |",
        f"| NEUTRAL | YELLOW | Mixed signal — could go either way |",
        f"| POOR FIT | RED | Loses to buy-and-hold by >5% OR sub-40% win rate AND negative expectancy |",
        f"",
        f"## Ranked by edge vs buy-and-hold",
        f"",
        f"| Ticker | Trades | W / L | Win % | Avg/Trade | Sum returns | B&H % | **Edge** | Avg win | Avg loss | Verdict |",
        f"|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        wl = f"{r['wins']} / {r['losses']}" if r["n_trades"] else "0 / 0"
        edge_str = f"**{r['edge']:+.2f}%**" if r["edge"] is not None else "—"
        md.append(
            f"| {r['ticker']} | {r['n_trades']} | {wl} | {r['win_rate']:.1f}% | "
            f"{r['avg_ret']:+.2f}% | {r['sum_ret']:+.2f}% | {r['buy_hold']:+.2f}% | "
            f"{edge_str} | {r['avg_win']:+.2f}% | {r['avg_loss']:+.2f}% | {r['verdict']} |"
        )

    if failed:
        md.extend([
            f"",
            f"## Skipped",
            f"",
            f"| Ticker | Reason |",
            f"|---|---|",
        ])
        for t, why in failed:
            md.append(f"| {t} | {why} |")

    # Cohort buckets
    good      = [r["ticker"] for r in rows if r["verdict"] == "GOOD FIT"]
    defensive = [r["ticker"] for r in rows if r["verdict"] == "DEFENSIVE FIT"]
    neutral   = [r["ticker"] for r in rows if r["verdict"] == "NEUTRAL"]
    poor      = [r["ticker"] for r in rows if r["verdict"] == "POOR FIT"]
    nosig     = [r["ticker"] for r in rows if r["verdict"] == "no signals"]

    md.extend([
        f"",
        f"## Cohort summary",
        f"",
        f"| Group | Count | Names |",
        f"|---|---|---|",
        f"| GOOD FIT (run JNSAR on these) | {len(good)} | {', '.join(good) or '—'} |",
        f"| DEFENSIVE FIT (downtrend protection) | {len(defensive)} | {', '.join(defensive) or '—'} |",
        f"| NEUTRAL | {len(neutral)} | {', '.join(neutral) or '—'} |",
        f"| POOR FIT (don't run JNSAR on these) | {len(poor)} | {', '.join(poor) or '—'} |",
        f"| No signals fired | {len(nosig)} | {', '.join(nosig) or '—'} |",
        f"",
        f"## How to read the table",
        f"",
        f"- **Trades**: number of complete long entry/exit cycles since Jan 1.",
        f"- **Win rate**: % of trades with positive return (entry → exit).",
        f"- **Avg/Trade**: arithmetic mean return per trade. Positive = profitable system.",
        f"- **Sum returns**: simple sum of all per-trade returns. Approximates total system return.",
        f"- **B&H %**: buy-and-hold return over the same window (Jan 2 to today).",
        f"- **Edge**: Sum returns − B&H. Positive = JNSAR outperformed buy-and-hold.",
        f"- **Avg win / loss**: typical magnitude per winning / losing trade. Bigger avg win than avg loss = asymmetric payoff (good).",
    ])

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out_path}")

    # Console summary
    print()
    print(f"{'Rank':<5} {'Ticker':<7} {'Trades':<7} {'Win%':<6} {'Sum%':<8} {'B&H%':<8} {'Edge%':<8} {'Verdict':<15}")
    print("-" * 85)
    for i, r in enumerate(rows, 1):
        edge_str = f"{r['edge']:+.2f}" if r['edge'] is not None else "—"
        print(f"{i:<5} {r['ticker']:<7} {r['n_trades']:<7} {r['win_rate']:<6.1f} {r['sum_ret']:+7.2f} {r['buy_hold']:+7.2f} {edge_str:<8} {r['verdict']:<15}")
    if failed:
        print(f"\nSkipped: {failed}")


if __name__ == "__main__":
    main()
