"""LB backtest — score the portfolio manager's hit rate against forward returns.

Tier 3C of the validation stack. Reads past LB verdicts and asks the meta-
question for the whole system:

  Have the LB's actions actually been right? Are BUY calls outperforming the
  benchmark? Are EXIT calls avoiding drawdowns? Are HOLD calls noise?

Sources of verdicts:
  1. `data/snapshots/<date>/judgments_portfolio.jsonl`  (pre-3A history works)
  2. `data/scratchpad/<date>/<run_id>.jsonl`            (post-3A, role=='pm')

For each verdict we look up forward returns at N-day windows (default 5, 20,
60), the benchmark return over the same window, and classify hit/miss by a
rule per action:

  BUY/ADD  — hit if excess return > 0
  HOLD     — hit if |excess| < 5%  (small move = HOLD was reasonable)
  TRIM     — hit if excess < 0     (avoiding underperformance)
  EXIT     — hit if absolute return < 0
  AVOID    — hit if excess < 0     (same as TRIM but for non-held names)

The benchmark is QQQ by default (matches scanner/multi_agent default).

Output:
  data/backtest/lb_backtest_<today>.jsonl  — one row per (run_date, ticker, window)
  data/backtest/lb_backtest_<today>.md     — human-readable summary

CLI:
    python lb_backtest.py                 # default windows 5, 20, 60
    python lb_backtest.py --windows 5,10  # custom windows
    python lb_backtest.py --benchmark SPY # default QQQ
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_fetch import DATA_DIR, fetch_many


SNAPSHOTS = DATA_DIR / "snapshots"
SCRATCHPAD = DATA_DIR / "scratchpad"
BACKTEST_DIR = DATA_DIR / "backtest"

DEFAULT_WINDOWS = (5, 20, 60)
DEFAULT_BENCHMARK = "QQQ"
HOLD_BAND_PCT = 5.0  # |excess| within this band -> HOLD was correct


# ---------- Verdict discovery -------------------------------------------------

def _iter_jsonl(p: Path):
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def discover_verdicts() -> list[dict]:
    """Return [{run_date, ticker, account, action, score, confidence, source}]
    pulled from every available history file. Deduplicates by (run_date, ticker, account)
    keeping the most-recent source.
    """
    seen: dict[tuple, dict] = {}

    # 1. judgments_portfolio.jsonl (pre-3A schema)
    for snap_dir in sorted(SNAPSHOTS.glob("*")):
        if not snap_dir.is_dir():
            continue
        run_date = snap_dir.name
        for rec in _iter_jsonl(snap_dir / "judgments_portfolio.jsonl"):
            pm = rec.get("result", {}).get("pm", {})
            if "action" not in pm:
                continue
            key = (run_date, rec["ticker"], rec.get("account", "?"))
            seen[key] = {
                "run_date": run_date,
                "ticker":   rec["ticker"],
                "account":  rec.get("account", "?"),
                "action":   pm["action"],
                "score":    pm.get("final_score"),
                "confidence": pm.get("confidence"),
                "source":   "judgments_portfolio",
            }

    # 2. scratchpad PM records (post-3A) — may add tickers from watchlist_judge
    #    runs that aren't in the portfolio_judge history.
    for date_dir in sorted(SCRATCHPAD.glob("*")):
        if not date_dir.is_dir():
            continue
        run_date = date_dir.name
        for jsonl in date_dir.glob("*.jsonl"):
            for rec in _iter_jsonl(jsonl):
                if rec.get("role") != "pm":
                    continue
                ticker = rec.get("ticker")
                out = rec.get("output", {})
                action = out.get("action")
                if not ticker or not action:
                    continue
                # account is not on scratchpad records; use run_kind as a tag
                key = (run_date, ticker, rec.get("run_kind", "scratchpad"))
                # Don't clobber a portfolio_judge record with a watchlist_judge one
                if key in seen and seen[key]["source"] == "judgments_portfolio":
                    continue
                seen[key] = {
                    "run_date":   run_date,
                    "ticker":     ticker,
                    "account":    rec.get("run_kind", "scratchpad"),
                    "action":     action,
                    "score":      out.get("final_score"),
                    "confidence": out.get("confidence"),
                    "source":     f"scratchpad:{rec['run_id']}",
                }

    return list(seen.values())


# ---------- Forward returns ---------------------------------------------------

def _forward_return(closes: pd.Series, start_date, n_days: int) -> float | None:
    """closes is a Series indexed by datetime, sorted ascending."""
    # locate the run_date bar (or the nearest prior trading day)
    idx = closes.index.searchsorted(start_date, side="right") - 1
    if idx < 0 or idx >= len(closes):
        return None
    target_idx = idx + n_days
    if target_idx >= len(closes):
        return None
    p0 = float(closes.iloc[idx])
    p1 = float(closes.iloc[target_idx])
    if p0 <= 0:
        return None
    return (p1 / p0 - 1) * 100


def _classify(action: str, ticker_ret: float, excess_ret: float) -> bool:
    """Hit/miss rule per action."""
    if action in ("BUY", "ADD"):
        return excess_ret > 0
    if action == "HOLD":
        return abs(excess_ret) < HOLD_BAND_PCT
    if action == "TRIM":
        return excess_ret < 0
    if action == "EXIT":
        return ticker_ret < 0
    if action == "AVOID":
        return excess_ret < 0
    return False


# ---------- Main --------------------------------------------------------------

def run(windows=DEFAULT_WINDOWS, benchmark=DEFAULT_BENCHMARK) -> dict:
    verdicts = discover_verdicts()
    if not verdicts:
        print("No historical verdicts found.")
        return {}

    print(f"Discovered {len(verdicts)} verdicts across "
          f"{len({v['run_date'] for v in verdicts})} run-dates.")

    tickers = sorted({v["ticker"] for v in verdicts} | {benchmark})
    print(f"Loading price data for {len(tickers)} tickers...")
    raw = fetch_many(tickers, force=False)
    if benchmark not in raw:
        raise SystemExit(f"Benchmark {benchmark} not available in fetched data.")
    bench_closes = raw[benchmark]["close"].sort_index()

    today_str = datetime.now().strftime("%Y-%m-%d")
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = BACKTEST_DIR / f"lb_backtest_{today_str}.jsonl"
    out_md    = BACKTEST_DIR / f"lb_backtest_{today_str}.md"

    rows: list[dict] = []
    skipped_no_data    = 0
    skipped_immature   = 0

    for v in verdicts:
        t = v["ticker"]
        if t not in raw:
            skipped_no_data += 1
            continue
        closes = raw[t]["close"].sort_index()
        run_dt = pd.Timestamp(v["run_date"])

        for w in windows:
            tr = _forward_return(closes, run_dt, w)
            br = _forward_return(bench_closes, run_dt, w)
            if tr is None or br is None:
                skipped_immature += 1
                continue
            excess = tr - br
            hit = _classify(v["action"], tr, excess)
            rows.append({
                "run_date":          v["run_date"],
                "ticker":            t,
                "account":           v["account"],
                "action":            v["action"],
                "score":             v["score"],
                "confidence":        v["confidence"],
                "window_days":       w,
                "ticker_return_pct": round(tr, 2),
                "benchmark":         benchmark,
                "benchmark_return_pct": round(br, 2),
                "excess_pct":        round(excess, 2),
                "hit":               bool(hit),
                "source":            v["source"],
            })

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(rows)} backtest rows to {out_jsonl}")
    print(f"  skipped: {skipped_no_data} no-price-data, {skipped_immature} window-not-matured")

    # ---- Aggregate ----------------------------------------------------------
    md_lines = [
        f"# LB Backtest — as of {today_str}",
        "",
        f"- **Benchmark**: {benchmark}",
        f"- **Hit rules**: BUY/ADD = excess > 0; HOLD = |excess| < {HOLD_BAND_PCT}%; TRIM/AVOID = excess < 0; EXIT = return < 0",
        f"- **Verdicts**: {len(verdicts)} ({sum(1 for v in verdicts if v['source'].startswith('scratchpad'))} from scratchpad)",
        f"- **Rows scored**: {len(rows)} (skipped {skipped_immature} window-immature, {skipped_no_data} no-price)",
        "",
    ]

    # By action × window
    by_aw: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_aw[(r["action"], r["window_days"])].append(r)

    md_lines.append("## Hit rate by action and window")
    md_lines.append("")
    md_lines.append("| Action | Window | N | Hit % | Mean excess % | Mean ticker % |")
    md_lines.append("|---|---|---|---|---|---|")
    for action in ("BUY", "ADD", "HOLD", "TRIM", "EXIT", "AVOID"):
        for w in windows:
            rs = by_aw.get((action, w))
            if not rs:
                continue
            n = len(rs)
            hit_pct = sum(1 for r in rs if r["hit"]) / n * 100
            mean_excess = sum(r["excess_pct"] for r in rs) / n
            mean_ticker = sum(r["ticker_return_pct"] for r in rs) / n
            md_lines.append(f"| {action} | {w}d | {n} | {hit_pct:.0f}% | {mean_excess:+.2f}% | {mean_ticker:+.2f}% |")
    md_lines.append("")

    # Worst calls (largest negative excess on a BUY/ADD/HOLD, or largest positive on EXIT/TRIM/AVOID)
    def _badness(r):
        a = r["action"]
        return -r["excess_pct"] if a in ("BUY", "ADD", "HOLD") else r["ticker_return_pct"]
    worst = sorted(rows, key=_badness, reverse=True)[:10]
    md_lines.append("## Top 10 worst calls")
    md_lines.append("")
    md_lines.append("| Date | Ticker | Action | Window | Ticker % | Bench % | Excess % | Hit |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    for r in worst:
        md_lines.append(
            f"| {r['run_date']} | {r['ticker']} | {r['action']} | {r['window_days']}d | "
            f"{r['ticker_return_pct']:+.2f}% | {r['benchmark_return_pct']:+.2f}% | "
            f"{r['excess_pct']:+.2f}% | {'OK' if r['hit'] else 'MISS'} |"
        )

    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote summary to {out_md}")

    # ---- stdout summary -----------------------------------------------------
    print("\n=== HIT RATE BY ACTION ===")
    for action in ("BUY", "ADD", "HOLD", "TRIM", "EXIT", "AVOID"):
        wins = sum(1 for r in rows if r["action"] == action and r["hit"])
        total = sum(1 for r in rows if r["action"] == action)
        if not total:
            continue
        print(f"  {action:6s}  {wins:>3d}/{total:<3d}  {wins/total*100:>5.1f}%")

    return {"rows": rows, "out_jsonl": str(out_jsonl), "out_md": str(out_md)}


if __name__ == "__main__":
    args = sys.argv[1:]
    windows = DEFAULT_WINDOWS
    benchmark = DEFAULT_BENCHMARK
    for i, a in enumerate(args):
        if a == "--windows" and i + 1 < len(args):
            windows = tuple(int(x) for x in args[i + 1].split(","))
        elif a == "--benchmark" and i + 1 < len(args):
            benchmark = args[i + 1]
    run(windows=windows, benchmark=benchmark)
