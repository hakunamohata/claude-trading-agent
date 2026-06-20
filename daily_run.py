"""Daily orchestrator — chains the daily ritual into one command.

Sequence (each step is a subprocess so one failure does not abort the rest):
  1. refresh.py            — fetch latest OHLCV for the universe
  2. scanner.py            — produce watchlist.parquet (top 30 by composite score)
  3. options_tracker.py    — re-price short calls, write options_state.parquet
  4. portfolio_judge.py    — (OPTIONAL, --judge) 7-agent LB panel on portfolio
  5. todays_actions.py     — synthesize the one-pager into todays_actions.md

Stages 1-3 + 5 are the cheap daily path (~$0 of LLM tokens).
Stage 4 is the expensive path (~$0.70-$1.10 per run) — opt in with --judge.

Output:
  - data/snapshots/<today>/daily_run.log           — per-step timing + stdout/stderr
  - data/snapshots/<today>/todays_actions.md       — the one-pager (printed at end)
  - data/snapshots/<today>/watchlist.parquet       — scanner output
  - data/snapshots/<today>/options_state.parquet   — options snapshot
  - data/snapshots/<today>/judgments_portfolio.jsonl (only when --judge)

CLI:
    python daily_run.py                # cheap daily (no LB judgments)
    python daily_run.py --judge        # add portfolio_judge.py (7-agent LB)
    python daily_run.py --refresh      # force data refresh (passes --refresh)
    python daily_run.py --top 50       # scanner top N
    python daily_run.py --skip-refresh # skip stage 1 (already-fresh data)
"""

from __future__ import annotations
import sys
import io
import subprocess
import time
from pathlib import Path
from datetime import datetime
import zoneinfo

# Force UTF-8 on Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ET = zoneinfo.ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parent
PYTHON = sys.executable  # use the same interpreter the orchestrator runs under


def _today_dir() -> Path:
    d = REPO / "data" / "snapshots" / datetime.now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _market_phase_warning() -> str | None:
    """Warn if invoked during regular market hours — today's bar will be partial."""
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return None
    h = now_et.hour + now_et.minute / 60.0
    if 9.5 <= h < 16:
        return (
            "WARNING: market is open (US/Eastern). Today's bar will be PARTIAL; "
            "scanner signals and A/D reads will be unreliable. Re-run after 16:00 ET."
        )
    return None


def _run_step(name: str, cmd: list[str], log_path: Path, abort_on_fail: bool) -> tuple[bool, float]:
    """Run a step as a subprocess, tee output to log + stdout. Returns (ok, seconds)."""
    print(f"\n{'='*72}\n[STEP] {name}\n  cmd: {' '.join(cmd)}\n{'='*72}", flush=True)
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as e:
        elapsed = time.time() - start
        msg = f"  [FAIL] could not launch: {e}"
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[{name}] FAIL ({elapsed:.1f}s): {e}\n")
        if abort_on_fail:
            sys.exit(1)
        return False, elapsed

    elapsed = time.time() - start
    out = proc.stdout or ""
    err = proc.stderr or ""
    if out:
        print(out, flush=True)
    if err:
        print(err, file=sys.stderr, flush=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n{'='*72}\n[{name}] {'OK' if proc.returncode == 0 else 'FAIL'} "
                f"({elapsed:.1f}s) rc={proc.returncode}\n")
        f.write(f"cmd: {' '.join(cmd)}\n")
        f.write(f"--- stdout ---\n{out}\n--- stderr ---\n{err}\n")

    ok = proc.returncode == 0
    if not ok:
        print(f"  [FAIL] {name} exited rc={proc.returncode} ({elapsed:.1f}s)", flush=True)
        if abort_on_fail:
            sys.exit(proc.returncode)
    else:
        print(f"  [OK]   {name} ({elapsed:.1f}s)", flush=True)
    return ok, elapsed


def main(argv: list[str]) -> int:
    run_judge = "--judge" in argv
    force_refresh = "--refresh" in argv
    skip_refresh = "--skip-refresh" in argv
    top_n = 30
    for i, a in enumerate(argv):
        if a == "--top" and i + 1 < len(argv):
            top_n = int(argv[i + 1])

    today_dir = _today_dir()
    log_path = today_dir / "daily_run.log"

    started_at = datetime.now()
    header = (
        f"\n=== Daily run @ {started_at.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(judge={run_judge}, refresh={force_refresh}, top={top_n}) ==="
    )
    print(header, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")

    warn = _market_phase_warning()
    if warn:
        print(f"\n{warn}", flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(warn + "\n")

    results: list[tuple[str, bool, float]] = []

    # ---- 1. refresh ----
    if not skip_refresh:
        ok, dt = _run_step("refresh", [PYTHON, "refresh.py"], log_path, abort_on_fail=False)
        results.append(("refresh", ok, dt))
    else:
        print("\n[STEP] refresh — skipped (--skip-refresh)", flush=True)

    # ---- 2. scanner ----
    scan_cmd = [PYTHON, "scanner.py", "--top", str(top_n)]
    if force_refresh:
        scan_cmd.append("--refresh")
    ok, dt = _run_step("scanner", scan_cmd, log_path, abort_on_fail=True)
    results.append(("scanner", ok, dt))

    # ---- 3. options_tracker ----
    ok, dt = _run_step(
        "options_tracker", [PYTHON, "options_tracker.py"], log_path, abort_on_fail=False
    )
    results.append(("options_tracker", ok, dt))

    # ---- 4. cc_income_engine (multi-name covered-call recommendations) ----
    ok, dt = _run_step(
        "cc_income_engine", [PYTHON, "cc_income_engine.py"], log_path, abort_on_fail=False
    )
    results.append(("cc_income_engine", ok, dt))

    # ---- 5. cc_buywrite (screener for unowned buy-write candidates) ----
    ok, dt = _run_step(
        "cc_buywrite", [PYTHON, "cc_buywrite.py"], log_path, abort_on_fail=False
    )
    results.append(("cc_buywrite", ok, dt))

    # ---- 6. qqq_leaps_dipbuy (dip-buy signal + open-LEAPS status) ----
    ok, dt = _run_step(
        "qqq_leaps_dipbuy",
        [PYTHON, "qqq_leaps_dipbuy.py", "--backtest", "--years", "5"],
        log_path, abort_on_fail=False,
    )
    results.append(("qqq_leaps_dipbuy", ok, dt))

    # ---- 7. portfolio_judge (optional, expensive) ----
    if run_judge:
        ok, dt = _run_step(
            "portfolio_judge",
            [PYTHON, "portfolio_judge.py", "--investor-agents"],
            log_path,
            abort_on_fail=False,
        )
        results.append(("portfolio_judge", ok, dt))

    # ---- 8. todays_actions (runs last so it sees outputs from earlier stages) ----
    ok, dt = _run_step(
        "todays_actions", [PYTHON, "todays_actions.py"], log_path, abort_on_fail=False
    )
    results.append(("todays_actions", ok, dt))

    # ---- Summary ----
    total = sum(dt for _, _, dt in results)
    print(f"\n{'='*72}\n[SUMMARY] Daily run finished in {total:.1f}s", flush=True)
    for name, ok, dt in results:
        status = "OK" if ok else "FAIL"
        print(f"  {status:4s}  {name:18s} {dt:6.1f}s", flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[SUMMARY] {total:.1f}s\n")
        for name, ok, dt in results:
            f.write(f"  {'OK' if ok else 'FAIL':4s}  {name:18s} {dt:6.1f}s\n")

    # ---- Tail: print today's one-pager ----
    one_pager = today_dir / "todays_actions.md"
    if one_pager.exists():
        print(f"\n{'='*72}\n[ONE-PAGER] {one_pager}\n{'='*72}\n", flush=True)
        print(one_pager.read_text(encoding="utf-8"), flush=True)
    else:
        print(f"\n[WARN] {one_pager} not found — todays_actions step may have failed.", flush=True)

    # Non-zero exit only if scanner or todays_actions failed (the two critical paths)
    critical_failed = any(
        (not ok) and name in ("scanner", "todays_actions")
        for name, ok, _ in results
    )
    return 1 if critical_failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
