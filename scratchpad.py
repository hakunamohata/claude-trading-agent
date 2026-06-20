"""Scratchpad — append-only JSONL log of every multi-agent call.

One record per specialist invocation captures: the input payload the model saw,
the parsed output it returned, tokens, latency, and the run it belonged to.
This is the ground-truth audit trail that downstream tooling depends on:

  Tier 3B (Critic agent)   — needs the full reasoning chain to flag "where did
                              LB get it wrong?"
  Tier 3C (LB backtest)    — needs final verdicts + the inputs at decision time
                              so we can roll forward N days and measure outcome.

Design:
  - Caller opens a run with `start_run(kind, args)`, which returns a run_id and
    enables logging at module scope.
  - Every `multi_agent._call_agent(...)` invocation inside the run appends one
    JSON line to `data/scratchpad/<date>/<run_id>.jsonl`.
  - Caller closes the run with `end_run()`, which writes a sibling manifest
    capturing totals (tickers, calls, tokens, est. cost, duration).
  - Outside an active run, `log_call(...)` is a no-op, so ad-hoc test calls
    from a REPL or one-off scripts don't litter the directory.

Cost estimation: uses a tiny per-model price table; misses are logged at $0.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from data_fetch import DATA_DIR

SCRATCHPAD_DIR = DATA_DIR / "scratchpad"


# Per-model pricing in $/1M tokens (input, output). Used for cost estimates only.
# Update when models change. Misses fall back to 0 (still logged).
_PRICING = {
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-opus-4-6":   (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00,  5.00),
}


_active: dict | None = None  # module-level "current run" handle


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _est_cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    price_in, price_out = _PRICING.get(model.split(":")[0], (0.0, 0.0))
    return (in_tok / 1_000_000) * price_in + (out_tok / 1_000_000) * price_out


def start_run(kind: str, args: dict | None = None) -> str:
    """Begin logging multi-agent calls under a new run. Returns the run_id.

    Idempotent within a process — a second call returns the existing run_id
    and does NOT open a second file. Caller is responsible for end_run().
    """
    global _active
    if _active is not None:
        return _active["run_id"]

    date = _today()
    run_id = f"{date}-{kind}-{uuid.uuid4().hex[:6]}"
    out_dir = SCRATCHPAD_DIR / date
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{run_id}.jsonl"

    _active = {
        "run_id":      run_id,
        "kind":        kind,
        "args":        args or {},
        "started_at":  datetime.now().isoformat(timespec="seconds"),
        "started_t":   time.perf_counter(),
        "jsonl_path":  jsonl_path,
        "calls":       0,
        "in_tokens":   0,
        "out_tokens":  0,
        "tickers":     set(),
        "roles":       {},   # role -> count
    }
    return run_id


def is_active() -> bool:
    return _active is not None


def current_run_id() -> str | None:
    return _active["run_id"] if _active else None


def log_call(*, role: str, ticker: str | None, model: str,
             payload: dict, output: dict | Any,
             input_tokens: int, output_tokens: int,
             latency_ms: int) -> None:
    """Append one JSONL line capturing this agent call. No-op outside a run."""
    if _active is None:
        return

    # Pydantic models -> dict
    if hasattr(output, "model_dump"):
        output = output.model_dump()

    rec = {
        "ts":            datetime.now().isoformat(timespec="seconds"),
        "run_id":        _active["run_id"],
        "run_kind":      _active["kind"],
        "role":          role,
        "ticker":        ticker,
        "model":         model,
        "input_tokens":  int(input_tokens),
        "output_tokens": int(output_tokens),
        "latency_ms":    int(latency_ms),
        "payload":       payload,
        "output":        output,
    }
    with _active["jsonl_path"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")

    _active["calls"] += 1
    _active["in_tokens"]  += int(input_tokens)
    _active["out_tokens"] += int(output_tokens)
    if ticker:
        _active["tickers"].add(ticker)
    _active["roles"][role] = _active["roles"].get(role, 0) + 1


def end_run() -> dict | None:
    """Close the active run, write a manifest, return the summary dict."""
    global _active
    if _active is None:
        return None

    duration_s = time.perf_counter() - _active["started_t"]
    # Estimate cost from totals; we lose per-model granularity but this is a
    # ~1-line approximation that's good enough for run-level budgeting. Real
    # cost-per-call lives in the JSONL.
    model = _active["args"].get("model", os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7"))
    est = _est_cost_usd(model, _active["in_tokens"], _active["out_tokens"])

    manifest = {
        "run_id":       _active["run_id"],
        "kind":         _active["kind"],
        "args":         _active["args"],
        "started_at":   _active["started_at"],
        "ended_at":     datetime.now().isoformat(timespec="seconds"),
        "duration_s":   round(duration_s, 2),
        "calls":        _active["calls"],
        "tickers":      sorted(_active["tickers"]),
        "ticker_count": len(_active["tickers"]),
        "roles":        _active["roles"],
        "input_tokens":  _active["in_tokens"],
        "output_tokens": _active["out_tokens"],
        "total_tokens":  _active["in_tokens"] + _active["out_tokens"],
        "est_cost_usd": round(est, 4),
        "jsonl_path":   str(_active["jsonl_path"]),
    }
    manifest_path = _active["jsonl_path"].with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _active = None
    return manifest


# ----- Read-side helpers (used by Tier 3B/3C later) ------------------------

def load_run(run_id: str) -> tuple[dict, list[dict]]:
    """Load (manifest, records) for a past run. Date is parsed from run_id."""
    date = run_id.split("-", 3)[:3]
    date_str = "-".join(date)
    base = SCRATCHPAD_DIR / date_str / run_id
    manifest = json.loads((base.with_suffix(".manifest.json")).read_text(encoding="utf-8"))
    records = []
    with base.with_suffix(".jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return manifest, records


def list_runs(date: str | None = None) -> list[str]:
    """List all run_ids on a given date (defaults to today)."""
    d = SCRATCHPAD_DIR / (date or _today())
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))


if __name__ == "__main__":
    # Tiny smoke test: list runs from today.
    for r in list_runs():
        print(r)
