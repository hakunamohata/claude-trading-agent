"""Daily snapshot manager — persists each day's scan outputs to a versioned dir.

Schema for each day at `data/snapshots/YYYY-MM-DD/`:
  manifest.json       — run metadata (timestamp, git commit, model, universe size)
  universe.parquet    — tickers scanned today + market cap / liquidity tags
  signals.parquet     — filter outputs (active + setting up) per (ticker, mode)
  watchlist.parquet   — pre-breakout shortlist (top N ranked by setup score)
  judgments.jsonl     — Claude / multi-agent judgments (one JSON per line)
  portfolio.parquet   — holdings as of snapshot

Empty files / missing keys are OK — phases fill the schema incrementally as they
come online (Phase 4 writes manifest only; Phase 1 fills universe + watchlist;
Phase 2 fills judgments).

Use `python snapshot.py` to take a snapshot now.
"""

from __future__ import annotations
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

from data_fetch import DATA_DIR

SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)


SCHEMA_FILES = [
    "manifest.json",
    "universe.parquet",
    "signals.parquet",
    "watchlist.parquet",
    "judgments.jsonl",
    "portfolio.parquet",
]


def _git_short_sha() -> str | None:
    """Current git commit short SHA, or None if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def snapshot_dir(date: str | None = None) -> Path:
    """Return (and create) the snapshot directory for a given date (default: today)."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    d = SNAPSHOTS_DIR / date
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(date: str | None = None, **extra: Any) -> Path:
    """Write the manifest.json for a snapshot. Idempotent — overwrites if exists."""
    d = snapshot_dir(date)
    manifest = {
        "snapshot_date": date or datetime.now().strftime("%Y-%m-%d"),
        "taken_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_short_sha(),
        "schema_files": SCHEMA_FILES,
    }
    manifest.update(extra)
    p = d / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2))
    return p


def save_df(name: str, df: pd.DataFrame, date: str | None = None) -> Path:
    """Save a DataFrame to the snapshot dir as parquet."""
    p = snapshot_dir(date) / name
    if not p.suffix == ".parquet":
        p = p.with_suffix(".parquet")
    df.to_parquet(p)
    return p


def save_jsonl(name: str, records: list[dict], date: str | None = None) -> Path:
    """Save a list of dicts as JSONL."""
    p = snapshot_dir(date) / name
    if not p.suffix == ".jsonl":
        p = p.with_suffix(".jsonl")
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    return p


def load_df(name: str, date: str) -> pd.DataFrame | None:
    """Load a DataFrame from a past snapshot. None if missing."""
    p = SNAPSHOTS_DIR / date / name
    if not p.suffix:
        p = p.with_suffix(".parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_jsonl(name: str, date: str) -> list[dict] | None:
    p = SNAPSHOTS_DIR / date / name
    if not p.suffix:
        p = p.with_suffix(".jsonl")
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def list_snapshots() -> list[str]:
    return sorted([d.name for d in SNAPSHOTS_DIR.iterdir() if d.is_dir()])


if __name__ == "__main__":
    p = write_manifest()
    print(f"Wrote snapshot manifest: {p}")
    print(f"Snapshots so far: {list_snapshots()}")
