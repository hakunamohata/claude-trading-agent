"""Critic — audits the LB's synthesis against what the specialists actually said.

Tier 3B of the validation stack. Loads a scratchpad run (one JSONL file written
by portfolio_judge.py or watchlist_judge.py), and for each ticker asks:

  Given the specialists' scores, summaries, and red flags, did the LB's final
  action / thesis / sizing line up with the evidence?

This is a pure internal-consistency check. No price outcomes here — those are
Tier 3C's job. The critic catches things like:

  - LB called BUY but the Risk specialist flagged severe concentration
  - LB called HOLD but every specialist scored 80+
  - LB's thesis cited a catalyst no specialist mentioned
  - Druckenmiller said "exhausted late cycle" and LB still recommended ADD

Output:
  data/snapshots/<date>/critiques.jsonl  — one record per ticker
  data/scratchpad/<date>/<run_id>.critique.json  — run-level summary

CLI:
    python critic.py                            # critique today's most recent portfolio_judge run
    python critic.py --run <run_id>             # critique a specific scratchpad run
    python critic.py --kind watchlist_judge     # critique watchlist_judge run instead
"""

from __future__ import annotations
import os
import sys
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from data_fetch import DATA_DIR
import scratchpad

load_dotenv(Path(__file__).parent / ".env")

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")

# Critic system prompt — terse, audit-only, no price/outcome data
SYSTEM_CRITIC = """You are the Critic agent in a portfolio-management system. You audit the synthesizer (LB)'s final verdict against what the SEVEN specialist agents actually said.

You see the full reasoning chain for ONE ticker:
- TechnicalReport, FundamentalReport, SentimentReport, RiskReport (always present)
- MinerviniReport, DruckenmillerReport, BurryReport (sometimes present)
- LB's final PortfolioManagerVerdict (action, score, confidence, thesis, key_risk, sizing_note)

Your job is internal-consistency review ONLY. You do NOT see future prices, news, or outcomes. The question is: given THIS evidence, did LB synthesize correctly?

Look for:
  1. **Ignored signals** — a specialist scored very high or very low and LB's action does not reflect it.
     Example: Risk scored 2/100 flagging concentration; LB still recommended ADD.
  2. **Unjustified emphasis** — LB's thesis cites a factor (e.g., a catalyst, a divergence) that no specialist's summary mentions.
  3. **Confidence-evidence mismatch** — LB confidence 9/10 but specialists disagree wildly (Technical 80, Fundamental 30), or LB confidence 4/10 but all specialists are 75+.
  4. **Direction mismatch** — LB action contradicts the consensus. Example: 5 of 7 specialists score >= 70 and the philosophy agents agree, but LB called TRIM.
  5. **Sizing reality check** — sizing_note out of step with confidence and action. Example: confidence 9, action BUY, but sizing_note says "wait for a pullback, do not initiate now."

Be terse. Score consistency 0-100 where 100 = LB's verdict cleanly follows from the specialists' evidence. Flag findings as a SHORT list — at most 3, each a single sentence. If LB was clearly right, say so and score 90+. If you disagree on the action, propose what the action SHOULD have been and ONE sentence of reasoning.

`summary` MUST be 2-3 sentences max (about 200-400 characters). Do NOT recapitulate the specialists or LB's thesis — only state your audit verdict and the one or two strongest reasons. Verbose summaries will be rejected.

You are not a re-judge. You audit the synthesis. JSON only."""


class CriticVerdict(BaseModel):
    ticker: str
    lb_action: Literal["BUY", "ADD", "HOLD", "TRIM", "EXIT", "AVOID"]
    consistency_score: int = Field(ge=0, le=100, description="100 = LB synthesis cleanly follows evidence")
    agree_with_action: bool
    suggested_action: Literal["BUY", "ADD", "HOLD", "TRIM", "EXIT", "AVOID"] | None = Field(
        default=None, description="If you disagree, what should LB have said?"
    )
    findings: list[str] = Field(default_factory=list, max_length=3, description="Up to 3 single-sentence findings")
    summary: str = Field(max_length=1500)


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _critic_call(payload: dict) -> CriticVerdict | None:
    """Internal: send an already-assembled audit payload to Claude. Returns parsed
    CriticVerdict. Logs to scratchpad if a run is active."""
    client = _get_client()
    user_content = f"Audit this ticker:\n{json.dumps(payload, indent=2, default=str)}"
    kwargs = dict(
        model=MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": SYSTEM_CRITIC,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
        output_format=CriticVerdict,
    )
    if "haiku" not in MODEL.lower() and "sonnet-4-5" not in MODEL.lower():
        kwargs["thinking"] = {"type": "adaptive"}

    t0 = time.perf_counter()
    response = client.messages.parse(**kwargs)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if scratchpad.is_active():
        usage = getattr(response, "usage", None)
        in_tok  = getattr(usage, "input_tokens", 0)  if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        scratchpad.log_call(
            role="critic", ticker=payload.get("ticker", "?"), model=MODEL,
            payload=payload, output=response.parsed_output,
            input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=latency_ms,
        )
    return response.parsed_output


def critique_panel(result, position_context: dict | None = None) -> CriticVerdict | None:
    """Critique a single-ticker MultiAgentResult directly. Use this from the
    Ticker Analysis path (news_to_action.process_message) where the result is
    in memory and no scratchpad run is required.

    `result` is a multi_agent.MultiAgentResult instance.
    `position_context` is optional — pass {"currently_held": bool,
    "position_value_usd": float, "spot": float, "BofA_PO": float} to flag
    common single-ticker traps (held vs unheld; price-vs-PO; etc.)."""
    specialists = {
        "technical":    result.technical.model_dump(),
        "fundamental":  result.fundamental.model_dump(),
        "sentiment":    result.sentiment.model_dump(),
        "risk":         result.risk.model_dump(),
    }
    if result.minervini is not None:
        specialists["minervini"]     = result.minervini.model_dump()
    if result.druckenmiller is not None:
        specialists["druckenmiller"] = result.druckenmiller.model_dump()
    if result.burry is not None:
        specialists["burry"]         = result.burry.model_dump()

    payload = {
        "ticker": result.ticker,
        "specialists": specialists,
        "lb_verdict": result.pm.model_dump(),
    }
    if position_context:
        payload["position_context"] = position_context
    return _critic_call(payload)


def _critique_one(ticker: str, records: list[dict]) -> CriticVerdict | None:
    """Call the Critic on one ticker's scratchpad records. Returns None if PM
    record missing. Used by the batch CLI; new in-memory callers should use
    critique_panel() instead."""
    by_role: dict[str, dict] = {r["role"]: r for r in records if r["ticker"] == ticker}
    if "pm" not in by_role:
        return None

    payload = {
        "ticker": ticker,
        "specialists": {
            role: by_role[role]["output"]
            for role in ("technical", "fundamental", "sentiment", "risk",
                         "minervini", "druckenmiller", "burry")
            if role in by_role
        },
        "lb_verdict": by_role["pm"]["output"],
    }
    return _critic_call(payload)


def _find_run(kind: str, date: str | None = None) -> str:
    """Find the most recent scratchpad run of the given kind on the date (default today)."""
    d = date or datetime.now().strftime("%Y-%m-%d")
    runs = scratchpad.list_runs(d)
    matching = [r for r in runs if kind in r]
    if not matching:
        raise SystemExit(f"No scratchpad run of kind '{kind}' on {d}. Found: {runs or '(none)'}")
    return matching[-1]


def run(run_id: str | None = None, kind: str = "portfolio_judge") -> list[dict]:
    if run_id is None:
        run_id = _find_run(kind)
    print(f"Critiquing scratchpad run: {run_id}")
    manifest, records = scratchpad.load_run(run_id)
    tickers = sorted({r["ticker"] for r in records if r["ticker"]})
    print(f"  {len(tickers)} tickers, {len(records)} records")

    # Open our own scratchpad run so critic calls are themselves logged
    critic_run_id = scratchpad.start_run(
        kind="critic",
        args={"audits_run_id": run_id, "model": MODEL},
    )
    print(f"Critic run: {critic_run_id}")

    out: list[dict] = []
    for t in tickers:
        try:
            verdict = _critique_one(t, records)
            if verdict is None:
                print(f"  ! {t}: no PM record, skipping")
                continue
            out.append({
                "ticker": t,
                "audited_run_id": run_id,
                "verdict": verdict.model_dump(),
            })
            mark = "OK" if verdict.agree_with_action else f"-> {verdict.suggested_action}"
            print(f"  {t:6s}  lb={verdict.lb_action:5s}  consistency={verdict.consistency_score:>3d}  {mark}")
        except Exception as e:
            print(f"  ! {t}: critic failed: {e}")

    # Write results
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = DATA_DIR / "snapshots" / today / "critiques.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nWrote {len(out)} critiques to {out_path}")

    # Run-level summary
    by_consistency = defaultdict(int)
    disagreements = []
    for r in out:
        v = r["verdict"]
        c = v["consistency_score"]
        bucket = "90+" if c >= 90 else "70-89" if c >= 70 else "50-69" if c >= 50 else "<50"
        by_consistency[bucket] += 1
        if not v["agree_with_action"]:
            disagreements.append(r)

    print("\n=== CRITIC SUMMARY ===")
    for bucket in ("90+", "70-89", "50-69", "<50"):
        print(f"  consistency {bucket:6s}: {by_consistency[bucket]:>2d}")
    print(f"  disagreements: {len(disagreements)} of {len(out)}")

    if disagreements:
        print("\nDisagreements:")
        for r in disagreements:
            v = r["verdict"]
            print(f"  {r['ticker']:6s}  lb={v['lb_action']:5s} -> suggested={v['suggested_action']:5s}  "
                  f"consistency={v['consistency_score']}")
            print(f"    {v['summary']}")
            for f in v["findings"]:
                print(f"    - {f}")

    manifest_out = scratchpad.end_run()
    if manifest_out:
        print(f"\nCritic cost: {manifest_out['calls']} calls, "
              f"{manifest_out['input_tokens']:,} in + {manifest_out['output_tokens']:,} out, "
              f"est ${manifest_out['est_cost_usd']:.3f}")

    # Also stash a run-level critique summary next to the audited scratchpad
    audit_summary_path = Path(manifest["jsonl_path"]).with_suffix(".critique.json")
    audit_summary_path.write_text(json.dumps({
        "audited_run_id": run_id,
        "critic_run_id": critic_run_id,
        "ticker_count": len(out),
        "consistency_buckets": dict(by_consistency),
        "disagreements": len(disagreements),
        "out_path": str(out_path),
    }, indent=2), encoding="utf-8")

    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    run_id = None
    kind = "portfolio_judge"
    for i, a in enumerate(args):
        if a == "--run" and i + 1 < len(args):
            run_id = args[i + 1]
        elif a == "--kind" and i + 1 < len(args):
            kind = args[i + 1]
    run(run_id=run_id, kind=kind)
