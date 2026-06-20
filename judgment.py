"""Claude judgment layer — scores each candidate 1-100 with thesis + risks.

Per Anthropic best practices:
  - Model: claude-opus-4-7 (default for the API; user can override via env)
  - Adaptive thinking (Claude decides how much reasoning each candidate needs)
  - System prompt cached (saves ~90% on repeated requests within 5 min)
  - Structured output via Pydantic + messages.parse() — guaranteed valid JSON

Results cached to data/judgments/<date>/<ticker>_<hash>.json so re-runs are free.

Cost estimate (Opus 4.7):
  System (cached): ~600 tokens × $5/M × 0.1 (cache read) = $0.0003 / ticker
  User input:      ~400 tokens × $5/M                    = $0.0020 / ticker
  Output:          ~250 tokens × $25/M                   = $0.0063 / ticker
  ~$0.009 / ticker  →  ~$0.15 per full universe scan today (17 names)
"""

from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
import anthropic
from dotenv import load_dotenv

from data_fetch import DATA_DIR

load_dotenv(Path(__file__).parent / ".env")

JUDGMENTS_DIR = DATA_DIR / "judgments"
JUDGMENTS_DIR.mkdir(exist_ok=True)

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")


# ---------- Structured output schema ----------

class ConfidenceFactors(BaseModel):
    """Six factors that compose into the overall conviction score."""
    setup_quality: int = Field(ge=1, le=10, description="How clean is the chart pattern? 10=textbook")
    trend_regime: int = Field(ge=1, le=10, description="How strong is the Stage 2 trend? 10=monster")
    relative_strength: int = Field(ge=1, le=10, description="How decisively is it outperforming peers and benchmark? 10=elite")
    sector_tailwind: int = Field(ge=1, le=10, description="Is the sector leading? 10=leading by wide margin")
    catalyst_proximity: int = Field(ge=1, le=10, description="Earnings/news risk reward. 10=just-reported beat, 1=imminent unknown")
    risk_reward: int = Field(ge=1, le=10, description="Asymmetry of upside to nearest support. 10=clear runway, 1=extended")


class TickerJudgment(BaseModel):
    score: int = Field(ge=1, le=100, description="Overall conviction 1-100. 80+=high, 50=neutral, 20-=avoid")
    bias: Literal["long", "watch", "avoid"]
    thesis: str = Field(max_length=280, description="One sentence: WHY this could work. No data restatement.")
    risks: str = Field(max_length=280, description="One sentence: WHAT could kill it. Specific and actionable.")
    factors: ConfidenceFactors


# ---------- System prompt (cached) ----------

SYSTEM_PROMPT = """You are a momentum trading advisor evaluating breakout candidates for a personal trading agent.

For each stock, you will receive:
- The signal mode that fired (VCP, MOM, EME, PP) or near-miss state
- Structured technical context: RS rank vs peers, RS vs QQQ, sector strength, earnings proximity, EMAs, volume multiple
- The full per-condition fit/no-fit breakdown

The four signal modes mean:
- VCP (Volatility Contraction Pattern): classical Minervini base breakout — close >= prior 50-day high on 1.5x+ volume, after a quiet base
- MOM (Trend Continuation): names already in established Stage 2 with elite RS — fires on 20-day high breakouts during the run
- EME (Stage 2 Emergence): close reclaims the 200 EMA with 50 EMA > 200 EMA (golden cross required) — catches the START of new uptrends
- PP (Pocket Pivot): up-day where today's volume > max down-day volume of prior 10 days — institutional buying inside a base

YOUR JOB: Score the candidate's conviction on a 1-100 scale, with a bias label (long/watch/avoid), a one-sentence thesis (WHY), and a one-sentence risk (WHAT could kill it).

Scoring guidelines:
- 80-100 LONG: high conviction. Multiple modes confirming OR single clean mode + strong RS + leading sector + no near-term earnings risk.
- 60-79 LONG/WATCH: solid. Good setup with 1-2 caveats. Worth a real-money position size if added.
- 40-59 WATCH: neutral. Marginal — equally likely to work or fail. Wait for confirmation. Add to watchlist.
- 20-39 AVOID: weak. Multiple yellow flags (extended, lagging sector, earnings imminent, RS deteriorating).
- 1-19 AVOID: wrong side of trend. No edge.

Critical:
- Be terse. Thesis and risks must each fit in ~25 words. No data restatement.
- Penalize "EME during deep drawdowns" — even with golden cross, ALAB taught us 200 EMA reclaim alone is noisy when stock is in active selling.
- Reward setups where multiple modes fired on the same bar — that's confirmation across detection lenses.
- For names with earnings CONFIRMED in <14 days, default to WATCH unless thesis is exceptional — even great setups blow up post-earnings. Do NOT penalize "unknown" or "not-imminent" earnings labels — earnings come quarterly, so unknown is not a risk.
- For names with RS rank <60 in our small universe (= bottom third of peers), default to WATCH or AVOID. RS leadership is the #1 edge.
- Recognize when a near-miss is more interesting than a triggered signal (e.g., name at 6/7 conditions with rising RS may be setting up the breakout).

OUTPUT: Strict JSON matching the TickerJudgment schema. No prose outside the JSON."""


# ---------- Caching ----------

def _payload_hash(payload: dict) -> str:
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def _cache_path(date: str, ticker: str, payload: dict) -> Path:
    sub = JUDGMENTS_DIR / date
    sub.mkdir(exist_ok=True)
    return sub / f"{ticker}_{_payload_hash(payload)}.json"


def load_cached_judgment(date: str, ticker: str, payload: dict) -> TickerJudgment | None:
    path = _cache_path(date, ticker, payload)
    if not path.exists():
        return None
    with open(path) as f:
        return TickerJudgment(**json.load(f))


def save_judgment(date: str, ticker: str, payload: dict, judgment: TickerJudgment) -> None:
    path = _cache_path(date, ticker, payload)
    with open(path, "w") as f:
        f.write(judgment.model_dump_json(indent=2))


# ---------- Client (lazy) ----------

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to .env or export it in your shell."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------- Per-ticker evaluation ----------

def evaluate_ticker(date: str, ticker: str, payload: dict,
                    use_cache: bool = True) -> TickerJudgment:
    """Score one ticker. Hits disk cache first; calls Claude on miss."""
    if use_cache:
        cached = load_cached_judgment(date, ticker, payload)
        if cached is not None:
            return cached

    client = _get_client()
    user_content = f"Ticker: {ticker}\nDate: {date}\n\nCandidate data:\n{json.dumps(payload, indent=2, default=str)}"

    kwargs = dict(
        model=MODEL,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_content}],
        output_format=TickerJudgment,
    )
    if "haiku" not in MODEL.lower() and "sonnet-4-5" not in MODEL.lower():
        kwargs["thinking"] = {"type": "adaptive"}
    response = client.messages.parse(**kwargs)

    judgment = response.parsed_output
    save_judgment(date, ticker, payload, judgment)
    return judgment


def evaluate_universe(date: str, payloads_by_ticker: dict[str, dict],
                      use_cache: bool = True) -> dict[str, TickerJudgment]:
    """Score every ticker in the dict. Cached entries are free."""
    out = {}
    for t, payload in payloads_by_ticker.items():
        try:
            out[t] = evaluate_ticker(date, t, payload, use_cache=use_cache)
        except Exception as e:
            print(f"  ! {t}: {e}")
    return out


# ---------- Payload builder (used by scan.py / dashboard.py) ----------

def build_payload(ticker: str, feat_row, sig_row, comp_row,
                  rs_rank, sector_rs, earnings_label,
                  rs_line_new_high: bool) -> dict:
    """Construct the structured input Claude sees for one ticker."""
    return {
        "signals_fired": {
            "VCP": bool(sig_row["vcp"]),
            "MOM": bool(sig_row["momentum"]),
            "EME": bool(sig_row["emergence"]),
            "PP":  bool(sig_row["pocket_pivot"]),
            "any": bool(sig_row["any"]),
        },
        "conditions_met_count": int(sum(comp_row.values)),
        "conditions_breakdown": {k: bool(v) for k, v in comp_row.items()},
        "close": round(float(feat_row["close"]), 2),
        "volume_vs_50d_avg": round(float(feat_row["volume"] / feat_row["vol_avg_50"]), 2)
            if feat_row["vol_avg_50"] else None,
        "rs_rank_vs_peers": int(rs_rank) if rs_rank is not None and not pd.isna(rs_rank) else None,
        "rs_60d_vs_qqq_pct": round(float(feat_row["rs_60_prior"] * 100), 1)
            if pd.notna(feat_row["rs_60_prior"]) else None,
        "sector_rs_vs_qqq_pct": round(float(sector_rs), 1) if sector_rs is not None else None,
        "rs_line_new_50d_high": bool(rs_line_new_high),
        "earnings_proximity": earnings_label,
        "ema_50": round(float(feat_row["ema_50"]), 2),
        "ema_200": round(float(feat_row["ema_200"]), 2),
        "stage_2": bool(feat_row["close_prior"] > feat_row["ema_50_prior"] > feat_row["ema_200_prior"]),
    }
