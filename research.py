"""Web-search research agent — Claude reads live news, analyst actions, catalysts.

Uses Anthropic's server-side web_search_20260209 + web_fetch_20260209 tools to
investigate a ticker's recent developments beyond what yfinance exposes. Returns
structured output: catalyst summary, recent material developments, sentiment,
key risks, and source URLs.

Costs ~$0.05-0.15 per ticker (web search adds material cost). Use sparingly on
high-conviction shortlist names. Results cached per (date, ticker) for 24h.

CLI:
    python research.py NVDA               # one ticker
    python research.py NVDA AMD ARM       # multiple
    python research.py --shortlist        # top 10 of today's watchlist
    python research.py NVDA --refresh     # bypass cache
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
import anthropic
from dotenv import load_dotenv

from data_fetch import DATA_DIR

load_dotenv(Path(__file__).parent / ".env")

RESEARCH_DIR = DATA_DIR / "research"
RESEARCH_DIR.mkdir(exist_ok=True)

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")


# ============================================================
# Structured output schema
# ============================================================

class Source(BaseModel):
    url: str
    title: str = Field(max_length=200)


class ResearchReport(BaseModel):
    catalyst_summary: str = Field(
        max_length=1200,
        description="2-3 sentences on the most material catalysts driving the stock in the last 30 days",
    )
    recent_developments: list[str] = Field(
        max_length=6,
        description="Bulleted material events: earnings, guidance, analyst actions, M&A, regulatory, product launches",
    )
    sentiment: Literal["bullish", "neutral", "bearish"]
    key_risks: str = Field(
        max_length=1000,
        description="1-3 sentences on the specific risks/catalysts that could derail this thesis in the next 30 days",
    )
    pending_catalysts: list[str] = Field(
        max_length=4,
        description="Upcoming events (next 30-60 days) that could move the stock",
    )
    sources: list[Source] = Field(
        max_length=6,
        description="URLs cited in the research, with their titles",
    )


# ============================================================
# Client + system prompt
# ============================================================

SYSTEM_PROMPT = """You are a research agent investigating a stock for a momentum trader.

Your job: use web_search and web_fetch to find material recent developments affecting this stock, then return a STRUCTURED summary.

Investigation priorities (in order):
1. Last 30 days of material news — earnings results, guidance changes, M&A, regulatory, key partnerships
2. Recent analyst actions (last 14 days) — upgrades, downgrades, price target changes from major brokers
3. Sector/theme developments — competitor moves, supply chain news, regulatory shifts affecting the industry
4. Upcoming pending catalysts (next 30-60 days) — confirmed events that could move the stock

Search strategy: be efficient. Run 3-5 targeted searches max. Suggested queries:
  - "<TICKER> earnings results"
  - "<TICKER> analyst upgrade downgrade"
  - "<TICKER> news <current_month>"
  - "<TICKER> guidance forecast"

For each material finding, cite the source URL. Avoid generic market commentary or hype articles — focus on factual developments.

Output: ResearchReport JSON. Be terse and factual. No filler."""


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ============================================================
# Cache
# ============================================================

def _cache_path(ticker: str, date: str | None = None) -> Path:
    date = date or datetime.now().strftime("%Y-%m-%d")
    d = RESEARCH_DIR / date
    d.mkdir(exist_ok=True)
    return d / f"{ticker}.json"


def load_cached(ticker: str, date: str | None = None) -> ResearchReport | None:
    p = _cache_path(ticker, date)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return ResearchReport(**data)
    except Exception:
        return None


def save_research(ticker: str, report: ResearchReport, date: str | None = None) -> None:
    p = _cache_path(ticker, date)
    p.write_text(report.model_dump_json(indent=2))


# ============================================================
# Research call
# ============================================================

def research_ticker(ticker: str, use_cache: bool = True,
                    extra_context: str | None = None) -> ResearchReport:
    """Research a single ticker. Returns structured ResearchReport.

    extra_context: optional string with additional info (e.g. "user holds X shares
                   at Y cost basis", "earnings in 14 days") — helps Claude focus.
    """
    if use_cache:
        cached = load_cached(ticker)
        if cached is not None:
            return cached

    client = _get_client()

    user_content = f"Research ticker: **{ticker}**.\nDate: {datetime.now().strftime('%Y-%m-%d')}"
    if extra_context:
        user_content += f"\n\nAdditional context:\n{extra_context}"
    user_content += (
        "\n\nInvestigate recent developments and return a ResearchReport. "
        "Focus on material catalysts, not market commentary."
    )

    kwargs = dict(
        model=MODEL,
        max_tokens=4096,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[
            # allowed_callers=["direct"] keeps the tools working on Haiku 4.5,
            # which doesn't support programmatic tool calling (PTC).
            {"type": "web_search_20260209", "name": "web_search", "allowed_callers": ["direct"]},
            {"type": "web_fetch_20260209", "name": "web_fetch", "allowed_callers": ["direct"]},
        ],
        messages=[{"role": "user", "content": user_content}],
        output_format=ResearchReport,
    )
    # Adaptive thinking only on Opus 4.6+ / Sonnet 4.6; skip on Haiku
    if "haiku" not in MODEL.lower() and "sonnet-4-5" not in MODEL.lower():
        kwargs["thinking"] = {"type": "adaptive"}
    response = client.messages.parse(**kwargs)

    report = response.parsed_output
    save_research(ticker, report)
    return report


def research_shortlist(top_n: int = 10) -> dict[str, ResearchReport]:
    """Research the top N from today's scanner watchlist."""
    today = datetime.now().strftime("%Y-%m-%d")
    wl_path = DATA_DIR / "snapshots" / today / "watchlist.parquet"
    if not wl_path.exists():
        print("No watchlist found. Run scanner.py first.")
        return {}
    import pandas as pd
    wl = pd.read_parquet(wl_path)
    tickers = wl.head(top_n)["ticker"].tolist()
    return {t: research_ticker(t) for t in tickers}


# ============================================================
# CLI
# ============================================================

def _print_report(ticker: str, r: ResearchReport) -> None:
    print(f"\n{'='*60}")
    print(f"  {ticker}  —  sentiment: {r.sentiment.upper()}")
    print(f"{'='*60}")
    print(f"\nCATALYST SUMMARY:\n  {r.catalyst_summary}\n")
    if r.recent_developments:
        print("RECENT DEVELOPMENTS:")
        for d in r.recent_developments:
            print(f"  - {d}")
        print()
    if r.pending_catalysts:
        print("PENDING CATALYSTS (next 30-60d):")
        for c in r.pending_catalysts:
            print(f"  - {c}")
        print()
    print(f"KEY RISKS:\n  {r.key_risks}\n")
    if r.sources:
        print("SOURCES:")
        for s in r.sources:
            print(f"  - {s.title[:80]}\n    {s.url}")


if __name__ == "__main__":
    args = sys.argv[1:]
    refresh = "--refresh" in args
    shortlist = "--shortlist" in args
    tickers = [a for a in args if not a.startswith("--")]

    if shortlist:
        n = 10
        if "--top" in args:
            idx = args.index("--top")
            if idx + 1 < len(args):
                try:
                    n = int(args[idx + 1])
                except ValueError:
                    pass
        results = research_shortlist(top_n=n)
        for t, r in results.items():
            _print_report(t, r)
    elif tickers:
        for t in tickers:
            r = research_ticker(t, use_cache=not refresh)
            _print_report(t, r)
    else:
        print("Usage:")
        print("  python research.py TICKER [TICKER2 ...] [--refresh]")
        print("  python research.py --shortlist [--top N]")
