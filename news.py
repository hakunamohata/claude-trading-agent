"""Daily per-position news feed + Claude summarization.

For each held stock, pulls recent headlines from yfinance and summarizes with
Claude Haiku 4.5 (fast + cheap, ~$0.001/ticker).

Output structure per ticker:
  {
    "ticker": "MSFT",
    "headlines": [{"title": ..., "publisher": ..., "link": ..., "published": ...}, ...],
    "summary": "1-2 sentence Claude summary",
    "sentiment": "bullish" | "neutral" | "bearish",
    "key_catalyst": "earnings beat / guidance raise / downgrade / etc",
  }

CLI:
    python news.py              # pull news for all held positions
    python news.py MSFT NVDA    # specific tickers
"""

from __future__ import annotations
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Literal
import yfinance as yf
from pydantic import BaseModel, Field
import anthropic
from dotenv import load_dotenv
import os

from data_fetch import DATA_DIR

load_dotenv(Path(__file__).parent / ".env")

NEWS_CACHE_DIR = DATA_DIR / "news"
NEWS_CACHE_DIR.mkdir(exist_ok=True)

# Use Haiku 4.5 — fast + cheap for news summarization
HAIKU_MODEL = "claude-haiku-4-5"


class NewsSummary(BaseModel):
    summary: str = Field(max_length=240, description="1-2 sentences on the most material developments")
    sentiment: Literal["bullish", "neutral", "bearish"]
    key_catalyst: str = Field(max_length=120, description="The single most important driver — earnings, downgrade, partnership, etc. 'none' if nothing material.")


_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def fetch_headlines(ticker: str, max_items: int = 15) -> list[dict]:
    """Fetch recent headlines via yfinance. Returns list of {title, publisher, link, published}."""
    try:
        t = yf.Ticker(ticker)
        raw = t.news or []
    except Exception:
        return []

    out = []
    for item in raw[:max_items]:
        # yfinance news shape varies; defensive parsing
        content = item.get("content", item)
        title = content.get("title") or item.get("title") or ""
        publisher = (content.get("provider") or {}).get("displayName") or item.get("publisher", "")
        link = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
        published = content.get("pubDate") or item.get("providerPublishTime", "")
        if title:
            out.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": str(published),
            })
    return out


SUMMARIZER_SYSTEM = """You summarize stock news for a momentum trader. Given a list of recent headlines for a ticker, produce:
1. summary: 1-2 sentences on the MOST MATERIAL developments (earnings, guidance, downgrades, partnerships, regulatory). Ignore generic market commentary.
2. sentiment: bullish / neutral / bearish
3. key_catalyst: the single most important driver. Use 'none' if no material news.

Be terse and factual. JSON only."""


def summarize_headlines(ticker: str, headlines: list[dict]) -> NewsSummary | None:
    if not headlines:
        return None
    client = _get_client()
    headlines_text = "\n".join(f"- {h['title']} ({h['publisher']})" for h in headlines)
    user_msg = f"Ticker: {ticker}\n\nHeadlines:\n{headlines_text}"

    try:
        response = client.messages.parse(
            model=HAIKU_MODEL,
            max_tokens=512,
            system=[{
                "type": "text",
                "text": SUMMARIZER_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_msg}],
            output_format=NewsSummary,
        )
        return response.parsed_output
    except Exception as e:
        print(f"  ! {ticker} summarize error: {e}")
        return None


def get_news(ticker: str, force_refresh: bool = False) -> dict:
    """Fetch + summarize today's news for one ticker. Cached per day."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_path = NEWS_CACHE_DIR / f"{ticker}_{today}.json"
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text())

    headlines = fetch_headlines(ticker)
    summary = summarize_headlines(ticker, headlines)

    result = {
        "ticker": ticker,
        "as_of": today,
        "headlines": headlines,
        "summary": summary.summary if summary else "no recent news",
        "sentiment": summary.sentiment if summary else "neutral",
        "key_catalyst": summary.key_catalyst if summary else "none",
    }
    cache_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def news_for_portfolio() -> list[dict]:
    """Run news fetch + Claude summarization on all held equity tickers."""
    import pandas as pd
    pf = pd.read_parquet(DATA_DIR.parent / "holdings" / "positions_current.parquet")
    NON_EQUITY = {"FDRXX", "CASH_ROTH", "CASH_HSA", "CASH_TOD"}
    tickers = sorted(set(pf["ticker"].dropna()) - NON_EQUITY)
    print(f"Pulling news + summarizing for {len(tickers)} tickers (~$0.001/ticker via Haiku)...")
    results = []
    for t in tickers:
        try:
            r = get_news(t)
            sentiment_marker = {"bullish": "+", "bearish": "-", "neutral": "="}.get(r["sentiment"], "?")
            print(f"  [{sentiment_marker}] {t:6s}  {r['summary'][:80]}")
            results.append(r)
        except Exception as e:
            print(f"  ! {t}: {e}")
    return results


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh = "--refresh" in sys.argv

    if args:
        for t in args:
            r = get_news(t, force_refresh=refresh)
            print(f"\n=== {t} ===")
            print(f"Summary: {r['summary']}")
            print(f"Sentiment: {r['sentiment']}")
            print(f"Catalyst: {r['key_catalyst']}")
            print(f"Headlines ({len(r['headlines'])}):")
            for h in r['headlines'][:5]:
                print(f"  - {h['title']}")
    else:
        news_for_portfolio()
