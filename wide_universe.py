"""Build a ~700-1000 name tradeable equity universe.

Sources (in order):
  1. S&P 500 — scraped from Wikipedia
  2. Nasdaq 100 — scraped from Wikipedia
  3. Curated momentum names — hand-maintained list of recent IPOs and
     mid-cap movers that aren't in the major indices yet (NBIS, ALAB, CRWV,
     IREN, BLSH, etc. + portfolio holdings)

Each scrape is cached to `data/universe/<source>_YYYY-MM-DD.parquet` so that:
  - We don't hit Wikipedia on every scan
  - If Wikipedia structure changes and a scrape fails, we fall back to the
    most recent cached copy

Output: `data/universe/wide_universe.parquet` with columns:
  ticker, name, source, sector (optional), gics_industry (optional)
"""

from __future__ import annotations
from io import StringIO
from pathlib import Path
from datetime import datetime
import pandas as pd
import urllib.request

from data_fetch import DATA_DIR


def _fetch_html(url: str) -> str:
    """Fetch URL with a real User-Agent so Wikipedia doesn't 403."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


UNIVERSE_DIR = DATA_DIR / "universe"
UNIVERSE_DIR.mkdir(exist_ok=True)


# Hand-curated names that aren't in S&P / Nasdaq 100 but are momentum-relevant.
# Includes user's portfolio holdings + recent IPOs + AI infra plays.
CURATED_MOMENTUM = [
    # AI infra / semis
    "NBIS", "ALAB", "CRWV", "IREN", "DRAM",
    # Recent IPOs / mid-cap movers
    "ARM", "BLSH", "IONQ", "DOCN", "UPST", "SHOP", "SNDK", "MU",
    # Other momentum candidates (mid-cap AI/tech)
    "RGTI", "QBTS",            # quantum
    "PLTR", "SMCI",            # AI infra (also in indices but worth direct)
    "SOFI", "HOOD",             # fintech mid-cap
    "RKLB", "ASTS", "PL",       # space
    "VRT", "ETN", "PWR",        # power/energy infra (AI tailwind)
    "WULF", "CIFR", "RIOT", "MARA",  # Bitcoin miners (often have HPC pivots)
    "ANET", "DDOG",             # observability + networking
    "LITE", "COHR",             # photonics
    "SPCX",                     # SpaceX OTC proxy (may not have full data)
    "PANW", "CRWD",             # cybersecurity
]


def _scrape_sp500() -> pd.DataFrame:
    """Scrape S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(StringIO(_fetch_html(url)))
    # First table is the constituents list with columns:
    # Symbol, Security, GICS Sector, GICS Sub-Industry, ...
    df = tables[0]
    df.columns = [c.strip() for c in df.columns]
    # Wikipedia column names sometimes shift — be defensive
    sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    name_col = "Security" if "Security" in df.columns else df.columns[1]
    sector_col = "GICS Sector" if "GICS Sector" in df.columns else None
    industry_col = "GICS Sub-Industry" if "GICS Sub-Industry" in df.columns else None

    out = pd.DataFrame({
        "ticker": df[sym_col].str.replace(".", "-", regex=False),  # BRK.B -> BRK-B (yfinance)
        "name": df[name_col],
        "source": "S&P 500",
        "sector": df[sector_col] if sector_col else None,
        "industry": df[industry_col] if industry_col else None,
    })
    return out


def _scrape_nasdaq100() -> pd.DataFrame:
    """Scrape Nasdaq 100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(StringIO(_fetch_html(url)))
    # Nasdaq 100 constituents table — usually one with Ticker + Company columns.
    # Find it by inspecting column names.
    for t in tables:
        cols = [c.strip() if isinstance(c, str) else c for c in t.columns]
        if any("Ticker" in str(c) for c in cols) or any("Symbol" in str(c) for c in cols):
            df = t.copy()
            df.columns = [str(c).strip() for c in df.columns]
            sym_col = next((c for c in df.columns if "Ticker" in c or "Symbol" in c), None)
            name_col = next((c for c in df.columns if "Company" in c or "Security" in c), None)
            sector_col = next((c for c in df.columns if "Sector" in c or "GICS" in c), None)
            if sym_col is None or name_col is None:
                continue
            return pd.DataFrame({
                "ticker": df[sym_col].str.replace(".", "-", regex=False),
                "name": df[name_col],
                "source": "Nasdaq 100",
                "sector": df[sector_col] if sector_col else None,
                "industry": None,
            })
    raise RuntimeError("Nasdaq 100 table not found in Wikipedia tables")


def _cached_or_scrape(name: str, scraper) -> pd.DataFrame:
    """Try cached file first; if missing or older than 7 days, re-scrape with fallback."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache = UNIVERSE_DIR / f"{name}_{today}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    try:
        df = scraper()
        df.to_parquet(cache)
        return df
    except Exception as e:
        print(f"  ! {name} scrape failed ({e}); falling back to most recent cached copy")
        # Find most recent cached file
        cached = sorted(UNIVERSE_DIR.glob(f"{name}_*.parquet"))
        if cached:
            return pd.read_parquet(cached[-1])
        raise RuntimeError(f"{name} scrape failed and no cached fallback available") from e


def build_wide_universe(refresh: bool = False) -> pd.DataFrame:
    """Build (or load cached) the wide universe."""
    out_path = UNIVERSE_DIR / "wide_universe.parquet"
    if not refresh and out_path.exists():
        return pd.read_parquet(out_path)

    print("Scraping S&P 500...")
    sp = _cached_or_scrape("sp500", _scrape_sp500)
    print(f"  {len(sp)} tickers")

    print("Scraping Nasdaq 100...")
    ndx = _cached_or_scrape("nasdaq100", _scrape_nasdaq100)
    print(f"  {len(ndx)} tickers")

    print(f"Adding {len(CURATED_MOMENTUM)} curated momentum names...")
    curated = pd.DataFrame({
        "ticker": CURATED_MOMENTUM,
        "name": CURATED_MOMENTUM,
        "source": "curated_momentum",
        "sector": None,
        "industry": None,
    })

    # Combine + dedupe (keep first occurrence — S&P > Nasdaq > curated in priority)
    all_df = pd.concat([sp, ndx, curated], ignore_index=True)
    all_df = all_df.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)

    # Clean up — yfinance can't handle certain symbols
    bad = ["BF.B", "BRK.B"]  # use BF-B, BRK-B which we already converted
    all_df = all_df[~all_df["ticker"].isin(bad)]

    print(f"\nTotal wide universe: {len(all_df)} unique tickers")
    all_df.to_parquet(out_path)
    return all_df


if __name__ == "__main__":
    df = build_wide_universe(refresh=True)
    print(f"\nSource breakdown:")
    print(df["source"].value_counts().to_string())
    print(f"\nFirst 20 tickers:")
    print(df.head(20).to_string(index=False))
