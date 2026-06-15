"""Force-refresh OHLCV data for the universe.

When to run:
  - Pre-market (~8 AM ET): yesterday's close is fully baked, today not yet started
  - After market close (~5 PM ET): today's close is fully baked
  - Intraday: today's bar will be PARTIAL — close=last price, volume=partial-day,
              high/low=intraday extremes so far. Filter results from a partial
              bar are not trustworthy — see warning below.

Usage:
    python refresh.py                 # refresh entire universe
    python refresh.py NBIS ALAB MU    # refresh specific tickers only
"""

from __future__ import annotations
import sys
from datetime import datetime
import zoneinfo

from universe import ALL_TICKERS
from data_fetch import fetch_many


ET = zoneinfo.ZoneInfo("America/New_York")


def market_phase(now_et: datetime) -> str:
    """Return a label describing what the most-recent bar likely represents."""
    if now_et.weekday() >= 5:
        return "weekend — most recent bar is Friday's close (complete)"
    h = now_et.hour + now_et.minute / 60.0
    if h < 9.5:
        return "pre-market — most recent bar is yesterday's close (complete)"
    if h < 16:
        return f"MARKET OPEN — today's bar is PARTIAL (volume & close not final)"
    if h < 20:
        return "after-hours — today's bar is today's close (complete)"
    return "overnight — today's bar is today's close (complete)"


def main(tickers: list[str] | None = None) -> None:
    tickers = tickers or ALL_TICKERS
    now_et = datetime.now(ET)

    print(f"Refresh run: {now_et.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"Phase: {market_phase(now_et)}")
    print(f"Refreshing {len(tickers)} tickers...\n")

    data = fetch_many(tickers, force=True)

    today = now_et.date()
    for t in sorted(data.keys()):
        df = data[t]
        last_date = df.index[-1].date()
        last_close = df["close"].iloc[-1]
        last_volume = df["volume"].iloc[-1]
        marker = ""
        if last_date == today and 9.5 <= now_et.hour + now_et.minute / 60.0 < 16:
            marker = " <-- PARTIAL bar"
        print(f"  {t:6s} {last_date}  close=${last_close:>9.2f}  vol={int(last_volume):>13,}{marker}")

    if any(d["close"].iloc[-1] and df.index[-1].date() == today for t, d in data.items() if (df := d) is not None) \
            and 9.5 <= now_et.hour + now_et.minute / 60.0 < 16:
        print("\n  WARNING: market is open. Today's bar is partial — running")
        print("  scan.py against this will give unreliable signal/volume reads.")
        print("  Re-run refresh.py after 4 PM ET for end-of-day signals.")


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(args)
