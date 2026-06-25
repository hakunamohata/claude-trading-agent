"""Overnight futures snapshot — NQ, ES, RTY, CL, GC via yfinance.

Pulls the most recent close of each major futures contract relative to its
prior close. The resulting `pct` is the overnight bid: a +2% on NQ=F vs a
-0.5% cash close on QQQ tells you the regular session is mispriced for what
comes at the open.

Key derivation:
  - `nq_es_spread_pct` = NQ% - ES%. > +1% = semi-led / tech-led bid.
                                    < -1% = semi-distressed / tech-led sell.
                                    [-1, +1] = broad-based move.

CLI:
    python futures.py        # print current futures snapshot
"""

from __future__ import annotations
from typing import Optional
import yfinance as yf

FUTURES = {
    "NQ=F": "Nasdaq-100",
    "ES=F": "S&P 500",
    "RTY=F": "Russell 2000",
    "CL=F": "WTI Crude",
    "GC=F": "Gold",
}


def _pull_one(symbol: str) -> Optional[dict]:
    try:
        h = yf.Ticker(symbol).history(period="5d", interval="1d")
        if len(h) < 2:
            return None
        last = float(h["Close"].iloc[-1])
        prev = float(h["Close"].iloc[-2])
        return {
            "symbol": symbol,
            "name": FUTURES.get(symbol, symbol),
            "prev": round(prev, 2),
            "last": round(last, 2),
            "pct": round((last / prev - 1) * 100, 2),
        }
    except Exception:
        return None


def snapshot() -> dict:
    """Return futures snapshot for NQ, ES, RTY, CL, GC plus a derived lean."""
    data = {}
    for sym in FUTURES:
        row = _pull_one(sym)
        if row is not None:
            data[sym] = row

    nq = data.get("NQ=F", {}).get("pct")
    es = data.get("ES=F", {}).get("pct")
    spread = (nq - es) if (nq is not None and es is not None) else None

    if spread is None:
        lean = "unknown"
        lean_note = "futures data unavailable"
    elif spread > 1.0:
        lean = "semi-led-up" if nq > 0 else "semi-relative-strength"
        lean_note = f"NQ {nq:+.2f}% vs ES {es:+.2f}% — tech/semi leading"
    elif spread < -1.0:
        lean = "semi-led-down" if nq < 0 else "semi-relative-weakness"
        lean_note = f"NQ {nq:+.2f}% vs ES {es:+.2f}% — tech/semi lagging"
    else:
        lean = "broad-based"
        lean_note = f"NQ {nq:+.2f}% vs ES {es:+.2f}% — broad-based"

    return {
        "data": data,
        "nq_pct": nq,
        "es_pct": es,
        "nq_es_spread_pct": round(spread, 2) if spread is not None else None,
        "lean": lean,
        "lean_note": lean_note,
    }


def futures_one_liner(snap: dict | None = None) -> str:
    """Short human-readable summary for banners/reports."""
    snap = snap if snap is not None else snapshot()
    if snap.get("nq_pct") is None:
        return "Futures: unavailable"
    nq = snap["nq_pct"]
    es = snap["es_pct"]
    rty = snap["data"].get("RTY=F", {}).get("pct")
    parts = [f"NQ {nq:+.2f}%", f"ES {es:+.2f}%"]
    if rty is not None:
        parts.append(f"RTY {rty:+.2f}%")
    note = snap["lean_note"]
    return f"Futures: {' / '.join(parts)} ({note})"


def delay_sells_signal(snap: dict | None = None, threshold: float = 1.5) -> tuple[bool, str]:
    """Return (should_delay, reason).

    If NQ futures gap > +threshold%, the regular session will likely open
    above today's close — selling at the open captures the gap rather than
    selling into today's weakness.
    """
    snap = snap if snap is not None else snapshot()
    nq = snap.get("nq_pct")
    if nq is None:
        return False, "no futures data"
    if nq > threshold:
        return True, f"NQ futures +{nq:.2f}% — defer sells; market gaps up at open"
    return False, f"NQ futures {nq:+.2f}% — no defer"


def cc_premium_tag(snap: dict | None = None) -> tuple[str | None, str | None]:
    """Return (tag, explanation) for CC ticket headers.

    Premium pricing reflects today's close. If futures point materially
    higher than -1%, tomorrow's open premium on the same strike will be
    richer (when gapping up) — fill at the open, not at today's mid.
    """
    snap = snap if snap is not None else snapshot()
    nq = snap.get("nq_pct")
    if nq is None:
        return None, None
    if nq > 1.5:
        return ("PREMIUM ~+20-40% RICHER AT OPEN",
                f"NQ +{nq:.2f}% — semis gap higher → call premiums expand. "
                f"Wait for the open instead of filling at today's mid.")
    if nq > 0.5:
        return ("Premium likely slightly richer at open",
                f"NQ +{nq:.2f}% — mild gap up; premium 5-10% higher.")
    if nq < -1.5:
        return ("PREMIUM LIKELY POOR AT OPEN",
                f"NQ {nq:.2f}% — semis gap lower → premiums compress. "
                f"Consider waiting for an intraday bounce.")
    return None, None


if __name__ == "__main__":
    import json
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    s = snapshot()
    print(json.dumps(s, indent=2))
    print()
    print(futures_one_liner(s))
    delay, reason = delay_sells_signal(s)
    print(f"Delay sells: {delay} ({reason})")
    tag, note = cc_premium_tag(s)
    if tag:
        print(f"CC tag: {tag}")
        print(f"  {note}")
