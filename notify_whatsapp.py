"""WhatsApp push notifications via CallMeBot.

CallMeBot is a free, no-signup gateway for sending WhatsApp messages from
scripts. It runs over the WhatsApp web protocol — perfect for personal daily
push, not for production at scale.

Setup (one-time, ~2 minutes):
  1. Add the number +34 644 64 38 26 to your phone's contacts as "CallMeBot"
  2. From your WhatsApp, send the message: I allow callmebot to send me messages
  3. Within seconds you'll get a reply with your API key (looks like 7-digit number)
  4. Add to .env:
       WHATSAPP_PHONE=15551234567        # your number with country code, no + or spaces
       WHATSAPP_API_KEY=1234567           # the key CallMeBot sent
  5. Test: python notify_whatsapp.py "hello"

CallMeBot URL format:
  https://api.callmebot.com/whatsapp.php?phone=NUMBER&text=TEXT&apikey=KEY

Daily usage:
  python notify_whatsapp.py --daily-summary
  Or wired into daily_run.py automatically.

Notes:
  - CallMeBot is a third-party service. Don't push anything sensitive.
  - There's a rate limit (~1 message every few seconds). Don't loop.
  - Messages over ~1500 characters may be truncated by the gateway.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PHONE = os.environ.get("WHATSAPP_PHONE")
API_KEY = os.environ.get("WHATSAPP_API_KEY")
API_URL = "https://api.callmebot.com/whatsapp.php"

MAX_LEN = 1400  # CallMeBot starts dropping characters above ~1500


def is_configured() -> bool:
    return bool(PHONE and API_KEY)


def send_message(text: str) -> tuple[bool, str]:
    """Send a WhatsApp message. Returns (ok, info)."""
    if not is_configured():
        return False, "WHATSAPP_PHONE / WHATSAPP_API_KEY not set in .env"
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN - 20] + "\n…[truncated]"
    try:
        r = requests.get(
            API_URL,
            params={"phone": PHONE, "apikey": API_KEY, "text": text},
            timeout=15,
        )
        if r.status_code == 200 and "Message queued" in r.text:
            return True, "queued"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"Exception: {e}"


def format_daily_summary() -> str:
    """Compose a daily push covering: macro, CC trades, TRIM-as-CC edge,
    buy-write candidates, options book snapshot.

    Pulls live numbers from today's snapshot artifacts so what you see in
    WhatsApp matches what's in the markdown reports.
    """
    from datetime import datetime
    import json, re
    from data_fetch import DATA_DIR

    today = datetime.now().strftime("%Y-%m-%d")
    snap = DATA_DIR / "snapshots" / today

    lines: list[str] = [f"Trading Agent {today}"]

    # ----- Macro --------------------------------------------------------
    try:
        from macro_gate import compute_regime
        regime = compute_regime()
        score = regime["composite_score"]
        label = regime["regime_label"]
        lines.append(f"\nMacro: {score:.0f}/100 {label}")
    except Exception:
        pass

    # ----- CC trades (today's owned-position recommendations) -----------
    cc_md = snap / "cc_income.md"
    cc_summary = _extract_cc_trades(cc_md)
    if cc_summary:
        lines.append(f"\n{cc_summary}")

    # ----- TRIM-as-CC edge ---------------------------------------------
    trim_summary = _extract_trim_edges(cc_md)
    if trim_summary:
        lines.append(f"\n{trim_summary}")

    # ----- Buy-write candidates ----------------------------------------
    bw_summary = _extract_buywrite(snap / "cc_buywrite.md")
    if bw_summary:
        lines.append(f"\n{bw_summary}")

    # ----- Options book snapshot ---------------------------------------
    try:
        from options import price_user_options
        opts = price_user_options()
        if not opts.empty:
            pnl_col = "current_contract_pnl_usd" if "current_contract_pnl_usd" in opts.columns else "total_pnl_usd"
            total_pnl = int(opts[pnl_col].sum())
            theta = int(-opts["theta_per_day"].sum())
            lines.append(f"\nBook: {len(opts)} short opts, ${total_pnl:+,} P&L, ${theta}/day theta")
            # Assignment risk: any short call with delta>=0.40 OR DTE<=7
            risky = opts[(opts["delta"].abs() >= 0.40) | (opts["dte"] <= 7)]
            if not risky.empty:
                for _, r in risky.iterrows():
                    lines.append(f"⚠️ {r['ticker']} {int(r['strike'])}{r['type'][0]} "
                                 f"Δ{r['delta']:.2f} {int(r['dte'])}d")
    except Exception:
        pass

    return "\n".join(lines)


def _extract_cc_trades(md_path: Path) -> str:
    """Pull the recommended-trades table from cc_income.md and condense to
    one line per trade."""
    if not md_path.exists():
        return ""
    text = md_path.read_text(encoding="utf-8")
    # Find the table after "Today's recommended trades"
    m = _section_table(text, "Today's recommended trades")
    if not m:
        return ""
    rows = _parse_md_table(m)
    if not rows:
        return ""
    out = ["CC trades:"]
    for r in rows:
        ticker = r.get("Ticker", "").strip("*")
        n = r.get("N", "")
        strike = r.get("Strike", "").strip("$")
        expiry = r.get("Expiry", "")[5:]  # MM-DD
        verdict = r.get("Verdict", "").strip()
        annual = r.get("Annual $", "")
        out.append(f"• {ticker} {n}×${strike}C {expiry} {verdict} ({annual})")
    # Total coverage line
    cov = _grep_line(text, "Projected annual income:")
    if cov:
        out.append(cov.replace("- ", "").replace("**", "").strip())
    return "\n".join(out)


def _extract_trim_edges(md_path: Path) -> str:
    """Pull the TRIM-as-CC edge dollar amounts from cc_income.md."""
    if not md_path.exists():
        return ""
    text = md_path.read_text(encoding="utf-8")
    import re
    # Looking for "Expected value: $X (vs spot sell $Y) — +$Z edge"
    matches = re.findall(r"\*\*(TRIM-as-CC|EXIT-as-CC) (\w+)\*\*.*?\*\*\+?\$([\-\d,]+) edge\*\*", text, re.DOTALL)
    if not matches:
        return ""
    out = ["TRIM/EXIT-as-CC edge:"]
    for tag, ticker, edge in matches:
        out.append(f"• {ticker} ({tag}): +${edge} vs spot sell")
    return "\n".join(out)


def _extract_buywrite(md_path: Path) -> str:
    """Pull top buy-write candidates from cc_buywrite.md."""
    if not md_path.exists():
        return ""
    text = md_path.read_text(encoding="utf-8")
    m = _section_table(text, "Top buy-write candidates")
    if not m:
        return ""
    rows = _parse_md_table(m)
    if not rows:
        return ""
    out = ["Buy-write picks:"]
    for r in rows[:3]:
        ticker = r.get("Ticker", "").strip("*")
        spot = r.get("Spot", "")
        yield_ = r.get("Annual yield", "").strip("*")
        cost = r.get("Cost", "")
        out.append(f"• {ticker} {spot} → {yield_} yield ({cost} cap)")
    return "\n".join(out)


def _section_table(text: str, after_header_substring: str) -> str | None:
    """Return the markdown table that appears after a header containing the substring."""
    if after_header_substring not in text:
        return None
    after = text.split(after_header_substring, 1)[1]
    # Table starts at the first line beginning with '|'
    lines = after.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("|")), None)
    if start is None:
        return None
    end = start
    while end < len(lines) and lines[end].strip().startswith("|"):
        end += 1
    return "\n".join(lines[start:end])


def _parse_md_table(table_text: str) -> list[dict]:
    """Parse a markdown table into a list of dicts."""
    rows = [l.strip() for l in table_text.splitlines() if l.strip().startswith("|")]
    if len(rows) < 3:
        return []
    headers = [c.strip() for c in rows[0].strip("|").split("|")]
    out = []
    for r in rows[2:]:  # skip header + separator
        cells = [c.strip() for c in r.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        out.append(dict(zip(headers, cells)))
    return out


def _grep_line(text: str, substring: str) -> str:
    for line in text.splitlines():
        if substring in line:
            return line
    return ""


if __name__ == "__main__":
    args = sys.argv[1:]
    if not is_configured():
        print("WhatsApp not configured. See module docstring.")
        sys.exit(1)

    if "--daily-summary" in args:
        msg = format_daily_summary()
        print(msg)
        print()
        print(f"Sending to {PHONE} via CallMeBot ({len(msg)} chars)...")
        ok, info = send_message(msg)
        print("Sent." if ok else f"Failed: {info}")
    elif args:
        msg = " ".join(args)
        ok, info = send_message(msg)
        print("Sent." if ok else f"Failed: {info}")
    else:
        ok, info = send_message("Trading agent test ping.")
        print("Test sent." if ok else f"Failed: {info}")
