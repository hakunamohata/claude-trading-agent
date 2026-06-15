"""Telegram bot push notifications for trading agent.

Setup (one-time):
  1. Open Telegram → search @BotFather → /newbot → choose name → save the TOKEN
  2. Send any message to your new bot (e.g. "hi")
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates in browser
     → find your numeric `chat.id` in the JSON
  4. Add to .env:
       TELEGRAM_BOT_TOKEN=...
       TELEGRAM_CHAT_ID=...
  5. Test: `python notify_telegram.py "hello from trading agent"`

Daily usage:
  Called automatically from daily_run.py (Phase F if/when wired).
  Standalone: `python notify_telegram.py --daily-summary`
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else None


def is_configured() -> bool:
    return bool(BOT_TOKEN and CHAT_ID)


def send_message(text: str, parse_mode: str | None = "Markdown") -> bool:
    """Send a Telegram message. Silently returns False if not configured."""
    if not is_configured():
        return False
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(f"{API_BASE}/sendMessage", json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def format_daily_summary() -> str:
    """Compose a daily push: macro regime + options book + portfolio action highlights."""
    from datetime import datetime
    lines = [f"*Trading Agent — {datetime.now().strftime('%Y-%m-%d')}*", ""]

    # Macro regime
    try:
        from macro_gate import compute_regime
        regime = compute_regime()
        score = regime["composite_score"]
        label = regime["regime_label"]
        lines.append(f"*Macro:* {score:.0f}/100 — _{label}_")
        if regime["vix"]["value"]:
            lines.append(f"  VIX {regime['vix']['value']:.1f} · Breadth {regime['breadth']['pct_above_50_ema']:.0f}% > 50EMA")
        lines.append("")
    except Exception as e:
        lines.append(f"_Macro: error ({e})_")
        lines.append("")

    # Options book
    try:
        from options import price_user_options
        opts = price_user_options()
        if not opts.empty:
            total_pnl = int(opts["total_pnl_usd"].sum())
            theta_per_day = int(-opts["theta_per_day"].sum())
            lines.append(f"*Options book:* {len(opts)} positions, ${total_pnl:,} open P&L, ${theta_per_day}/day theta")
            # Surface expiring soon (DTE <=7) and approaching strike (within 5%)
            expiring = opts[opts["dte"] <= 7]
            at_strike = opts[opts["moneyness_pct"].abs() <= 5]
            if not expiring.empty:
                lines.append("  Expiring soon:")
                for _, r in expiring.iterrows():
                    lines.append(f"    {r['ticker']} {r['strike']:.0f}{r['type'][0]} ({r['dte']}d) — close ${r['current_mid']:.2f}, P&L ${int(r['total_pnl_usd']):,}")
            if not at_strike.empty:
                lines.append("  At/near strike:")
                for _, r in at_strike.iterrows():
                    cov = "NAKED" if "NAKED" in r["coverage"] else "covered"
                    lines.append(f"    {r['ticker']} {r['strike']:.0f}{r['type'][0]} delta {r['delta']:.2f} ({cov})")
            lines.append("")
    except Exception as e:
        lines.append(f"_Options: error ({e})_")
        lines.append("")

    # Portfolio actions from latest snapshot
    try:
        import json
        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y-%m-%d")
        from data_fetch import DATA_DIR
        jpath = DATA_DIR / "snapshots" / today / "judgments_portfolio.jsonl"
        if jpath.exists():
            records = [json.loads(l) for l in jpath.read_text(encoding="utf-8").splitlines() if l.strip()]
            by_action = {}
            for r in records:
                a = r["result"]["pm"]["action"]
                by_action.setdefault(a, []).append(r)
            actionable = []
            for a in ["EXIT", "TRIM", "ADD", "BUY"]:
                if a in by_action:
                    n = len(by_action[a])
                    tot = sum(r["value"] for r in by_action[a] if r["value"] == r["value"])  # NaN-safe
                    actionable.append(f"{a} ({n}, ${tot:,.0f})")
            if actionable:
                lines.append(f"*Portfolio actions:* {' · '.join(actionable)}")
                lines.append("")
    except Exception as e:
        lines.append(f"_Portfolio: error ({e})_")

    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not is_configured():
        print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.")
        print("See module docstring for setup steps.")
        sys.exit(1)

    if "--daily-summary" in args:
        msg = format_daily_summary()
        print(msg)
        print()
        print(f"Sending to chat {CHAT_ID}...")
        ok = send_message(msg)
        print("Sent." if ok else "Failed.")
    elif args:
        msg = " ".join(args)
        ok = send_message(msg)
        print("Sent." if ok else "Failed.")
    else:
        # Test ping
        ok = send_message("Trading agent test ping.")
        print("Test message sent." if ok else "Failed — check token + chat_id.")
