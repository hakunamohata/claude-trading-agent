"""News → Action — turn a free-text news/alert message into a BUY/SELL/HOLD
verdict by running the LB multi-agent panel on the tickers it mentions.

Designed for the caktusjxck WhatsApp feed (or any similar source), but works
on any text containing tickers.

THREE INPUT MODES — same backend:

  1. CLI / paste / pipe (works today, no setup):
       python news_to_action.py "MU breaking out on HBM tailwind, watch $115"
       echo "MU breaking out..." | python news_to_action.py
       python news_to_action.py --read-clipboard

  2. HTTP receiver (for iOS Shortcut share-target):
       python news_to_action.py --serve 8000
       # iPhone POSTs to http://<laptop-ip>:8000/analyze
       #   body: {"message": "..."}
       # returns JSON with verdicts; optionally also pushes a WhatsApp reply

  3. File watcher (for whatsapp-web.js or any future automation):
       python news_to_action.py --watch data/inbox/caktusjxck.txt
       # processes any new line appended to the file

Pipeline per message:
  - Extract tickers ($AAPL form + bare-caps matched against known universe)
  - For each ticker: build features, fetch earnings, compute sector RS,
    look up portfolio position context, run 4-agent panel (~$0.025/ticker)
  - Compose a compact verdict per ticker
  - For new (unowned) names: also surface a buy-write candidate from the CC engine
  - For held names: surface earnings/CC overlay
  - Format for CLI (full) or WhatsApp (compact)

CLI flags:
    --serve <port>          HTTP receiver mode (default port 8000)
    --watch <path>          File watcher mode
    --read-clipboard        Read message from system clipboard
    --no-llm                Skip LB panel — ticker extraction only (cheap dry-run)
    --no-push               Do not push verdict back via WhatsApp
    --compact               Force compact (WhatsApp-style) output
"""

from __future__ import annotations
import io
import json
import os
import re
import sys
import time
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from data_fetch import DATA_DIR, fetch_many
from breakout import (
    build_features, any_breakout_signal, signal_components, compute_universe_rs_rank,
)
from earnings import build_earnings_cache, days_to_earnings, earnings_proximity_label
from sector import compute_sector_strength
from universe import ALL_TICKERS, BENCHMARK, SECTOR_ETFS, TICKER_TO_SECTOR
from multi_agent import evaluate_full
from portfolio import HOLDINGS_DIR, ACCOUNT_LABEL, TRADE_ELIGIBLE_ACCOUNTS, LOCKED_POSITIONS


# ---------- Ticker extraction -----------------------------------------------

# Words that look like tickers but aren't in any real-world news text we care
# about. ALL/ONE/OUT are real tickers but rarely the subject of a news alert,
# so we exclude unless prefixed with $.
_BARE_CAPS_BLACKLIST = {
    "A", "AT", "BE", "I", "IF", "IN", "IS", "IT", "OR", "SO", "TO", "WE",
    "ALL", "AND", "ARE", "BUT", "CAN", "FOR", "GET", "HIS", "HOW", "MAY",
    "NOT", "NOW", "ONE", "OUR", "OUT", "PER", "SEE", "TBD", "THE", "WAS",
    "WHO", "WHY", "YES", "YOU", "USD", "EUR", "GBP", "ATH", "ATL", "DTE",
    "EOD", "EOM", "EOY", "EPS", "IPO", "IRR", "LBO", "OTM", "ITM", "OPEX",
    "PE", "PT", "PEG", "ROE", "ROI", "ROIC", "RSI", "SMA", "EMA", "MACD",
    "FOMC", "CPI", "PPI", "GDP", "VIX", "ETF", "NYSE", "OTC", "AI", "ML",
    "IV", "OI", "PM", "AM", "QQQ",
}


def _load_known_universe() -> set[str]:
    """Tickers we trust as 'real' for bare-caps fallback extraction.
    Pulls the wide universe (~537 names: SP500 + N100 + sectors) so news
    mentions of any major US name are recognised. Falls back to the small
    core list if the wide-universe builder fails."""
    universe = set(ALL_TICKERS or [])
    try:
        from wide_universe import build_wide_universe
        wide = build_wide_universe(refresh=False)
        universe |= set(wide["ticker"].tolist())
    except Exception:
        pass
    try:
        import user_config
        universe |= set(getattr(user_config, "WATCHLIST", []))
        universe |= set(getattr(user_config, "PERSONAL_PORTFOLIO_TICKERS", []))
    except Exception:
        pass
    return universe


_KNOWN_UNIVERSE = None


def known_universe() -> set[str]:
    global _KNOWN_UNIVERSE
    if _KNOWN_UNIVERSE is None:
        _KNOWN_UNIVERSE = _load_known_universe()
    return _KNOWN_UNIVERSE


def extract_tickers(text: str) -> list[str]:
    """Return unique tickers in order of first appearance.

    Rules:
      1. Any token of the form $XXXX (1-5 uppercase letters) is always a ticker.
      2. Any bare 2-5 uppercase letter run is a ticker ONLY if it's in the
         known universe AND not in the blacklist of common abbreviations.
    """
    found: list[str] = []
    seen: set[str] = set()

    # $TICKER form — always treated as ticker
    for m in re.finditer(r"\$([A-Z]{1,5})\b", text):
        t = m.group(1)
        if t not in seen:
            seen.add(t); found.append(t)

    # Bare caps fallback — only if it matches known universe
    universe = known_universe()
    for m in re.finditer(r"\b([A-Z]{2,5})\b", text):
        t = m.group(1)
        if t in seen or t in _BARE_CAPS_BLACKLIST:
            continue
        if t in universe:
            seen.add(t); found.append(t)

    return found


# ---------- Portfolio context lookup ---------------------------------------

def _portfolio_context_for(ticker: str) -> dict:
    """Return shares held, accounts, value%, locked? for one ticker.
    Returns {'held': False, ...} if not owned."""
    try:
        import user_config
        positions = []
        for (acct, t, qty, price, value, basis) in getattr(user_config, "HOLDINGS_CURRENT", []):
            if t == ticker:
                positions.append({"account_id": acct, "shares": float(qty),
                                  "value": float(value), "basis": basis})
        if not positions:
            return {"held": False}
        total_value = sum(p["value"] for p in positions)
        total_shares = sum(p["shares"] for p in positions)
        # Best-effort total household value for %
        try:
            pf = pd.read_parquet(HOLDINGS_DIR / "positions_current.parquet")
            pf_total = float(pf["value"].sum())
        except Exception:
            pf_total = None
        return {
            "held":          True,
            "shares":        total_shares,
            "total_value":   total_value,
            "pct_household": (total_value / pf_total * 100) if pf_total else None,
            "accounts":      [{"label": ACCOUNT_LABEL.get(p["account_id"], p["account_id"]),
                               "id": p["account_id"], "shares": p["shares"], "value": p["value"]}
                              for p in positions],
            "locked":        ticker in LOCKED_POSITIONS,
        }
    except Exception:
        return {"held": False}


# ---------- Per-ticker analysis --------------------------------------------

def analyze_ticker(ticker: str, raw: dict, rs_df: pd.DataFrame,
                   sector_strength: pd.DataFrame, earnings_df: pd.DataFrame,
                   deep: bool = False) -> dict:
    """Run the LB panel on one ticker with full feature context.

    deep=False: 4 functional agents (Technical, Fundamental, Sentiment, Risk) + LB
    deep=True : 4 functional + 3 legendary investor agents (Minervini, Druckenmiller,
                Burry) + macro regime score + cached research report + LB synthesizer
    """
    if ticker not in raw:
        return {"ticker": ticker, "error": f"no price data"}
    df = raw[ticker]
    df = df.dropna(subset=["close"])
    if df.empty:
        return {"ticker": ticker, "error": "no valid bars"}

    bench = raw[BENCHMARK]["close"] if BENCHMARK in raw else None
    if bench is None:
        return {"ticker": ticker, "error": "benchmark missing"}

    try:
        rs_series = rs_df[ticker] if ticker in rs_df.columns else None
        feat = build_features(df, bench, rs_rank_series=rs_series)
        feat = feat.dropna(subset=["close"])
        if feat.empty:
            return {"ticker": ticker, "error": "no features"}
        latest = feat.index[-1]
        feat_row = feat.loc[latest]
        sig = any_breakout_signal(feat)
        comp = signal_components(feat)
        sig_row = sig.loc[latest]
        comp_row = comp.loc[latest]

        sec_id = TICKER_TO_SECTOR.get(ticker)
        sec_str = None
        if sec_id and sec_id in sector_strength.columns and latest in sector_strength.index:
            v = sector_strength.loc[latest, sec_id]
            sec_str = float(v) if pd.notna(v) else None

        days_e = days_to_earnings(earnings_df, ticker, as_of=latest)
        earn_lbl = earnings_proximity_label(days_e)

        pos = _portfolio_context_for(ticker)
        value = pos.get("total_value", 0.0) if pos.get("held") else 0.0
        try:
            pf = pd.read_parquet(HOLDINGS_DIR / "positions_current.parquet")
            pf_total = float(pf["value"].sum())
        except Exception:
            pf_total = 1_000_000.0  # safe fallback for ratio

        account_id = pos.get("accounts", [{}])[0].get("id") if pos.get("held") else None
        trade_eligible = (account_id in TRADE_ELIGIBLE_ACCOUNTS) if account_id else True

        # Deep-mode extras: macro regime, live/cached research report, theme tag
        macro_score = None
        macro_label = None
        research_report = None
        research_error = None
        theme_tag = "neutral"
        if deep:
            try:
                from macro_gate import compute_regime
                m = compute_regime()
                # macro_gate returns composite_score / regime_label — not score / label
                macro_score = float(m.get("composite_score")) if m and m.get("composite_score") is not None else None
                macro_label = m.get("regime_label")
            except Exception as e:
                macro_label = f"unavailable ({e})"
            # Forward-looking catalysts: pull from cache; if missing, FIRE a live
            # web-research call so LB sees real-world context (analyst actions,
            # contract wins, M&A, regulatory, theme strength) — not just charts.
            try:
                from research import load_cached, research_ticker
                cached = load_cached(ticker)
                if cached is not None:
                    research_report = cached.model_dump() if hasattr(cached, "model_dump") else dict(cached)
                else:
                    r = research_ticker(ticker, use_cache=True)
                    research_report = r.model_dump() if hasattr(r, "model_dump") else dict(r)
            except Exception as e:
                research_error = str(e)
            # Theme tag — if research has a sentiment + sector context, surface it
            try:
                from universe import sector_name as _sn, TICKER_TO_SECTOR as _t2s
                sec_id = _t2s.get(ticker)
                sec_lbl = _sn(sec_id) if sec_id else None
                if sec_lbl:
                    theme_tag = sec_lbl
            except Exception:
                pass

        close_series = df["close"] if deep else None

        result = evaluate_full(
            ticker=ticker,
            feat_row=feat_row, sig_row=sig_row, comp_row=comp_row,
            earnings_label=earn_lbl, sector_rs=sec_str,
            position_value_usd=value, total_portfolio_usd=pf_total,
            account_type=ACCOUNT_LABEL.get(account_id, "n/a") if account_id else "not held",
            trade_eligible=trade_eligible,
            ticker_in_locked=(ticker in LOCKED_POSITIONS),
            include_investor_agents=deep,
            close_series=close_series,
            macro_score=macro_score,
            theme_tag=theme_tag,
            research_report=research_report,
        )

        out = {
            "ticker":   ticker,
            "spot":     float(feat_row["close"]),
            "earnings_label": earn_lbl,
            "days_to_earnings": int(days_e) if days_e is not None else None,
            "position": pos,
            "signals_fired": [k for k in ("vcp", "momentum", "emergence", "pocket_pivot") if bool(sig_row.get(k, False))],
            "rs_rank": int(feat_row["rs_rank"]) if pd.notna(feat_row.get("rs_rank")) else None,
            "lb":       result.pm.model_dump(),
            "panel":    {
                "technical":   result.technical.score,
                "fundamental": result.fundamental.score,
                "sentiment":   result.sentiment.score,
                "risk":        result.risk.score,
            },
            "deep": deep,
        }
        if deep:
            if result.minervini is not None:
                out["minervini"]     = result.minervini.model_dump()
            if result.druckenmiller is not None:
                out["druckenmiller"] = result.druckenmiller.model_dump()
            if result.burry is not None:
                out["burry"]         = result.burry.model_dump()
            out["macro"] = {"score": macro_score, "label": macro_label} if (macro_score is not None or macro_label) else None
            out["theme_tag"] = theme_tag
            out["research_present"] = research_report is not None
            if research_report is not None:
                out["research"] = research_report
            elif research_error:
                out["research_error"] = research_error
        return out
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ---------- Buy-write side-look (for unowned names) ------------------------

def _buywrite_alt(ticker: str) -> dict | None:
    """If the ticker is NOT held and looks attractive, fetch one buy-write
    candidate so the verdict can offer 'consider buy-write entry'."""
    try:
        from cc_buywrite import score_buywrite_for_ticker
        from cc_income_engine import compute_technical_features
        raw = fetch_many([ticker, BENCHMARK], force=False)
        bench = raw[BENCHMARK]["close"]
        tech = compute_technical_features(ticker, raw, bench)
        spot = tech.get("spot")
        if not spot:
            return None
        earn_df = build_earnings_cache([ticker])
        ne = earn_df.loc[ticker, "next_earnings"] if ticker in earn_df.index else None
        ne = pd.Timestamp(ne) if ne is not None and not pd.isna(ne) else None
        cands = score_buywrite_for_ticker(
            ticker, spot, dte_min=25, dte_max=50,
            delta_min=0.20, delta_max=0.30,
            tech=tech, next_earnings=ne, max_cost=25_000,
        )
        if not cands:
            return None
        c = cands[0]
        return {
            "strike": c["strike"], "expiry": c["expiry"], "dte": c["dte"],
            "premium": c["mid"], "annualized_yield_pct": c["annualized_cash_yield_pct"],
            "adj_pop": c["adjusted_pop"], "verdict": c["risk_verdict"],
            "cost_per_contract": c["cost_per_contract"],
        }
    except Exception:
        return None


# ---------- Process one message ---------------------------------------------

def process_message(text: str, run_llm: bool = True, want_buywrite: bool = True,
                    deep: bool = False) -> dict:
    tickers = extract_tickers(text)
    if not tickers:
        return {"message": text, "tickers": [], "verdicts": [], "warning": "no tickers detected"}

    verdicts: list[dict] = []
    if not run_llm:
        for t in tickers:
            verdicts.append({"ticker": t, "position": _portfolio_context_for(t)})
        return {"message": text, "tickers": tickers, "verdicts": verdicts, "llm_skipped": True}

    # Load context once for all tickers
    needed = sorted(set(tickers) | set(ALL_TICKERS) | {BENCHMARK})
    raw = fetch_many(needed, force=False)
    equity_closes = {t: df["close"] for t, df in raw.items()
                     if t not in SECTOR_ETFS and t != BENCHMARK}
    rs_df = compute_universe_rs_rank(equity_closes)
    sector_strength = compute_sector_strength(raw, broad_benchmark=BENCHMARK)
    earnings_df = build_earnings_cache(tickers)

    for t in tickers:
        v = analyze_ticker(t, raw, rs_df, sector_strength, earnings_df, deep=deep)
        # If not held and the LB action is BUY/ADD, fetch a buy-write alternative
        if want_buywrite and not v.get("error") and not v.get("position", {}).get("held"):
            if v.get("lb", {}).get("action") in ("BUY", "ADD"):
                v["buywrite_alt"] = _buywrite_alt(t)
        verdicts.append(v)

    return {"message": text, "tickers": tickers, "verdicts": verdicts}


# ---------- Output formatters ----------------------------------------------

def format_cli(result: dict) -> str:
    lines = []
    msg = result.get("message", "")
    lines.append(f"=== Message ({datetime.now().strftime('%H:%M:%S')}) ===")
    lines.append(msg[:300])
    lines.append("")
    if not result.get("tickers"):
        lines.append("(no tickers detected)")
        return "\n".join(lines)
    lines.append(f"Tickers: {', '.join(result['tickers'])}")
    lines.append("")
    for v in result["verdicts"]:
        if v.get("error"):
            lines.append(f"--- {v['ticker']} — ERROR: {v['error']}")
            continue
        if "lb" not in v:
            pos = v.get("position", {})
            held = ("held" if pos.get("held") else "not held")
            lines.append(f"--- {v['ticker']} — {held} (LLM skipped)")
            continue
        lb = v["lb"]
        p  = v["panel"]
        pos = v.get("position", {})
        lines.append(f"--- {v['ticker']} @ ${v['spot']:.2f} ---")
        lines.append(f"  LB: {lb['action']} (score {lb['final_score']}/100, conviction {lb['confidence']}/10)")
        lines.append(f"  Panel: Tech {p['technical']}  Fund {p['fundamental']}  Sent {p['sentiment']}  Risk {p['risk']}")
        sig = v.get("signals_fired") or []
        rs_rank = v.get("rs_rank")
        line = "  Signals: " + ("/".join(s.upper() for s in sig) if sig else "-")
        if rs_rank:
            line += f"   RS {rs_rank}"
        lines.append(line)
        lines.append(f"  Thesis: {lb['thesis']}")
        lines.append(f"  Risk:   {lb['key_risk']}")
        lines.append(f"  Sizing: {lb['sizing_note']}")
        if pos.get("held"):
            for a in pos["accounts"]:
                lines.append(f"  Held: {a['shares']:.0f} sh in {a['label']} (${a['value']:,.0f})")
            if pos.get("locked"):
                lines.append(f"  🔒 LOCKED")
        days_e = v.get("days_to_earnings")
        if days_e is not None and days_e <= 14:
            lines.append(f"  📅 Earnings in {days_e}d")
        if v.get("buywrite_alt"):
            bw = v["buywrite_alt"]
            lines.append(f"  Buy-write alt: ${int(bw['strike'])}C {bw['expiry']} @ ${bw['premium']:.2f} "
                         f"→ {bw['annualized_yield_pct']:.1f}% yield, adj POP {bw['adj_pop']:.0f}% ({bw['verdict']})")
        lines.append("")
    return "\n".join(lines)


def format_compact(result: dict) -> str:
    """Compact for WhatsApp / iPhone notification. Aim under 800 chars."""
    if not result.get("tickers"):
        return "No tickers found in message."
    out = []
    for v in result["verdicts"]:
        if v.get("error"):
            out.append(f"{v['ticker']}: ERR {v['error'][:60]}")
            continue
        lb = v["lb"]
        pos = v.get("position", {})
        line = f"{v['ticker']} → {lb['action']} (LB {lb['final_score']}/{lb['confidence']})"
        if pos.get("held"):
            shares = int(pos["shares"])
            pct = pos.get("pct_household")
            held = f"holding {shares}sh"
            if pct is not None:
                held += f" ({pct:.1f}%)"
            if pos.get("locked"):
                held += " 🔒"
            line += f"\n  {held}"
        else:
            line += "\n  not held"
        line += f"\n  {lb['thesis'][:160]}"
        days_e = v.get("days_to_earnings")
        if days_e is not None and days_e <= 14:
            line += f"\n  📅 earn {days_e}d"
        if v.get("buywrite_alt"):
            bw = v["buywrite_alt"]
            line += f"\n  BW alt: ${int(bw['strike'])}C {bw['expiry']} {bw['annualized_yield_pct']:.0f}% yld ({bw['verdict']})"
        out.append(line)
    return "\n\n".join(out)


# ---------- HTTP server ----------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict | str, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        if isinstance(body, dict):
            self.wfile.write(json.dumps(body, default=str).encode("utf-8"))
        else:
            self.wfile.write(body.encode("utf-8"))

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} {fmt%args}\n")

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"ok": True, "time": datetime.now().isoformat()})
        else:
            self._send(404, {"error": "POST /analyze with {message: ...}"})

    def do_POST(self):
        if self.path not in ("/analyze", "/"):
            self._send(404, {"error": "use /analyze"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace") if length else ""
            ctype = self.headers.get("Content-Type", "")
            if "application/json" in ctype:
                payload = json.loads(raw) if raw else {}
                message = payload.get("message", "")
            else:
                message = raw
            if not message.strip():
                self._send(400, {"error": "empty message"})
                return
            result = process_message(message)
            cli = format_cli(result)
            compact = format_compact(result)
            response = {
                "ok": True,
                "tickers": result["tickers"],
                "verdicts": result["verdicts"],
                "summary_compact": compact,
                "summary_cli": cli,
            }
            # Optional: push compact verdict back via WhatsApp
            if os.environ.get("NEWS_TO_ACTION_PUSH_WHATSAPP", "").lower() in ("1", "true", "yes"):
                try:
                    from notify_whatsapp import send_message
                    ok, info = send_message(compact)
                    response["pushed_whatsapp"] = ok
                    response["push_info"] = info
                except Exception as e:
                    response["pushed_whatsapp"] = False
                    response["push_info"] = str(e)
            self._send(200, response)
        except Exception as e:
            self._send(500, {"error": str(e)})


def serve(port: int):
    server = HTTPServer(("0.0.0.0", port), _Handler)
    print(f"news_to_action listening on 0.0.0.0:{port}")
    print("Endpoints:")
    print(f"  GET  /healthz")
    print(f"  POST /analyze   body: {{\"message\": \"...\"}}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


# ---------- File watcher ----------------------------------------------------

def watch_file(path: Path, poll_seconds: float = 2.0):
    """Tail an inbox file. Each new non-empty line is treated as a message."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    print(f"Watching {path}; press Ctrl+C to stop.")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, io.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_seconds)
                continue
            line = line.strip()
            if not line:
                continue
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] new message: {line[:120]}")
            try:
                result = process_message(line)
                print(format_cli(result))
            except Exception as e:
                print(f"  ! processing failed: {e}")


# ---------- Main -----------------------------------------------------------

def main(argv: list[str]) -> int:
    if "--serve" in argv:
        i = argv.index("--serve")
        port = int(argv[i + 1]) if i + 1 < len(argv) else 8000
        serve(port)
        return 0

    if "--watch" in argv:
        i = argv.index("--watch")
        if i + 1 >= len(argv):
            print("--watch requires a path"); return 1
        watch_file(Path(argv[i + 1]))
        return 0

    # CLI / pipe / paste / clipboard
    run_llm = "--no-llm" not in argv
    compact = "--compact" in argv

    message = ""
    if "--read-clipboard" in argv:
        try:
            import subprocess
            if sys.platform == "win32":
                # PowerShell Get-Clipboard
                r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
                                   capture_output=True, text=True, timeout=5)
                message = r.stdout.strip()
            elif sys.platform == "darwin":
                r = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
                message = r.stdout.strip()
            else:
                r = subprocess.run(["xclip", "-selection", "clipboard", "-o"],
                                   capture_output=True, text=True, timeout=5)
                message = r.stdout.strip()
        except Exception as e:
            print(f"Clipboard read failed: {e}"); return 1
    else:
        # Positional args = the message; or read stdin
        args_msg = [a for a in argv if not a.startswith("--")]
        if args_msg:
            message = " ".join(args_msg)
        elif not sys.stdin.isatty():
            message = sys.stdin.read().strip()

    if not message:
        print(__doc__)
        return 1

    result = process_message(message, run_llm=run_llm)
    print(format_compact(result) if compact else format_cli(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
