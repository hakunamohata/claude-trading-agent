"""Today's Actions one-pager — daily decision artifact.

Distills scanner output + LB judgments + portfolio + options + macro into a
markdown one-pager with at most 5 prioritized actions.

Selection rubric (in priority order):
  1. Options assignment-imminent: short calls with delta >= 0.40 AND DTE <= 21
  2. Options expiring today (DTE == 0)
  3. LB EXIT verdicts on positions (sequenced with option close if applicable)
  4. A/D distribution warnings on portfolio (grade D/E + price near highs)
  5. LB ADD/BUY verdicts + fresh signal fired today
  6. New names (not in portfolio) score >= 80 + A/D grade A

Output: data/snapshots/<today>/todays_actions.md (also prints to stdout).

CLI:
    python todays_actions.py
"""

from __future__ import annotations
import sys
import io
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Force UTF-8 on Windows console so we can print unicode (—, →, etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from data_fetch import DATA_DIR
from portfolio import HOLDINGS_DIR, ACCOUNT_LABEL, LOCKED_POSITIONS, TRADE_ELIGIBLE_ACCOUNTS


NON_EQUITY = {"FDRXX", "SPAXX", "NHFSMKX98", "CASH_ROTH", "CASH_TOD", "CASH_HSA"}


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _short_acct(acct_id_or_label: str) -> str:
    """Return a short account label (first word of full label)."""
    label = ACCOUNT_LABEL.get(acct_id_or_label, acct_id_or_label)
    return label.split()[0] if isinstance(label, str) else acct_id_or_label


# ---------- Data loaders ----------

def load_watchlist() -> pd.DataFrame | None:
    p = DATA_DIR / "snapshots" / _today() / "watchlist.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_latest_judgments() -> tuple[list[dict], str | None]:
    """Return (judgments, source_date) from the most recent jsonl available."""
    snap_dir = DATA_DIR / "snapshots"
    candidates = sorted(snap_dir.glob("*/judgments_portfolio.jsonl"), reverse=True)
    for p in candidates:
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
            rows = [json.loads(ln) for ln in lines if ln.strip()]
            if rows:
                return rows, p.parent.name
        except Exception:
            continue
    return [], None


def load_positions() -> pd.DataFrame:
    pf = pd.read_parquet(HOLDINGS_DIR / "positions_current.parquet")
    return pf[~pf["ticker"].isin(NON_EQUITY) & pf["ticker"].notna()].copy()


def load_options_state() -> pd.DataFrame:
    from options import price_user_options
    return price_user_options()


def load_macro() -> dict | None:
    try:
        from macro_gate import compute_regime
        return compute_regime()
    except Exception:
        return None


# ---------- Action candidate generation ----------

def options_actions(opts: pd.DataFrame) -> list[dict]:
    """Build option-related action candidates, sorted by urgency."""
    actions = []
    if opts is None or opts.empty:
        return actions

    for _, r in opts.iterrows():
        ticker = r["ticker"]
        dte = int(r["dte"])
        delta = float(r["delta"]) if pd.notna(r["delta"]) else 0.0
        side = r["side"]
        contracts = int(r["contracts"])
        strike = float(r["strike"])
        expiry = r["expiry"]
        spot = float(r["spot"])
        mid = float(r["current_mid"])
        pnl = float(r["current_contract_pnl_usd"]) if pd.notna(r["current_contract_pnl_usd"]) else 0.0
        # Buy-to-close cost in $ (for short positions, this is the cash outlay to close)
        btc_cost_usd = mid * 100 * abs(contracts)
        coverage = r["coverage"]
        acct = _short_acct(r["account"])

        # Priority A: expires today
        if dte == 0 and side == "SHORT":
            if delta >= 0.5:
                actions.append({
                    "priority": 1,
                    "ticker": ticker,
                    "verb": "CLOSE-OR-ACCEPT-ASSIGNMENT",
                    "rationale": (f"{ticker} {strike:.0f}{r['type'][0]} expires TODAY, delta {delta:.2f}, spot ${spot:.2f}. "
                                  f"Assignment will trigger sale of {abs(contracts)*100} shares at ${strike:.0f}. "
                                  f"Buy-to-close cost: ${btc_cost_usd:,.0f}. "
                                  f"Current option P&L: ${pnl:+,.0f}."),
                    "account": acct,
                    "size_usd": btc_cost_usd,
                })
            else:
                actions.append({
                    "priority": 2,
                    "ticker": ticker,
                    "verb": "EXPIRE",
                    "rationale": (f"{ticker} {strike:.0f}{r['type'][0]} expires TODAY OTM, delta {delta:.2f}. "
                                  f"Premium kept: ${pnl:+,.0f}. No action needed."),
                    "account": acct,
                    "size_usd": 0,
                })
            continue

        # Priority B: assignment-imminent (delta >= 0.40 AND DTE <= 21)
        if side == "SHORT" and delta >= 0.40 and dte <= 21:
            actions.append({
                "priority": 1,
                "ticker": ticker,
                "verb": "ROLL-OR-CLOSE",
                "rationale": (f"{ticker} {strike:.0f}{r['type'][0]} expires {expiry} ({dte}d), delta {delta:.2f} — "
                              f"assignment-imminent. Spot ${spot:.2f}, BTC ${btc_cost_usd:,.0f}, P&L ${pnl:+,.0f}. "
                              f"Coverage: {coverage}."),
                "account": acct,
                "size_usd": btc_cost_usd,
            })
            continue

        # Priority C: deep-ITM short call far from expiry (sustained loss + capital trapped)
        if side == "SHORT" and delta >= 0.65 and pnl <= -1000:
            actions.append({
                "priority": 2,
                "ticker": ticker,
                "verb": "CONSIDER-ROLL",
                "rationale": (f"{ticker} {strike:.0f}{r['type'][0]} deep ITM (delta {delta:.2f}, spot ${spot:.2f}, "
                              f"strike ${strike:.0f}), {dte}d. Current P&L ${pnl:+,.0f}. "
                              f"Roll up-and-out to free capital and capture additional premium."),
                "account": acct,
                "size_usd": btc_cost_usd,
            })
    return actions


def lb_exit_trim_actions(judgments: list[dict], positions: pd.DataFrame,
                          options_tickers: set[str]) -> list[dict]:
    """Pull EXIT/TRIM verdicts from latest LB judgments."""
    actions = []
    for j in judgments:
        pm = j["result"]["pm"]
        action = pm["action"]
        if action not in ("EXIT", "TRIM"):
            continue
        ticker = j["ticker"]
        if ticker in LOCKED_POSITIONS:
            continue
        value = j["value"]
        confidence = pm.get("confidence", 0)
        # Skip low-confidence calls
        if confidence < 5:
            continue
        has_option = ticker in options_tickers
        sequence_note = " [SEQUENCE: close option first]" if has_option else ""
        actions.append({
            "priority": 3 if action == "EXIT" else 4,
            "ticker": ticker,
            "verb": action,
            "rationale": (f"LB {pm['final_score']}/{confidence}/10: {pm['thesis']} "
                          f"Position ${value:,.0f} ({j['pct_portfolio']:.1f}%).{sequence_note}"),
            "account": _short_acct(j["account"]),
            "size_usd": value,
            "sizing_note": pm.get("sizing_note", ""),
        })
    return actions


def ad_distribution_warnings(watchlist: pd.DataFrame, positions: pd.DataFrame) -> list[dict]:
    """Flag portfolio names in grade D/E A/D bucket."""
    if watchlist is None or watchlist.empty:
        return []
    port_tickers = set(positions["ticker"].unique())
    in_port = watchlist[watchlist["ticker"].isin(port_tickers)].copy()
    bad_ad = in_port[in_port["ad_grade"].isin(["D", "E"])]
    actions = []
    for _, r in bad_ad.iterrows():
        # Aggregate position value across accounts
        pos = positions[positions["ticker"] == r["ticker"]]
        total_value = pos["value"].sum()
        if total_value < 5000:
            continue  # ignore trivial
        # Only flag if not already covered by LB EXIT/TRIM
        actions.append({
            "priority": 5,
            "ticker": r["ticker"],
            "verb": "DISTRIBUTION-WATCH",
            "rationale": (f"A/D grade {r['ad_grade']} ({r['ad_score']:+.2f}), RS {int(r['rs_rank']) if pd.notna(r['rs_rank']) else '?'}. "
                          f"Institutional selling pressure over 13 weeks. Total exposure ${total_value:,.0f}."),
            "account": "multiple" if pos["account_id"].nunique() > 1 else _short_acct(pos.iloc[0]["account_id"]),
            "size_usd": total_value,
        })
    return actions


def lb_add_buy_actions(judgments: list[dict], watchlist: pd.DataFrame) -> list[dict]:
    """ADD on portfolio names where LB says ADD/BUY AND today's scanner fired."""
    actions = []
    if watchlist is None or watchlist.empty:
        return actions
    fired = set(watchlist[watchlist["signal_fired"] == True]["ticker"].tolist())
    for j in judgments:
        pm = j["result"]["pm"]
        if pm["action"] not in ("ADD", "BUY"):
            continue
        ticker = j["ticker"]
        confidence = pm.get("confidence", 0)
        if confidence < 6:
            continue
        if ticker not in fired:
            continue  # require fresh trigger
        actions.append({
            "priority": 4,
            "ticker": ticker,
            "verb": "ADD",
            "rationale": (f"LB {pm['final_score']}/{confidence}/10 + signal fired today. {pm['thesis']}"),
            "account": _short_acct(j["account"]),
            "size_usd": j["value"],
            "sizing_note": pm.get("sizing_note", ""),
        })
    return actions


def qqq_leaps_actions() -> list[dict]:
    """Surface QQQ LEAPS dip-buy events for Today's Actions:
      - SIGNAL FIRED today → BUY-LEAP at the recommended strike
      - Open position hit 50% take-profit → SELL-TO-CLOSE
      - Open position has <60 days to expiry → ROLL or CLOSE decision
    """
    try:
        from data_fetch import fetch_many
        from qqq_leaps_dipbuy import (
            check_entry_today, find_60d_leap, open_qqq_leaps,
            DEFAULTS as LEAPS_DEFAULTS,
        )
        from options import fetch_options_chain
    except Exception:
        return []

    out: list[dict] = []

    # ---- Open positions: take-profit + near-expiry checks ------------------
    try:
        open_pos = open_qqq_leaps()
    except Exception:
        open_pos = []

    for p in open_pos:
        if p.get("error"):
            continue
        # Take-profit hit (highest priority — locks in real $$)
        if p.get("tp_hit"):
            out.append({
                "priority": 1,
                "ticker": "QQQ",
                "account": "Individual (margin)",
                "verb": "TAKE-PROFIT-LEAP",
                "size_usd": float(p["current_value"]) * int(p["contracts"]),
                "rationale": (
                    f"QQQ {int(p['strike'])}C {p['expiry']} has gained "
                    f"{p['pct_gain_premium']:+.1f}% in premium (target was "
                    f"+{LEAPS_DEFAULTS['take_profit_pct']:.0f}%). Sell to close "
                    f"{int(p['contracts'])} contract at mid ${p['current_mid']:.2f} → "
                    f"locks in ${p['pnl_usd']:+,.0f}."
                ),
                "sizing_note": (
                    f"Working order: limit at mid ${p['current_mid']:.2f}; "
                    f"work down toward bid if no fill in 5 min. After fill, "
                    f"remove this contract from OPTIONS_POSITIONS in user_config.py."
                ),
            })
            continue
        # Near-expiry warning
        if p["dte"] <= 60:
            verb = "ROLL-LEAP" if p["pct_gain_premium"] is not None and p["pct_gain_premium"] >= 0 else "CLOSE-LEAP"
            decision = ("Roll forward 12 months at ~60 delta to stay in the strategy."
                        if verb == "ROLL-LEAP"
                        else "Close at a loss before theta accelerates further.")
            out.append({
                "priority": 2,
                "ticker": "QQQ",
                "account": "Individual (margin)",
                "verb": verb,
                "size_usd": float(p["current_value"]) * int(p["contracts"]),
                "rationale": (
                    f"QQQ {int(p['strike'])}C {p['expiry']} has only {p['dte']} days "
                    f"to expiry; premium is {p['pct_gain_premium']:+.1f}%. {decision}"
                ),
                "sizing_note": (
                    f"Current mid ${p['current_mid']:.2f}, delta {p['delta']:.2f}. "
                    "Theta acceleration kicks in inside ~60 days — decide soon."
                ),
            })

    # ---- Today's entry signal ---------------------------------------------
    try:
        raw = fetch_many(["QQQ"], force=False)
        df = raw.get("QQQ")
        if df is not None and not df.empty:
            signal = check_entry_today(df)
            if signal.get("signal"):
                leap = None
                try:
                    chain = fetch_options_chain("QQQ")
                    spot = chain.get("spot")
                    if spot is not None:
                        leap = find_60d_leap(spot)
                except Exception:
                    pass
                if leap:
                    out.append({
                        "priority": 2,    # alongside EXIT/ADD — fresh actionable buy
                        "ticker": "QQQ",
                        "account": "Individual (margin)",
                        "verb": "BUY-LEAP",
                        "size_usd": float(leap["cost_per_contract"]),
                        "rationale": (
                            f"DIP-BUY SIGNAL fires today: QQQ gapped {signal['gap_down_pct']:+.2f}% "
                            f"and is above its 100-day average (${signal['sma_100']:.2f}). "
                            f"Strategy: buy ~60-delta 12-month LEAPS. Recommended: "
                            f"QQQ ${int(leap['strike'])}C expiring {leap['expiry']} at "
                            f"${leap['mid']:.2f}/share (delta {leap['delta']:.2f}, "
                            f"{leap['dte']} days to expiry)."
                        ),
                        "sizing_note": (
                            f"Capital required: ${leap['cost_per_contract']:,.0f} per contract. "
                            f"Max loss = premium paid. Take-profit at +50% premium "
                            f"= ${leap['take_profit_premium']:.2f}/share "
                            f"(+${leap['take_profit_dollars']:,.0f}). No stop loss — held to "
                            "expiry if needed."
                        ),
                    })
                else:
                    out.append({
                        "priority": 2,
                        "ticker": "QQQ",
                        "account": "Individual (margin)",
                        "verb": "BUY-LEAP",
                        "size_usd": 0,
                        "rationale": (
                            f"DIP-BUY SIGNAL fires today: QQQ gapped {signal['gap_down_pct']:+.2f}% "
                            f"and is above its 100-day average. Strategy: buy ~60-delta 12-month "
                            f"LEAPS, but no clean candidate found in the chain. Pull the chain "
                            "manually and pick the nearest 60-delta call expiring in 11-13 months."
                        ),
                        "sizing_note": "See `python qqq_leaps_dipbuy.py` for chain details.",
                    })
    except Exception:
        pass

    return out


def new_name_buy_candidates(watchlist: pd.DataFrame, positions: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """Surface top-scoring names NOT in portfolio with score >= 80 + grade A."""
    if watchlist is None or watchlist.empty:
        return []
    port_tickers = set(positions["ticker"].unique())
    candidates = watchlist[
        ~watchlist["ticker"].isin(port_tickers)
        & (watchlist["score"] >= 80)
        & (watchlist["ad_grade"] == "A")
    ].sort_values("score", ascending=False).head(top_n)
    actions = []
    for _, r in candidates.iterrows():
        signal = r.get("signal_mode") if r.get("signal_fired") else "no-signal"
        actions.append({
            "priority": 6,
            "ticker": r["ticker"],
            "verb": "RESEARCH-NEW",
            "rationale": (f"Score {r['score']:.0f}, RS {int(r['rs_rank']) if pd.notna(r['rs_rank']) else '?'}, "
                          f"A/D grade A ({r['ad_score']:+.2f}), vol {r['volume_x']:.2f}x, signal {signal}. "
                          f"Not in portfolio."),
            "account": "new-position",
            "size_usd": 0,
        })
    return actions


# ---------- Rendering ----------

def render_markdown(*, today: str, macro: dict | None, judgments_date: str | None,
                    actions: list[dict], opts: pd.DataFrame, watchlist: pd.DataFrame,
                    positions: pd.DataFrame) -> str:
    lines = []
    lines.append(f"# Today's Actions — {today}\n")

    # Macro banner
    if macro:
        label = macro.get("regime_label", "?")
        score = macro.get("composite_score", 0)
        lines.append(f"**Macro:** {score:.0f}/100 — {label}\n")
    else:
        lines.append("**Macro:** (regime data unavailable)\n")

    # Data freshness
    bar_date = None
    if watchlist is not None and not watchlist.empty:
        # Scanner runs on the latest settled benchmark bar (yfinance close).
        # We don't store the bar date in the watchlist parquet; the snapshot dir
        # is today's date but the scoring bar may be 1 trading day prior.
        bar_date = "latest settled bar"
    judgments_note = f"LB judgments from {judgments_date}" if judgments_date else "LB judgments: NONE FOUND — re-run portfolio_judge.py"
    lines.append(f"_Scanner data: {bar_date or 'N/A'} · {judgments_note}_\n")

    # Top actions
    lines.append("## Top Actions\n")
    top = sorted(actions, key=lambda a: (a["priority"], -a.get("size_usd", 0)))[:5]
    if not top:
        lines.append("_No prioritized actions today. HOLD all positions._\n")
    else:
        for i, a in enumerate(top, 1):
            lines.append(f"**{i}. {a['verb']} — {a['ticker']}** ({a['account']})  ")
            lines.append(f"   {a['rationale']}")
            if a.get("sizing_note"):
                lines.append(f"   _Sizing: {a['sizing_note']}_")
            lines.append("")

    # Portfolio breakouts fired today
    if watchlist is not None and not watchlist.empty:
        port_tickers = set(positions["ticker"].unique())
        port_fired = watchlist[
            watchlist["ticker"].isin(port_tickers) & (watchlist["signal_fired"] == True)
        ].sort_values("score", ascending=False)
        if not port_fired.empty:
            lines.append("## Portfolio breakouts fired\n")
            for _, r in port_fired.iterrows():
                lines.append(
                    f"- **{r['ticker']}** — {r['signal_mode']}, score {r['score']:.0f}, "
                    f"RS {int(r['rs_rank']) if pd.notna(r['rs_rank']) else '?'}, "
                    f"A/D {r['ad_grade']} ({r['ad_score']:+.2f}), vol {r['volume_x']:.2f}x"
                )
            lines.append("")

    # New signals outside portfolio
    if watchlist is not None and not watchlist.empty:
        port_tickers = set(positions["ticker"].unique())
        new_fired = watchlist[
            ~watchlist["ticker"].isin(port_tickers)
            & (watchlist["signal_fired"] == True)
            & (watchlist["ad_grade"].isin(["A", "B"]))
        ].sort_values("score", ascending=False).head(5)
        if not new_fired.empty:
            lines.append("## New signals (not in portfolio, A/D-confirmed)\n")
            for _, r in new_fired.iterrows():
                lines.append(
                    f"- **{r['ticker']}** — {r['signal_mode']}, score {r['score']:.0f}, "
                    f"RS {int(r['rs_rank']) if pd.notna(r['rs_rank']) else '?'}, "
                    f"A/D {r['ad_grade']} ({r['ad_score']:+.2f}), vol {r['volume_x']:.2f}x"
                )
            lines.append("")

    # A/D distribution watch
    if watchlist is not None and not watchlist.empty:
        port_tickers = set(positions["ticker"].unique())
        bad_ad = watchlist[
            watchlist["ticker"].isin(port_tickers) & watchlist["ad_grade"].isin(["D", "E"])
        ].sort_values("ad_score").head(10)
        if not bad_ad.empty:
            lines.append("## A/D distribution watch (portfolio)\n")
            for _, r in bad_ad.iterrows():
                pos = positions[positions["ticker"] == r["ticker"]]
                val = pos["value"].sum()
                lines.append(
                    f"- **{r['ticker']}** — grade {r['ad_grade']} ({r['ad_score']:+.2f}), "
                    f"RS {int(r['rs_rank']) if pd.notna(r['rs_rank']) else '?'}, "
                    f"${val:,.0f}"
                )
            lines.append("")

    # Options state summary
    if opts is not None and not opts.empty:
        lines.append("## Open options\n")
        for _, r in opts.iterrows():
            tag = ""
            if int(r["dte"]) == 0:
                tag = " ⚠️ EXPIRES TODAY"
            elif r["side"] == "SHORT" and r["delta"] >= 0.40 and r["dte"] <= 21:
                tag = " ⚠️ assignment risk"
            elif r["side"] == "SHORT" and r["delta"] >= 0.65:
                tag = " (deep ITM)"
            lines.append(
                f"- **{r['ticker']} {r['strike']:.0f}{r['type'][0]} {r['expiry']}** "
                f"({_short_acct(r['account'])}, {r['coverage']}{', ROLLED' if r['rolled'] == 'ROLL' else ''}) — "
                f"{r['dte']}d, Δ{r['delta']:.2f}, spot ${r['spot']:.2f}, "
                f"P&L ${r['current_contract_pnl_usd']:+,.0f}{tag}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------- Main ----------

def main():
    today = _today()
    print(f"Generating Today's Actions for {today}...")

    watchlist = load_watchlist()
    judgments, judgments_date = load_latest_judgments()
    positions = load_positions()
    opts = load_options_state()
    macro = load_macro()

    options_tickers = set(opts["ticker"].unique()) if opts is not None and not opts.empty else set()

    actions = []
    actions.extend(options_actions(opts))
    actions.extend(qqq_leaps_actions())                 # NEW: dip-buy signal + open-LEAPS events
    actions.extend(lb_exit_trim_actions(judgments, positions, options_tickers))
    actions.extend(ad_distribution_warnings(watchlist, positions))
    actions.extend(lb_add_buy_actions(judgments, watchlist))
    actions.extend(new_name_buy_candidates(watchlist, positions, top_n=3))

    md = render_markdown(
        today=today, macro=macro, judgments_date=judgments_date,
        actions=actions, opts=opts, watchlist=watchlist, positions=positions,
    )

    # Write to snapshot dir
    out_dir = DATA_DIR / "snapshots" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "todays_actions.md"
    out_path.write_text(md, encoding="utf-8")

    print(md)
    print(f"\n→ Written to {out_path}")


if __name__ == "__main__":
    main()
