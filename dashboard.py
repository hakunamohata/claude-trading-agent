"""Streamlit dashboard for the breakout agent.

Run with:
    streamlit run dashboard.py

Pages:
  1. Today's Brief — active signals + names setting up + raw scan output
  2. Chart Browser — pick a ticker, see candles with EMAs, volume, signal markers
  3. Backtest Explorer — historical signals over the back-test window with fwd returns
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from universe import ALL_TICKERS, TARGETS, BENCHMARK, SECTOR_ETFS, TICKER_TO_SECTOR
from data_fetch import fetch_many, DATA_DIR
from breakout import (
    build_features, any_breakout_signal, signal_components,
    compute_universe_rs_rank,
)
from earnings import build_earnings_cache, days_to_earnings, earnings_proximity_label
from sector import compute_sector_strength, sector_strength_label
from judgment import (
    build_payload, evaluate_ticker, load_cached_judgment, TickerJudgment,
)
from portfolio import (
    HOLDINGS_DIR, ACCOUNT_LABEL, ACCOUNT_TAX_STATUS,
    INDIVIDUAL_MARGIN_SNAPSHOT, margin_annual_cost, margin_summary,
    position_action, TRADE_ELIGIBLE_ACCOUNTS, LOCKED_POSITIONS, MAX_POSITION_PCT,
)
import json as _json
from data_fetch import DATA_DIR as _DATA_DIR


st.set_page_config(
    page_title="Breakout Agent",
    page_icon=None,
    layout="wide",
)


# ---------- Shared data loading (cached for Streamlit reruns) ----------

@st.cache_data(ttl=900)
def load_universe_data(force_refresh: bool = False):
    """Fetch OHLCV for the full universe. Cached for 15 min."""
    return fetch_many(ALL_TICKERS, force=force_refresh)


@st.cache_data(ttl=900)
def build_all_features():
    raw = load_universe_data()
    bench = raw[BENCHMARK]["close"]

    # Cross-sectional RS rank over equity tickers only (skip ETFs)
    equity_closes = {
        t: df["close"] for t, df in raw.items()
        if t not in SECTOR_ETFS and t != BENCHMARK
    }
    rs_rank_df = compute_universe_rs_rank(equity_closes)

    # Sector strength matrix
    sector_strength = compute_sector_strength(raw, broad_benchmark=BENCHMARK)

    feats, sigs, comps = {}, {}, {}
    for t, df in raw.items():
        if t == BENCHMARK or t in SECTOR_ETFS:
            continue
        rs_rank_series = rs_rank_df[t] if t in rs_rank_df.columns else None
        f = build_features(df, bench, rs_rank_series=rs_rank_series)
        feats[t] = f
        sigs[t] = any_breakout_signal(f)
        comps[t] = signal_components(f)
    return raw, feats, sigs, comps, sector_strength


@st.cache_data(ttl=86400)
def load_earnings():
    equity_tickers = [t for t in ALL_TICKERS if t not in SECTOR_ETFS and t != BENCHMARK]
    return build_earnings_cache(equity_tickers)


# ---------- Column tooltips ----------
# Hover any column header to see what it means. Edit values here to update
# all tables at once.

COLUMN_HELP: dict[str, str] = {
    "Ticker": "Stock symbol",
    "Mode": ("Which filter mode fired. "
             "VCP = volatility contraction breakout (classic Minervini). "
             "MOM = trend continuation in established uptrend. "
             "EME = Stage-2 emergence (close reclaims 200 EMA, golden cross required). "
             "PP = pocket pivot (today's volume > max down-day volume of prior 10 days, inside or near a base)."),
    "Close": "Most recent close price (split-adjusted via yfinance auto_adjust=True)",
    "Vol×": ("Today's volume divided by the 50-day average volume. "
                 "1.5x is the classical threshold for a confirmed breakout."),
    "RS rank": ("IBD-style cross-sectional rank, 1-99 (higher = stronger). "
                "Composite return = 40% × 1-month + 20% each of 3/6/12-month, "
                "ranked across all equity peers in the universe on each day. "
                "Note: with only ~17 names the bucketing is coarse (≈6 percentile per name). "
                "Wider universe will sharpen this."),
    "RS 60d %": ("Stock's 60-day return minus QQQ's 60-day return, in percentage points. "
                 "Positive = outperforming the benchmark."),
    "Sector RS %": ("Stock's sector ETF (SOXX for semis, XLK for software, etc.) 60-day return "
                    "minus QQQ 60-day return. Positive = the sector is leading the market."),
    "Earnings in": ("Days-to-next-earnings label. "
                    "imminent = <=7d (risk-off setups extending into earnings), "
                    "soon = 8-21d, "
                    "far = >21d, "
                    "passed = the date pulled from yfinance has already gone by, "
                    "unknown = yfinance returned no calendar (common for ETFs / small/recent IPOs)."),
    "50 EMA": "50-day exponential moving average — medium-term trend reference",
    "200 EMA": ("200-day exponential moving average — long-term trend reference. "
                "Price above = Stage 2 territory (Weinstein)."),
    "Score": "How many of the 7 strict VCP conditions are currently True (out of 7).",
    "Missing": "Which VCP conditions are currently False — explains why no signal fired.",
    "Date": "Bar date of the signal",
    "+5d %": "Forward 5-trading-day return after the signal — short-term follow-through quality",
    "+20d %": ("Forward 20-trading-day return after the signal. Primary measure of "
               "signal quality used in the regression test."),
    "n": "Number of signals in this row's group",
    "hit_rate": "Fraction of signals with +20d return > 0, expressed as percent",
    "median_fwd20": "Median forward-20d return across signals in the group",
    "Claude": ("Claude's conviction score 1-100. 80+ = high conviction long, "
               "50 = neutral/watch, <30 = avoid. Click row to see full thesis + risks."),
    "Bias": "Claude's directional bias: long / watch / avoid",
    "Thesis": "Claude's one-sentence rationale for WHY this could work",
    "Risks": "Claude's one-sentence call-out of WHAT could kill it",
    "Account": "Which account this position is held in",
    "Shares": "Number of shares held",
    "Value": "Current market value of this position",
    "% Port": "Position size as % of total portfolio value (across all accounts). Cap is 10% (MSFT exempt).",
    "Action": ("Recommended action per portfolio rules + Claude score: "
               "LOCKED = MSFT, do not touch · "
               "NOT-ACTIONABLE = not in a trade-eligible account (Roth IRA or 401k BrokerageLink only) · "
               "TRIM = over 10% concentration cap, sell down · "
               "AVOID = Claude score < 30 · "
               "BUY = Claude score >= 80 AND position < 5% · "
               "HOLD = everything else"),
    "Cost Basis": "Total amount paid to acquire this position",
    "P&L": "Unrealized gain/loss vs cost basis",
    "P&L %": "Unrealized gain/loss as % of cost basis",
    "Stage 2": "True if close > 50 EMA > 200 EMA — the textbook uptrend regime",
}


# ---------- Portfolio data loading ----------

@st.cache_data(ttl=300)
def load_portfolio() -> pd.DataFrame:
    p = HOLDINGS_DIR / "positions_current.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(ttl=3600)
def load_macro_regime():
    """Cached macro regime read (computing is slow due to S&P 500 EMA scan)."""
    try:
        from macro_gate import compute_regime
        return compute_regime()
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_lb_judgments(date_str: str) -> dict:
    """Read judgments_portfolio.jsonl from today's snapshot.
    Returns dict keyed by ticker -> result dict (with LB + investor agents)."""
    p = _DATA_DIR / "snapshots" / date_str / "judgments_portfolio.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = _json.loads(line)
            out[r["ticker"]] = r
        except Exception:
            continue
    return out


@st.cache_data(ttl=300)
def load_research(date_str: str, ticker: str) -> dict | None:
    p = _DATA_DIR / "research" / date_str / f"{ticker}.json"
    if not p.exists():
        return None
    try:
        return _json.loads(p.read_text())
    except Exception:
        return None


def _column_config_for(df: pd.DataFrame) -> dict:
    """Build a column_config dict for st.dataframe from COLUMN_HELP."""
    return {
        col: st.column_config.Column(col, help=COLUMN_HELP.get(col, ""))
        for col in df.columns
    }


def _mode_label(sig_row) -> str:
    modes = []
    if sig_row["vcp"]: modes.append("VCP")
    if sig_row["momentum"]: modes.append("MOM")
    if sig_row["emergence"]: modes.append("EME")
    if sig_row["pocket_pivot"]: modes.append("PP")
    return "/".join(modes) if modes else ""


def _sector_strength_today(sector_strength: pd.DataFrame, ticker: str, latest_date) -> float | None:
    sector = TICKER_TO_SECTOR.get(ticker)
    if sector is None or sector not in sector_strength.columns:
        return None
    if latest_date not in sector_strength.index:
        return None
    val = sector_strength.loc[latest_date, sector]
    return float(val) if pd.notna(val) else None


# ---------- Sidebar ----------

st.sidebar.title("Breakout Agent")
page = st.sidebar.radio("Page", ["My Portfolio", "Today's Brief", "Chart Browser", "Backtest Explorer"])

if st.sidebar.button("Force refresh data"):
    load_universe_data.clear()
    build_all_features.clear()
    fetch_many(ALL_TICKERS, force=True)
    st.sidebar.success("Data refreshed")
    st.rerun()

raw, feats, sigs, comps, sector_strength = build_all_features()
earnings_df = load_earnings()
portfolio_df = load_portfolio()
latest_date = max(df.index[-1] for df in raw.values())
st.sidebar.caption(f"Most-recent bar: **{latest_date.date()}**")
st.sidebar.caption(f"Universe size: **{len(ALL_TICKERS) - 1}** + benchmark")


# =============================================================
# PAGE 1 — Today's Brief
# =============================================================

# =============================================================
# PAGE 0 — My Portfolio
# =============================================================

if page == "My Portfolio":
    st.title("My Portfolio — Rebalance View")

    if portfolio_df.empty:
        st.error("No portfolio data found. Run `python portfolio.py` to parse holdings.")
        st.stop()

    # ---- Macro regime banner ----
    regime = load_macro_regime()
    if regime and regime.get("composite_score") is not None:
        score = regime["composite_score"]
        label = regime["regime_label"]
        # Pick color tone based on score
        if score >= 70: bg = "#1b4332"  # green-tinted
        elif score >= 50: bg = "#2d3748"  # neutral
        else: bg = "#5c1f1f"  # red-tinted
        st.markdown(
            f"<div style='background-color:{bg};padding:10px 14px;border-radius:6px;margin-bottom:14px;'>"
            f"<strong>Macro regime:</strong> {score:.0f}/100 — <em>{label}</em><br>"
            f"<small>VIX {regime['vix']['value']:.1f} · "
            f"{regime['breadth']['pct_above_50_ema']:.0f}% of S&P > 50 EMA · "
            f"{regime['breadth']['pct_above_200_ema']:.0f}% > 200 EMA · "
            f"HYG {'above' if regime['credit']['hyg_above_200_ema'] else 'below'} 200 EMA</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---- Account-level summary ----
    total_value = portfolio_df["value"].sum()

    # Margin-adjusted total: subtract margin debt from gross holdings
    margin_debt = abs(INDIVIDUAL_MARGIN_SNAPSHOT["net_debit"])
    individual_equity = INDIVIDUAL_MARGIN_SNAPSHOT["account_equity"]
    indiv_gross = portfolio_df[portfolio_df["account_id"] == "X38822884"]["value"].sum()
    net_household = total_value - margin_debt

    st.subheader("Account overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross holdings", f"${total_value:,.0f}")
    c2.metric("Margin debt", f"${margin_debt:,.0f}",
              delta=f"-${margin_annual_cost():,.0f}/yr interest", delta_color="inverse")
    c3.metric("Net household equity", f"${net_household:,.0f}")
    c4.metric("# of unique tickers", f"{portfolio_df['ticker'].nunique()}")

    account_summary = (
        portfolio_df.groupby(["account_label", "tax_status"])
        .agg(positions=("ticker", "nunique"), value=("value", "sum"))
        .reset_index()
        .sort_values("value", ascending=False)
    )
    account_summary["% of gross"] = (account_summary["value"] / total_value * 100).round(1)
    st.dataframe(account_summary, hide_index=True, use_container_width=True)

    # ---- Margin status ----
    with st.expander("Margin status (Individual account)"):
        m = INDIVIDUAL_MARGIN_SNAPSHOT
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Net equity", f"${m['account_equity']:,.0f}")
        col_a.metric("Equity ratio", f"{m['equity_pct']:.1f}%")
        col_b.metric("Stocks (gross)", f"${m['margin_market_value']:,.0f}")
        col_b.metric("Cash", f"${m['cash_market_value']:,.0f}")
        col_c.metric("Margin buying power", f"${m['margin_buying_power']:,.0f}")
        col_c.metric("Non-margin buying power", f"${m['non_margin_buying_power']:,.0f}")
        st.caption(f"Margin rate: **{m['margin_interest_rate_pct']:.2f}% / year**. "
                   f"Daily accrual: ${m['margin_interest_accrued_daily']:.2f}. "
                   f"Annualized cost: **${margin_annual_cost():,.0f}**.")

    st.divider()

    # ---- Claude scoring on trade-eligible positions ----
    if "portfolio_claude" not in st.session_state:
        st.session_state["portfolio_claude"] = {}

    cols = st.columns([1, 1, 4])
    with cols[0]:
        score_btn = st.button("Score trade-eligible positions with LB")
    with cols[1]:
        bypass = st.checkbox("Bypass cache", key="pf_bypass")

    # ---- Load LB (multi-agent) judgments + research from today's snapshot ----
    today_str = str(latest_date.date())
    lb_judgments = load_lb_judgments(today_str)
    if lb_judgments:
        st.caption(f"Found multi-agent judgments for {len(lb_judgments)} positions in today's snapshot "
                   f"(run `python portfolio_judge.py` to refresh)")

    # Build payloads for trade-eligible positions
    payloads = {}
    latest = latest_date
    for _, row in portfolio_df.iterrows():
        t = row["ticker"]
        acct = row["account_id"]
        if not (acct in TRADE_ELIGIBLE_ACCOUNTS and t not in LOCKED_POSITIONS):
            continue
        if t not in feats:
            continue
        if latest not in feats[t].index:
            continue
        feat_row = feats[t].loc[latest]
        sig_row = sigs[t].loc[latest]
        comp_row = comps[t].loc[latest]
        rs_rank = feat_row.get("rs_rank")
        rs_rank_v = int(rs_rank) if pd.notna(rs_rank) else None
        sec_str = _sector_strength_today(sector_strength, t, latest)
        days_e = days_to_earnings(earnings_df, t, as_of=latest)
        earnings_lbl = earnings_proximity_label(days_e)
        rs_line_nh = bool(feat_row.get("rs_line_new_high", False))

        p = build_payload(t, feat_row, sig_row, comp_row,
                          rs_rank=rs_rank_v, sector_rs=sec_str,
                          earnings_label=earnings_lbl, rs_line_new_high=rs_line_nh)
        # Add position-specific context
        p["current_position_value_usd"] = round(float(row["value"]), 2)
        p["current_position_pct_of_portfolio"] = round(float(row["value"] / total_value * 100), 2)
        p["account_type"] = row["account_label"]
        if pd.notna(row.get("cost_basis")):
            p["unrealized_pnl_pct"] = round(float((row["value"] - row["cost_basis"]) / row["cost_basis"] * 100), 1)
        payloads[(t, acct)] = p

    if score_btn:
        progress = st.progress(0, text="Scoring positions...")
        results = {}
        total_n = len(payloads)
        for i, ((t, acct), p) in enumerate(payloads.items()):
            progress.progress((i + 1) / max(total_n, 1), text=f"Scoring {t}")
            try:
                key = f"{t}__{acct}"
                j = evaluate_ticker(str(latest.date()), key, p, use_cache=not bypass)
                results[(t, acct)] = j
            except Exception as e:
                st.warning(f"{t} ({acct}): {e}")
        st.session_state["portfolio_claude"] = results
        progress.empty()
        st.success(f"Scored {len(results)} positions")

    # Load cached judgments if not yet computed
    if not st.session_state["portfolio_claude"]:
        for (t, acct), p in payloads.items():
            cached = load_cached_judgment(str(latest.date()), f"{t}__{acct}", p)
            if cached:
                st.session_state["portfolio_claude"][(t, acct)] = cached

    claude_pf = st.session_state["portfolio_claude"]

    # ---- Holdings table with actions ----
    st.subheader("Holdings — current state + recommended action")

    rows = []
    for _, row in portfolio_df.iterrows():
        t = row["ticker"]
        acct = row["account_id"]
        value = float(row["value"])
        pct = (value / total_value * 100) if total_value > 0 else 0
        j = claude_pf.get((t, acct))
        claude_score = j.score if j else None
        action = position_action(t, acct, value, total_value, claude_score)
        # Stage 2 lookup
        stage2 = None
        if t in feats and latest in feats[t].index:
            f = feats[t].loc[latest]
            stage2 = bool(f["close_prior"] > f["ema_50_prior"] > f["ema_200_prior"])

        # P&L if cost basis available
        cb = float(row["cost_basis"]) if pd.notna(row.get("cost_basis")) else None
        pnl = (value - cb) if cb is not None else None
        pnl_pct = (pnl / cb * 100) if cb and cb > 0 else None

        rows.append({
            "Ticker": t,
            "Account": ACCOUNT_LABEL.get(acct, acct).split(" ")[0],  # short label
            "Shares": round(float(row["quantity"]), 2),
            "Value": round(value, 0),
            "% Port": round(pct, 1),
            "Action": action,
            "Claude": claude_score,
            "Bias": j.bias if j else None,
            "Stage 2": stage2,
            "Cost Basis": round(cb, 0) if cb is not None else None,
            "P&L %": round(pnl_pct, 1) if pnl_pct is not None else None,
        })

    df_hold = pd.DataFrame(rows)
    # Sort by action priority then value descending
    action_priority = {"TRIM": 0, "AVOID": 1, "BUY": 2, "HOLD": 3, "NOT-ACTIONABLE": 4, "LOCKED": 5}
    df_hold["_sort"] = df_hold["Action"].map(action_priority).fillna(99)
    df_hold = df_hold.sort_values(["_sort", "Value"], ascending=[True, False]).drop(columns="_sort")

    st.dataframe(df_hold, hide_index=True, use_container_width=True,
                 column_config=_column_config_for(df_hold))

    # ---- Action summary at bottom ----
    st.subheader("Action summary (trade-eligible)")

    trims = df_hold[df_hold["Action"] == "TRIM"]
    if not trims.empty:
        st.markdown("**TRIM — over 10% concentration cap:**")
        for _, r in trims.iterrows():
            target_value = total_value * MAX_POSITION_PCT / 100
            excess = r["Value"] - target_value
            st.write(f"  • {r['Ticker']} ({r['Account']}): ${r['Value']:,.0f} ({r['% Port']:.1f}%) "
                     f"— target ${target_value:,.0f} ({MAX_POSITION_PCT}%), sell ~${excess:,.0f}")

    avoids = df_hold[df_hold["Action"] == "AVOID"]
    if not avoids.empty:
        st.markdown("**AVOID — Claude score < 30:**")
        for _, r in avoids.iterrows():
            st.write(f"  • {r['Ticker']} ({r['Account']}): ${r['Value']:,.0f}, score {r['Claude']}, P&L {r['P&L %']:+.1f}%")

    buys = df_hold[df_hold["Action"] == "BUY"]
    if not buys.empty:
        st.markdown("**BUY — Claude score >= 80 AND under 5%:**")
        for _, r in buys.iterrows():
            st.write(f"  • {r['Ticker']} ({r['Account']}): ${r['Value']:,.0f} ({r['% Port']:.1f}%), score {r['Claude']}")

    # ---- Sector allocation ----
    from universe import TICKER_TO_SECTOR
    df_hold["sector"] = df_hold["Ticker"].map(TICKER_TO_SECTOR).fillna("Other/Cash")
    sec_alloc = df_hold.groupby("sector")["Value"].sum().sort_values(ascending=False).reset_index()
    sec_alloc["% of total"] = (sec_alloc["Value"] / total_value * 100).round(1)
    st.subheader("Sector allocation")
    st.dataframe(sec_alloc, hide_index=True, use_container_width=True)

    # ---- Per-position deep-dives (single-agent quick OR LB multi-agent if available) ----
    if claude_pf or lb_judgments:
        st.subheader("Per-position analysis")
        # Iterate by single-agent score where available, otherwise by LB score
        all_keys = set(claude_pf.keys())
        for ticker in lb_judgments.keys():
            for (t, acct) in payloads.keys():
                if t == ticker:
                    all_keys.add((t, acct))
                    break

        def _sort_key(kv):
            (t, acct) = kv if isinstance(kv, tuple) else (kv, "")
            if (t, acct) in claude_pf:
                return -claude_pf[(t, acct)].score
            if t in lb_judgments:
                return -lb_judgments[t]["result"]["pm"]["final_score"]
            return 0

        for key in sorted(all_keys, key=_sort_key):
            (t, acct) = key
            j_quick = claude_pf.get((t, acct))
            j_lb = lb_judgments.get(t)
            research = load_research(today_str, t)

            # Build header
            if j_lb:
                pm = j_lb["result"]["pm"]
                header = f"{t} ({ACCOUNT_LABEL.get(acct, acct).split(' ')[0]}) — LB {pm['final_score']}/100 · {pm['action']} · conf {pm['confidence']}/10"
            elif j_quick:
                header = f"{t} ({ACCOUNT_LABEL.get(acct, acct).split(' ')[0]}) — quick {j_quick.score}/100 · {j_quick.bias.upper()}"
            else:
                header = f"{t} ({ACCOUNT_LABEL.get(acct, acct).split(' ')[0]})"

            with st.expander(header):
                # LB verdict (preferred display when available)
                if j_lb:
                    pm = j_lb["result"]["pm"]
                    st.markdown(f"### LB synthesizer (final)")
                    st.markdown(f"**Thesis:** {pm['thesis']}")
                    st.markdown(f"**Key risk:** {pm['key_risk']}")
                    st.markdown(f"**Sizing:** {pm['sizing_note']}")
                    st.markdown("---")

                    # Specialist agent scores
                    st.markdown("**Specialist scores:**")
                    cols_s = st.columns(4)
                    r = j_lb["result"]
                    for col, (name, key) in zip(cols_s, [
                        ("Technical", "technical"),
                        ("Fundamental", "fundamental"),
                        ("Sentiment", "sentiment"),
                        ("Risk", "risk"),
                    ]):
                        if r.get(key):
                            col.metric(name, f"{r[key]['score']}/100")

                    # Investor philosophy scores (if present)
                    if any(r.get(k) for k in ("minervini", "druckenmiller", "burry")):
                        st.markdown("**Investor philosophies:**")
                        cols_i = st.columns(3)
                        if r.get("minervini"):
                            m = r["minervini"]
                            cols_i[0].metric("Minervini",
                                             f"{m['score']}/100",
                                             delta=f"VCP={m['vcp_grade']} · entry {m['entry_proximity']}/10",
                                             delta_color="off")
                            cols_i[0].caption(m["summary"])
                        if r.get("druckenmiller"):
                            d = r["druckenmiller"]
                            cols_i[1].metric("Druckenmiller",
                                             f"{d['score']}/100",
                                             delta=f"cycle: {d['cycle_position']}",
                                             delta_color="off")
                            cols_i[1].caption(d["summary"])
                        if r.get("burry"):
                            b = r["burry"]
                            cols_i[2].metric("Burry",
                                             f"{b['score']}/100",
                                             delta=f"mean-rev {b['mean_reversion_probability_pct']}% · ext {b['extension_risk']}/10",
                                             delta_color="off")
                            cols_i[2].caption(b["summary"])

                # Research summary (if present)
                if research:
                    st.markdown("---")
                    st.markdown(f"### Live research")
                    sent_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(research.get("sentiment"), "")
                    st.markdown(f"{sent_emoji} **Sentiment:** {research.get('sentiment', '?')}")
                    st.markdown(f"**Catalyst summary:** {research.get('catalyst_summary', '')}")
                    if research.get("recent_developments"):
                        st.markdown("**Recent developments:**")
                        for d in research["recent_developments"][:5]:
                            st.markdown(f"- {d}")
                    if research.get("pending_catalysts"):
                        st.markdown("**Pending catalysts:**")
                        for c in research["pending_catalysts"][:4]:
                            st.markdown(f"- {c}")
                    st.markdown(f"**Key risks:** {research.get('key_risks', '')}")
                    if research.get("sources"):
                        with st.expander("Sources"):
                            for s in research["sources"][:6]:
                                st.markdown(f"- [{s['title'][:80]}]({s['url']})")

                # Single-agent fallback (when LB judgments missing)
                if j_quick and not j_lb:
                    st.markdown("### Quick single-agent score")
                    st.markdown(f"**Thesis:** {j_quick.thesis}")
                    st.markdown(f"**Risks:** {j_quick.risks}")
                    f = j_quick.factors
                    fcols = st.columns(6)
                    for col, (n, v) in zip(fcols, [
                        ("Setup", f.setup_quality), ("Trend", f.trend_regime),
                        ("RS", f.relative_strength), ("Sector", f.sector_tailwind),
                        ("Catalyst", f.catalyst_proximity), ("Risk/Rew", f.risk_reward),
                    ]):
                        col.metric(n, f"{v}/10")


elif page == "Today's Brief":
    st.title(f"Today's Brief — {latest_date.date()}")

    # Claude judgment session state
    if "claude_results" not in st.session_state:
        st.session_state["claude_results"] = {}

    judge_cols = st.columns([1, 1, 4])
    with judge_cols[0]:
        run_judge = st.button("Compute Claude scores")
    with judge_cols[1]:
        force_recompute = st.checkbox("Bypass cache", value=False)

    # Build per-ticker payloads once — used both for table display and for Claude
    payloads_for_judging: dict[str, dict] = {}

    active_rows, setup_rows = [], []
    for t in feats:
        if latest_date not in sigs[t].index:
            continue
        sig_row = sigs[t].loc[latest_date]
        feat_row = feats[t].loc[latest_date]
        comp_row = comps[t].loc[latest_date]
        vol_x = feat_row["volume"] / feat_row["vol_avg_50"] if feat_row["vol_avg_50"] else None
        rs_60 = feat_row["rs_60_prior"] * 100 if pd.notna(feat_row["rs_60_prior"]) else None

        rs_rank = feat_row.get("rs_rank")
        rs_rank_str = int(rs_rank) if pd.notna(rs_rank) else None
        sec_str = _sector_strength_today(sector_strength, t, latest_date)
        days_e = days_to_earnings(earnings_df, t, as_of=latest_date)
        earnings_lbl = earnings_proximity_label(days_e)
        rs_line_nh = bool(feat_row.get("rs_line_new_high", False))

        # Build payload for Claude (whether or not we call this run)
        score = int(comp_row.sum())
        if sig_row["any"] or 5 <= score <= 6:
            payloads_for_judging[t] = build_payload(
                t, feat_row, sig_row, comp_row,
                rs_rank=rs_rank_str, sector_rs=sec_str,
                earnings_label=earnings_lbl,
                rs_line_new_high=rs_line_nh,
            )

        if sig_row["any"]:
            active_rows.append({
                "Ticker": t,
                "Mode": _mode_label(sig_row),
                "Close": round(feat_row["close"], 2),
                "Vol×": round(vol_x, 2) if vol_x is not None else None,
                "RS rank": rs_rank_str,
                "RS 60d %": round(rs_60, 1) if rs_60 is not None else None,
                "Sector RS %": round(sec_str, 1) if sec_str is not None else None,
                "Earnings in": earnings_lbl,
                "50 EMA": round(feat_row["ema_50"], 2),
                "200 EMA": round(feat_row["ema_200"], 2),
            })
        else:
            score = int(comp_row.sum())
            if 5 <= score <= 6:
                missing = [c for c, v in comp_row.items() if not v]
                setup_rows.append({
                    "Ticker": t,
                    "Score": f"{score}/7",
                    "Close": round(feat_row["close"], 2),
                    "Vol×": round(vol_x, 2) if vol_x is not None else None,
                    "RS rank": rs_rank_str,
                    "RS 60d %": round(rs_60, 1) if rs_60 is not None else None,
                    "Sector RS %": round(sec_str, 1) if sec_str is not None else None,
                    "Earnings in": earnings_lbl,
                    "Missing": ", ".join(missing),
                })

    # Run Claude if button pressed
    if run_judge:
        progress = st.progress(0, text="Calling Claude (Opus 4.7)...")
        results: dict[str, TickerJudgment] = {}
        total = len(payloads_for_judging)
        for i, (t, payload) in enumerate(payloads_for_judging.items()):
            progress.progress((i + 1) / max(total, 1), text=f"Scoring {t} ({i+1}/{total})")
            try:
                j = evaluate_ticker(str(latest_date.date()), t, payload, use_cache=not force_recompute)
                results[t] = j
            except Exception as e:
                st.warning(f"{t}: {e}")
        st.session_state["claude_results"] = results
        progress.empty()
        st.success(f"Scored {len(results)} candidates")

    # Load previously cached judgments (so the column appears even before button press if cached)
    if not st.session_state["claude_results"]:
        for t, payload in payloads_for_judging.items():
            cached = load_cached_judgment(str(latest_date.date()), t, payload)
            if cached is not None:
                st.session_state["claude_results"][t] = cached

    claude_results = st.session_state["claude_results"]

    def _add_claude_cols(rows: list[dict]) -> list[dict]:
        for r in rows:
            j = claude_results.get(r["Ticker"])
            if j is not None:
                r["Claude"] = j.score
                r["Bias"] = j.bias
                r["Thesis"] = j.thesis
            else:
                r["Claude"] = None
                r["Bias"] = None
                r["Thesis"] = None
        return rows

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Active signals")
        if active_rows:
            df_a = pd.DataFrame(_add_claude_cols(active_rows))
            # Sort by Claude score descending if available
            if df_a["Claude"].notna().any():
                df_a = df_a.sort_values("Claude", ascending=False, na_position="last")
            st.dataframe(df_a, hide_index=True, use_container_width=True,
                         column_config=_column_config_for(df_a))
        else:
            st.info("No signals fired on the most recent bar.")

    with col2:
        st.subheader("Setting up (5-6 of 7)")
        if setup_rows:
            df_s = pd.DataFrame(_add_claude_cols(setup_rows))
            if df_s["Claude"].notna().any():
                df_s = df_s.sort_values("Claude", ascending=False, na_position="last")
            st.dataframe(df_s, hide_index=True, use_container_width=True,
                         column_config=_column_config_for(df_s))
        else:
            st.info("No setups in the watchlist.")

    # Detailed Claude per-ticker view
    if claude_results:
        st.subheader("Claude per-ticker analysis")
        for t in sorted(claude_results.keys(), key=lambda x: -claude_results[x].score):
            j = claude_results[t]
            with st.expander(f"{t}  —  score {j.score} ({j.bias.upper()})"):
                st.markdown(f"**Thesis:** {j.thesis}")
                st.markdown(f"**Risks:** {j.risks}")
                f = j.factors
                fcols = st.columns(6)
                for col, (name, val) in zip(fcols, [
                    ("Setup", f.setup_quality),
                    ("Trend", f.trend_regime),
                    ("RS", f.relative_strength),
                    ("Sector", f.sector_tailwind),
                    ("Catalyst", f.catalyst_proximity),
                    ("Risk/Rew", f.risk_reward),
                ]):
                    col.metric(name, f"{val}/10")

    st.divider()
    st.caption("Mode legend: **VCP** = volatility contraction breakout · "
               "**MOM** = trend-continuation in established uptrend · "
               "**EME** = Stage-2 emergence (reclaim of 200 EMA) · "
               "**PP** = pocket pivot (institutional buy inside base)")
    st.caption("Context: **RS rank** = cross-sectional 1-99 vs universe peers · "
               "**RS 60d %** = stock 60d return − QQQ 60d return · "
               "**Sector RS %** = sector ETF 60d − QQQ 60d")


# =============================================================
# PAGE 2 — Chart Browser
# =============================================================

elif page == "Chart Browser":
    st.title("Chart Browser")

    tickers = sorted([t for t in feats.keys()])
    default_ix = tickers.index("NBIS") if "NBIS" in tickers else 0
    ticker = st.selectbox("Ticker", tickers, index=default_ix)
    months = st.slider("Months of history", 3, 24, 6)

    f = feats[ticker]
    s = sigs[ticker]
    window_start = latest_date - pd.DateOffset(months=months)
    fw = f[f.index >= window_start].copy()
    sw = s[s.index >= window_start].copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=("Price + EMAs", "Volume"),
    )

    fig.add_trace(go.Candlestick(
        x=fw.index, open=fw["open"], high=fw["high"], low=fw["low"], close=fw["close"],
        name="Price", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fw.index, y=fw["ema_50"], name="50 EMA",
        line=dict(color="orange", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fw.index, y=fw["ema_200"], name="200 EMA",
        line=dict(color="red", width=1.5),
    ), row=1, col=1)

    # 9/21 SMA cloud — fill between
    if "sma_9" in fw.columns and "sma_21" in fw.columns:
        fig.add_trace(go.Scatter(
            x=fw.index, y=fw["sma_21"], name="21 SMA",
            line=dict(color="rgba(0,0,0,0.0)", width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        # Use the modal cloud color across the window (predominant regime)
        cloud_green = float(fw["cloud_bullish"].mean()) if "cloud_bullish" in fw.columns else 0.5
        fill_color = "rgba(46, 204, 113, 0.15)" if cloud_green > 0.5 else "rgba(231, 76, 60, 0.15)"
        fig.add_trace(go.Scatter(
            x=fw.index, y=fw["sma_9"], name="9 SMA",
            line=dict(color="rgba(0,0,0,0.0)", width=0),
            fill="tonexty", fillcolor=fill_color,
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # Signal markers — different shapes for each mode
    for mode, color, label in [
        ("vcp", "#1f77b4", "VCP"),
        ("momentum", "#2ca02c", "MOM"),
        ("emergence", "#9467bd", "EME"),
        ("pocket_pivot", "#ff7f0e", "PP"),
    ]:
        sig_days = sw[sw[mode]]
        if not sig_days.empty:
            ys = fw.loc[sig_days.index, "low"] * 0.97
            fig.add_trace(go.Scatter(
                x=sig_days.index, y=ys, mode="markers", name=label,
                marker=dict(symbol="triangle-up", size=14, color=color,
                            line=dict(width=1, color="black")),
            ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=fw.index, y=fw["volume"], name="Volume",
        marker=dict(color="lightgray"), showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=fw.index, y=fw["vol_avg_50"], name="50d avg vol",
        line=dict(color="black", width=1, dash="dot"), showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Fit/no-fit panel: per-day condition breakdown ----
    st.subheader(f"{ticker} — fit/no-fit on most recent bar")
    last_bar = fw.index[-1]
    comp_row = comps[ticker].loc[last_bar]
    feat_row = fw.loc[last_bar]
    sig_row = sw.loc[last_bar]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**VCP-mode conditions (need all 7)**")
        labels = {
            "breakout_high": f"Close >= prior 50d high  ({feat_row['close']:.2f} vs {feat_row.get('high_50_prior', float('nan')):.2f})",
            "volume_surge": f"Volume >= 1.5x 50d avg  ({(feat_row['volume']/feat_row['vol_avg_50']):.2f}x)" if feat_row['vol_avg_50'] else "Volume — n/a",
            "top_third": f"Close in top third of range",
            "quiet_base": f"ATR% prior bar in bottom 35% of 120d  ({feat_row.get('atr_pct_prior', 0)*100:.2f}% vs {feat_row.get('atr_pct_q35_120', 0)*100:.2f}%)",
            "stage2": f"Stage 2: close > 50 EMA > 200 EMA",
            "ema_rising": f"50 EMA rising over 10 days  ({feat_row.get('ema_50_slope10', 0):.2f})",
            "rs_positive": f"RS 60d > 0 vs QQQ  ({feat_row.get('rs_60_prior', 0)*100:+.1f}%)",
        }
        for k, lbl in labels.items():
            mark = "PASS" if comp_row.get(k, False) else "FAIL"
            st.write(f"  [{mark}]  {lbl}")

    with col_b:
        st.markdown("**Which modes fired?**")
        modes_status = [
            ("VCP (Volatility Contraction)", sig_row["vcp"]),
            ("MOM (Trend Continuation)", sig_row["momentum"]),
            ("EME (Stage-2 Emergence)", sig_row["emergence"]),
            ("PP  (Pocket Pivot)", sig_row["pocket_pivot"]),
        ]
        for name, fired in modes_status:
            st.write(f"  [{'FIRED' if fired else '----'}]  {name}")

        st.markdown("\n**Context**")
        rs_rank = feat_row.get("rs_rank")
        st.write(f"  RS rank (cross-sectional 1-99): "
                 f"**{int(rs_rank) if pd.notna(rs_rank) else 'n/a'}**")
        st.write(f"  RS 60d vs QQQ: "
                 f"**{feat_row.get('rs_60_prior', float('nan'))*100:+.1f}%**")
        rs_line_nh = "YES" if feat_row.get("rs_line_new_high", False) else "no"
        st.write(f"  RS line at new 50d high: **{rs_line_nh}**")
        sec_str = _sector_strength_today(sector_strength, ticker, last_bar)
        st.write(f"  Sector RS vs QQQ: "
                 f"**{f'{sec_str:+.1f}%' if sec_str is not None else 'n/a'}**  "
                 f"({sector_strength_label(sec_str)})")
        days_e = days_to_earnings(earnings_df, ticker, as_of=last_bar)
        st.write(f"  Earnings: **{earnings_proximity_label(days_e)}**")

    st.divider()

    # Signal table for this ticker
    triggers = sw[sw["any"]]
    if not triggers.empty:
        rows = []
        for d in triggers.index:
            sig_row = sw.loc[d]
            f_row = fw.loc[d]
            fwd5 = (fw["close"].shift(-5) / fw["close"] - 1).loc[d]
            fwd20 = (fw["close"].shift(-20) / fw["close"] - 1).loc[d]
            rows.append({
                "Date": d.date(),
                "Mode": _mode_label(sig_row),
                "Close": round(f_row["close"], 2),
                "Vol×": round(f_row["volume"] / f_row["vol_avg_50"], 2),
                "+5d %": round(fwd5 * 100, 1) if pd.notna(fwd5) else None,
                "+20d %": round(fwd20 * 100, 1) if pd.notna(fwd20) else None,
            })
        st.subheader(f"{ticker} signals in window")
        df_t = pd.DataFrame(rows)
        st.dataframe(df_t, hide_index=True, use_container_width=True,
                     column_config=_column_config_for(df_t))


# =============================================================
# PAGE 3 — Backtest Explorer
# =============================================================

elif page == "Backtest Explorer":
    st.title("Backtest Explorer")
    months = st.slider("Window (months back from latest bar)", 1, 24, 6)
    window_start = latest_date - pd.DateOffset(months=months)

    rows = []
    for t, sig_df in sigs.items():
        sd = sig_df[(sig_df.index >= window_start) & sig_df["any"]]
        for d in sd.index:
            f_row = feats[t].loc[d]
            sig_row = sig_df.loc[d]
            fwd5 = (feats[t]["close"].shift(-5) / feats[t]["close"] - 1).loc[d]
            fwd20 = (feats[t]["close"].shift(-20) / feats[t]["close"] - 1).loc[d]
            rows.append({
                "Date": d.date(),
                "Ticker": t,
                "Mode": _mode_label(sig_row),
                "Close": round(f_row["close"], 2),
                "Vol×": round(f_row["volume"] / f_row["vol_avg_50"], 2),
                "RS 60d %": round(f_row["rs_60_prior"] * 100, 1) if pd.notna(f_row["rs_60_prior"]) else None,
                "+5d %": round(fwd5 * 100, 1) if pd.notna(fwd5) else None,
                "+20d %": round(fwd20 * 100, 1) if pd.notna(fwd20) else None,
            })

    if not rows:
        st.info("No signals in window.")
    else:
        df = pd.DataFrame(rows).sort_values("Date", ascending=False)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total signals", len(df))
        c2.metric("Unique names", df["Ticker"].nunique())
        c3.metric("Mode VCP / MOM / EME",
                  f"{(df['Mode']=='VCP').sum()} / "
                  f"{(df['Mode']=='MOM').sum()} / "
                  f"{(df['Mode']=='EME').sum()}")
        c4.metric("Median +20d %", f"{df['+20d %'].median():.1f}%")

        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config=_column_config_for(df))

        # Hit rate per mode (forward 20d > 0)
        st.subheader("Hit rate per mode (forward 20d > 0)")
        valid = df.dropna(subset=["+20d %"])
        hits = valid.groupby("Mode").agg(
            n=("+20d %", "size"),
            hit_rate=("+20d %", lambda s: (s > 0).mean()),
            median_fwd20=("+20d %", "median"),
        ).reset_index()
        hits["hit_rate"] = (hits["hit_rate"] * 100).round(1)
        hits["median_fwd20"] = hits["median_fwd20"].round(1)
        st.dataframe(hits, hide_index=True, use_container_width=True,
                     column_config=_column_config_for(hits))
