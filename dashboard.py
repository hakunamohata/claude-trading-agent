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

from universe import ALL_TICKERS, TARGETS, BENCHMARK, SECTOR_ETFS, TICKER_TO_SECTOR, sector_name
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
    "Sector": "Human-readable sector classification (e.g. Semiconductors, Technology, Financials)",
    "Sector ETF": "Underlying sector ETF used for sector-relative-strength tagging (SOXX, XLK, etc.)",
    "Sector RS %": ("Stock's sector ETF 60-day return minus QQQ 60-day return. "
                    "Positive = the sector is leading the market."),
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


def _latest_snapshot_date_with(filename: str) -> str | None:
    """Find the most recent snapshot directory that contains `filename`.
    Returns the date string, or None if none found."""
    base = _DATA_DIR / "snapshots"
    if not base.exists():
        return None
    candidates = sorted([d.name for d in base.iterdir() if d.is_dir()], reverse=True)
    for d in candidates:
        if (base / d / filename).exists():
            return d
    return None


def _latest_research_date_with(ticker: str) -> str | None:
    base = _DATA_DIR / "research"
    if not base.exists():
        return None
    candidates = sorted([d.name for d in base.iterdir() if d.is_dir()], reverse=True)
    for d in candidates:
        if (base / d / f"{ticker}.json").exists():
            return d
    return None


@st.cache_data(ttl=300)
def load_lb_judgments(date_str: str | None = None) -> tuple[dict, str | None]:
    """Read judgments_portfolio.jsonl from today's snapshot (or the most recent
    snapshot directory that has it). Returns (mapping, date_used)."""
    used = date_str
    p = _DATA_DIR / "snapshots" / (date_str or "_nope_") / "judgments_portfolio.jsonl"
    if not p.exists():
        used = _latest_snapshot_date_with("judgments_portfolio.jsonl")
        if not used:
            return {}, None
        p = _DATA_DIR / "snapshots" / used / "judgments_portfolio.jsonl"
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = _json.loads(line)
            out[r["ticker"]] = r
        except Exception:
            continue
    return out, used


@st.cache_data(ttl=300)
def load_research(date_str: str | None, ticker: str) -> dict | None:
    """Load research for a ticker — try given date first, then most recent.
    Attaches `_loaded_from_date` to the returned dict so callers can show
    freshness."""
    if date_str:
        p = _DATA_DIR / "research" / date_str / f"{ticker}.json"
        if p.exists():
            try:
                d = _json.loads(p.read_text())
                d["_loaded_from_date"] = date_str
                return d
            except Exception:
                pass
    used = _latest_research_date_with(ticker)
    if not used:
        return None
    try:
        d = _json.loads((_DATA_DIR / "research" / used / f"{ticker}.json").read_text())
        d["_loaded_from_date"] = used
        return d
    except Exception:
        return None


def _freshness_label(date_str: str | None, today_str: str | None = None) -> str:
    """Return a 'today' / 'N days old' / 'N weeks old' label for a YYYY-MM-DD."""
    if not date_str:
        return "no date"
    try:
        from datetime import datetime as _dt
        d  = _dt.strptime(date_str, "%Y-%m-%d").date()
        t  = (_dt.strptime(today_str, "%Y-%m-%d").date()
              if today_str else _dt.now().date())
        delta = (t - d).days
    except Exception:
        return date_str
    if delta <= 0:
        return "today"
    if delta == 1:
        return "1 day ago"
    if delta < 7:
        return f"{delta} days ago"
    if delta < 30:
        weeks = delta // 7
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    months = delta // 30
    return f"{months} month{'s' if months != 1 else ''} ago"


def _freshness_color(date_str: str | None, today_str: str | None = None) -> str:
    """Color-code by age. Fresh=green, week-old=yellow, stale=red."""
    if not date_str:
        return "#777"
    try:
        from datetime import datetime as _dt
        d  = _dt.strptime(date_str, "%Y-%m-%d").date()
        t  = (_dt.strptime(today_str, "%Y-%m-%d").date()
              if today_str else _dt.now().date())
        delta = (t - d).days
    except Exception:
        return "#777"
    if delta <= 1:    return "#1b5e20"   # fresh — green
    if delta <= 7:    return "#e65100"   # week-old — orange
    if delta <= 30:   return "#b71c1c"   # stale — red
    return "#5d4037"                      # very old — brown


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
page = st.sidebar.radio("Page", [
    "Today's Brief",
    "My Portfolio",
    "Ticker Analysis",
    "Today's Actions",
    "Trade Ticket",
    "Covered Calls",
    "QQQ LEAPS Dip-Buy",
    "LB Backtest",
    "Research Done",
    "Chart Browser",
    "Backtest Explorer",
    "Glossary",
])

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

    # ---- Load LB (multi-agent) judgments + research from snapshot ----
    from datetime import datetime as _dt
    calendar_today = _dt.now().strftime("%Y-%m-%d")
    lb_judgments, lb_date_used = load_lb_judgments(calendar_today)
    today_str = lb_date_used or calendar_today
    if lb_judgments:
        date_note = "today's snapshot" if lb_date_used == calendar_today else f"snapshot from {lb_date_used}"
        st.caption(f"Found LB multi-agent judgments for {len(lb_judgments)} positions in {date_note} "
                   f"(run `python portfolio_judge.py` to refresh)")
    else:
        st.warning("No LB judgments found in any snapshot. Run `python portfolio_judge.py` to populate.")

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
    df_hold["Sector ETF"] = df_hold["Ticker"].map(TICKER_TO_SECTOR).fillna("Other/Cash")
    df_hold["Sector"] = df_hold["Sector ETF"].map(lambda x: sector_name(x) if x != "Other/Cash" else "Other/Cash")
    sec_alloc = (
        df_hold.groupby(["Sector", "Sector ETF"])["Value"].sum()
        .sort_values(ascending=False).reset_index()
    )
    sec_alloc["% of total"] = (sec_alloc["Value"] / total_value * 100).round(1)
    st.subheader("Sector allocation")
    st.dataframe(sec_alloc, hide_index=True, use_container_width=True)

    # ---- Pointer to Research Done page ----
    if lb_judgments or claude_pf:
        n_lb = len(lb_judgments)
        n_quick = len(claude_pf)
        st.info(f"🔍 Per-position deep-dive analysis (LB synthesizer + 4 specialists + 3 investor "
                f"philosophies + live web research) lives on the **Research Done** page. "
                f"{n_lb} positions have LB judgments, {n_quick} have single-agent scores.")


elif page == "Research Done":
    st.title("Research Done — per-position deep-dive")

    # ---- Macro regime banner ----
    regime = load_macro_regime()
    if regime and regime.get("composite_score") is not None:
        score = regime["composite_score"]
        label = regime["regime_label"]
        if score >= 70: bg = "#1b4332"
        elif score >= 50: bg = "#2d3748"
        else: bg = "#5c1f1f"
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

    # ---- Load LB judgments ----
    from datetime import datetime as _dt
    calendar_today = _dt.now().strftime("%Y-%m-%d")
    lb_judgments, lb_date_used = load_lb_judgments(calendar_today)

    if not lb_judgments:
        st.warning(
            "No LB judgments found yet. Run one of these to populate:\n\n"
            "```\npython portfolio_judge.py                            # 4 agents only\n"
            "python portfolio_judge.py --investor-agents               # 7 agents\n"
            "python portfolio_judge.py --investor-agents --with-research  # 7 + live research\n```"
        )
        st.stop()

    lb_age = _freshness_label(lb_date_used, calendar_today)
    lb_color = _freshness_color(lb_date_used, calendar_today)
    st.markdown(
        f"<div style='display:inline-block;padding:6px 12px;border-radius:6px;"
        f"background-color:{lb_color};color:#fff;margin-bottom:10px;'>"
        f"<strong>LB judgments:</strong> {len(lb_judgments)} tickers · "
        f"analyzed <strong>{lb_date_used}</strong> ({lb_age})"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- Build per-ticker account lookup ----
    ticker_to_account = {}
    if not portfolio_df.empty:
        for _, row in portfolio_df.iterrows():
            t = row["ticker"]
            acct = row["account_id"]
            ticker_to_account.setdefault(t, []).append(acct)

    # ---- Action filter + sort ----
    actions = sorted(set(j["result"]["pm"]["action"] for j in lb_judgments.values()))
    filt_cols = st.columns([3, 3, 3])
    with filt_cols[0]:
        action_filter = st.multiselect(
            "Filter by action",
            options=actions,
            default=actions,
        )
    with filt_cols[1]:
        sort_by = st.selectbox(
            "Sort by",
            options=[
                "LB score (high → low)",
                "LB score (low → high)",
                "Research date (newest first)",
                "Research date (oldest first)",
                "LB date (newest first)",
                "LB date (oldest first)",
                "Ticker (A → Z)",
                "Conviction (high → low)",
            ],
            index=0,
        )

    # ---- Pre-compute per-ticker dates (research only; LB date is same for all) ----
    research_dates: dict[str, str] = {}
    for t in lb_judgments:
        rd = load_research(lb_date_used, t)
        research_dates[t] = rd.get("_loaded_from_date") if rd else ""

    def _sort_key(kv):
        t, j = kv
        pm = j["result"]["pm"]
        score = pm.get("final_score", 0)
        conf  = pm.get("confidence", 0)
        rdate = research_dates.get(t, "")  # YYYY-MM-DD sorts lexically by recency
        if sort_by == "LB score (high → low)":
            return (-score, t)
        if sort_by == "LB score (low → high)":
            return (score, t)
        if sort_by == "Research date (newest first)":
            # Empty strings sort to end when reversed
            return (rdate == "", -ord(rdate[0]) if rdate else 0, rdate * -1 if False else rdate)
        if sort_by == "Research date (oldest first)":
            return (rdate == "", rdate, t)
        if sort_by == "LB date (newest first)":
            return (lb_date_used or "", t)
        if sort_by == "LB date (oldest first)":
            return (lb_date_used or "", t)
        if sort_by == "Ticker (A → Z)":
            return (t,)
        if sort_by == "Conviction (high → low)":
            return (-conf, -score, t)
        return (-score, t)

    sorted_judgments = sorted(lb_judgments.items(), key=_sort_key)
    if sort_by == "Research date (newest first)":
        # Bring tickers WITH a date first, sorted desc by date
        with_date    = [(t, j) for t, j in lb_judgments.items() if research_dates.get(t)]
        without_date = [(t, j) for t, j in lb_judgments.items() if not research_dates.get(t)]
        with_date.sort(key=lambda kv: research_dates[kv[0]], reverse=True)
        without_date.sort(key=lambda kv: kv[0])
        sorted_judgments = with_date + without_date

    for ticker, j_lb in sorted_judgments:
        pm = j_lb["result"]["pm"]
        if pm["action"] not in action_filter:
            continue

        accounts = ticker_to_account.get(ticker, ["—"])
        acct_label = ACCOUNT_LABEL.get(accounts[0], accounts[0]).split(" ")[0] if accounts else "—"

        bias_emoji = {
            "strong_long": "🟢🟢", "long": "🟢",
            "watch": "🟡", "trim": "🟠", "avoid": "🔴",
        }.get(pm.get("bias", "watch"), "")

        # Peek at research date (without rendering body yet) for header freshness
        _peek = load_research(lb_date_used, ticker)
        _res_date = _peek.get("_loaded_from_date") if _peek else None
        _res_age = _freshness_label(_res_date, calendar_today) if _res_date else "no research"
        header = (f"{bias_emoji} {ticker} ({acct_label}) — "
                  f"LB {pm['final_score']}/100 · {pm['action']} · conf {pm['confidence']}/10  "
                  f"· LB {lb_date_used} ({_freshness_label(lb_date_used, calendar_today)})  "
                  f"· research: {_res_age}")

        with st.expander(header):
            # LB synthesizer (always show)
            st.markdown(f"### LB synthesizer (final)")
            st.markdown(f"**Thesis:** {pm['thesis']}")
            st.markdown(f"**Key risk:** {pm['key_risk']}")
            st.markdown(f"**Sizing:** {pm['sizing_note']}")
            st.markdown("---")

            r = j_lb["result"]

            # Specialist scores
            st.markdown("**Specialist scores:**")
            cols_s = st.columns(4)
            for col, (name, key) in zip(cols_s, [
                ("Technical", "technical"),
                ("Fundamental", "fundamental"),
                ("Sentiment", "sentiment"),
                ("Risk", "risk"),
            ]):
                if r.get(key):
                    col.metric(name, f"{r[key]['score']}/100")
                    col.caption(r[key]["summary"])

            # Investor philosophy scores
            if any(r.get(k) for k in ("minervini", "druckenmiller", "burry")):
                st.markdown("**Investor philosophies:**")
                cols_i = st.columns(3)
                if r.get("minervini"):
                    m = r["minervini"]
                    cols_i[0].metric(
                        "Minervini",
                        f"{m['score']}/100",
                        delta=f"VCP={m['vcp_grade']} · entry {m['entry_proximity']}/10",
                        delta_color="off",
                    )
                    cols_i[0].caption(m["summary"])
                if r.get("druckenmiller"):
                    d = r["druckenmiller"]
                    cols_i[1].metric(
                        "Druckenmiller",
                        f"{d['score']}/100",
                        delta=f"cycle: {d['cycle_position']}",
                        delta_color="off",
                    )
                    cols_i[1].caption(d["summary"])
                if r.get("burry"):
                    b = r["burry"]
                    cols_i[2].metric(
                        "Burry",
                        f"{b['score']}/100",
                        delta=f"mean-rev {b['mean_reversion_probability_pct']}% · ext {b['extension_risk']}/10",
                        delta_color="off",
                    )
                    cols_i[2].caption(b["summary"])

            # Live research
            research = load_research(lb_date_used, ticker)
            if research:
                st.markdown("---")
                res_date = research.get("_loaded_from_date")
                res_age = _freshness_label(res_date, calendar_today)
                res_color = _freshness_color(res_date, calendar_today)
                st.markdown(
                    f"### Live research &nbsp;"
                    f"<span style='background-color:{res_color};color:#fff;"
                    f"padding:3px 10px;border-radius:4px;font-size:0.85em;'>"
                    f"{res_date} · {res_age}</span>",
                    unsafe_allow_html=True,
                )
                sent_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(
                    research.get("sentiment"), ""
                )
                st.markdown(f"{sent_emoji} **Sentiment:** {research.get('sentiment', '?')}")
                st.markdown(f"**Catalyst summary:** {research.get('catalyst_summary', '')}")
                if research.get("recent_developments"):
                    st.markdown("**Recent developments:**")
                    for dev in research["recent_developments"][:6]:
                        st.markdown(f"- {dev}")
                if research.get("pending_catalysts"):
                    st.markdown("**Pending catalysts (next 30-60d):**")
                    for c in research["pending_catalysts"][:4]:
                        st.markdown(f"- {c}")
                st.markdown(f"**Key risks:** {research.get('key_risks', '')}")
                if research.get("sources"):
                    with st.expander("Sources"):
                        for s in research["sources"][:6]:
                            st.markdown(f"- [{s['title'][:90]}]({s['url']})")


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
                "Sector": sector_name(TICKER_TO_SECTOR.get(t)) or "—",
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
                    "Sector": sector_name(TICKER_TO_SECTOR.get(t)) or "—",
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


# =============================================================
# Markdown-artifact pages — render the files cc_income_engine.py,
# cc_buywrite.py, msft_income.py, todays_actions.py, etc. write daily.
# =============================================================

def _list_snapshots_with(filename: str) -> list[str]:
    """Sorted descending list of snapshot dates that contain `filename`."""
    root = DATA_DIR / "snapshots"
    if not root.exists():
        return []
    out = []
    for d in sorted(root.glob("*"), reverse=True):
        if d.is_dir() and (d / filename).exists():
            out.append(d.name)
    return out


# ---------- Smart markdown renderer with color-coded tables ---------------

# Row background colors by rating value. Lower-priority (lighter) tints for
# Streamlit's default light theme. Match on uppercase, strip emojis/whitespace.
_ROW_COLORS = {
    "GREEN":          "#d4edda",   # safe
    "SAFE":           "#d4edda",
    "YELLOW":         "#fff3cd",   # caution
    "MODERATE":       "#fff3cd",
    "AGGRESSIVE":     "#ffe5b4",   # peach
    "EARNINGS-RISK":  "#ffd699",   # orange — discrete event risk
    "RED":            "#f8d7da",   # danger
    "DANGEROUS":      "#f8d7da",
}
_VERDICT_COLS = {"color", "verdict"}  # column headers we treat as colorizers


def _strip_md_inline(s: str) -> str:
    """Strip markdown bold/italic + emoji decorations for matching."""
    import re as _re
    s = _re.sub(r"\*+", "", s).strip()
    # Drop common emoji decorations the engine adds
    for ch in ("📅", "🔒", "🚨", "⚠️", "🔥"):
        s = s.replace(ch, "")
    return s.strip()


def _parse_md_tables(text: str) -> list:
    """Yield (kind, content) blocks where kind is 'md' (markdown text) or
    'table' (a dict with headers and rows lists)."""
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect start of a markdown table: a row of '|...|' followed by a
        # separator row (---|---|...)
        if line.lstrip().startswith("|") and i + 1 < len(lines) \
           and set(lines[i + 1].strip().replace("|", "").replace("-", "").replace(":", "").strip()) <= set():
            header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
            data_rows = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if len(cells) == len(header_cells):
                    data_rows.append(cells)
                j += 1
            blocks.append(("table", {"headers": header_cells, "rows": data_rows}))
            i = j
            continue
        # Regular markdown line — accumulate
        buf = [line]
        i += 1
        while i < len(lines) and not lines[i].lstrip().startswith("|"):
            buf.append(lines[i])
            i += 1
        blocks.append(("md", "\n".join(buf)))
    return blocks


def _render_smart_markdown(text: str):
    """Render markdown with color-coded tables. Falls back to st.markdown
    for any table that has no Color/Verdict column or for non-table prose."""
    import pandas as pd
    blocks = _parse_md_tables(text)
    for kind, content in blocks:
        if kind == "md":
            if content.strip():
                st.markdown(content)
            continue
        headers = content["headers"]
        rows = content["rows"]
        # Find a colorizer column (case-insensitive on the header label)
        color_idx = None
        for idx, h in enumerate(headers):
            if h.strip().lower() in _VERDICT_COLS:
                color_idx = idx
                break
        if color_idx is None or not rows:
            # No styling — render as plain markdown
            md = ["| " + " | ".join(headers) + " |",
                  "|" + "|".join(["---"] * len(headers)) + "|"]
            for r in rows:
                md.append("| " + " | ".join(r) + " |")
            st.markdown("\n".join(md))
            continue

        # Build DataFrame, keep verdict column for styling but drop in display
        df = pd.DataFrame(rows, columns=headers)
        # Extract uppercase, decoration-stripped verdict value per row
        verdict_clean = df[headers[color_idx]].astype(str).map(
            lambda v: _strip_md_inline(v).upper()
        )

        def _row_style(row):
            v = verdict_clean.loc[row.name]
            # Match the first known token
            for key, bg in _ROW_COLORS.items():
                if key in v:
                    return [f"background-color: {bg}; color: #111"] * len(row)
            return [""] * len(row)

        display = df.drop(columns=[headers[color_idx]])
        st.dataframe(
            display.style.apply(_row_style, axis=1),
            hide_index=True,
            use_container_width=True,
        )


def _render_artifact_page(title: str, filename: str,
                          empty_hint: str, command_hint: str,
                          show_title: bool = True, key_prefix: str = ""):
    """Generic page renderer: date picker + smart markdown view + raw-toggle.

    show_title=False suppresses the st.title (use when embedding inside a tab
    that already has its own header). key_prefix scopes the selectbox/toggle
    keys so multiple calls in the same page don't collide.
    """
    if show_title:
        st.title(title)
    snaps = _list_snapshots_with(filename)
    if not snaps:
        st.info(empty_hint)
        st.code(command_hint, language="bash")
        return
    chosen = st.selectbox(
        "Snapshot date",
        snaps,
        index=0,
        help="Pick a date to view that day's run.",
        key=f"snap_{key_prefix}_{filename}",
    )
    path = DATA_DIR / "snapshots" / chosen / filename
    text = path.read_text(encoding="utf-8")
    c2 = st.container()
    c2.caption(f"File: `{path}`")
    show_raw = c2.toggle("Show raw markdown source", value=False,
                         key=f"raw_{key_prefix}_{filename}")
    if show_raw:
        st.code(text, language="markdown")
    else:
        _render_smart_markdown(text)


def _discover_cc_tickers() -> list[str]:
    """Scan data/snapshots/*/ for files matching <ticker>_roll_flow.md OR
    <ticker>_trade_ticket.md and return the unique uppercase tickers."""
    snaps_dir = DATA_DIR / "snapshots"
    if not snaps_dir.exists():
        return []
    found: set[str] = set()
    for day_dir in snaps_dir.iterdir():
        if not day_dir.is_dir():
            continue
        for f in day_dir.iterdir():
            name = f.name.lower()
            for suffix in ("_roll_flow.md", "_trade_ticket.md"):
                if name.endswith(suffix):
                    ticker = name[: -len(suffix)].upper()
                    if ticker:
                        found.add(ticker)
                    break
    return sorted(found)


# =============================================================
# Today's Actions one-pager
# =============================================================

if page == "Today's Actions":
    _render_artifact_page(
        title="Today's Actions",
        filename="todays_actions.md",
        empty_hint="No `todays_actions.md` snapshots found yet.",
        command_hint="python todays_actions.py",
    )

# =============================================================
# Trade Ticket — portable execution sheet
# =============================================================

if page == "Trade Ticket":
    _render_artifact_page(
        title="Trade Ticket",
        filename="trade_ticket.md",
        empty_hint="No trade ticket yet. Once `cc_income_engine.py` and `cc_buywrite.py` run, "
                   "a trade ticket can be drafted from their output.",
        command_hint="# Today's ticket lives at data/snapshots/<date>/trade_ticket.md",
    )

# =============================================================
# Covered Calls — consolidated page (engines + per-ticker detail)
# =============================================================

if page == "Covered Calls":
    st.title("Covered Calls")
    st.caption("Income engine, buy-write screener, MSFT wheel, and per-ticker "
               "roll history + trade tickets — all in one place.")

    cc_tabs = st.tabs([
        "Income Engine",
        "Buy-Write Screener",
        "MSFT Wheel",
        "Per-Ticker Detail",
    ])

    with cc_tabs[0]:
        _render_artifact_page(
            title="CC Income Engine",
            filename="cc_income.md",
            empty_hint="No `cc_income.md` yet — run the engine to generate today's report.",
            command_hint="python cc_income_engine.py",
            show_title=False, key_prefix="cc_income",
        )

    with cc_tabs[1]:
        _render_artifact_page(
            title="Buy-Write Screener",
            filename="cc_buywrite.md",
            empty_hint="No `cc_buywrite.md` yet — run the screener to surface unowned buy-write candidates.",
            command_hint="python cc_buywrite.py",
            show_title=False, key_prefix="cc_buywrite",
        )

    with cc_tabs[2]:
        _render_artifact_page(
            title="MSFT Covered-Call Wheel",
            filename="msft_income.md",
            empty_hint="No `msft_income.md` yet — single-name MSFT report.",
            command_hint="python msft_income.py",
            show_title=False, key_prefix="msft_wheel",
        )

    with cc_tabs[3]:
        tickers = _discover_cc_tickers()
        if not tickers:
            st.info("No per-ticker roll history or trade tickets found yet. "
                    "Files are expected at "
                    "`data/snapshots/<date>/<ticker>_roll_flow.md` "
                    "and `data/snapshots/<date>/<ticker>_trade_ticket.md`.")
        else:
            chosen_ticker = st.selectbox(
                "Ticker", tickers, index=0, key="cc_ticker_pick",
                help="Tickers are auto-discovered from snapshot filenames.",
            )
            ticker_lower = chosen_ticker.lower()
            detail_tabs = st.tabs(["Roll History", "Trade Ticket"])
            with detail_tabs[0]:
                _render_artifact_page(
                    title=f"{chosen_ticker} Roll History",
                    filename=f"{ticker_lower}_roll_flow.md",
                    empty_hint=f"No `{ticker_lower}_roll_flow.md` snapshots found for {chosen_ticker}.",
                    command_hint=f"# See data/snapshots/<date>/{ticker_lower}_roll_flow.md",
                    show_title=False, key_prefix=f"roll_{ticker_lower}",
                )
            with detail_tabs[1]:
                _render_artifact_page(
                    title=f"{chosen_ticker} Trade Ticket",
                    filename=f"{ticker_lower}_trade_ticket.md",
                    empty_hint=f"No `{ticker_lower}_trade_ticket.md` snapshots found for {chosen_ticker}.",
                    command_hint=f"# See data/snapshots/<date>/{ticker_lower}_trade_ticket.md",
                    show_title=False, key_prefix=f"ticket_{ticker_lower}",
                )

# =============================================================
# QQQ LEAPS Dip-Buy
# =============================================================

if page == "QQQ LEAPS Dip-Buy":
    _render_artifact_page(
        title="QQQ LEAPS Dip-Buy",
        filename="qqq_leaps_dipbuy.md",
        empty_hint="No `qqq_leaps_dipbuy.md` yet — run the engine to generate today's report.",
        command_hint="python qqq_leaps_dipbuy.py --backtest --years 5",
    )

# =============================================================
# LB Backtest — lives in data/backtest/, not in snapshots/
# =============================================================

if page == "LB Backtest":
    st.title("LB Backtest")
    bt_dir = DATA_DIR / "backtest"
    md_files = sorted(bt_dir.glob("lb_backtest_*.md"), reverse=True) if bt_dir.exists() else []
    if not md_files:
        st.info("No backtest runs yet.")
        st.code("python lb_backtest.py", language="bash")
    else:
        labels = [p.stem.replace("lb_backtest_", "") for p in md_files]
        chosen = st.selectbox("Backtest as-of date", labels, index=0)
        path = bt_dir / f"lb_backtest_{chosen}.md"
        st.caption(f"File: `{path}`")
        if st.toggle("Show raw markdown source", value=False):
            st.code(path.read_text(encoding="utf-8"), language="markdown")
        else:
            st.markdown(path.read_text(encoding="utf-8"))

        # Also surface the JSONL rows as a sortable table
        jsonl = bt_dir / f"lb_backtest_{chosen}.jsonl"
        if jsonl.exists():
            st.subheader("Per-row data")
            rows = [_json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, use_container_width=True)

# =============================================================
# Glossary — render glossary.md with the smart renderer
# =============================================================

if page == "Glossary":
    glossary_path = Path(__file__).parent / "glossary.md"
    if not glossary_path.exists():
        st.title("Glossary")
        st.warning("`glossary.md` not found in repo root.")
    else:
        _render_smart_markdown(glossary_path.read_text(encoding="utf-8"))


# =============================================================
# Ticker Analysis — type a ticker, run the full LB multi-agent panel
# =============================================================

_ACTION_BG = {
    "BUY":   "#1b5e20", "ADD":   "#1b5e20",
    "HOLD":  "#5d4037",
    "TRIM":  "#e65100", "EXIT":  "#b71c1c", "AVOID": "#b71c1c",
}

def _action_pill(action: str) -> str:
    bg = _ACTION_BG.get(action.upper(), "#37474f")
    return (f"<span style='background-color:{bg};color:#fff;"
            f"padding:6px 14px;border-radius:6px;font-weight:700;"
            f"font-size:1.1em;letter-spacing:0.5px;'>{action}</span>")


TA_HISTORY_DIR = DATA_DIR / "ticker_analysis"
TA_HISTORY_DIR.mkdir(exist_ok=True, parents=True)


def _ta_save_result(ticker: str, result: dict, deep: bool) -> Path:
    from datetime import datetime as _dt
    now = _dt.now()
    day_dir = TA_HISTORY_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(exist_ok=True, parents=True)
    fname = f"{ticker}_{now.strftime('%H%M%S')}_{'deep' if deep else 'fast'}.json"
    path = day_dir / fname
    payload = {
        "ticker": ticker,
        "timestamp": now.isoformat(timespec="seconds"),
        "deep": deep,
        "result": result,
    }
    path.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _ta_list_history(limit: int = 100) -> list[dict]:
    rows = []
    if not TA_HISTORY_DIR.exists():
        return rows
    files = sorted(TA_HISTORY_DIR.glob("*/*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[:limit]:
        try:
            payload = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        verdicts = payload.get("result", {}).get("verdicts", [])
        first = verdicts[0] if verdicts else {}
        lb = first.get("lb", {}) if isinstance(first, dict) else {}
        rows.append({
            "path": p,
            "ticker": payload.get("ticker") or first.get("ticker") or "?",
            "timestamp": payload.get("timestamp", ""),
            "deep": payload.get("deep", False),
            "action": lb.get("action", "—"),
            "score": lb.get("final_score"),
            "confidence": lb.get("confidence"),
        })
    return rows


def _render_ticker_verdict(v: dict):
    """Render one LB verdict block from news_to_action.process_message output.
    Supports both fast (4-agent) and deep (4 + Minervini + Druckenmiller + Burry)
    modes, plus macro context and research-report presence."""
    ticker = v.get("ticker", "?")
    if v.get("error"):
        st.error(f"**{ticker}** — {v['error']}")
        return

    if "lb" not in v:
        st.subheader(ticker)
        pos = v.get("position", {})
        if pos.get("held"):
            st.info(f"Held — {int(pos['shares'])} shares "
                    f"(${pos['total_value']:,.0f}, "
                    f"{pos.get('pct_household', 0):.1f}% of household)")
        else:
            st.info("Not held.")
        st.caption("LLM panel skipped — toggle 'Run LLM panel' to get full verdict.")
        return

    lb = v["lb"]
    panel = v["panel"]
    pos = v.get("position", {})
    spot = v.get("spot")
    is_deep = v.get("deep", False)

    head_left, head_right = st.columns([3, 2])
    head_left.subheader(f"{ticker} — ${spot:.2f}" if spot else ticker)
    head_right.markdown(_action_pill(lb["action"]), unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("LB score",     f"{lb['final_score']}/100")
    k2.metric("Conviction",   f"{lb['confidence']}/10")
    rs = v.get("rs_rank")
    k3.metric("RS rank",      f"{rs}" if rs is not None else "—")
    days_e = v.get("days_to_earnings")
    k4.metric("Earnings in",  f"{days_e}d" if days_e is not None else "—",
              help=v.get("earnings_label", ""))

    st.markdown("**Functional panel**")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Technical",   f"{panel['technical']}/100")
    p2.metric("Fundamental", f"{panel['fundamental']}/100")
    p3.metric("Sentiment",   f"{panel['sentiment']}/100")
    p4.metric("Risk",        f"{panel['risk']}/100")

    if is_deep:
        st.markdown("**Legendary investor lenses**")
        l1, l2, l3 = st.columns(3)
        mv = v.get("minervini") or {}
        dr = v.get("druckenmiller") or {}
        br = v.get("burry") or {}
        l1.metric("Minervini (VCP/SEPA)", f"{mv.get('score','—')}/100",
                  help=f"VCP grade {mv.get('vcp_grade','—')} · stage-2 {mv.get('stage_2_strength','—')}/10 · pocket pivot {mv.get('pocket_pivot_quality','—')}/10")
        l2.metric("Druckenmiller (macro/theme)", f"{dr.get('score','—')}/100",
                  help=f"Cycle: {dr.get('cycle_position','—')} · macro {dr.get('macro_alignment','—')}/10 · theme {dr.get('theme_strength','—')}/10")
        l3.metric("Burry (contrarian/mean-rev)", f"{br.get('score','—')}/100",
                  help=f"Extension {br.get('extension_risk','—')}/10 · mean-rev prob {br.get('mean_reversion_probability_pct','—')}%")

        m = v.get("macro")
        theme = v.get("theme_tag")
        if m or theme:
            bits = []
            if m and m.get("label"):
                if m.get("score") is not None:
                    bits.append(f"**Macro regime:** {m['label']} ({m['score']:.0f}/100)")
                else:
                    bits.append(f"**Macro regime:** {m['label']}")
            if theme and theme != "neutral":
                bits.append(f"**Sector/theme:** {theme}")
            st.caption("  ·  ".join(bits))

        # Forward-looking research block (web search results)
        research = v.get("research")
        if research:
            sentiment = research.get("sentiment", "neutral")
            sent_color = {"bullish": "#1b5e20", "neutral": "#5d4037", "bearish": "#b71c1c"}.get(sentiment, "#37474f")
            st.markdown(
                f"**Forward-looking catalysts** &nbsp;"
                f"<span style='background-color:{sent_color};color:#fff;padding:3px 10px;"
                f"border-radius:4px;font-size:0.85em;'>{sentiment.upper()}</span>",
                unsafe_allow_html=True,
            )
            cs = research.get("catalyst_summary")
            if cs:
                st.markdown(f"_{cs}_")
            rd = research.get("recent_developments") or []
            if rd:
                st.markdown("**Recent developments (last 30 days):**")
                for d in rd:
                    st.markdown(f"- {d}")
            pc = research.get("pending_catalysts") or []
            if pc:
                st.markdown("**Pending catalysts (next 30-60 days):**")
                for c in pc:
                    st.markdown(f"- {c}")
            kr = research.get("key_risks")
            if kr:
                st.markdown(f"**Research-level risks:** {kr}")
            srcs = research.get("sources") or []
            if srcs:
                with st.expander(f"Sources ({len(srcs)})"):
                    for s in srcs:
                        st.markdown(f"- [{s.get('title','(no title)')}]({s.get('url','')})")
        elif v.get("research_error"):
            st.caption(f"⚠ Research call failed: {v['research_error']}")
        else:
            st.caption("No research report — LB ran without web-research context.")

        with st.expander("Lens summaries (Minervini / Druckenmiller / Burry)"):
            if mv.get("summary"):
                st.markdown(f"**Minervini.** {mv['summary']}")
            if dr.get("summary"):
                st.markdown(f"**Druckenmiller.** {dr['summary']}")
            if br.get("summary"):
                st.markdown(f"**Burry.** {br['summary']}")

    sigs = v.get("signals_fired") or []
    if sigs:
        st.markdown("**Signals fired:** " + "  ".join(f"`{s.upper()}`" for s in sigs))

    st.markdown(f"**Thesis.** {lb['thesis']}")
    st.markdown(f"**Key risk.** {lb['key_risk']}")
    st.markdown(f"**Sizing.** {lb['sizing_note']}")

    if pos.get("held"):
        rows = []
        for a in pos.get("accounts", []):
            rows.append({"Account": a["label"], "Shares": int(a["shares"]),
                         "Value": f"${a['value']:,.0f}"})
        pct = pos.get("pct_household")
        held_caption = f"Held in {len(rows)} account(s)"
        if pct is not None:
            held_caption += f" — {pct:.1f}% of household"
        if pos.get("locked"):
            held_caption += " — LOCKED"
        st.markdown(f"**{held_caption}**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("Not held.")

    bw = v.get("buywrite_alt")
    if bw:
        st.markdown("**Buy-write alternative (own + write call):**")
        st.markdown(
            f"Strike **${int(bw['strike'])}**, expiry **{bw['expiry']}** ({bw['dte']} DTE) — "
            f"premium ${bw['premium']:.2f}, annualized yield "
            f"{bw['annualized_yield_pct']:.1f}%, adjusted probability of profit "
            f"{bw['adj_pop']:.0f}% — **{bw['verdict']}**. "
            f"Cost per contract ≈ ${bw['cost_per_contract']:,.0f}."
        )


if page == "Ticker Analysis":
    st.title("Ticker Analysis")
    st.caption(
        "Type a ticker — DEEP mode runs the full intel stack: "
        "Technical · Fundamental · Sentiment · Risk · "
        "**Minervini (VCP/SEPA)** · **Druckenmiller (macro/theme)** · **Burry (contrarian)** · "
        "+ macro regime + cached research + LB Portfolio Manager synthesizer. "
        "Cost ≈ $0.04 deep / $0.025 fast per ticker."
    )

    col_t, col_d, col_l, col_b = st.columns([2, 1, 1, 1])
    raw_input = col_t.text_input(
        "Ticker symbol",
        value="",
        placeholder="e.g. INTC, NBIS, AAPL, $TSLA",
        key="ta_ticker_input",
    )
    ticker = raw_input.strip().upper().lstrip("$")
    deep_mode = col_d.checkbox(
        "Deep panel", value=True,
        help="On = 7 agents + macro + research. Off = 4 agents + LB only.",
    )
    run_llm = col_l.checkbox(
        "Run LLM", value=True,
        help="Off = position context + signals only, no LLM tokens.",
    )
    want_buywrite = col_b.checkbox(
        "Buy-write alt", value=True,
        help="If not held and panel says BUY/ADD, also fetch a buy-write candidate.",
    )

    run_clicked = st.button(
        "Run analysis", type="primary", disabled=not ticker,
        use_container_width=False,
    )

    if run_clicked and ticker:
        spinner_msg = f"Analyzing {ticker} — {'deep' if deep_mode else 'fast'} panel..."
        with st.spinner(spinner_msg):
            try:
                from news_to_action import process_message
                result = process_message(
                    f"${ticker}",
                    run_llm=run_llm,
                    want_buywrite=want_buywrite,
                    deep=deep_mode and run_llm,
                )
                saved = _ta_save_result(ticker, result, deep=deep_mode and run_llm)
                st.session_state["ta_result"] = result
                st.session_state["ta_last_ticker"] = ticker
                st.session_state["ta_last_saved"] = str(saved)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.session_state.pop("ta_result", None)

    # ---- History panel ----
    st.divider()
    hist = _ta_list_history(limit=200)
    if hist:
        with st.expander(f"Analysis history ({len(hist)} runs)", expanded=False):
            hist_df = pd.DataFrame([{
                "When": h["timestamp"],
                "Ticker": h["ticker"],
                "Mode": "deep" if h["deep"] else "fast",
                "Action": h["action"],
                "LB score": h["score"],
                "Conviction": h["confidence"],
                "File": h["path"].name,
            } for h in hist])
            st.dataframe(hist_df, hide_index=True, use_container_width=True)

            options = [f"{h['timestamp']} — {h['ticker']} ({'deep' if h['deep'] else 'fast'}) → {h['action']}"
                       for h in hist]
            pick_col, btn_col = st.columns([4, 1])
            choice = pick_col.selectbox("Pick a prior analysis to reload",
                                        options, index=0, key="ta_history_pick")
            load_clicked = btn_col.button("Load", use_container_width=True,
                                          key="ta_history_load")
            if load_clicked and choice:
                idx = options.index(choice)
                try:
                    payload = _json.loads(hist[idx]["path"].read_text(encoding="utf-8"))
                    st.session_state["ta_result"] = payload.get("result", {})
                    st.session_state["ta_last_ticker"] = payload.get("ticker", "")
                    st.session_state["ta_last_saved"] = str(hist[idx]["path"])
                    st.success(f"Loaded {hist[idx]['ticker']} from {hist[idx]['timestamp']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not reload: {e}")
    else:
        st.caption("No analyses saved yet — they'll appear here after your first run.")

    # ---- Current/loaded result ----
    result = st.session_state.get("ta_result")
    if result:
        last = st.session_state.get("ta_last_ticker", "")
        saved_path = st.session_state.get("ta_last_saved", "")
        st.caption(f"Showing: **{last}**" + (f" · saved to `{saved_path}`" if saved_path else ""))

        if not result.get("tickers"):
            st.warning(result.get("warning", "No tickers were extracted from the input."))
        else:
            for v in result["verdicts"]:
                st.divider()
                _render_ticker_verdict(v)

        with st.expander("Show raw JSON result"):
            st.json(result)
    else:
        st.info("Enter a ticker and click **Run analysis** to start.")
