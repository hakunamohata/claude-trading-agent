"""Multi-agent judgment system — Technical / Fundamental / Sentiment / Risk +
a Portfolio Manager that synthesizes them.

KEY DESIGN PRINCIPLE: each agent sees DIFFERENT inputs so they don't converge.
If you pass the same payload to all four, you've built one agent four times.

  Technical agent     → chart features only (EMAs, signals, RS line, ATR, VCP cond)
  Fundamental agent   → earnings + sector + valuation signals only
  Sentiment agent     → price action + volume signatures over last 20d only
  Risk agent          → full portfolio context, "what breaks this?"
  Portfolio Manager   → all four reports + position context, final call

Each agent uses Opus 4.7 with adaptive thinking + cached system prompts.

Cost: ~5 × $0.005 = ~$0.025 per ticker. 30-name watchlist = ~$0.75 per full scan.
"""

from __future__ import annotations
import os
from typing import Literal
from pathlib import Path
from pydantic import BaseModel, Field
import anthropic
from dotenv import load_dotenv

from data_fetch import DATA_DIR

load_dotenv(Path(__file__).parent / ".env")

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")
MAJ_DIR = DATA_DIR / "multi_agent"
MAJ_DIR.mkdir(exist_ok=True)


# ============================================================
# Pydantic schemas — each agent's output
# ============================================================

class TechnicalReport(BaseModel):
    score: int = Field(ge=0, le=100, description="Technical conviction 0-100")
    chart_strength: int = Field(ge=0, le=10)
    signal_quality: int = Field(ge=0, le=10)
    rs_quality: int = Field(ge=0, le=10)
    trend_health: int = Field(ge=0, le=10)
    summary: str = Field(max_length=240)


class FundamentalReport(BaseModel):
    score: int = Field(ge=0, le=100, description="Catalyst/setup conviction 0-100")
    earnings_safety: int = Field(ge=0, le=10, description="10 = no earnings risk near term")
    sector_tailwind: int = Field(ge=0, le=10)
    catalyst_potential: int = Field(ge=0, le=10)
    summary: str = Field(max_length=240)


class SentimentReport(BaseModel):
    score: int = Field(ge=0, le=100, description="Flow/sentiment conviction 0-100")
    accumulation_signature: int = Field(ge=0, le=10)
    momentum_strength: int = Field(ge=0, le=10)
    extension_safety: int = Field(ge=0, le=10, description="10 = not extended, room to run")
    summary: str = Field(max_length=240)


class RiskReport(BaseModel):
    score: int = Field(ge=0, le=100, description="Risk-adjusted attractiveness")
    concentration_risk: int = Field(ge=0, le=10, description="0 = severe concentration, 10 = fine")
    correlation_risk: int = Field(ge=0, le=10)
    position_size_risk: int = Field(ge=0, le=10)
    summary: str = Field(max_length=240)


class MinerviniReport(BaseModel):
    """Mark Minervini lens: VCP, Stage 2 leadership, tight bases, pocket pivots."""
    score: int = Field(ge=0, le=100, description="SEPA / VCP setup quality")
    vcp_grade: Literal["A", "B", "C", "D", "F"] = Field(description="Letter grade for base quality")
    stage_2_strength: int = Field(ge=0, le=10)
    pocket_pivot_quality: int = Field(ge=0, le=10)
    entry_proximity: int = Field(ge=0, le=10, description="10 = at the pivot, 0 = no entry yet")
    summary: str = Field(max_length=240)


class DruckenmillerReport(BaseModel):
    """Stan Druckenmiller lens: macro tailwinds, thematic strength, position-in-cycle."""
    score: int = Field(ge=0, le=100)
    macro_alignment: int = Field(ge=0, le=10, description="Is broader regime supportive?")
    theme_strength: int = Field(ge=0, le=10, description="AI infra / memory cycle / etc strength")
    sector_leadership: int = Field(ge=0, le=10)
    cycle_position: Literal["early", "mid", "late", "exhausted"]
    summary: str = Field(max_length=240)


class BurryReport(BaseModel):
    """Michael Burry lens: contrarian, extension risk, mean-reversion probability."""
    score: int = Field(ge=0, le=100, description="Contrarian conviction; HIGH = wait/avoid, LOW = enter")
    extension_risk: int = Field(ge=0, le=10, description="10 = severely extended, prime for reversion")
    rs_divergence_risk: int = Field(ge=0, le=10, description="10 = RS rolling over while price still up")
    mean_reversion_probability_pct: int = Field(ge=0, le=100)
    summary: str = Field(max_length=240)


class PortfolioManagerVerdict(BaseModel):
    final_score: int = Field(ge=0, le=100)
    bias: Literal["strong_long", "long", "watch", "trim", "avoid"]
    action: Literal["BUY", "ADD", "HOLD", "TRIM", "EXIT", "AVOID"]
    confidence: int = Field(ge=1, le=10)
    thesis: str = Field(max_length=280)
    key_risk: str = Field(max_length=280)
    sizing_note: str = Field(max_length=240, description="Position-sizing guidance vs current")


class MultiAgentResult(BaseModel):
    ticker: str
    technical: TechnicalReport
    fundamental: FundamentalReport
    sentiment: SentimentReport
    risk: RiskReport
    minervini: MinerviniReport | None = None
    druckenmiller: DruckenmillerReport | None = None
    burry: BurryReport | None = None
    pm: PortfolioManagerVerdict


# ============================================================
# System prompts — concise + role-specific
# ============================================================

SYSTEM_TECHNICAL = """You are the Technical Analysis agent in a momentum trading system. You see ONLY chart indicators — no fundamentals, no portfolio context, no news. Your job: evaluate this candidate's TECHNICAL setup quality on a 0-100 scale.

Inputs you receive:
- VCP/MOM/EME/PP signal modes (which fired today, near-misses)
- EMAs (50, 200) and Stage 2 confirmation
- RS rank (1-99 vs peers), RS vs QQQ, RS line at new highs
- ATR contraction, volume vs avg, 9/21 SMA cloud

Score 0-100 weighted by:
  chart_strength: clarity of the chart pattern (base, breakout, base-on-base)
  signal_quality: how clean and confirmed the signal mode is
  rs_quality: top-decile RS leadership signal
  trend_health: Stage 2 strength, EMA stacking, 50 EMA slope

Be terse. Summary = 1 sentence. JSON only."""


SYSTEM_FUNDAMENTAL = """You are the Fundamental & Catalyst agent. You see ONLY earnings proximity, sector strength, and catalyst potential — no chart pattern data. Your job: evaluate fundamental/catalyst quality on a 0-100 scale.

Inputs you receive:
- Earnings proximity (imminent / soon / far / unknown)
- Sector ETF strength vs QQQ
- RS rank percentile
- Stock's 60d outperformance vs QQQ (a fundamentals-adjacent signal)

Penalize:
  - Earnings within 14 days of a CONFIRMED date (extension blow-up risk)
  - Lagging sector (sector_rs < 0%)

Do NOT penalize:
  - "Unknown" or "not-imminent" earnings dates — earnings come on a regular ~3-month cadence, so missing exact date data is not a risk factor. Treat unknown as neutral / far.

Reward:
  - Recent post-earnings breakout (catalyst already digested)
  - Leading sector (10%+ vs QQQ)
  - Top-decile RS = institutional sponsorship

Score 0-100 weighted by earnings_safety, sector_tailwind, catalyst_potential.
Be terse. JSON only."""


SYSTEM_SENTIMENT = """You are the Sentiment & Flow agent. You see ONLY recent price action and volume signatures — no fundamentals, no chart patterns. Your job: evaluate institutional flow/sentiment on a 0-100 scale.

Inputs you receive:
- Volume multiple today vs 50-day avg
- VCP "quiet base" / volume-dry-up signals
- Pocket pivot (today's volume > max down-day volume of prior 10 days)
- Top-third-of-range close (closing strength)
- 9/21 SMA cloud state (bullish/bearish)

Reward:
  - Accumulation: VDU before breakout, pocket pivot inside base
  - Up-day on 1.5x+ volume = institutional bid
  - Bullish cloud regime

Penalize:
  - Extended above 50 EMA (extension risk)
  - Light volume on supposed breakouts (not confirmed)
  - Bearish cloud regime

Score 0-100 weighted by accumulation_signature, momentum_strength, extension_safety.
Be terse. JSON only."""


SYSTEM_RISK = """You are the Risk agent. You see ONLY portfolio context — current position size, account type, household concentration. Your job: evaluate position-level risk on a 0-100 scale (higher = more attractive risk-adjusted).

Inputs you receive:
- Current position $ value and % of total portfolio
- Account type (taxable/tax-deferred/tax-free)
- Household-level rules: MSFT locked, 10% max per position (MSFT exempt)
- Trade-eligible accounts: Roth IRA + 401k BrokerageLink

Penalize:
  - Position over 10% (concentration risk)
  - Position in non-trade-eligible account (cannot rebalance)
  - High correlation with existing major holdings (semis with NVDA, etc.)

Reward:
  - Position well under 5% (room to add)
  - Account allows free rebalancing (tax-advantaged)
  - Diversifying exposure vs existing book

Score 0-100. Be terse. JSON only."""


SYSTEM_MINERVINI = """You are the Mark Minervini investor agent. Your lens: SEPA (Specific Entry Point Analysis) + VCP (Volatility Contraction Pattern) + Stage 2 leadership + pocket pivots.

You see ONLY: VCP/PP signals, Stage 2 confirmation, EMA structure, RS rank, volume signature, extension percent above 50 EMA.

Score 0-100 on SEPA conviction:
  - vcp_grade: A=textbook quiet base + tight contraction, B=decent base, C=loose/wide, D=no real base, F=in a downtrend
  - stage_2_strength: 50EMA>200EMA, both rising, price respecting 21 EMA on pullbacks (10 = perfect)
  - pocket_pivot_quality: up-day volume vs max down-day volume of prior 10 (10 = clear institutional print)
  - entry_proximity: how close to a clean entry pivot (10 = at the pivot, 0 = far/extended)

Penalize: extension >25% above 50 EMA (no entry), no Stage 2, weak RS rank (<70).
Reward: tight contraction + RS rank >85 + pocket pivot + entry at 21/50 EMA.

Be terse, factual. JSON only."""


SYSTEM_DRUCKENMILLER = """You are the Stan Druckenmiller investor agent. Your lens: macro tailwinds, thematic strength, position-in-cycle. You think in terms of "what's the BIG story?" not chart patterns.

You see ONLY: macro regime score (0-100), sector ETF strength, RS vs benchmark, theme membership context.

Score 0-100 on macro/thematic conviction:
  - macro_alignment: is the broader regime supportive? (10 = clear risk-on with rate trajectory aligned)
  - theme_strength: is the name riding a powerful multi-year theme? (10 = clear secular winner like AI infra cycle 2024-2026, memory super-cycle)
  - sector_leadership: is its sector ETF leading?
  - cycle_position: where in the move are we? (early=fresh breakout, mid=trending, late=blow-off coming, exhausted=topped)

Penalize: lagging sector, broken macro (high VIX + falling breadth), late-cycle blow-offs.
Reward: leading sector + early/mid-cycle + thematic megatrend.

Be terse. JSON only."""


SYSTEM_BURRY = """You are the Michael Burry contrarian investor agent. Your lens: where's the over-extension, where's the divergence, what's likely to mean-revert.

You see ONLY: extension percent above 50/200 EMAs, RS rank trajectory, recent N-day return magnitude, volume on rallies vs pullbacks.

Score 0-100 on CONTRARIAN conviction (higher = stronger signal to AVOID or WAIT):
  - extension_risk: 10 = severely extended (40%+ above 50EMA), prime mean-reversion candidate; 0 = at the 50 EMA, healthy
  - rs_divergence_risk: 10 = RS rolling over while price still up (classic distribution); 0 = RS confirming
  - mean_reversion_probability_pct: rough probability stock pulls back 15%+ within 30 days

Reward (low score): names at the 50 EMA after a healthy pullback with RS holding up.
Penalize (high score): parabolic moves, +40% in last 30 days, RS divergence, light-volume rallies.

Burry would say: "the best entry is when everyone hates the name and it's bottoming, not when CNBC is talking about it." Score with that bias.

Be terse, factual. JSON only."""


SYSTEM_PM = """You are the Portfolio Manager. You see reports from up to SEVEN specialist agents (Technical, Fundamental, Sentiment, Risk, plus optional Minervini, Druckenmiller, Burry), an optional `live_research_report` with current catalysts/news/analyst actions, and full position context. Your job: synthesize them into a final action.

If `live_research_report` is present, weight it heavily — it contains current real-world catalysts the indicator-only agents can't see (insider sales, analyst PT changes, M&A, pending events). Recent material developments should shift your sizing and timing notes.

You must NOT just average their scores. Weigh them by quality:
  - If Technical says "no signal fired today" + Sentiment says "extended, light volume", final action is HOLD or TRIM regardless of Fundamental
  - If Risk flags concentration, even a 90 Technical doesn't justify ADD — recommend TRIM
  - If Fundamental flags earnings in 7 days, downgrade BUY to HOLD until after earnings
  - If all functional specialists (Tech/Fund/Sent/Risk) are 70+ AND position is under 5%, ADD or BUY is justified
  - When investor philosophy agents are included:
    * Minervini high (80+) + entry_proximity 8+ confirms a buyable setup
    * Druckenmiller flags "late" or "exhausted" cycle position → downgrade BUY toward HOLD
    * Burry high (70+) extension_risk → trim or wait for pullback even on otherwise great names
    * The three philosophies often disagree — that's intentional. Use them to size confidence, not as votes.

Actions:
  BUY    — new position; not currently held; >=80 conviction
  ADD    — increase existing position; <5% of portfolio; >=80 conviction
  HOLD   — keep as-is
  TRIM   — reduce position size (concentration or weakening setup)
  EXIT   — close position entirely (broken trend or persistent weakness)
  AVOID  — would not initiate, would not add

Be terse. thesis = 1 sentence WHY, key_risk = 1 sentence WHAT could break this. sizing_note = how much to trade (e.g., "trim 30% of position", "add 1/3 size on a pullback to 50 EMA"). JSON only."""


# ============================================================
# Client
# ============================================================

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _call_agent(system_prompt: str, user_payload: dict, schema: type[BaseModel]) -> BaseModel:
    """One Claude call with adaptive thinking + cached system prompt + Pydantic output."""
    client = _get_client()
    import json
    user_content = f"Candidate:\n{json.dumps(user_payload, indent=2, default=str)}"
    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_content}],
        output_format=schema,
    )
    return response.parsed_output


# ============================================================
# Per-agent payload builders — these are the KEY to differentiation
# ============================================================

def _technical_payload(ticker: str, feat_row, sig_row, comp_row) -> dict:
    return {
        "ticker": ticker,
        "signals_fired": {k: bool(sig_row[k]) for k in ("vcp", "momentum", "emergence", "pocket_pivot", "any")},
        "vcp_conditions_met": int(comp_row.sum()),
        "vcp_conditions_breakdown": {k: bool(v) for k, v in comp_row.items()},
        "rs_rank_1_99": int(feat_row["rs_rank"]) if pd_notna(feat_row.get("rs_rank")) else None,
        "rs_line_new_50d_high": bool(feat_row.get("rs_line_new_high", False)),
        "stage_2_confirmed": bool(feat_row["close_prior"] > feat_row["ema_50_prior"] > feat_row["ema_200_prior"]),
        "ema_50_slope_over_10d": round(float(feat_row.get("ema_50_slope10", 0)), 2),
        "ema_50": round(float(feat_row["ema_50"]), 2),
        "ema_200": round(float(feat_row["ema_200"]), 2),
        "sma_9_above_sma_21": bool(feat_row.get("cloud_bullish", False)),
        "atr_pct": round(float(feat_row["atr_pct"]) * 100, 2),
        "close": round(float(feat_row["close"]), 2),
    }


def _fundamental_payload(ticker: str, feat_row, earnings_label: str, sector_rs: float | None) -> dict:
    return {
        "ticker": ticker,
        "earnings_proximity": earnings_label,
        "sector_rs_vs_qqq_60d_pct": round(float(sector_rs), 1) if sector_rs is not None else None,
        "stock_rs_vs_qqq_60d_pct": round(float(feat_row["rs_60_prior"]) * 100, 1) if pd_notna(feat_row.get("rs_60_prior")) else None,
        "rs_rank_vs_peers_1_99": int(feat_row["rs_rank"]) if pd_notna(feat_row.get("rs_rank")) else None,
    }


def _sentiment_payload(ticker: str, feat_row, sig_row) -> dict:
    return {
        "ticker": ticker,
        "volume_today_vs_50d_avg": round(float(feat_row["volume"] / feat_row["vol_avg_50"]), 2)
            if feat_row.get("vol_avg_50") else None,
        "pocket_pivot_fired": bool(sig_row["pocket_pivot"]),
        "quiet_base_pre_breakout": bool(feat_row["atr_pct_prior"] <= feat_row["atr_pct_q35_120"])
            if pd_notna(feat_row.get("atr_pct_prior")) and pd_notna(feat_row.get("atr_pct_q35_120")) else None,
        "closed_in_top_third_of_range": bool(feat_row.get("close_in_top_third", False)),
        "sma_9_above_sma_21": bool(feat_row.get("cloud_bullish", False)),
        "pct_above_50_ema": round(float((feat_row["close"] - feat_row["ema_50"]) / feat_row["ema_50"] * 100), 1),
    }


def _risk_payload(ticker: str, position_value_usd: float, total_portfolio_usd: float,
                  account_type: str, trade_eligible: bool, ticker_in_locked: bool) -> dict:
    return {
        "ticker": ticker,
        "current_position_value_usd": round(position_value_usd, 2),
        "current_position_pct_of_portfolio": round(position_value_usd / total_portfolio_usd * 100, 2) if total_portfolio_usd > 0 else 0,
        "max_position_pct_household_rule": 10.0,
        "account_type": account_type,
        "trade_eligible_account": trade_eligible,
        "ticker_locked": ticker_in_locked,
    }


def _pm_payload(ticker: str, reports: dict, position_value_usd: float, total_portfolio_usd: float,
                trade_eligible: bool, research_report: dict | None = None) -> dict:
    payload = {
        "ticker": ticker,
        "current_position_value_usd": round(position_value_usd, 2),
        "current_position_pct_of_portfolio": round(position_value_usd / total_portfolio_usd * 100, 2) if total_portfolio_usd > 0 else 0,
        "trade_eligible_account": trade_eligible,
        "technical_report": reports["technical"].model_dump(),
        "fundamental_report": reports["fundamental"].model_dump(),
        "sentiment_report": reports["sentiment"].model_dump(),
        "risk_report": reports["risk"].model_dump(),
    }
    # Optional investor philosophy reports
    for name in ("minervini", "druckenmiller", "burry"):
        if reports.get(name) is not None:
            payload[f"{name}_report"] = reports[name].model_dump()
    # Optional live research (catalysts, news, analyst actions)
    if research_report is not None:
        payload["live_research_report"] = research_report
    return payload


# ============================================================
# Investor philosophy payload builders
# ============================================================

def _minervini_payload(ticker, feat_row, sig_row, comp_row) -> dict:
    """Chart pattern + Stage 2 + RS + volume — Minervini's specific lens."""
    return {
        "ticker": ticker,
        "signals": {k: bool(sig_row[k]) for k in ("vcp", "momentum", "emergence", "pocket_pivot")},
        "vcp_conditions_met": int(comp_row.sum()),
        "stage_2_confirmed": bool(feat_row["close_prior"] > feat_row["ema_50_prior"] > feat_row["ema_200_prior"]),
        "rs_rank_1_99": int(feat_row["rs_rank"]) if pd_notna(feat_row.get("rs_rank")) else None,
        "ema_50_slope_10d": round(float(feat_row.get("ema_50_slope10", 0)), 2),
        "pct_above_50_ema": round(float((feat_row["close"] - feat_row["ema_50"]) / feat_row["ema_50"] * 100), 1),
        "volume_x_vs_50d": round(float(feat_row["volume"] / feat_row["vol_avg_50"]), 2)
            if feat_row.get("vol_avg_50") else None,
        "atr_pct": round(float(feat_row["atr_pct"]) * 100, 2),
        "atr_pct_in_bottom_35_of_120d": bool(feat_row["atr_pct_prior"] <= feat_row["atr_pct_q35_120"])
            if pd_notna(feat_row.get("atr_pct_prior")) and pd_notna(feat_row.get("atr_pct_q35_120")) else None,
    }


def _druckenmiller_payload(ticker, feat_row, sector_rs, theme_tag, macro_score) -> dict:
    """Macro + sector + theme — Druckenmiller's big-picture lens."""
    return {
        "ticker": ticker,
        "macro_regime_score_0_100": macro_score,
        "sector_rs_vs_qqq_60d_pct": round(float(sector_rs), 1) if sector_rs is not None else None,
        "stock_rs_vs_qqq_60d_pct": round(float(feat_row["rs_60_prior"]) * 100, 1) if pd_notna(feat_row.get("rs_60_prior")) else None,
        "rs_rank_vs_peers_1_99": int(feat_row["rs_rank"]) if pd_notna(feat_row.get("rs_rank")) else None,
        "theme_tag": theme_tag,  # e.g. "AI infra", "memory super-cycle", "neutral"
    }


def _burry_payload(ticker, feat_row, close_series) -> dict:
    """Extension + divergence + recent run — Burry's contrarian lens."""
    pct_above_50 = float((feat_row["close"] - feat_row["ema_50"]) / feat_row["ema_50"] * 100)
    pct_above_200 = float((feat_row["close"] - feat_row["ema_200"]) / feat_row["ema_200"] * 100)
    # Recent 30-day return
    if len(close_series) >= 21:
        ret_30d = float(feat_row["close"] / close_series.iloc[-22] - 1) * 100
    else:
        ret_30d = None
    return {
        "ticker": ticker,
        "pct_above_50_ema": round(pct_above_50, 1),
        "pct_above_200_ema": round(pct_above_200, 1),
        "stock_30d_return_pct": round(ret_30d, 1) if ret_30d is not None else None,
        "rs_rank_vs_peers_1_99": int(feat_row["rs_rank"]) if pd_notna(feat_row.get("rs_rank")) else None,
        "rs_line_at_new_high": bool(feat_row.get("rs_line_new_high", False)),
        "volume_x_vs_50d": round(float(feat_row["volume"] / feat_row["vol_avg_50"]), 2)
            if feat_row.get("vol_avg_50") else None,
        "atr_pct": round(float(feat_row["atr_pct"]) * 100, 2),
    }


# Helper — pd.notna without depending on top-level import
def pd_notna(x):
    try:
        import pandas as pd
        return pd.notna(x)
    except Exception:
        return x is not None


# ============================================================
# Orchestrator
# ============================================================

def evaluate_full(ticker: str,
                  feat_row, sig_row, comp_row,
                  earnings_label: str,
                  sector_rs: float | None,
                  position_value_usd: float,
                  total_portfolio_usd: float,
                  account_type: str,
                  trade_eligible: bool,
                  ticker_in_locked: bool,
                  include_investor_agents: bool = False,
                  close_series=None,
                  macro_score: float | None = None,
                  theme_tag: str = "neutral",
                  research_report: dict | None = None) -> MultiAgentResult:
    """Run the 4 functional agents + PM. Optionally include 3 investor philosophy
    agents (Minervini, Druckenmiller, Burry) if include_investor_agents=True.

    Additional inputs needed for investor agents:
      close_series: pd.Series of the ticker's daily closes (for Burry 30d return)
      macro_score: 0-100 from macro_gate.compute_regime() (for Druckenmiller)
      theme_tag: e.g. "AI infra" — applied as a soft category hint to Druckenmiller
    """
    tech = _call_agent(SYSTEM_TECHNICAL, _technical_payload(ticker, feat_row, sig_row, comp_row), TechnicalReport)
    fund = _call_agent(SYSTEM_FUNDAMENTAL, _fundamental_payload(ticker, feat_row, earnings_label, sector_rs), FundamentalReport)
    sent = _call_agent(SYSTEM_SENTIMENT, _sentiment_payload(ticker, feat_row, sig_row), SentimentReport)
    risk = _call_agent(SYSTEM_RISK, _risk_payload(ticker, position_value_usd, total_portfolio_usd,
                                                  account_type, trade_eligible, ticker_in_locked), RiskReport)

    reports = {"technical": tech, "fundamental": fund, "sentiment": sent, "risk": risk}

    if include_investor_agents:
        try:
            minervini = _call_agent(SYSTEM_MINERVINI, _minervini_payload(ticker, feat_row, sig_row, comp_row), MinerviniReport)
            reports["minervini"] = minervini
        except Exception as e:
            print(f"  ! minervini agent error: {e}")
        try:
            drucken = _call_agent(SYSTEM_DRUCKENMILLER, _druckenmiller_payload(ticker, feat_row, sector_rs, theme_tag, macro_score), DruckenmillerReport)
            reports["druckenmiller"] = drucken
        except Exception as e:
            print(f"  ! druckenmiller agent error: {e}")
        if close_series is not None:
            try:
                burry = _call_agent(SYSTEM_BURRY, _burry_payload(ticker, feat_row, close_series), BurryReport)
                reports["burry"] = burry
            except Exception as e:
                print(f"  ! burry agent error: {e}")

    pm = _call_agent(SYSTEM_PM, _pm_payload(ticker, reports, position_value_usd, total_portfolio_usd,
                                            trade_eligible, research_report=research_report),
                     PortfolioManagerVerdict)

    return MultiAgentResult(
        ticker=ticker, technical=tech, fundamental=fund,
        sentiment=sent, risk=risk,
        minervini=reports.get("minervini"),
        druckenmiller=reports.get("druckenmiller"),
        burry=reports.get("burry"),
        pm=pm,
    )
