# Claude Trading Agent

A personal momentum-trading and options-income agent that scans ~540 US stocks daily, reasons over them with **7 specialist Claude agents + a Portfolio Manager synthesizer**, runs covered-call income engines on your held positions, screens for buy-write entries on new names, and produces specific BUY / ADD / HOLD / TRIM / EXIT / AVOID recommendations with position-sizing guidance.

Built for **swing / position traders running covered-call income strategies** who want a daily forward-looking watchlist plus disciplined rebalance advice on what they already hold — not a day-trading platform.

> ⚠️ This is a personal research tool, not financial advice. It generates ranked recommendations with rationale; **you make every trade decision yourself**, and you execute manually with your broker.

---

## What it does

### 1. Identifies tomorrow's leaders today (forward-looking scanner)

Scans S&P 500 + Nasdaq 100 + a curated mid-cap momentum list daily. Ranks each name by a composite of:
- **Clean base** — ATR contraction, proximity to 52-week high
- **RS strength** — IBD-style cross-sectional rank + RS line trajectory
- **Trend regime** — Stage 2 confirmation, EMA stacking, 50-EMA slope health
- **Pre-breakout proximity** — how close to triggering an actual signal
- **Accumulation signature** — 13-week A/D grade, pocket-pivot detection, volume signatures

Outputs a daily top-30 watchlist sorted by setup quality.

### 2. Reasons with up to 7 specialist Claude agents + LB synthesizer per candidate

Each agent receives **different inputs** so they don't converge to the same answer:

| Agent | Sees only | Asks |
|---|---|---|
| **Technical** | Chart features, EMAs, signals, RS line, ATR | "Is this chart constructive?" |
| **Fundamental** | Earnings proximity, sector RS, RS rank | "Is the catalyst environment favorable?" |
| **Sentiment / Flow** | Volume signatures, 13-week A/D grade, pocket pivots, extension | "Are institutions buying?" |
| **Risk** | Portfolio context (concentration, account, holdings) | "What breaks this for this user?" |
| **Minervini** *(opt-in)* | VCP setups, Stage 2, pocket pivots, entry proximity | "Is this a textbook SEPA / VCP setup?" |
| **Druckenmiller** *(opt-in)* | Macro regime, sector ETF strength, cycle position, theme tag | "Is the macro backdrop supportive?" |
| **Burry** *(opt-in)* | Extension, RS divergence, mean-reversion probability | "Where's the over-extension / hidden risk?" |
| **LB Portfolio Manager** | All specialist reports + position context + live research | **Final action + sizing** |

The LB synthesizer ("Laxmi Bank", an ode to the user's ancestral bank — same role as a PM) ingests all specialists, an optional live web-research report, and full position context, then issues one of BUY / ADD / HOLD / TRIM / EXIT / AVOID with a thesis, key risk, and sizing note. A hard rule prevents HOLD/TRIM/EXIT verdicts on unheld names.

Uses Claude Opus 4.7 with adaptive thinking + prompt caching. Structured output via Pydantic so every response is guaranteed to fit the schema.

Costs (per ticker):
- 4-agent panel: **~$0.025**
- 7-agent panel + live research: **~$0.04–0.06**
- 30-name watchlist deep run: **~$1.50**

### 3. Live web research (catalyst-aware analysis)

`research.py` calls Claude with the Anthropic `web_search` + `web_fetch` server tools to investigate a ticker's recent material developments — earnings, guidance, analyst actions, M&A, regulatory, pending events — and returns a structured `ResearchReport` (catalyst summary, recent developments, sentiment, key risks, pending catalysts, source URLs). The LB synthesizer weights this heavily when present, so the panel can see forward-looking catalysts that pure chart indicators miss.

Auto-fires from the dashboard's "Ticker Analysis" page in deep mode if no cached report exists.

### 4. Covered-call income engines (multi-strategy)

Three engines for running short-call strategies on positions you already own:

- **`cc_income_engine.py`** — multi-name CC recommendations with a 10-signal risk overlay (earnings proximity, sector RS, A/D grade, extension, IV regime, etc.). Produces an annualized cash yield + adjusted probability of profit per candidate.
- **`cc_buywrite.py`** — screener for tickers you don't own that look attractive as buy-and-hold + write CC. Surfaces the underlying setup quality alongside the option economics.
- **`msft_income.py`** — single-name MSFT wheel engine for a $69K/yr passive-income target with a 50% capacity hard-cap.

All three feed a portable trade ticket and the dashboard's color-coded tables.

### 5. QQQ LEAPS dip-buy strategy

`qqq_leaps_dipbuy.py` implements a forward-looking stock-replacement strategy: QQQ gaps down ≥1% + closes above the 100-day SMA → buy the ~60-delta 12-month call. Take-profit at +50%, no stop. Backtest on 5 years of QQQ data showed 79% hit rate (rose ≥15% within 365 days).

### 6. Tracks your portfolio across multiple accounts

Parses Fidelity statements (PDF), transaction history (CSV), and live screenshots. Tracks margin debt + daily interest accrual + active short-option positions + cost basis + realized/unrealized P&L + cumulative roll P&L across option lineages. Supports tax-advantaged (Roth, 401k, HSA) and taxable (Individual, TOD) accounts with different recommendation behavior in each.

### 7. Generates rebalance recommendations honoring user-defined rules

Configurable per user via `user_config.py`:
- **Locked positions** — tickers the agent must never recommend selling (employer stock, etc.)
- **Concentration cap** — max % of household value per position (default 10%, locked exempt)
- **Trade-eligible accounts** — only recommend actions in accounts where trading is allowed
- **Options sequencing** — for stocks with open short calls, always close options first before recommending share sales
- **Rolled-option cumulative P&L** — track running net cash across an entire roll lineage (premium received on currently-open contract is already INSIDE cumulative — never add MTM on top)
- **Earnings cadence** — earnings come quarterly; "unknown earnings dates" are not a risk
- **Filter integrity** — 30 pinned regression-test winning signals must always continue to fire

### 8. Daily orchestrator + WhatsApp inbound/outbound

- **`daily_run.py`** — orchestrates the full daily routine: refresh → scanner → options_tracker → cc_income_engine → cc_buywrite → qqq_leaps_dipbuy → todays_actions. Runs end-to-end in ~2 min.
- **`news_to_action.py`** — turns a free-text news/alert message (e.g. from a WhatsApp feed) into a per-ticker LB verdict. Three input modes: CLI/paste, HTTP receiver (for iOS Shortcut share-target), file watcher.
- **`notify_whatsapp.py`** — CallMeBot outbound for daily summaries.

---

## How it works (architecture)

```
                              ┌───────────────────────────────────┐
                              │  user_config.py (gitignored)      │
                              │  - your holdings + cost basis     │
                              │  - account IDs + rules            │
                              │  - margin status                  │
                              │  - personal watchlist             │
                              │  - active option positions        │
                              └─────────────┬─────────────────────┘
                                            │ imported by
        ┌────────────┬──────────────┬───────┴─────┬────────────┬────────────┐
        ▼            ▼              ▼             ▼            ▼            ▼
   ┌────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐  ┌────────┐  ┌──────────┐
   │scanner │   │portfolio│   │multi_    │   │cc_income │  │research│  │dashboard │
   │(wide   │   │rebalance│   │agent     │   │+ cc_buy- │  │(web    │  │(Streamlit│
   │universe│   │engine   │   │(7 specs  │   │write +   │  │search) │  │ 12 pages)│
   │+ scan) │   │         │   │+ LB)     │   │msft_     │  │        │  │          │
   │        │   │         │   │          │   │wheel)    │  │        │  │          │
   └───┬────┘   └────┬────┘   └────┬─────┘   └────┬─────┘  └───┬────┘  └────┬─────┘
       │             │             │              │            │            │
       └─────────────┴─────────────┴──────────────┴────────────┴────────────┘
                                            │
                              ┌─────────────┼─────────────┐
                              ▼             ▼             ▼
                       data/snapshots/  data/research/  data/scratchpad/
                       (gitignored)     (gitignored)    (gitignored)
                       ├── watchlist.parquet
                       ├── judgments_portfolio.jsonl
                       ├── cc_income.md
                       ├── cc_buywrite.md
                       ├── qqq_leaps_dipbuy.md
                       ├── todays_actions.md
                       └── trade_ticket.md
```

**Framework code is generic.** All user-specific data lives in `user_config.py` and gitignored output directories. Anyone can fork this repo and run it on their own portfolio.

---

## Setup (new user)

```sh
# 1. Clone and install
git clone https://github.com/hakunamohata/claude-trading-agent
cd claude-trading-agent
pip install -r requirements.txt

# 2. Set up your config
cp user_config.example.py user_config.py
# edit user_config.py — see comments in the file for each field
# Fill in your account IDs, holdings, rules, watchlist, options positions

# 3. Set up Anthropic API key
cp .env.example .env
# edit .env — paste your API key from https://console.anthropic.com/settings/keys

# 4. Verify the framework is working
python regression_test.py     # should print: Passed: 30/30
```

## Daily usage

### Option A — orchestrator (recommended)

```sh
python daily_run.py            # refresh → scan → engines → today's actions (~2 min)
streamlit run dashboard.py     # opens browser at localhost:8501
```

### Option B — individual modules

```sh
python refresh.py                    # refresh OHLCV data (~30 sec)
python scanner.py                    # build today's top-30 watchlist (~30 sec)
python watchlist_judge.py --top 10   # multi-agent on top 10 (~$0.25 in API)
python portfolio_judge.py            # multi-agent on your holdings (~$0.70 in API)
python portfolio_judge.py --investor-agents --with-research   # full 7-agent + research
python cc_income_engine.py           # covered-call recommendations
python cc_buywrite.py                # buy-write screener for unowned names
python qqq_leaps_dipbuy.py           # QQQ LEAPS signal check
python todays_actions.py             # one-pager action list
```

### Dashboard pages (12)

1. **Today's Brief** *(default landing)* — fired signals, near-misses, watchlist
2. **My Portfolio** — holdings, rebalance actions, sector breakdown, scores
3. **Ticker Analysis** — type any ticker, run 7-agent deep panel + research; persistent history per run
4. **Today's Actions** — one-pager from the orchestrator
5. **Trade Ticket** — portable execution sheet from the engines
6. **Covered Calls** — tabbed: Income Engine / Buy-Write Screener / MSFT Wheel / Per-Ticker Detail (roll history + trade ticket dropdown)
7. **QQQ LEAPS Dip-Buy** — strategy backtest + live signal status
8. **LB Backtest** — LB synthesizer hit-rate vs forward returns
9. **Research Done** — per-position LB judgments with freshness dates + sort options
10. **Chart Browser** — candles, EMAs, 9/21 SMA cloud, signal markers
11. **Backtest Explorer** — historical signal performance per mode
12. **Glossary** — terminology reference (delta, POP, IV, A/D grade, MTM, etc.)

---

## What it doesn't do (deliberately)

| Capability | Why not |
|---|---|
| Place trades automatically | Out of scope. You execute manually on your broker. |
| Day-trade / intraday signals | Designed for swing/position trades on end-of-day data |
| Tax-loss harvesting automation | Surfaces the opportunity, doesn't auto-execute |
| Time Flies / calendar spreads | Stock-focused for now; calendar/diagonal screener is on the roadmap |
| Broker API integration | No retail API for most brokers; positions are updated by editing `user_config.py` or pasting screenshots |

---

## Key files

| File | Purpose |
|---|---|
| `breakout.py` | Indicators + 4 signal modes (VCP, MOM, EME, PP) + 9/21 SMA cloud + RS rank + 13-week A/D |
| `scanner.py` | Daily multi-dimensional scanner → top-30 watchlist |
| `multi_agent.py` | 7-agent specialist judgment (4 functional + 3 investor lenses) + LB Portfolio Manager |
| `research.py` | Anthropic web_search-backed live catalyst research |
| `critic.py` | Self-validation pass over judgments |
| `scratchpad.py` | JSONL logger of multi-agent calls (tokens, latency, payloads) |
| `judgment.py` | Single-agent Claude scoring (simpler/cheaper alternative) |
| `portfolio.py` | Statement / CSV parser, margin status, position-action rules engine |
| `portfolio_judge.py` | Run multi-agent on all held positions (with `--investor-agents` and `--with-research` flags) |
| `watchlist_judge.py` | Run multi-agent on the scanner's top watchlist |
| `cc_income_engine.py` | Multi-name covered-call recommendation engine with 10-signal risk overlay |
| `cc_buywrite.py` | Buy-write screener for unowned tickers |
| `msft_income.py` | MSFT-specific covered-call wheel |
| `qqq_leaps_dipbuy.py` | QQQ gap-down + above-100-SMA LEAPS dip-buy strategy |
| `todays_actions.py` | One-pager action synthesizer |
| `daily_run.py` | Orchestrator (refresh → scanner → engines → actions) |
| `lb_backtest.py` | LB synthesizer backtest harness |
| `news_to_action.py` | Free-text news → ticker extraction → LB panel (CLI/HTTP/file-watcher modes) |
| `notify_whatsapp.py` | CallMeBot WhatsApp outbound |
| `options.py` / `options_tracker.py` | Black-Scholes Greeks + option position P&L |
| `dashboard.py` | Streamlit UI (12 pages) |
| `wide_universe.py` | Wikipedia scraper for S&P 500 + Nasdaq 100 with cached fallback |
| `snapshot.py` | Versioned daily archive manager |
| `sector.py` | Sector ETF strength computation |
| `macro_gate.py` | VIX + breadth + credit composite regime score |
| `earnings.py` | Next-earnings calendar (confirmed or estimated from last + 90d cycle) |
| `regression_test.py` | 30 pinned winning signals — runs after every change |
| `data_fetch.py` | yfinance OHLCV wrapper + parquet cache |
| `glossary.md` | Terminology reference rendered in the dashboard |
| `user_config.example.py` | Template for personal config (copy → `user_config.py`) |

## Signal modes

| Mode | Trigger | Catches |
|---|---|---|
| **VCP** (Volatility Contraction Pattern) | Quiet base + volume surge + price > prior 50d high | Classical Minervini-style breakouts |
| **MOM** (Trend Continuation) | Stage 2 + elite RS + 20-day high breakout | Names already in strong uptrends |
| **EME** (Stage-2 Emergence) | Close reclaims 200 EMA + golden cross + volume | Start of new uptrends |
| **PP** (Pocket Pivot) | Up-day with volume > max down-day volume of prior 10d | Institutional buying inside bases |

Each mode is independently regression-tested against historically winning signals to ensure changes don't regress.

---

## Cost

API calls run on **Claude Opus 4.7** with adaptive thinking + prompt caching. Typical daily run:

- Multi-agent on portfolio (~30 positions, 7 agents): **~$1.20**
- Multi-agent on watchlist (top 10, 7 agents): **~$0.40**
- Live research (5 tickers, web_search): **~$0.50**
- Combined daily cost: **~$2/day** running every weekday → **~$44/month**

Cheaper option: switch `ANTHROPIC_MODEL` in `.env` to `claude-haiku-4-5` for ~5× cost reduction with some quality tradeoff. Or use `--investor-agents` off for routine days.

---

## Privacy

- **Your financial data never leaves your machine.** All holdings + rules + analysis run locally.
- `user_config.py` is gitignored. So is `.env`, `holdings/`, `data/snapshots/`, `data/research/`, `data/news/`, `data/options/`, `data/multi_agent/`, `data/backtest/`, `data/scratchpad/`, `data/ticker_analysis/`, and all cached OHLCV.
- The public repo contains zero personal financial information — only framework code.
- API requests to Anthropic include the structured payloads (ticker + indicators + position size in $) but not account numbers, names, or anything personally identifying.
- The `research.py` web_search calls hit Anthropic's hosted browsing tool, which fetches public web pages — the ticker is the only thing identified.

---

## Background

Built iteratively with the help of [Claude Code](https://claude.ai/code). The architecture takes inspiration from:

- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) — multi-agent investor philosophy idea, here split into 4 functional specialists + 3 legendary investor lenses + a Portfolio Manager
- Mark Minervini's Volatility Contraction Pattern + Pocket Pivot mechanics (Minervini agent)
- Stan Druckenmiller's macro/theme-driven sizing (Druckenmiller agent)
- Michael Burry's contrarian / mean-reversion lens (Burry agent)
- Stan Weinstein's Stage 2 / Stage 4 trend framework
- IBD-style cross-sectional Relative Strength ranking
- William O'Neil's CAN SLIM principles (the "leadership" emphasis)

---

## License

MIT. See LICENSE (or treat as MIT until a file is added).

---

## Disclaimer

This software is for educational and personal research use only. It does not constitute financial, investment, tax, or legal advice. **Past signal performance does not predict future results.** Trade at your own risk. The author and contributors accept no liability for any losses incurred from using this software or acting on its recommendations.
