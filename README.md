# Claude Trading Agent

A personal momentum-trading agent that scans ~540 US stocks daily, reasons over them with **5 specialist Claude agents**, analyzes your household portfolio across multiple brokerage accounts, and produces specific BUY / ADD / HOLD / TRIM / EXIT / AVOID recommendations with position-sizing guidance.

Built for **swing / position traders** who want a daily forward-looking watchlist plus disciplined rebalance advice on what they already hold — not a day-trading platform.

> ⚠️ This is a personal research tool, not financial advice. It generates ranked recommendations with rationale; **you make every trade decision yourself**, and you execute manually with your broker.

---

## What it does

### 1. Identifies tomorrow's leaders today (forward-looking scanner)

Scans S&P 500 + Nasdaq 100 + a curated mid-cap momentum list daily. Ranks each name by a composite of:
- **Clean base** — ATR contraction, proximity to 52-week high
- **RS strength** — IBD-style cross-sectional rank + RS line trajectory
- **Trend regime** — Stage 2 confirmation, EMA stacking, 50-EMA slope health
- **Pre-breakout proximity** — how close to triggering an actual signal
- **Volume signature** — accumulation / volume-dry-up patterns

Outputs a daily top-30 watchlist sorted by setup quality.

### 2. Reasons with 5 specialist Claude agents per candidate

Each agent receives **different inputs** so they don't converge to the same answer:

| Agent | Sees only | Asks |
|---|---|---|
| **Technical** | Chart features, EMAs, signals, RS line, ATR | "Is this chart constructive?" |
| **Fundamental** | Earnings proximity, sector RS, RS rank | "Is the catalyst environment favorable?" |
| **Sentiment** | Volume signatures, accumulation pattern, extension | "Are institutions buying?" |
| **Risk** | Portfolio context (concentration, account, holdings) | "What breaks this for this user?" |
| **Portfolio Manager** | The 4 reports + position context | **Final action + sizing** |

Uses Claude Opus 4.7 with adaptive thinking + prompt caching. Structured output via Pydantic so every response is guaranteed to fit the schema. Cost: ~$0.025 per ticker, ~$0.75 for a 30-name watchlist run.

### 3. Tracks your portfolio across multiple accounts

Parses Fidelity statements (PDF), transaction history (CSV), and live screenshots. Tracks margin debt + daily interest accrual + active short-option positions + cost basis + realized/unrealized P&L. Supports tax-advantaged (Roth, 401k, HSA) and taxable (Individual, TOD) accounts with different recommendation behavior in each.

### 4. Generates rebalance recommendations honoring user-defined rules

Configurable per user via `user_config.py`:
- **Locked positions** — tickers the agent must never recommend selling (employer stock, etc.)
- **Concentration cap** — max % of household value per position (default 10%, locked exempt)
- **Trade-eligible accounts** — only recommend actions in accounts where trading is allowed
- **Options sequencing** — for stocks with open short calls, always close options first before recommending share sales
- **Earnings cadence** — earnings come quarterly; "unknown earnings dates" are not a risk
- **Filter integrity** — 30 pinned regression-test winning signals must always continue to fire

---

## How it works (architecture)

```
                              ┌───────────────────────────────────┐
                              │  user_config.py (gitignored)      │
                              │  - your holdings + cost basis     │
                              │  - account IDs + rules            │
                              │  - margin status                  │
                              │  - personal watchlist             │
                              └─────────────┬─────────────────────┘
                                            │ imported by
              ┌─────────────┬───────────────┴─┬─────────────────┐
              ▼             ▼                 ▼                 ▼
       ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐
       │ scanner  │  │ portfolio    │  │ multi_agent │  │ dashboard   │
       │ (wide    │  │ rebalance    │  │ (5 Claude   │  │ (Streamlit) │
       │ universe)│  │ rules engine │  │ specialists)│  │             │
       └────┬─────┘  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘
            │               │                  │                │
            └───────────────┴──────────────────┴────────────────┘
                                            │
                                            ▼
                                  data/snapshots/<date>/
                                  ├── watchlist.parquet
                                  ├── judgments_portfolio.jsonl
                                  ├── judgments.jsonl
                                  └── manifest.json
                                  (versioned daily archive)
```

**Framework code is generic.** All user-specific data lives in `user_config.py` (gitignored). Anyone can fork this repo and run it on their own portfolio.

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
# Fill in your account IDs, holdings, rules, watchlist

# 3. Set up Anthropic API key
cp .env.example .env
# edit .env — paste your API key from https://console.anthropic.com/settings/keys

# 4. Verify the framework is working
python regression_test.py     # should print: Passed: 30/30
```

## Daily usage

```sh
# Pre-market or after-close routine:
python refresh.py                    # refresh OHLCV data (~30 sec)
python scanner.py                    # build today's top-30 watchlist (~30 sec)
python watchlist_judge.py --top 10   # multi-agent on top 10 (~$0.25 in API)
python portfolio_judge.py            # multi-agent on your holdings (~$0.70 in API)

# Then view results:
streamlit run dashboard.py           # opens browser at localhost:8501
```

The dashboard has 4 pages:
- **My Portfolio** — holdings, rebalance actions, sector breakdown, Claude scores per position
- **Today's Brief** — names that fired signals today, near-misses setting up, watchlist
- **Chart Browser** — pick any ticker, see candles with EMAs + 9/21 SMA cloud + signal markers
- **Backtest Explorer** — historical signal performance, hit rate per mode

---

## What it doesn't do (deliberately)

| Capability | Why not |
|---|---|
| Place trades automatically | Out of scope. You execute manually on your broker. |
| Day-trade / intraday signals | Designed for swing/position trades on end-of-day data |
| Real-time alerts / push notifications | Could be added (Telegram bot, etc.) — not yet built |
| Tax-loss harvesting automation | Surfaces the opportunity, doesn't auto-execute |
| Options strategies (Time Flies, etc.) | Stock-focused for now |
| News / catalyst hunting via web search | Would add Anthropic web search / Exa / Tavily; not yet built |
| Broker API integration | No retail API for most brokers; positions are updated by editing `user_config.py` |

---

## Key files

| File | Purpose |
|---|---|
| `breakout.py` | Indicators + 4 signal modes (VCP, MOM, EME, PP) + 9/21 SMA cloud + RS rank |
| `scanner.py` | Daily multi-dimensional scanner → top-30 watchlist |
| `multi_agent.py` | 5-agent specialist judgment (Technical / Fundamental / Sentiment / Risk + PM) |
| `judgment.py` | Single-agent Claude scoring (simpler/cheaper alternative) |
| `portfolio.py` | Statement / CSV parser, margin status, position-action rules engine |
| `portfolio_judge.py` | Run multi-agent on all your held positions |
| `watchlist_judge.py` | Run multi-agent on the scanner's top watchlist |
| `dashboard.py` | Streamlit UI (4 pages) |
| `wide_universe.py` | Wikipedia scraper for S&P 500 + Nasdaq 100 with cached fallback |
| `snapshot.py` | Versioned daily archive manager |
| `sector.py` | Sector ETF strength computation |
| `earnings.py` | Next-earnings calendar (confirmed or estimated from last + 90d cycle) |
| `regression_test.py` | 30 pinned winning signals — runs after every change |
| `data_fetch.py` | yfinance OHLCV wrapper + parquet cache |
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

- Multi-agent on portfolio (~30 positions): **~$0.70**
- Multi-agent on watchlist (top 10): **~$0.25**
- Combined daily cost: **~$1/day** running every weekday → **~$22/month**

Cheaper option: switch `ANTHROPIC_MODEL` in `.env` to `claude-haiku-4-5` for ~5× cost reduction with some quality tradeoff.

---

## Privacy

- **Your financial data never leaves your machine.** All holdings + rules + analysis run locally.
- `user_config.py` is gitignored. So is `.env`, `holdings/`, `data/snapshots/`, `data/judgments/`, and all cached OHLCV.
- The public repo contains zero personal financial information — only framework code.
- API requests to Anthropic include the structured payloads (ticker + indicators + position size in $) but not account numbers, names, or anything personally identifying.

---

## Background

Built iteratively over a single session with the help of [Claude Code](https://claude.ai/code). The architecture takes inspiration from:

- [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) — multi-agent investor philosophy idea, simplified here to 4 functional specialists + a Portfolio Manager
- Mark Minervini's Volatility Contraction Pattern + Pocket Pivot mechanics
- Stan Weinstein's Stage 2 / Stage 4 trend framework
- IBD-style cross-sectional Relative Strength ranking
- William O'Neil's CAN SLIM principles (the "leadership" emphasis)

---

## License

MIT. See LICENSE (or treat as MIT until a file is added).

---

## Disclaimer

This software is for educational and personal research use only. It does not constitute financial, investment, tax, or legal advice. **Past signal performance does not predict future results.** Trade at your own risk. The author and contributors accept no liability for any losses incurred from using this software or acting on its recommendations.
