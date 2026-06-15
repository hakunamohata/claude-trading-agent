# Personal Trading Agent — System Overview

A daily-driver agent that scans ~540 US stocks for breakout setups, reasons over them with 5 specialist Claude agents, analyzes the user's household portfolio, and produces specific trade recommendations with sizing and sequencing — all while respecting individualized rules (locked positions, concentration caps, options sequencing, trade-eligible accounts only).

---

## Setup (new user)

```sh
git clone <repo>
cd stocktrading
pip install -r requirements.txt

# Copy the user-config template and fill in your own holdings + account IDs + rules
cp user_config.example.py user_config.py
# edit user_config.py — see comments in the file

# Put your Anthropic API key in .env
cp .env.example .env
# edit .env — paste your API key from console.anthropic.com

# Run regression test to verify framework is working
python regression_test.py
```

Once `user_config.py` and `.env` are filled in, all scripts (`scanner.py`, `portfolio_judge.py`, `dashboard.py`, etc.) read your personal data from `user_config.py`. The framework code itself contains no personal financial data.

**What's user-specific (lives in `user_config.py`, gitignored):**
- Account IDs and labels
- Your holdings (with cost basis)
- Margin status snapshot
- Trade-eligible accounts, locked positions, max-position-pct
- Watchlist + extra portfolio tickers + sector overrides

**What's framework (committed, generic):**
- All scanner, indicator, scoring, multi-agent logic
- S&P 500 + Nasdaq 100 universe
- 4 breakout filter modes, 9/21 SMA cloud, RS rank computation
- Dashboard UI
- Statement / CSV parsers

---

## What it does

**1. Identifies tomorrow's leaders today.**
Scanner runs across S&P 500 + Nasdaq 100 + curated momentum names. Ranks each by a composite of *clean base*, *RS strength*, *trend regime*, *pre-breakout proximity*, and *volume signature*. Outputs a daily top-30 watchlist.

**2. Reasons with 5 Claude agents (Opus 4.7) per name.**
Each agent sees **different** inputs to avoid convergence:
- **Technical** — only chart features, EMAs, signals, RS line
- **Fundamental** — only earnings + sector + RS
- **Sentiment** — only price action + volume signatures
- **Risk** — only portfolio context (concentration, account type)
- **Portfolio Manager** — synthesizes the four reports into final BUY/ADD/HOLD/TRIM/EXIT/AVOID + position sizing guidance

**3. Tracks the user's full household portfolio across 6 Fidelity accounts.**
401k BrokerageLink, Roth IRA, HSA, Individual TOD, Individual margin (MSFT), 529. Parses statements + CSV transactions + live screenshots. Tracks margin debt, daily interest accrual, active covered-call positions, cost basis, P&L.

**4. Generates rebalance recommendations honoring 7 hard rules.**
- MSFT is locked (employer concentration accepted by user)
- 10% max position size, % of total household
- Trade-eligible accounts only = Roth IRA + 401k BrokerageLink
- Close short options BEFORE selling underlying shares (sequence + cost)
- Earnings cadence is predictable (quarterly) — unknown earnings are not a risk
- Existing breakout filter wins (30 pinned regressions) never get lost in edits
- Forward-looking — focus on pre-breakout candidates, not just reactive signals

---

## How to use it daily

| When | Command | What you get |
|---|---|---|
| Morning, pre-market | `python refresh.py` | Latest OHLCV across universe, cached |
| Morning | `python scanner.py` | Top-30 watchlist written to today's snapshot |
| Morning | `python watchlist_judge.py --top 10` | Multi-agent runs on top 10 watchlist (~$0.25 in API calls) |
| Anytime | `python portfolio_judge.py` | Multi-agent on all 27 non-MSFT held positions (~$0.70) |
| In browser | `streamlit run dashboard.py` → localhost:8501 | Visual: Portfolio page, Chart Browser with signals, Backtest Explorer, Today's Brief |
| After any code change | `python regression_test.py` | Asserts 30 known winning signals still fire |

---

## The 4 phases delivered (per Amit's playbook)

| Phase | Module | What it does |
|---|---|---|
| 1 — Scanner | `scanner.py`, `wide_universe.py` | Pre-breakout candidate generation across 537 names |
| 2 — Multi-agent | `multi_agent.py`, `portfolio_judge.py` | 4 specialist agents + PM, structured Pydantic output, prompt caching |
| 3 — Indicators | `breakout.py` | 4 breakout modes (VCP/MOM/EME/PP), 9/21 SMA cloud, RS rank, RS line, sector strength |
| 4 — Snapshot + git | `snapshot.py`, `data/snapshots/<date>/` | Versioned daily archives so "run for a month, then ask agent to improve" actually works |

---

## What this system does NOT do (yet, deferred)

- Place trades (you execute on Fidelity manually)
- Options strategies (Time Flies Spread, MSFT covered-call income to offset margin interest — deferred)
- News / catalyst hunting via web search (dexter-style research agent)
- Tax-loss harvesting automation
- Real-time alerts (push notifications when a signal fires)

---

## Files at a glance

```
data_fetch.py      OHLCV fetch + parquet cache
breakout.py        Indicators + 4 signal modes + EMA cloud + RS rank
sector.py          Sector ETF strength tagging
earnings.py        Next-earnings date (confirmed or estimated last+90d)
wide_universe.py   ~540-name universe (S&P 500 + Nasdaq 100 + curated)
universe.py        Small universe + sector mapping + portfolio holdings list
scanner.py         Daily pre-breakout scanner → top-30 watchlist
judgment.py        Single-agent Claude scoring
multi_agent.py     5-agent specialist judgment (Technical/Fundamental/Sentiment/Risk + PM)
portfolio.py       Statement + CSV parser, margin status, position-action rules
portfolio_judge.py Multi-agent run on all held positions
score_portfolio.py Single-agent run on all held positions
watchlist_judge.py Multi-agent run on scanner watchlist
snapshot.py        Daily versioned archive manager
dashboard.py       Streamlit UI (Portfolio, Today's Brief, Chart Browser, Backtest)
regression_test.py 30 pinned winning signals — runs after every change
refresh.py         Force-refresh OHLCV
scan.py            Lightweight daily scan (small universe)
backtest.py        Historical filter back-test
diagnose_targets.py Per-ticker forward-return diagnostic

data/
  ├── <TICKER>.parquet   cached OHLCV
  ├── universe/          Wikipedia scrapes (S&P 500, Nasdaq 100)
  ├── earnings.parquet   earnings calendar cache
  ├── scans/             legacy daily scan output
  ├── snapshots/<date>/  versioned daily archives
  │   ├── manifest.json
  │   ├── watchlist.parquet
  │   ├── judgments_portfolio.jsonl
  │   └── ...
  └── judgments/         Claude single-agent cache
holdings/                Fidelity statements + screenshots + parsed positions
```

---

## Current state

See `data/snapshots/<today>/` for the latest scan + multi-agent judgments. The dashboard's Portfolio page shows a live view from `user_config.py`.
