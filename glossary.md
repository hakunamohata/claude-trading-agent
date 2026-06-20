# Glossary

Reference for the terms used across the covered-call income engine, buy-write screener, scanner, and judgment system. Sorted by category.

---

## Options — the basics

**Delta (Δ)**
The option's price sensitivity to a $1 move in the underlying stock. A call with delta 0.30 gains $0.30 in value for every $1 the underlying rises. For covered-call sellers, two practical interpretations:
1. **Model-implied assignment probability**. Delta 0.30 ≈ 30% chance the option finishes in-the-money at expiry (shares get called away).
2. **Buyback sensitivity**. If the underlying rallies, delta tells you how much the option's price climbs against you per dollar of underlying move.

**Probability of profit**
The chance you keep the full premium when selling a call. Approximately `1 − delta`. A delta-0.30 short call has ~70% probability of profit.
- **BS POP** (Black-Scholes Probability of profit) — the textbook value, assumes the stock follows a random walk. Ignores trends, momentum, recent earnings moves.
- **Adjusted POP** — what the engine actually trades on. Starts from BS POP, then applies a 10-signal overlay covering: extension above 50-day moving average, recent return, accumulation/distribution grade, whether expiry spans earnings, growth since last earnings, prior-year same-quarter earnings move. **Always trust adjusted POP over BS POP.**

**Days to expiration**
Calendar days remaining until the contract expires. The wheel strategy targets **25-50 days** at entry — enough time-decay (theta) acceleration without locking into one strike too long.

**Strike**
The price at which the call buyer can exercise. Above the current spot for out-of-the-money calls.

**Premium**
The dollar amount you collect per share when selling the contract. Multiplied by 100 for total per contract.

**Open interest**
The total number of these exact contracts (same strike + same expiry) currently outstanding across the market. Higher means more market participants → tighter bid/ask spreads → easier fills. **Under 100 = caution; you may get filled below mid.**

**Today's option volume**
How many of THIS contract have traded today. Confirms there's an active market right now, not just stale resting orders. **>100 = active.**

**Implied volatility**
The market's forward-looking, annualized forecast for how much the underlying will move. Higher implied volatility = bigger premium for the seller (good), but also implies the market expects bigger underlying moves (bad — higher assignment risk). MSFT's typical implied volatility runs 25-30%; readings above 35% are elevated.

**Theta**
Daily time-decay — the dollar amount the option's price falls per calendar day if everything else stays still. Positive for the seller. Reported as "theta per day."

**Mark-to-Market (MTM)**
The current market value of an open position, used to compute "if I closed this right now, what's my unrealized profit or loss?" For a short call: `MTM = premium received − current mid-price to buy back`. Positive MTM means the option got cheaper since you sold (good for seller). Negative MTM means it got more expensive (bad). MTM is **unrealized** — it's a snapshot that only becomes "real" when the position closes.

**Realized vs. Unrealized P&L**
- **Realized** = locked in. The cash hit your account when you closed the trade (or it expired). Cannot move.
- **Unrealized** = on paper. The current MTM of any open position. Moves every day until you close or it expires.
A rolled trade has BOTH: realized P&L from the prior legs that were closed during each roll, plus unrealized MTM on the current still-open leg.

---

## Risk verdicts

The engine assigns each candidate a verdict based on adjusted probability of profit + underlying technical conditions.

| Verdict | Meaning | Dashboard color |
|---|---|---|
| **🟢 SAFE** | Adjusted probability of profit ≥ 65%, GREEN-rated. Underlying technicals are favorable (downtrend, distribution, or just neutral). Trade with confidence. | Green |
| **🟡 MODERATE** | Adjusted probability of profit 55-65%, YELLOW-rated. Some technical caution flags but still tradeable. Monitor. | Yellow |
| **🟠 AGGRESSIVE** | Negative technical adjustment (extended, momentum, accumulation continuing). Higher yield possible but expect to manage assignment. | Peach |
| **📅 EARNINGS-RISK** | Expiry spans the next earnings date. Binary event risk — gap + implied-volatility crush. Locked underlyings (MSFT) should usually pick a pre-earnings expiry instead. | Orange |
| **🔴 DANGEROUS** | Strongly negative technicals — extended >25% above 50-day average, recent momentum, A-grade accumulation. Black-Scholes massively understates real assignment risk. Consider skipping. | Red |

---

## Scanner & breakout terminology

**Volatility contraction pattern (VCP)**
A Mark Minervini setup: the stock makes higher highs while range/volume tighten. 7 conditions are tracked; full strict signal fires when all 7 are met simultaneously.

**Pocket pivot**
Today's volume exceeds the highest down-day volume of the prior 10 days, with price closing in the upper part of the day's range. Inside a base = institutional accumulation signal.

**Momentum mode**
Continuation pattern — stock already in an uptrend, today's bar confirms continuation. Less rigorous than VCP but useful for established trends.

**Emergence mode**
Stage 2 entry — price reclaims the 200-day average for the first time, with a golden cross (50 EMA crossing above 200 EMA) recently confirmed.

**Relative strength rank (1-99)**
IBD-style cross-sectional rank — composite return weighted 40% × 1-month + 20% each of 3/6/12-month, then ranked across all equity peers in the universe. **>=80** = top decile = institutional sponsorship signature.

**Accumulation/Distribution grade**
13-week, dollar-volume-weighted ratio of up-day volume vs. down-day volume. Quantifies whether institutions are net buyers or net sellers of the name.

| Grade | Reading |
|---|---|
| A | Heavy accumulation — call sellers should expect continued upside follow-through |
| B | Mild accumulation |
| C | Neutral / mixed |
| D | Mild distribution — limited upside follow-through likely |
| E | Heavy distribution — call sellers benefit; topping pattern possible |

**Stage 2 confirmation**
Weinstein-style: price > 50-day average > 200-day average, both averages rising. The "in-trend" regime where momentum strategies work.

**ATR (Average True Range)**
A volatility measure. Low ATR % = tight base. Used as a VCP condition and as a normalizer for volatility comparisons across names.

---

## LB judgment system

**LB (Laxmi Bank)**
The chief synthesizer agent in the multi-agent pipeline. Takes specialist reports from Technical, Fundamental, Sentiment, Risk (and optionally Minervini, Druckenmiller, Burry) agents and produces a final action recommendation.

| LB Action | Meaning |
|---|---|
| **BUY** | New position recommended; >=80 conviction, not currently held |
| **ADD** | Increase existing position; <5% of portfolio; >=80 conviction |
| **HOLD** | Keep as-is, no change |
| **TRIM** | Reduce position size — too concentrated or weakening setup |
| **EXIT** | Close the position entirely — broken trend or persistent weakness |
| **AVOID** | Wouldn't initiate or add at current price |

**Final score (0-100)**
LB's overall conviction. 80+ = strong action, 50-70 = nuanced view, <50 = avoid/exit territory.

**Confidence (1-10)**
How sure LB is of the call given specialist agreement. 8+ = specialists agree, 4 or below = mixed signals.

**🔒 (locked underlying)**
The position is marked LOCKED in user_config.py — never sell shares. For MSFT, all CC activity must respect this — roll out and up if delta drifts above 0.40 with <14 days to expiry; don't let it assign.

---

## TRIM-as-CC / EXIT-as-CC

When LB rates a position TRIM or EXIT, the income engine offers a covered-call alternative to a spot sell. The math compares:
- **Spot sell value** = shares × current price (cash today, position closed)
- **CC expected value** = (probability of profit × (spot value + premium)) + ((1 − probability of profit) × (strike + premium) × shares)
- **Edge** = CC expected value − spot sell value (positive means CC beats the spot sell on average)

Use the CC route whenever the edge is positive. If not assigned, the premium effectively averages down your basis. If assigned, you exit at strike + premium (typically higher than spot).

---

## Risk overlay adjustments

The 10 signals the engine combines to convert Black-Scholes POP into adjusted POP. Positive = safer for the call seller, negative = more dangerous.

| Signal | Adjustment |
|---|---|
| Stock <0% vs 50 EMA (downtrend) | **+5** |
| Stock >25% above 50 EMA (extended) | **−15** |
| Stock >10% above 50 EMA | −5 |
| 5-day return >+10% (momentum) | −10 |
| 5-day return <−3% (weakness) | +5 |
| Accumulation/Distribution grade A or B | −5 |
| Accumulation/Distribution grade D or E | +5 |
| Expiry spans next earnings | −10 |
| Stock up >25% since last earnings AND spans earnings | −5 |
| Stock down >15% since last earnings AND spans earnings | +5 |
| Prior-year same-quarter earnings move ≥8% AND spans earnings | −5 |

Adjusted POP = clip(BS POP + total adjustment, 0, 100).

---

## Other terms

**Capacity (covered calls)**
The maximum number of contracts you can write against a holding. `floor(shares / 100)`. The engine applies a **50% hard cap** on top of that to leave roll headroom.

**Wheel strategy**
Sell out-of-the-money calls each cycle (25-50 days to expiry), let them expire worthless or buy them back near zero, then sell new calls. The wheel turns continuously and produces predictable annualized income on shares you hold.

**Buy-write**
Buy 100 shares + simultaneously sell one covered call at a strike modestly above the entry price. The premium gives you downside protection equal to the premium captured.
