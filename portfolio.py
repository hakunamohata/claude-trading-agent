"""Statement / CSV / screenshot parser + rules engine for portfolio analysis.

GENERIC FRAMEWORK CODE — no user-specific data here. All holdings, account IDs,
rules, and margin info live in `user_config.py` (gitignored). To set up for a
new user, copy `user_config.example.py` to `user_config.py` and fill it in.

Outputs:
  - holdings/positions_may31.parquet  — past-snapshot holdings (pre-reconciliation)
  - holdings/june_activity.parquet    — transactions parsed from CSV
  - holdings/positions_current.parquet — reconciled current positions
"""

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import pdfplumber

HOLDINGS_DIR = Path(__file__).parent / "holdings"


# ============================================================
# Load user-specific data
# ============================================================

try:
    import user_config
except ImportError as e:
    raise ImportError(
        "user_config.py not found. Copy user_config.example.py to user_config.py "
        "and fill in your account IDs, holdings, and rules."
    ) from e

# Re-exported user-config values for backward compatibility with existing imports
INDIVIDUAL_MARGIN_SNAPSHOT = user_config.INDIVIDUAL_MARGIN_SNAPSHOT
TRADE_ELIGIBLE_ACCOUNTS = user_config.TRADE_ELIGIBLE_ACCOUNTS
TAXABLE_ACCOUNTS = user_config.TAXABLE_ACCOUNTS
LOCKED_POSITIONS = user_config.LOCKED_POSITIONS
MAX_POSITION_PCT = user_config.MAX_POSITION_PCT
HOLDINGS_CURRENT = user_config.HOLDINGS_CURRENT
HOLDINGS_MAY31 = user_config.HOLDINGS_MAY31
HOLDINGS_HARDCODED = HOLDINGS_CURRENT + HOLDINGS_MAY31

# Derive label / tax-status lookups from ACCOUNT_INFO
ACCOUNT_LABEL = {k: v["label"] for k, v in user_config.ACCOUNT_INFO.items()}
ACCOUNT_TAX_STATUS = {k: v["tax_status"] for k, v in user_config.ACCOUNT_INFO.items()}


def margin_annual_cost() -> float:
    """Annualized margin interest at current rate ($/year). Zero if no margin account."""
    if not INDIVIDUAL_MARGIN_SNAPSHOT:
        return 0.0
    s = INDIVIDUAL_MARGIN_SNAPSHOT
    return abs(s["net_debit"]) * s["margin_interest_rate_pct"] / 100


def margin_summary() -> str:
    """Human-readable summary of margin status. Empty string if no margin account."""
    if not INDIVIDUAL_MARGIN_SNAPSHOT:
        return "No margin account configured."
    s = INDIVIDUAL_MARGIN_SNAPSHOT
    return (
        f"Margin account ({s['account_id']}) — net equity ${s['account_equity']:,.0f}\n"
        f"  Stocks (margin):     ${s['margin_market_value']:,.0f}\n"
        f"  Cash:                ${s['cash_market_value']:,.0f}\n"
        f"  Short options:       ${s['option_market_value']:,.0f}\n"
        f"  Margin debt:         ${s['net_debit']:,.0f}\n"
        f"  Equity ratio:        {s['equity_pct']:.1f}%\n"
        f"  Margin interest:     {s['margin_interest_rate_pct']:.2f}% / year = "
        f"~${margin_annual_cost():,.0f}/year\n"
        f"  Daily accrual:       ${s['margin_interest_accrued_daily']:.2f}/day"
    )


def is_trade_eligible(account_id: str, ticker: str) -> bool:
    """Returns True if the rebalance engine may recommend trades on this (account, ticker)."""
    if ticker in LOCKED_POSITIONS:
        return False
    return account_id in TRADE_ELIGIBLE_ACCOUNTS


def position_action(ticker: str, account_id: str, value: float,
                    total_portfolio_value: float, claude_score: int | None = None) -> str:
    """Suggest TRIM / HOLD / WATCH / BUY based on rules.

    Action vocabulary:
      LOCKED  — MSFT, do not touch
      NOT-ACTIONABLE — outside trade-eligible accounts (HSA, 529, Individual etc.)
      TRIM    — over 10% concentration cap
      AVOID   — Claude score < 30
      BUY     — Claude score >= 80, position under 5% of total
      HOLD    — everything else
    """
    if ticker in LOCKED_POSITIONS:
        return "LOCKED"
    if account_id not in TRADE_ELIGIBLE_ACCOUNTS:
        return "NOT-ACTIONABLE"
    pct = (value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
    if pct > MAX_POSITION_PCT:
        return "TRIM"
    if claude_score is not None:
        if claude_score < 30:
            return "AVOID"
        if claude_score >= 80 and pct < 5.0:
            return "BUY"
    return "HOLD"


def _ticker_safe(line: str, ticker: str) -> bool:
    """Filter out obvious false positive matches (option contracts, footnotes)."""
    if "CALL" in line or "PUT" in line:
        return False
    if "You Bought" in line or "You Sold" in line:
        return False
    return True


def parse_statement_2_holdings(pdf_path: Path) -> pd.DataFrame:
    """Extract per-account holdings from the multi-account statement.

    Returns DataFrame: account_id, ticker, quantity, price, value, cost_basis
    """
    # Pattern matches: "<NAME> (<TICKER>) <begin_val> <qty> <price> <end_val> <cost> <gain>"
    # Numbers may have commas + 2-4 decimal places. Skip negative-quantity (short/options) rows.
    holding_pat = re.compile(
        r'\(([A-Z]{1,5})\)\s+'                         # (TICKER)
        r'([\d,]+\.\d{2})\s+'                          # begin value
        r'([\d,]+\.\d{1,3})\s+'                        # quantity (positive only)
        r'([\d,]+\.\d{2,4})\s+'                        # price per unit
        r'([\d,]+\.\d{2})\s+'                          # ending value
        r'([\d,]+\.\d{2})'                             # cost
    )

    rows = []
    current_acct = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Track account context
            m_acc = re.search(r'Account\s*#\s*([\w-]{8,})', text)
            if m_acc:
                current_acct = m_acc.group(1).replace("-", "")

            for line in text.split("\n"):
                m = holding_pat.search(line)
                if not m or not current_acct:
                    continue
                ticker = m.group(1)
                if len(ticker) <= 1:
                    continue
                if not _ticker_safe(line, ticker):
                    continue
                # Skip obvious junk tickers from headers (e.g. ASE, SEC indicators)
                if ticker in ("US", "NYSE", "EAI", "EY", "ETF", "ADS", "ORD", "COM", "SHS", "CL", "NPV"):
                    continue

                def _num(s): return float(s.replace(",", ""))
                rows.append({
                    "account_id": current_acct,
                    "account_label": ACCOUNT_LABEL.get(current_acct, current_acct),
                    "tax_status": ACCOUNT_TAX_STATUS.get(current_acct, "unknown"),
                    "ticker": ticker,
                    "quantity": _num(m.group(3)),
                    "price_5_31": _num(m.group(4)),
                    "value": _num(m.group(5)),
                    "cost_basis": _num(m.group(6)),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # De-dupe (some rows might be repeated across pages)
    df = df.drop_duplicates(subset=["account_id", "ticker"], keep="first").reset_index(drop=True)
    return df


def build_hardcoded_holdings() -> pd.DataFrame:
    rows = []
    for acct, ticker, qty, price, value, cost in HOLDINGS_HARDCODED:
        rows.append({
            "account_id": acct,
            "account_label": ACCOUNT_LABEL.get(acct, acct),
            "tax_status": ACCOUNT_TAX_STATUS.get(acct, "unknown"),
            "ticker": ticker,
            "quantity": qty,
            "price_5_31": price,
            "value": value,
            "cost_basis": cost,
        })
    return pd.DataFrame(rows)


def parse_june_activity(csv_path: Path) -> pd.DataFrame:
    """Parse the 30-day history CSV, extract only stock buy/sell transactions in June."""
    # CSV has variable column counts (footnote rows have different shape).
    # Use Python engine and skip bad lines to be forgiving.
    df = pd.read_csv(csv_path, skiprows=2, engine="python", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    df["Run Date"] = pd.to_datetime(df["Run Date"], errors="coerce")
    # June 2026 only
    june_only = df[df["Run Date"].dt.month == 6].copy()
    # Stock buy/sell only (skip options, cash, dividends)
    action = june_only["Action"].fillna("")
    is_stock_trade = (
        action.str.contains("YOU BOUGHT", case=False, na=False)
        | action.str.contains("YOU SOLD", case=False, na=False)
        | action.str.contains("CONVERSION SHARES DEPOSITED", case=False, na=False)
    )
    is_option = action.str.contains("CALL|PUT", regex=True, na=False)
    # Stock trades only — exclude options
    june_stock = june_only[is_stock_trade & ~is_option].copy()

    # Normalize columns
    out = june_stock[["Run Date", "Account", "Account Number", "Symbol",
                      "Description", "Price", "Quantity", "Amount"]].copy()
    out = out.rename(columns={
        "Run Date": "date",
        "Account": "account_name",
        "Account Number": "account_id",
        "Symbol": "ticker",
        "Description": "description",
        "Price": "price",
        "Quantity": "quantity",
        "Amount": "amount",
    })
    out["account_id"] = out["account_id"].astype(str).str.strip()
    out["ticker"] = out["ticker"].fillna("").str.strip()
    return out.reset_index(drop=True)


def reconcile(positions_5_31: pd.DataFrame, june_activity: pd.DataFrame,
              skip_accounts: set[str] | None = None) -> pd.DataFrame:
    """Apply June stock transactions to May 31 positions to get current holdings.

    `skip_accounts` is a set of account_ids whose positions are ALREADY at today's
    snapshot — June activity will not be applied to them.
    """
    skip = skip_accounts or set()
    current = positions_5_31.copy()

    if not june_activity.empty:
        # Drop June activity for accounts already at today's snapshot
        june_filtered = june_activity[~june_activity["account_id"].isin(skip)]
        if june_filtered.empty:
            return current

        deltas = (
            june_filtered.dropna(subset=["ticker"])
            .query("ticker != ''")
            .groupby(["account_id", "ticker"])
            .agg(qty_change=("quantity", "sum"), notional=("amount", "sum"))
            .reset_index()
        )

        merged = current.merge(deltas, on=["account_id", "ticker"], how="outer")
        merged["quantity"] = merged["quantity"].fillna(0) + merged["qty_change"].fillna(0)
        merged.loc[merged["account_label"].isna(), "account_label"] = merged["account_id"].map(ACCOUNT_LABEL)
        merged.loc[merged["tax_status"].isna(), "tax_status"] = merged["account_id"].map(ACCOUNT_TAX_STATUS)
        current = merged[merged["quantity"].abs() > 0.001].reset_index(drop=True)
        current = current.drop(columns=["qty_change", "notional"], errors="ignore")

    return current


def accounts_at_current() -> set[str]:
    """Returns the set of account_ids whose HOLDINGS_CURRENT rows are at today's snapshot."""
    return {row[0] for row in HOLDINGS_CURRENT}


def all_unique_tickers(positions: pd.DataFrame) -> list[str]:
    """Tickers from positions, minus cash/money-market funds, for universe expansion."""
    NON_EQUITY = {"FDRXX", "SPAXX", "NHFSMKX98", "CASH_ROTH", "CASH_TOD", "CASH_HSA"}
    return sorted(set(positions["ticker"].dropna()) - NON_EQUITY)


if __name__ == "__main__":
    print("Loading hand-mapped holdings from Statements 1 and 2...")
    h_hard = build_hardcoded_holdings()
    print(f"  Hand-mapped: {len(h_hard)} positions, total value ${h_hard['value'].sum():,.0f}")

    print("\nAttempting auto-parse of multi-account statement (best-effort)...")
    h_auto = parse_statement_2_holdings(HOLDINGS_DIR / "Statement5312026 (1).pdf")
    print(f"  Auto-parsed: {len(h_auto)} positions across {h_auto['account_id'].nunique() if not h_auto.empty else 0} accounts")

    # Merge — hardcoded wins on conflicts (better quality)
    if not h_auto.empty:
        keys_hard = set(zip(h_hard["account_id"], h_hard["ticker"]))
        h_auto_new = h_auto[~h_auto.apply(lambda r: (r["account_id"], r["ticker"]) in keys_hard, axis=1)]
        positions_5_31 = pd.concat([h_hard, h_auto_new], ignore_index=True)
        print(f"  Auto-parser added {len(h_auto_new)} positions not in hand-mapped set")
    else:
        positions_5_31 = h_hard
    positions_5_31.to_parquet(HOLDINGS_DIR / "positions_may31.parquet")
    print(f"\nMay 31 combined: {len(positions_5_31)} positions, ${positions_5_31['value'].sum():,.0f}")

    print("\nParsing June activity from CSV...")
    june = parse_june_activity(HOLDINGS_DIR / "Accounts_History.csv")
    june.to_parquet(HOLDINGS_DIR / "june_activity.parquet")
    print(f"  June stock trades: {len(june)} rows")

    print("\nReconciling May 31 holdings + June trades -> current holdings...")
    print(f"  (Skipping accounts already at today's snapshot: {accounts_at_current()})")
    current = reconcile(positions_5_31, june, skip_accounts=accounts_at_current())
    current.to_parquet(HOLDINGS_DIR / "positions_current.parquet")
    print(f"  Current positions: {len(current)} rows")

    print("\nPer-account summary:")
    summary = current.groupby(["account_id", "account_label", "tax_status"]).agg(
        positions=("ticker", "nunique"),
        value=("value", "sum"),
    ).reset_index()
    print(summary.to_string(index=False))

    print(f"\nUnique tickers across portfolio: {len(all_unique_tickers(current))}")
    print(f"Tickers: {', '.join(all_unique_tickers(current))}")

    print("\n" + "=" * 60)
    print(margin_summary())
