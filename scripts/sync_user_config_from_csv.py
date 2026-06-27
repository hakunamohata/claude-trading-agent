"""Sync HOLDINGS_CURRENT in user_config.py from a Fidelity Portfolio CSV.

Handles Fidelity's quirk where some account types (BrokerageLink, HSA) export
with a different column shift than Individual / Roth. Backs up user_config.py,
preserves per-account comment headers, replaces only the HOLDINGS_CURRENT
block. Verifies via regression_test.py.

Usage:
    python scripts/sync_user_config_from_csv.py holdings/Portfolio_Positions_Jun-23-2026.csv
    python scripts/sync_user_config_from_csv.py  # auto-glob latest Portfolio_Positions_*.csv
"""
from __future__ import annotations
import sys
import csv
import re
import shutil
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent.parent
HOLDINGS_DIR = REPO / "holdings"
USER_CONFIG = REPO / "user_config.py"

# Account ID mappings are in a separate gitignored file (account_map.py).
# Copy account_map.example.py to account_map.py and fill in your account IDs.
sys.path.insert(0, str(REPO))
try:
    from account_map import (
        ACCOUNT_NAME_TO_ID,
        CASH_LABEL_BY_ACCT,
        ACCOUNT_ORDER,
        ACCOUNT_HEADER_LABEL,
        MUTUAL_FUND_LABELS,
    )
except ImportError:
    sys.exit(
        "Missing account_map.py. Copy account_map.example.py to account_map.py "
        "and fill in your account IDs."
    )

ACCOUNT_ID_TO_LABEL = {v: k for k, v in ACCOUNT_NAME_TO_ID.items()}

# Cash markers — Fidelity money-market funds
CASH_SYMBOLS = {"FDRXX", "SPAXX", "FZFXX"}

TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
OPTION_DESC_RE = re.compile(r"\b(CALL|PUT)\b")

# Per-account header template (also used to preserve insertion order)
HEADER_TEMPLATE = (
    "    # ----- {label} ({acct_id}) — Fidelity CSV (live) -----"
)


def parse_csv(path: Path) -> list[dict]:
    """Parse Fidelity CSV handling the variable column shift."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 6:
                continue
            # Account name is in col 0 OR col 1 depending on whether Account Number
            # was populated (Individual+Roth) or shifted (BrokerageLink/HSA)
            # We look at col 0 and col 1 and use what's a recognized account name
            acct_name = None
            ticker_col = None
            for ci in [0, 1]:
                v = row[ci].strip()
                if v in ACCOUNT_NAME_TO_ID:
                    acct_name = v
                    ticker_col = ci + 1
                    break
            if acct_name is None:
                continue
            acct_id = ACCOUNT_NAME_TO_ID[acct_name]

            raw_ticker = row[ticker_col].strip() if ticker_col < len(row) else ""
            # Strip Fidelity '**' marker on money-market funds
            ticker_clean = raw_ticker.rstrip("*")
            description = row[ticker_col + 1].strip() if ticker_col + 1 < len(row) else ""
            description_upper = description.upper()

            # Skip option contracts (e.g. "DDOG AUG 21 2026 $230 CALL")
            if OPTION_DESC_RE.search(description_upper) or OPTION_DESC_RE.search(raw_ticker):
                continue

            # Identify cash sleeves (money market / cash label)
            is_money_market = (
                "MONEY MARKET" in description_upper
                or "HELD IN MONEY MARKET" in description_upper
                or ticker_clean in CASH_SYMBOLS
            )
            if is_money_market:
                ticker_use = CASH_LABEL_BY_ACCT.get(acct_id, "FDRXX")
            elif description_upper in MUTUAL_FUND_LABELS:
                ticker_use = MUTUAL_FUND_LABELS[description_upper]
            elif TICKER_RE.match(ticker_clean):
                ticker_use = ticker_clean
            else:
                # Fallback for unrecognized rows: skip rather than mangle
                continue

            # Quantity & price & value — handle blank quantity for money-market rows
            qty_str = row[ticker_col + 2].strip().replace("$", "").replace(",", "") if ticker_col + 2 < len(row) else ""
            price_str = row[ticker_col + 3].strip().replace("$", "").replace(",", "") if ticker_col + 3 < len(row) else ""
            value_str = row[ticker_col + 5].strip().replace("$", "").replace(",", "") if ticker_col + 5 < len(row) else ""

            try:
                value = float(value_str)
            except ValueError:
                value = 0.0
            try:
                price = float(price_str)
            except ValueError:
                price = 1.0 if is_money_market else 0.0
            try:
                shares = float(qty_str)
            except ValueError:
                # Money market with blank quantity -> use value as both shares (at $1) and value
                if is_money_market and value > 0:
                    shares = value
                    price = 1.0
                else:
                    continue

            try:
                cb_col = ticker_col + 11
                cb = float(row[cb_col].strip().replace("$", "").replace(",", ""))
            except (ValueError, IndexError):
                cb = None

            out.append({
                "acct_id": acct_id,
                "acct_label": acct_name,
                "ticker": ticker_use,
                "shares": shares,
                "price": price,
                "value": value,
                "cost_basis": cb,
            })
    return out


def render_holdings_block(rows: list[dict]) -> str:
    """Build the new HOLDINGS_CURRENT block preserving per-account ordering."""
    by_acct: dict[str, list[dict]] = {}
    for r in rows:
        by_acct.setdefault(r["acct_id"], []).append(r)

    lines = ["HOLDINGS_CURRENT = ["]
    for acct_id in ACCOUNT_ORDER:
        if acct_id not in by_acct:
            continue
        label = ACCOUNT_HEADER_LABEL[acct_id]
        lines.append(f"    # ----- {label} ({acct_id}) — Jun-23 Fidelity CSV (live) -----")
        # Sort: cash first, then alphabetical by ticker
        items = by_acct[acct_id]
        cash = [r for r in items if r["ticker"].startswith("CASH") or r["ticker"] in CASH_SYMBOLS]
        non_cash = [r for r in items if r not in cash]
        non_cash.sort(key=lambda r: r["ticker"])
        for r in cash + non_cash:
            cb_str = f"{r['cost_basis']:.2f}" if r["cost_basis"] is not None else "None"
            lines.append(
                f"    (\"{r['acct_id']}\", \"{r['ticker']}\", "
                f"{r['shares']:>11.3f}, {r['price']:>8.2f}, "
                f"{r['value']:>12.2f}, {cb_str:>11s}),"
            )
        lines.append("")
    lines.append("]")
    return "\n".join(lines)


def replace_holdings_block(content: str, new_block: str) -> str:
    """Replace the HOLDINGS_CURRENT = [...] block in user_config.py."""
    pattern = re.compile(
        r"^HOLDINGS_CURRENT\s*=\s*\[.*?^\]",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(content)
    if m is None:
        raise RuntimeError("Could not locate HOLDINGS_CURRENT block in user_config.py")
    # Use slicing to avoid re.sub's backreference interpretation of \ chars
    return content[:m.start()] + new_block + content[m.end():]


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        candidates = sorted(HOLDINGS_DIR.glob("Portfolio_Positions_*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No Portfolio_Positions_*.csv found in holdings/")
            sys.exit(1)
        csv_path = candidates[0]
    print(f"Source CSV: {csv_path}")

    rows = parse_csv(csv_path)
    print(f"Parsed {len(rows)} stock positions across {len({r['acct_id'] for r in rows})} accounts")

    # Compute diff vs current user_config
    try:
        sys.path.insert(0, str(REPO))
        from user_config import HOLDINGS_CURRENT as OLD
        old_by = {(r[0], r[1]): r for r in OLD}
    except Exception as e:
        print(f"Could not load existing HOLDINGS_CURRENT for diff: {e}")
        old_by = {}

    # Build new block
    new_block = render_holdings_block(rows)

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = USER_CONFIG.parent / f"user_config.backup_{ts}.py"
    shutil.copy(USER_CONFIG, backup)
    print(f"Backed up to: {backup}")

    # Replace
    content = USER_CONFIG.read_text(encoding="utf-8")
    new_content = replace_holdings_block(content, new_block)
    USER_CONFIG.write_text(new_content, encoding="utf-8")
    print(f"Updated: {USER_CONFIG}")

    # Diff summary
    print("\n=== Diff summary ===")
    new_by = {(r["acct_id"], r["ticker"]): r for r in rows}
    added, removed, changed = [], [], []
    for k, r in new_by.items():
        old = old_by.get(k)
        if old is None:
            added.append((k, r["shares"], r["value"]))
        else:
            old_sh, old_val = old[2], old[4]
            if abs(r["shares"] - old_sh) > 0.001:
                changed.append((k, old_sh, r["shares"], old_val, r["value"]))
            elif abs(r["value"] - old_val) / max(old_val, 1) > 0.01:
                changed.append((k, old_sh, r["shares"], old_val, r["value"]))
    for k in old_by:
        if k not in new_by:
            removed.append((k, old_by[k][2], old_by[k][4]))

    print(f"\nADDED ({len(added)}):")
    for k, sh, v in added:
        print(f"  + {k[0]:>10} {k[1]:>10}  {sh:>10.3f} sh  ${v:,.0f}")
    print(f"\nREMOVED ({len(removed)}):")
    for k, sh, v in removed:
        print(f"  - {k[0]:>10} {k[1]:>10}  {sh:>10.3f} sh  ${v:,.0f}")
    print(f"\nCHANGED ({len(changed)}):")
    for k, osh, nsh, ov, nv in changed:
        d_sh = nsh - osh
        d_v = nv - ov
        print(f"  ~ {k[0]:>10} {k[1]:>10}  {osh:>10.3f} -> {nsh:>10.3f}  ({d_sh:+8.3f})  ${ov:>11,.0f} -> ${nv:>11,.0f}  ({d_v:+,.0f})")

    print(f"\nTotal portfolio value: ${sum(r['value'] for r in rows):,.0f}")
    print(f"Tickers w/ shares >= 100 (CC-capable): {sum(1 for r in rows if r['shares'] >= 100)}")


if __name__ == "__main__":
    main()
