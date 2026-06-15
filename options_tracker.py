"""Daily options state tracker — saves snapshots, diffs against yesterday,
flags new strikes / IV moves / theta decay.

Workflow:
  1. Run daily after market close: `python options_tracker.py`
  2. Saves today's options state to data/snapshots/<today>/options_state.parquet
  3. Compares to yesterday's snapshot, prints alerts for:
       - New strikes that opened in the chains we care about
       - IV moves >5 vol points on any held position
       - Theta accrued since yesterday's mark
       - Positions approaching expiry (DTE < 7)
       - Coverage status changes
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from data_fetch import DATA_DIR
from snapshot import snapshot_dir, save_df, load_df, list_snapshots
from options import price_user_options, fetch_options_chain


def save_today_state(date: str | None = None) -> pd.DataFrame:
    """Compute current state of all open option positions, save to snapshot."""
    df = price_user_options()
    if df.empty:
        return df
    df["snapshot_date"] = date or datetime.now().strftime("%Y-%m-%d")
    save_df("options_state", df, date=date)
    return df


def _yesterday_snapshot() -> pd.DataFrame | None:
    """Find the most recent prior snapshot."""
    snaps = list_snapshots()
    today = datetime.now().strftime("%Y-%m-%d")
    prior = [s for s in snaps if s < today]
    if not prior:
        return None
    return load_df("options_state", prior[-1])


def diff_against_yesterday(today_df: pd.DataFrame) -> dict:
    """Compare today's state to yesterday's. Returns alerts dict."""
    yesterday = _yesterday_snapshot()
    alerts: dict[str, list] = {
        "expiring_soon": [],
        "approaching_strike": [],
        "iv_moves": [],
        "theta_collected": [],
        "no_history": [],
    }

    if yesterday is None:
        alerts["no_history"].append("First snapshot — no yesterday data to compare.")

    for _, row in today_df.iterrows():
        key = (row["ticker"], row["strike"], row["expiry"], row["type"])

        # Expiring soon
        if row["dte"] <= 7:
            alerts["expiring_soon"].append({
                "position": f"{row['ticker']} {row['strike']:.0f}{row['type'][0]} {row['expiry']}",
                "dte": row["dte"],
                "moneyness_pct": row["moneyness_pct"],
                "current_mid": row["current_mid"],
                "pnl": row["total_pnl_usd"],
            })

        # Approaching strike — within 5% of being ITM
        m = row["moneyness_pct"]
        if -5 <= m <= 5 and row["side"] == "SHORT":
            alerts["approaching_strike"].append({
                "position": f"{row['ticker']} {row['strike']:.0f}{row['type'][0]}",
                "spot": row["spot"],
                "strike": row["strike"],
                "moneyness_pct": m,
                "delta": row["delta"],
                "coverage": row["coverage"],
            })

        # IV moves and theta vs yesterday
        if yesterday is not None:
            prior = yesterday[
                (yesterday["ticker"] == row["ticker"])
                & (yesterday["strike"] == row["strike"])
                & (yesterday["expiry"] == row["expiry"])
            ]
            if not prior.empty:
                p = prior.iloc[0]
                iv_change = row["iv"] - p["iv"]
                if abs(iv_change) >= 5:
                    alerts["iv_moves"].append({
                        "position": f"{row['ticker']} {row['strike']:.0f}{row['type'][0]}",
                        "iv_yesterday": p["iv"],
                        "iv_today": row["iv"],
                        "change_pts": round(iv_change, 1),
                    })
                # Theta collected = change in P&L attributable to time decay
                pnl_change = row["total_pnl_usd"] - p["total_pnl_usd"]
                alerts["theta_collected"].append({
                    "position": f"{row['ticker']} {row['strike']:.0f}{row['type'][0]}",
                    "pnl_change_today_usd": int(pnl_change),
                    "expected_theta_per_day_usd": int(-row["theta_per_day"])
                        if row["side"] == "SHORT" else int(row["theta_per_day"]),
                })

    return alerts


def format_alerts_text(alerts: dict, today_df: pd.DataFrame) -> str:
    """Human-readable summary of state + alerts."""
    lines = []
    total_pnl = int(today_df["total_pnl_usd"].sum()) if not today_df.empty else 0
    daily_theta = int(-today_df["theta_per_day"].sum()) if not today_df.empty else 0
    lines.append(f"OPTIONS BOOK — {len(today_df)} positions, total open P&L ${total_pnl:,}")
    lines.append(f"Theta collected daily: ${daily_theta:,}/day  (from selling premium)")
    lines.append("")

    if alerts["expiring_soon"]:
        lines.append("EXPIRING SOON (<=7 days):")
        for a in alerts["expiring_soon"]:
            lines.append(f"  - {a['position']}  DTE={a['dte']}  moneyness={a['moneyness_pct']:+.1f}%  mark=${a['current_mid']:.2f}  P&L=${a['pnl']:,.0f}")
        lines.append("")

    if alerts["approaching_strike"]:
        lines.append("APPROACHING STRIKE (within 5%):")
        for a in alerts["approaching_strike"]:
            lines.append(f"  - {a['position']}  spot ${a['spot']:.2f} vs strike ${a['strike']:.0f}  delta {a['delta']:.2f}  {a['coverage']}")
        lines.append("")

    if alerts["iv_moves"]:
        lines.append("BIG IV MOVES (>=5 pts vs yesterday):")
        for a in alerts["iv_moves"]:
            arrow = "up" if a["change_pts"] > 0 else "down"
            lines.append(f"  - {a['position']}  IV {a['iv_yesterday']:.0f} -> {a['iv_today']:.0f}  ({arrow} {abs(a['change_pts'])} pts)")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Saving today's options state...")
    today_df = save_today_state()
    if today_df.empty:
        print("  (No options positions to track)")
    else:
        print(f"  Saved {len(today_df)} positions to snapshot")
        print()
        alerts = diff_against_yesterday(today_df)
        print(format_alerts_text(alerts, today_df))
