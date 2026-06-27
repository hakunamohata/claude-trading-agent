"""Template for `account_map.py` (gitignored).

Copy this to `account_map.py` and replace the placeholder IDs with your
real Fidelity account numbers. The sync script and any tool that maps
broker statement strings to internal account IDs read from `account_map.py`.

The keys (left-hand side) in ACCOUNT_NAME_TO_ID must match the exact
strings Fidelity writes in the "Account Name" column of the Portfolio
Positions CSV (case-sensitive, including spaces around hyphens).
"""

# Map Fidelity CSV "Account Name" string -> internal account_id of your choice.
# Internal IDs can be anything stable — Fidelity account numbers are convenient
# but a label like "ROTH" works too.
ACCOUNT_NAME_TO_ID = {
    "INDIVIDUAL":              "INDIV_MARGIN",
    "INDIVIDUAL - TOD":        "INDIV_TOD",
    "BROKERAGELINK":           "401K_BL",
    "ROTH IRA":                "ROTH",
    "HEALTH SAVINGS ACCOUNT":  "HSA",
    "ROLLOVER IRA":            "ROLLOVER",
    "INDIVIDUAL - 529":        "PLAN_529",
    "MICROSOFT 401K PLAN":     "401K_CORE",
}

# Per-account "cash sleeve" label used when description = "HELD IN MONEY MARKET"
CASH_LABEL_BY_ACCT = {
    "INDIV_MARGIN": "FDRXX",
    "ROTH":         "CASH_ROTH",
    "INDIV_TOD":    "CASH_TOD",
    "HSA":          "CASH_HSA",
    "ROLLOVER":     "CASH_ROLLOVER",
}

# Insertion order for HOLDINGS_CURRENT block
ACCOUNT_ORDER = [
    "401K_BL",
    "INDIV_MARGIN",
    "ROTH",
    "HSA",
    "INDIV_TOD",
    "ROLLOVER",
    "PLAN_529",
    "401K_CORE",
]

# Human-readable labels for the per-account comment headers in user_config.py
ACCOUNT_HEADER_LABEL = {
    "401K_BL":      "401k BrokerageLink",
    "INDIV_MARGIN": "Individual margin",
    "ROTH":         "Roth IRA",
    "HSA":          "HSA",
    "INDIV_TOD":    "Individual TOD",
    "ROLLOVER":     "Rollover IRA",
    "PLAN_529":     "529 Education Plan",
    "401K_CORE":    "Microsoft 401k Plan core funds",
}

# Mutual-fund / plan-asset description -> stable internal label.
# Add the descriptions Fidelity uses for your plan's mutual funds.
MUTUAL_FUND_LABELS = {
    # "FIDELITY 500 INDEX": "FXAIX",
    # "VANGUARD 500 INDEX TRUST": "VANG_500_INDEX",
}
