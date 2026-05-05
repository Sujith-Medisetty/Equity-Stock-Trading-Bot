"""
config.py — Central configuration and watchlist.

Two things live here:
1. Config  — every tunable number in the system. Change values here, nowhere else.
2. Watchlist — the 15 Nifty 50 stocks we trade, with sector mapping.

Why everything is in one place:
- Easy to review risk settings before going live
- No magic numbers scattered across files
- Switching paper → live is just one flag: PAPER_TRADE = False
"""

import os
import logging
from typing import Optional

# Third-party availability flags.
# The system degrades gracefully if packages are missing —
# data fetching falls back to mock OHLCV, Dhan orders are skipped.
try:
    import pandas as pd
    import numpy as np
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False
    print("Run: pip install dhanhq pandas numpy requests schedule")

try:
    from dhanhq import dhanhq
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("schedule not installed — pip install schedule")

# Single shared logger used by every module.
# All files do: from config import log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("trading_system.log"),  # persists across runs
        logging.StreamHandler()                      # also prints to terminal
    ]
)
log = logging.getLogger("TradingSystem")


class Config:
    """
    Every constant the system uses. Read-only at runtime — never mutate these
    during execution except PAPER_TRADE which main.py flips for live mode.

    Capital rules — designed for a ₹2L account:
    - Max ₹50k per stock (25% concentration limit)
    - Max ₹1500 loss per trade (0.75% of capital)
    - Max 4 open positions simultaneously
    - Max ₹6000 total portfolio risk at any time (sum of all open SL distances)

    Loss protection — cascading limits that pause trading:
    - ₹3000 daily loss  → stop for today
    - ₹6000 weekly loss → stop for the week
    - ₹10000 monthly    → stop for the month
    - ₹20000 drawdown   → full stop, review needed
    """

    # --- Dhan API — set these as environment variables, never hardcode ---
    DHAN_CLIENT_ID    = os.getenv("DHAN_CLIENT_ID",    "YOUR_CLIENT_ID")
    DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")

    # --- Capital rules ---
    TOTAL_CAPITAL           = 200000   # ₹2,00,000
    MAX_CAPITAL_PER_TRADE   = 50000    # ₹50,000 max per stock
    MAX_RISK_PER_TRADE      = 1500     # ₹1,500 max loss per trade
    MAX_SIMULTANEOUS_TRADES = 4        # max open positions
    MAX_PORTFOLIO_RISK      = 6000     # total open risk across all trades

    # --- Loss protection limits ---
    DAILY_LOSS_LIMIT   = 3000
    WEEKLY_LOSS_LIMIT  = 6000
    MONTHLY_LOSS_LIMIT = 10000
    MAX_DRAWDOWN       = 20000

    # --- Risk/reward ---
    MIN_RR_RATIO = 2.0   # minimum 1:2 — risk ₹1 to make ₹2

    # --- ATR multipliers for stop loss placement per strategy ---
    # Higher multiplier = wider SL = more breathing room but more risk per share
    ATR_MULT = {
        "SWING":    1.5,   # standard swing SL
        "BREAKOUT": 1.0,   # tight SL below consolidation low
        "PULLBACK": 1.0,   # SL just below the EMA being tested
        "FII_FLOW": 2.0,   # wider SL — FII driven moves are volatile
        "WEEK52":   1.5,   # SL just below the old 52W high
    }

    # --- Strategy priority (higher = checked first, wins ties) ---
    STRATEGY_PRIORITY = {
        "FII_FLOW": 5,   # strongest edge — institutional money behind it
        "WEEK52":   4,   # rare but powerful
        "BREAKOUT": 3,   # high probability with volume confirmation
        "PULLBACK": 2,   # lower risk entry in existing trend
        "SWING":    1,   # general trend following
    }

    # --- Technical indicator periods ---
    EMA_SHORT         = 20
    EMA_MED           = 50
    EMA_LONG          = 200
    RSI_PERIOD        = 14
    ATR_PERIOD        = 14
    MACD_FAST         = 12
    MACD_SLOW         = 26
    MACD_SIGNAL       = 9
    BB_PERIOD         = 20
    BB_STD            = 2
    VOLUME_AVG_PERIOD = 20

    # --- Market condition thresholds ---
    VIX_CALM              = 13    # markets relaxed, all strategies active
    VIX_NORMAL            = 17    # normal conditions
    VIX_NERVOUS           = 22    # reduce exposure, exit volatile trades
    VIX_PANIC             = 28    # go to cash, no new trades
    FII_FLOW_THRESHOLD_CR = 2000  # ₹2000 Cr net = meaningful FII activity
    FII_CONSECUTIVE_DAYS  = 3     # need 3+ days in same direction to confirm trend

    # --- Timing rules ---
    MARKET_OPEN_WAIT_MINS  = 15   # wait 15 mins after 9:15 before any entry
    FRIDAY_NO_ENTRY_HOUR   = 14   # no new entries after 2 PM Friday (weekend risk)
    MONDAY_NO_ENTRY_MINS   = 30   # wait 30 mins Monday morning (gap resolution)
    MAX_HOLD_DAYS          = 15   # force exit if trade open > 15 days without profit
    COOLDOWN_AFTER_LOSS_HR = 2    # wait 2 hours after a SL hit before next trade
    EVENT_EXIT_DAYS        = 5    # exit if earnings/results within 5 days

    # --- NSE equity delivery charges (2026 rates) ---
    # These are deducted from gross PNL to get net PNL
    STT_DELIVERY    = 0.001      # 0.1% on both buy and sell
    DP_CHARGE       = 15.34      # flat ₹15.34 per sell transaction (demat charge)
    EXCHANGE_CHARGE = 0.0000297  # NSE transaction charge
    STAMP_DUTY      = 0.00015    # 0.015% on buy side only
    GST_RATE        = 0.18       # 18% GST on exchange charges
    SEBI_CHARGE     = 0.000001   # SEBI regulatory fee

    # --- Tax rates ---
    STCG_RATE             = 0.20    # 20% Short Term Capital Gains tax
    CESS_RATE             = 0.04    # 4% health & education cess on tax
    EFFECTIVE_TAX         = 0.208   # combined: 20% + 4% cess
    ADVANCE_TAX_THRESHOLD = 10000   # pay advance tax if annual liability > ₹10k

    # --- NSE public API endpoints (no auth needed, cookie-based) ---
    NSE_BASE        = "https://www.nseindia.com"
    NSE_FII_DII     = "/api/fiidiiTradeReact"      # FII/DII daily activity
    NSE_VIX         = "/api/allIndices"             # India VIX + all indices
    NSE_EVENTS      = "/api/event-calendar"         # earnings calendar
    NSE_SECTOR_BASE = "/api/equity-stockIndices?index="
    NSE_HEADERS = {
        # NSE blocks bots — we mimic a browser to avoid getting blocked
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept":          "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/",
    }

    # --- System settings ---
    DB_PATH     = "trading.db"  # SQLite file — created automatically on first run
    PAPER_TRADE = True          # True = no real orders placed. Set False for live trading.


class Watchlist:
    """
    The 15 Nifty 50 stocks this system trades.

    Why only 15?
    - Deep familiarity beats broad coverage for a ₹2L account
    - All are high-liquidity large caps — easy to enter/exit without slippage
    - Spread across 9 sectors for diversification
    - All Tier 1 — no mid/small cap risk

    Sector mapping is used by StockScreener to enforce a rule:
    max 1 open trade per sector at any time. This prevents over-concentration
    e.g. holding both ICICIBANK and HDFCBANK simultaneously.
    """

    STOCKS = {
        # symbol: {sector, tier}
        "ICICIBANK":  {"sector": "BANKING",  "tier": 1},
        "HDFCBANK":   {"sector": "BANKING",  "tier": 1},
        "AXISBANK":   {"sector": "BANKING",  "tier": 1},
        "INFY":       {"sector": "IT",       "tier": 1},
        "HCLTECH":    {"sector": "IT",       "tier": 1},
        "TATAMOTORS": {"sector": "AUTO",     "tier": 1},
        "MARUTI":     {"sector": "AUTO",     "tier": 1},
        "RELIANCE":   {"sector": "ENERGY",   "tier": 1},
        "BHARTIARTL": {"sector": "TELECOM",  "tier": 1},
        "SUNPHARMA":  {"sector": "PHARMA",   "tier": 1},
        "BAJFINANCE": {"sector": "FINANCE",  "tier": 1},
        "LT":         {"sector": "INFRA",    "tier": 1},
        "ITC":        {"sector": "CONSUMER", "tier": 1},
        "TITAN":      {"sector": "CONSUMER", "tier": 1},
        "TCS":        {"sector": "IT",       "tier": 1},
    }

    # Nifty 50 index — fetched as the market benchmark for RS calculation
    NIFTY_SYMBOL      = "NIFTY 50"
    NIFTY_SECURITY_ID = "13"

    # NSE sector index names — used to pull sector-level FII activity
    SECTOR_INDICES = {
        "BANKING":  "NIFTY BANK",
        "IT":       "NIFTY IT",
        "AUTO":     "NIFTY AUTO",
        "PHARMA":   "NIFTY PHARMA",
        "ENERGY":   "NIFTY ENERGY",
        "INFRA":    "NIFTY INFRA",
        "CONSUMER": "NIFTY FMCG",
        "TELECOM":  "NIFTY MEDIA",
        "FINANCE":  "NIFTY FIN SERVICE",
    }

    @classmethod
    def get_symbols(cls):
        return list(cls.STOCKS.keys())

    @classmethod
    def get_sector(cls, symbol):
        return cls.STOCKS.get(symbol, {}).get("sector", "UNKNOWN")

    @classmethod
    def get_tier(cls, symbol):
        return cls.STOCKS.get(symbol, {}).get("tier", 2)
