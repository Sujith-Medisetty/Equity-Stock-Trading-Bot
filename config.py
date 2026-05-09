"""
config.py — Central configuration and watchlist.

Two things live here:
1. Config  — every tunable number in the system. Change values here, nowhere else.
2. Watchlist — the 15 Nifty 50 stocks we trade, with sector mapping.

Why everything is in one place:
- Easy to review risk settings before going live
- No magic numbers scattered across files
- PAPER_TRADE=True uses Upstox paper account; PAPER_TRADE=False is live; BACKTEST_MODE=True is fully offline
"""

import os
import logging
from typing import Optional

# Third-party availability flags.
# The system degrades gracefully if packages are missing —
# data fetching falls back to mock OHLCV, Upstox orders are skipped.
try:
    import pandas as pd
    import numpy as np
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False
    print("Run: pip install upstox-python-sdk upstox-totp pandas numpy requests pyotp schedule pnsea")

try:
    import upstox_client
    UPSTOX_AVAILABLE = True
except ImportError:
    UPSTOX_AVAILABLE = False
    print("upstox-python-sdk not installed — pip install upstox-python-sdk")

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
    during execution except PAPER_TRADE / BACKTEST_MODE which main.py flips.

    Capital rules (percentage-based, auto-scaled from live Upstox balance):
    - Max 25% of available capital per stock
    - Max 0.75% of available capital at risk per trade
    - Max 4 open positions simultaneously
    - Max 3% of available capital as total open portfolio risk

    Loss protection — cascading limits that pause trading:
    - ₹3000 daily loss  → stop for today
    - ₹6000 weekly loss → stop for the week
    - ₹10000 monthly    → stop for the month
    - ₹20000 drawdown   → full stop, review needed
    """

    # --- Upstox API credentials ---
    # Get these from https://developer.upstox.com after creating an app.
    # UPSTOX_MOBILE, UPSTOX_PIN, UPSTOX_TOTP_SECRET enable fully headless daily auth.
    # Without these, the bot prints a URL on startup that needs one manual browser login.
    UPSTOX_API_KEY      = os.getenv("UPSTOX_API_KEY",      "YOUR_API_KEY")
    UPSTOX_API_SECRET   = os.getenv("UPSTOX_API_SECRET",   "YOUR_API_SECRET")
    UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://127.0.0.1:8765/callback")
    # Credentials for automated headless login (avoids manual browser auth each day)
    UPSTOX_MOBILE       = os.getenv("UPSTOX_MOBILE",       "")   # 10-digit mobile (login username)
    UPSTOX_PASSWORD     = os.getenv("UPSTOX_PASSWORD",     "")   # Upstox account password
    UPSTOX_PIN          = os.getenv("UPSTOX_PIN",          "")   # 6-digit MPIN (trading PIN)
    UPSTOX_TOTP_SECRET  = os.getenv("UPSTOX_TOTP_SECRET",  "")   # base32 secret from Upstox 2FA setup

    # --- Upstox API rate limiting & parallel fetch ---
    # Upstox allows ~250 req/min = ~4 req/sec sustained.
    # max_workers=5 with rate limiter at 3.5 req/sec keeps us safely under the limit
    # even with burst concurrency. Increase carefully — 429s fall back to mock data.
    UPSTOX_MAX_WORKERS       = 5     # concurrent threads for parallel OHLCV fetching
    UPSTOX_RATE_LIMIT_PER_SEC = 3.5  # max API calls per second across all threads

    # --- Retry config (applies to all Upstox API calls) ---
    API_MAX_RETRIES   = 3    # total attempts per call (1 original + 2 retries)
    API_RETRY_BASE_S  = 1.0  # base delay in seconds before first retry
    API_RETRY_BACKOFF = 2.0  # exponential backoff multiplier (1s → 2s → 4s)

    # --- Capital rules ---
    # TOTAL_CAPITAL is used only in BACKTEST_MODE.
    # In live/sandbox mode the actual available balance is fetched from Upstox
    # via get_available_capital() on every step1 run, so limits auto-scale
    # with your real portfolio as positions are added or closed.
    TOTAL_CAPITAL           = 200000   # backtest fallback only

    # Percentage-based limits — applied against live available capital at runtime.
    CAPITAL_PER_TRADE_PCT   = 0.25     # max 25% of available capital per position
    PORTFOLIO_RISK_PCT      = 0.03     # max 3% total open risk across all trades

    MAX_SIMULTANEOUS_TRADES = 4        # hard ceiling — never more than this regardless of capital

    # No reserve deduction — the MIN_TRADE_CAPITAL floor already prevents trading
    # when capital is too low, so subtracting a reserve is redundant.
    CAPITAL_RESERVE         = 0

    # Account-level floor (trading capital after reserve): stop all new trades below this.
    # At ₹20k trading capital the position size cap (25% × 20k = ₹5k) falls below
    # MIN_POSITION_VALUE anyway, so trades would be skipped regardless — this is the
    # explicit system-level gate that prevents even attempting to size a trade.
    MIN_TRADE_CAPITAL       = 20000    # ₹20,000 trading capital (after reserve) floor

    # Position-level floor: even if account capital is healthy, a specific trade
    # can produce a tiny position (e.g. expensive stock, wide ATR → few shares).
    # Below ₹15,000 position value the DP charge alone (₹15.34) is >5% of a
    # typical profit target — not worth the risk for such a small return.
    MIN_POSITION_VALUE      = 15000    # ₹15,000 — minimum position size per trade

    # Minimum shares to make the 3-tier exit system meaningful.
    # Tier 2 sells floor(qty/2) shares. With qty < 3:
    #   qty=1 → tier 2 sells all 1 share, tier 3 never runs (position fully closed at tier 2)
    #   qty=2 → tier 2 sells 1, tier 3 trails just 1 share (profit too small to matter)
    #   qty=3 → tier 2 sells 1, tier 3 trails 2 shares properly ✓
    MIN_QUANTITY            = 3        # minimum shares — below this the tiered exit breaks down

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
    MAX_ENTRY_HOUR         = 13   # no new entries at or after 1:30 PM (holds overnight)
    MAX_ENTRY_MINUTE       = 30
    MAX_ENTRY_DRIFT_PCT    = 1.5  # skip entry if live price moved >1.5% from setup price
    # FII_FLOW and BREAKOUT strategies expect gap-ups — allow wider drift for these.
    MAX_ENTRY_DRIFT_PCT_WIDE = 3.0
    WIDE_DRIFT_STRATEGIES  = {"FII_FLOW", "BREAKOUT", "WEEK52"}
    EVENT_WARN_DAYS        = 10   # log warning if earnings/results within 10 days
    EVENT_EXIT_DAYS        = 5    # tighten SL to breakeven if event within 5 days (exit if at a loss)
    EVENT_FORCE_EXIT_DAYS  = 2    # force exit unconditionally if event within 2 days

    # --- Slippage simulation (paper mode only) ---
    # In live trading, fills are slightly worse than LTP due to spread and queue position.
    # This adds a penalty to paper entries and exits so paper PNL is more realistic.
    PAPER_SLIPPAGE_PCT     = 0.002  # 0.2% adverse slippage on paper fills

    # --- SL replacement safety ---
    SL_REPLACE_MAX_RETRIES = 3     # retry SL placement this many times before emergency sell
    SL_REPLACE_RETRY_DELAY = 2     # seconds between retries

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
    DB_PATH      = "trading.db"  # SQLite file — created automatically on first run

    # PAPER_TRADE = True  → uses Upstox paper/sandbox account (same API key, simulated fills).
    #                       Orders are placed but money doesn't move.
    # PAPER_TRADE = False → live mode, real money. Requires explicit --live confirmation.
    PAPER_TRADE  = True

    # BACKTEST_MODE = True → fully offline historical simulation.
    #                        No broker API calls at all. Uses mock/historical OHLCV data.
    #                        SL hits are checked manually against price each cycle.
    BACKTEST_MODE = False

    @staticmethod
    def risk_per_trade(trading_capital: float) -> float:
        """
        Fixed rupee risk per trade based on current trading capital (after reserve).
        Returns 0 when capital is at or below the floor — caller must skip the trade.

          > ₹80,000  → ₹1,500  (full risk, meaningful profit after tax)
          > ₹20,000  → ₹1,000  (reduced risk, still clears charges + STCG)
          ≤ ₹20,000  → ₹0      (below floor — no new trades)
        """
        if trading_capital > 80000:
            return 1500.0
        elif trading_capital > 20000:
            return 1000.0
        return 0.0

    @staticmethod
    def effective_max_trades(available_capital: float) -> int:
        """
        Returns how many simultaneous trades the current capital can support.
        Each position must be worth at least MIN_POSITION_VALUE, so the limit is
        floor(available_capital / MIN_POSITION_VALUE), capped at MAX_SIMULTANEOUS_TRADES.

        Examples (MIN_POSITION_VALUE=₹15k, ceiling=4):
          ₹2,00,000 → floor(200k/15k)=13 → 4
          ₹80,000   → floor(80k/15k)=5   → 4
          ₹60,000   → floor(60k/15k)=4   → 4
          ₹50,000   → floor(50k/15k)=3   → 3
          ₹30,000   → floor(30k/15k)=2   → 2
        """
        dynamic = max(1, int(available_capital / Config.MIN_POSITION_VALUE))
        return min(Config.MAX_SIMULTANEOUS_TRADES, dynamic)


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
