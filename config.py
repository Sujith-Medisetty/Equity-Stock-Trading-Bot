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

# Load .env file if present — lets you store credentials locally without exporting env vars.
# pip install python-dotenv   (silent no-op if the package or file is missing)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # override=False: real env vars always take precedence
except ImportError:
    pass

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
    # Upstox rate limits (source: upstox.com/developer/api-documentation/rate-limiting)
    # Standard APIs (candles, market data, portfolio, charges): 50 req/sec, 500 req/min
    # Order APIs  (place / cancel / modify):                    10 req/sec, 500 req/min
    # Shared hard cap for both categories:                      2,000 req / 30 min
    UPSTOX_RATE_LIMIT_DATA_PER_SEC  = 40   # standard APIs — safe buffer under 50/sec
    UPSTOX_RATE_LIMIT_ORDER_PER_SEC = 8    # order APIs  — safe buffer under 10/sec
    UPSTOX_RATE_LIMIT_PER_MIN       = 450  # per-minute cap for all calls (Upstox limit: 500/min)

    # --- Retry config (applies to all Upstox API calls) ---
    API_MAX_RETRIES   = 3    # total attempts per call (1 original + 2 retries)
    API_RETRY_BASE_S  = 1.0  # base delay in seconds before first retry
    API_RETRY_BACKOFF = 2.0  # exponential backoff multiplier (1s → 2s → 4s)

    # --- Capital rules ---
    # TOTAL_CAPITAL is used only in BACKTEST_MODE.
    # In live/sandbox mode the actual available balance is fetched from Upstox
    # via get_available_capital() on every step1 run, so limits auto-scale
    # with your real portfolio as positions are added or closed.
    TOTAL_CAPITAL           = 600000   # backtest fallback only

    # Percentage-based limits — applied against live available capital at runtime.
    CAPITAL_PER_TRADE_PCT   = 0.25     # max 25% of available capital per position
    PORTFOLIO_RISK_PCT      = 0.03     # max 3% total open risk across all trades

    MAX_SIMULTANEOUS_TRADES = 4        # hard ceiling — never more than this regardless of capital

    # Account-level floor: stop all new trades below this.
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

    # --- Loss protection limits (scaled 3× for ₹6L capital vs original ₹2L) ---
    DAILY_LOSS_LIMIT   = 9000
    WEEKLY_LOSS_LIMIT  = 18000
    MONTHLY_LOSS_LIMIT = 30000
    MAX_DRAWDOWN       = 60000

    # --- Risk/reward ---
    MIN_RR_RATIO     = 2.5   # minimum R:R for SWING / BREAKOUT / WEEK52
    PULLBACK_MIN_RR  = 1.5   # PULLBACK is mean-reversion: target = recent swing high ≈ 2.7×ATR away
                              # At 1.5 RR with 45% win rate: EV = +0.125 per trade (positive)
                              # At 2.5 RR target was 4.5×ATR — unreachable for large-caps → losses

    # --- ATR multipliers for stop loss placement per strategy ---
    # Higher multiplier = wider SL = more breathing room but more risk per share
    ATR_MULT = {
        "SWING":    2.0,   # SL wider — price already above EMA20 by definition
        "BREAKOUT": 1.5,   # breakouts retest before continuing, need room below the box
        "PULLBACK": 1.5,   # SL 1.5×ATR below EMA20 — logical: if EMA20 breaks, setup is invalid
        "WEEK52":   1.5,   # SL just below the old 52W high (now support)
    }

    # --- Strategy priority (higher = checked first, wins ties) ---
    # FII sector buying is now a score modifier (+15) inside PULLBACK/BREAKOUT,
    # not a separate strategy — so FII_FLOW is removed from this table.
    STRATEGY_PRIORITY = {
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
    VOLUME_AVG_PERIOD = 20

    # --- Market condition thresholds ---
    VIX_CALM              = 13    # markets relaxed, all strategies active
    VIX_NORMAL            = 17    # normal conditions
    VIX_NERVOUS           = 25    # block new entries AND tighten existing trailing SL to (price - 0.5×ATR)
    VIX_PANIC             = 29    # full exit all positions, go to cash
    FII_FLOW_THRESHOLD_CR = 2000  # ₹2000 Cr net = meaningful FII activity
    FII_CONSECUTIVE_DAYS  = 3     # need 3+ days in same direction to confirm trend

    # --- Timing rules ---
    MARKET_OPEN_WAIT_MINS  = 15   # wait 15 mins after 9:15 before any entry
    FRIDAY_NO_ENTRY_HOUR   = 14   # no new entries after 2 PM Friday (weekend risk)
    MONDAY_NO_ENTRY_MINS   = 30   # wait 30 mins Monday morning (gap resolution)
    MAX_HOLD_DAYS          = 40   # force exit if trade open > 40 days without profit (momentum cycles run 6-12 weeks)
    COOLDOWN_AFTER_LOSS_HR = 2    # wait 2 hours after a SL hit before next trade
    MAX_ENTRY_HOUR         = 13   # no new entries at or after 1:30 PM (holds overnight)
    MAX_ENTRY_MINUTE       = 30
    MAX_ENTRY_DRIFT_PCT    = 1.5  # skip entry if live price moved >1.5% from setup price
    # BREAKOUT and WEEK52 expect gap-ups on the entry signal — allow wider drift.
    MAX_ENTRY_DRIFT_PCT_WIDE = 3.0
    WIDE_DRIFT_STRATEGIES  = {"BREAKOUT", "WEEK52"}
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

    # --- NSE equity delivery charges — fallback rates (used only if Upstox brokerage API fails) ---
    STT_DELIVERY    = 0.001        # 0.1% on both buy and sell
    DP_CHARGE       = 15.34        # flat ₹15.34 per sell transaction (demat charge)
    EXCHANGE_CHARGE = 0.0000297    # NSE transaction charge (0.00297%)
    STAMP_DUTY      = 0.00015      # 0.015% on buy side only
    GST_RATE        = 0.18         # 18% GST on exchange charges
    SEBI_CHARGE     = 0.000001     # SEBI regulatory fee (₹10 per crore)
    CLEARING_CHARGE = 0.00000325   # NSE clearing charge (₹32.5 per crore, 0.000325%)
    IPFT_CHARGE     = 0.0000000001 # IPFT (₹1 per crore — negligible)

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

          > ₹3,00,000 → ₹3,000  (0.5% of ₹6L — proportionally same as ₹1,500 on ₹3L)
          > ₹80,000   → ₹1,500  (full risk, meaningful profit after tax)
          > ₹20,000   → ₹1,000  (reduced risk, still clears charges + STCG)
          ≤ ₹20,000   → ₹0      (below floor — no new trades)
        """
        if trading_capital > 300000:
            return 3000.0
        elif trading_capital > 80000:
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
    All Nifty 50 stocks except IT/tech sector (TCS, INFY, HCLTECH, WIPRO, TECHM).
    45 stocks across 15 sectors — gives the screener a wide pool to find setups in.

    Sector mapping enforces max 1 open trade per sector at any time, preventing
    over-concentration (e.g. holding ICICIBANK + HDFCBANK + SBIN simultaneously).
    With 6 banking stocks in the pool, only the best setup from that sector enters.
    """

    STOCKS = {
        # ── BANKING (6) ──────────────────────────────────────────────────────
        "ICICIBANK":  {"sector": "BANKING",      "tier": 1},
        "HDFCBANK":   {"sector": "BANKING",      "tier": 1},
        "AXISBANK":   {"sector": "BANKING",      "tier": 1},
        "SBIN":       {"sector": "BANKING",      "tier": 1},
        "KOTAKBANK":  {"sector": "BANKING",      "tier": 1},
        "INDUSINDBK": {"sector": "BANKING",      "tier": 1},

        # ── FINANCE (5) ──────────────────────────────────────────────────────
        "BAJFINANCE": {"sector": "FINANCE",        "tier": 1},
        "BAJAJFINSV": {"sector": "FINANCE",        "tier": 1},
        "SHRIRAMFIN": {"sector": "FINANCE",        "tier": 1},
        "MUTHOOTFIN": {"sector": "FINANCE",        "tier": 1},
        "CHOLAFIN":   {"sector": "FINANCE",        "tier": 1},

        # ── INSURANCE (2) ────────────────────────────────────────────────────
        "HDFCLIFE":   {"sector": "INSURANCE",      "tier": 1},
        "SBILIFE":    {"sector": "INSURANCE",      "tier": 1},

        # ── AUTO (7) ─────────────────────────────────────────────────────────
        "TATAMOTORS": {"sector": "AUTO",           "tier": 1},
        "MARUTI":     {"sector": "AUTO",           "tier": 1},
        "M&M":        {"sector": "AUTO",           "tier": 1},
        "BAJAJ-AUTO": {"sector": "AUTO",           "tier": 1},
        "HEROMOTOCO": {"sector": "AUTO",           "tier": 1},
        "EICHERMOT":  {"sector": "AUTO",           "tier": 1},
        "TVSMOTOR":   {"sector": "AUTO",           "tier": 1},

        # ── AUTO_COMP (4) — auto components / tyres ──────────────────────────
        "MOTHERSON":  {"sector": "AUTO_COMP",      "tier": 1},
        "MRF":        {"sector": "AUTO_COMP",      "tier": 1},
        "TIINDIA":    {"sector": "AUTO_COMP",      "tier": 1},
        "BOSCHLTD":   {"sector": "AUTO_COMP",      "tier": 1},

        # ── FMCG (5) ─────────────────────────────────────────────────────────
        "ITC":        {"sector": "FMCG",           "tier": 1},
        "HINDUNILVR": {"sector": "FMCG",           "tier": 1},
        "NESTLEIND":  {"sector": "FMCG",           "tier": 1},
        "BRITANNIA":  {"sector": "FMCG",           "tier": 1},
        "TATACONSUM": {"sector": "FMCG",           "tier": 1},

        # ── CONSUMER (6) ─────────────────────────────────────────────────────
        "TITAN":      {"sector": "CONSUMER",       "tier": 1},
        "ASIANPAINT": {"sector": "CONSUMER",       "tier": 1},
        "TRENT":      {"sector": "CONSUMER",       "tier": 1},
        "PAGEIND":    {"sector": "CONSUMER",       "tier": 1},
        "DMART":      {"sector": "CONSUMER",       "tier": 1},
        "ETERNAL":    {"sector": "CONSUMER",       "tier": 1},

        # ── CONSUMER_ELECT (3) — wires, fans, electronics manufacturing ──────
        "HAVELLS":    {"sector": "CONSUMER_ELECT", "tier": 1},
        "POLYCAB":    {"sector": "CONSUMER_ELECT", "tier": 1},
        "DIXON":      {"sector": "CONSUMER_ELECT", "tier": 1},

        # ── PHARMA (3) ───────────────────────────────────────────────────────
        "SUNPHARMA":  {"sector": "PHARMA",         "tier": 1},
        "CIPLA":      {"sector": "PHARMA",         "tier": 1},
        "DRREDDY":    {"sector": "PHARMA",         "tier": 1},

        # ── HEALTHCARE (4) ───────────────────────────────────────────────────
        "APOLLOHOSP": {"sector": "HEALTHCARE",     "tier": 1},
        "MAXHEALTH":  {"sector": "HEALTHCARE",     "tier": 1},
        "ALKEM":      {"sector": "HEALTHCARE",     "tier": 1},
        "TORNTPHARM": {"sector": "HEALTHCARE",     "tier": 1},

        # ── METALS (3) ───────────────────────────────────────────────────────
        "HINDALCO":   {"sector": "METALS",         "tier": 1},
        "TATASTEEL":  {"sector": "METALS",         "tier": 1},
        "JSWSTEEL":   {"sector": "METALS",         "tier": 1},

        # ── CHEMICALS (3) ────────────────────────────────────────────────────
        "PIDILITIND": {"sector": "CHEMICALS",      "tier": 1},
        "DEEPAKNTR":  {"sector": "CHEMICALS",      "tier": 1},
        "PIIND":      {"sector": "CHEMICALS",      "tier": 1},

        # ── ENERGY (6) ───────────────────────────────────────────────────────
        "RELIANCE":   {"sector": "ENERGY",         "tier": 1},
        "ONGC":       {"sector": "ENERGY",         "tier": 1},
        "NTPC":       {"sector": "ENERGY",         "tier": 1},
        "POWERGRID":  {"sector": "ENERGY",         "tier": 1},
        "COALINDIA":  {"sector": "ENERGY",         "tier": 1},
        "BPCL":       {"sector": "ENERGY",         "tier": 1},

        # ── POWER (3) — renewable/clean power separate from fossil ENERGY ────
        "TATAPOWER":  {"sector": "POWER",          "tier": 1},
        "NHPC":       {"sector": "POWER",          "tier": 1},
        "IREDA":      {"sector": "POWER",          "tier": 1},

        # ── CEMENT (2) ───────────────────────────────────────────────────────
        "ULTRACEMCO": {"sector": "CEMENT",         "tier": 1},
        "GRASIM":     {"sector": "CEMENT",         "tier": 1},

        # ── INFRA (2) ────────────────────────────────────────────────────────
        "LT":         {"sector": "INFRA",          "tier": 1},
        "ADANIPORTS": {"sector": "INFRA",          "tier": 1},

        # ── CAPGOODS (5) — industrial capital goods / automation ─────────────
        "SIEMENS":    {"sector": "CAPGOODS",       "tier": 1},
        "ABB":        {"sector": "CAPGOODS",       "tier": 1},
        "CUMMINSIND": {"sector": "CAPGOODS",       "tier": 1},
        "BHEL":       {"sector": "CAPGOODS",       "tier": 1},
        "CGPOWER":    {"sector": "CAPGOODS",       "tier": 1},

        # ── REALESTATE (3) ───────────────────────────────────────────────────
        "DLF":        {"sector": "REALESTATE",     "tier": 1},
        "GODREJPROP": {"sector": "REALESTATE",     "tier": 1},
        "OBEROIRLTY": {"sector": "REALESTATE",     "tier": 1},

        # ── CONGLOMERATE (1) ─────────────────────────────────────────────────
        "ADANIENT":   {"sector": "CONGLOMERATE",   "tier": 1},

        # ── TELECOM (1) ──────────────────────────────────────────────────────
        "BHARTIARTL": {"sector": "TELECOM",        "tier": 1},

        # ── DEFENCE (3) ──────────────────────────────────────────────────────
        "BEL":        {"sector": "DEFENCE",        "tier": 1},
        "HAL":        {"sector": "DEFENCE",        "tier": 1},
        "SOLARINDS":  {"sector": "DEFENCE",        "tier": 1},

        # ── RAILINFRA (2) — railway construction + financing PSUs ────────────
        "RVNL":       {"sector": "RAILINFRA",      "tier": 1},
        "IRFC":       {"sector": "RAILINFRA",      "tier": 1},
    }

    NIFTY_SYMBOL = "NIFTY 50"

    # NSE sector index names — used to pull sector-level FII activity
    SECTOR_INDICES = {
        "BANKING":        "NIFTY BANK",
        "FINANCE":        "NIFTY FIN SERVICE",
        "INSURANCE":      "NIFTY FIN SERVICE",
        "AUTO":           "NIFTY AUTO",
        "AUTO_COMP":      "NIFTY AUTO",
        "FMCG":           "NIFTY FMCG",
        "CONSUMER":       "NIFTY INDIA CONSUMPTION",
        "CONSUMER_ELECT": "NIFTY INDIA CONSUMPTION",
        "PHARMA":         "NIFTY PHARMA",
        "HEALTHCARE":     "NIFTY HEALTHCARE INDEX",
        "METALS":         "NIFTY METAL",
        "CHEMICALS":      "NIFTY CHEMICALS",
        "ENERGY":         "NIFTY ENERGY",
        "POWER":          "NIFTY ENERGY",
        "CEMENT":         "NIFTY INFRA",
        "INFRA":          "NIFTY INFRA",
        "CAPGOODS":       "NIFTY INDIA MANUFACTURING",
        "REALESTATE":     "NIFTY REALTY",
        "CONGLOMERATE":   "NIFTY 500",
        "TELECOM":        "NIFTY MEDIA",
        "DEFENCE":        "NIFTY INDIA DEFENCE",
        "RAILINFRA":      "NIFTY INFRASTRUCTURE",
    }

    @classmethod
    def get_symbols(cls):
        return list(cls.STOCKS.keys())

    @classmethod
    def get_sector(cls, symbol):
        return cls.STOCKS.get(symbol, {}).get("sector", "UNKNOWN")
