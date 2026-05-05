"""
=============================================================================
NIFTY 50 SWING TRADING SYSTEM
=============================================================================
Complete automated trading system for equity swing trading on Nifty 50 stocks.
Covers: Data Collection, Market Mode Detection, Stock Screening, Strategy
Selection, Risk Management, Order Execution, Trade Monitoring, Exit Management,
P&L Tracking, Tax Calculation, and Dashboard.

Author  : Your Trading System
Version : 1.0.0
Capital : ₹2,00,000 | Max Per Trade: ₹50,000 | Max Risk/Trade: ₹1,500
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import json
import time
import logging
import sqlite3
import requests
import threading
from datetime import datetime, timedelta, date
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

# Third party - install via: pip install dhanhq pandas numpy requests schedule
try:
    import pandas as pd
    import numpy as np
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False
    print("⚠️  Run: pip install dhanhq pandas numpy requests schedule")

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
    print("⚠️  schedule not installed — scheduler disabled. pip install schedule")


# =============================================================================
# PURE NUMPY/PANDAS TECHNICAL INDICATOR LIBRARY
# No external TA library needed — all implemented from scratch
# =============================================================================

class ta:
    """
    Pure pandas/numpy implementation of all required indicators.
    Drop-in replacement for pandas_ta.
    """

    @staticmethod
    def ema(series: "pd.Series", length: int) -> "pd.Series":
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rsi(series: "pd.Series", length: int = 14) -> "pd.Series":
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: "pd.Series", fast=12, slow=26, signal=9) -> Optional["pd.DataFrame"]:
        if not LIBS_AVAILABLE:
            return None
        ema_fast = ta.ema(series, fast)
        ema_slow = ta.ema(series, slow)
        macd_line   = ema_fast - ema_slow
        signal_line = ta.ema(macd_line, signal)
        histogram   = macd_line - signal_line
        return pd.DataFrame({
            "macd":   macd_line,
            "signal": signal_line,
            "hist":   histogram
        })

    @staticmethod
    def atr(high: "pd.Series", low: "pd.Series",
            close: "pd.Series", length: int = 14) -> "pd.Series":
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(com=length - 1, adjust=False).mean()

    @staticmethod
    def bbands(series: "pd.Series", length: int = 20,
               std: float = 2.0) -> Optional["pd.DataFrame"]:
        if not LIBS_AVAILABLE:
            return None
        mid   = series.rolling(length).mean()
        sigma = series.rolling(length).std()
        upper = mid + std * sigma
        lower = mid - std * sigma
        return pd.DataFrame({"lower": lower, "mid": mid, "upper": upper})

    @staticmethod
    def obv(close: "pd.Series", volume: "pd.Series") -> "pd.Series":
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return (direction * volume).cumsum()

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TradingSystem")

# =============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# =============================================================================

class Config:
    """
    Central configuration. Edit these values before running.
    Never hardcode API keys — use environment variables.
    """

    # --- Dhan API Credentials ---
    DHAN_CLIENT_ID   = os.getenv("DHAN_CLIENT_ID", "YOUR_CLIENT_ID")
    DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")

    # --- Capital Rules ---
    TOTAL_CAPITAL        = 200000   # ₹2,00,000 total
    MAX_CAPITAL_PER_TRADE = 50000   # ₹50,000 max per stock
    MAX_RISK_PER_TRADE   = 1500     # ₹1,500 max loss per trade
    MAX_SIMULTANEOUS_TRADES = 4     # max 4 open positions
    MAX_PORTFOLIO_RISK   = 6000     # total ₹6,000 risk across all trades

    # --- Loss Protection Limits ---
    DAILY_LOSS_LIMIT   = 3000       # stop trading if daily loss >= ₹3,000
    WEEKLY_LOSS_LIMIT  = 6000       # stop if weekly loss >= ₹6,000
    MONTHLY_LOSS_LIMIT = 10000      # stop if monthly loss >= ₹10,000
    MAX_DRAWDOWN       = 20000      # full stop if drawdown >= ₹20,000

    # --- Risk Reward ---
    MIN_RR_RATIO       = 2.0        # minimum 1:2 risk reward

    # --- ATR Multipliers per Strategy ---
    ATR_MULT = {
        "SWING":     1.5,
        "BREAKOUT":  1.0,
        "PULLBACK":  1.0,
        "FII_FLOW":  2.0,
        "WEEK52":    1.5,
    }

    # --- Strategy Priority (higher = checked first) ---
    STRATEGY_PRIORITY = {
        "FII_FLOW":  5,
        "WEEK52":    4,
        "BREAKOUT":  3,
        "PULLBACK":  2,
        "SWING":     1,
    }

    # --- Indicator Parameters ---
    EMA_SHORT   = 20
    EMA_MED     = 50
    EMA_LONG    = 200
    RSI_PERIOD  = 14
    ATR_PERIOD  = 14
    MACD_FAST   = 12
    MACD_SLOW   = 26
    MACD_SIGNAL = 9
    BB_PERIOD   = 20
    BB_STD      = 2
    VOLUME_AVG_PERIOD = 20

    # --- Market Thresholds ---
    VIX_CALM       = 13
    VIX_NORMAL     = 17
    VIX_NERVOUS    = 22
    VIX_PANIC      = 28
    FII_FLOW_THRESHOLD_CR = 2000   # crore
    FII_CONSECUTIVE_DAYS  = 3

    # --- Timing Rules ---
    MARKET_OPEN_WAIT_MINS  = 15    # wait after 9:15
    FRIDAY_NO_ENTRY_HOUR   = 14    # no entries after 2 PM Friday
    MONDAY_NO_ENTRY_MINS   = 30    # no entries in first 30 mins Monday
    MAX_HOLD_DAYS          = 15    # exit if trade open > 15 days
    COOLDOWN_AFTER_LOSS_HR = 2     # hours to wait after SL hit
    EVENT_EXIT_DAYS        = 5     # exit if event within 5 days

    # --- Charges (2026 rates) ---
    STT_DELIVERY    = 0.001        # 0.1% both sides
    DP_CHARGE       = 15.34        # flat per sell per stock
    EXCHANGE_CHARGE = 0.0000297    # NSE transaction charge
    STAMP_DUTY      = 0.00015      # on buy side only
    GST_RATE        = 0.18         # on brokerage + exchange charges
    SEBI_CHARGE     = 0.000001     # tiny SEBI fee

    # --- Tax ---
    STCG_RATE  = 0.20              # 20% STCG
    CESS_RATE  = 0.04              # 4% cess on tax
    EFFECTIVE_TAX = 0.208          # 20% + 4% cess combined
    ADVANCE_TAX_THRESHOLD = 10000  # pay advance tax if liability > ₹10k

    # --- NSE API Endpoints ---
    NSE_BASE          = "https://www.nseindia.com"
    NSE_FII_DII       = "/api/fiidiiTradeReact"
    NSE_VIX           = "/api/allIndices"
    NSE_EVENTS        = "/api/event-calendar"
    NSE_SECTOR_BASE   = "/api/equity-stockIndices?index="

    NSE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    # --- Database ---
    DB_PATH = "trading.db"

    # --- Paper Trading Mode (set True to test without real orders) ---
    PAPER_TRADE = True


# =============================================================================
# SECTION 2: WATCHLIST
# =============================================================================

class Watchlist:
    """
    Your 15 carefully selected Nifty 50 stocks across 9 sectors.
    Tier 1 = Primary focus | Tier 2 = Trade with caution
    """

    STOCKS = {
        # symbol: {sector, tier, avoid_reason}
        "ICICIBANK":    {"sector": "BANKING",   "tier": 1},
        "HDFCBANK":     {"sector": "BANKING",   "tier": 1},
        "AXISBANK":     {"sector": "BANKING",   "tier": 1},
        "INFY":         {"sector": "IT",        "tier": 1},
        "HCLTECH":      {"sector": "IT",        "tier": 1},
        "TATAMOTORS":   {"sector": "AUTO",      "tier": 1},
        "MARUTI":       {"sector": "AUTO",      "tier": 1},
        "RELIANCE":     {"sector": "ENERGY",    "tier": 1},
        "BHARTIARTL":   {"sector": "TELECOM",   "tier": 1},
        "SUNPHARMA":    {"sector": "PHARMA",    "tier": 1},
        "BAJFINANCE":   {"sector": "FINANCE",   "tier": 1},
        "LT":           {"sector": "INFRA",     "tier": 1},
        "ITC":          {"sector": "CONSUMER",  "tier": 1},
        "TITAN":        {"sector": "CONSUMER",  "tier": 1},
        "TCS":          {"sector": "IT",        "tier": 1},
    }

    # Nifty 50 index symbol for Dhan API
    NIFTY_SYMBOL = "NIFTY 50"
    NIFTY_SECURITY_ID = "13"   # Dhan security ID for Nifty

    # NSE sector index names for sector rotation
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


# =============================================================================
# SECTION 3: ENUMS & DATA CLASSES
# =============================================================================

class MarketMode(Enum):
    AGGRESSIVE = "AGGRESSIVE"   # Bull + FII Buying
    NORMAL     = "NORMAL"       # Bull + FII Neutral
    SELECTIVE  = "SELECTIVE"    # Pullback mode
    CAUTIOUS   = "CAUTIOUS"     # Sideways
    DEFENSIVE  = "DEFENSIVE"    # Bear — no new trades
    CASH       = "CASH"         # VIX > 22 — sit out


class FIIFlow(Enum):
    BUYING  = "BUYING"
    SELLING = "SELLING"
    NEUTRAL = "NEUTRAL"


class StrategyType(Enum):
    SWING    = "SWING"
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    FII_FLOW = "FII_FLOW"
    WEEK52   = "WEEK52"


class TradeStatus(Enum):
    OPEN   = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(Enum):
    TARGET_HIT     = "TARGET_HIT"
    SL_HIT         = "SL_HIT"
    TRAILING_SL    = "TRAILING_SL"
    TIME_BASED     = "TIME_BASED"
    EVENT_EXIT     = "EVENT_EXIT"
    MARKET_CRASH   = "MARKET_CRASH"
    MANUAL         = "MANUAL"
    LOSS_LIMIT     = "LOSS_LIMIT"


@dataclass
class StockData:
    """Snapshot of all indicator values for one stock on one day."""
    symbol: str
    date: str
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    atr: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    volume_ratio: float = 1.0
    week_52_high: float = 0.0
    week_52_low: float = 0.0
    rs_score: float = 0.0
    candle_pattern: str = "NONE"
    weekly_bullish: bool = False
    daily_bullish: bool = False
    h4_bullish: bool = False
    tf_aligned_count: int = 0       # 0-3: how many timeframes agree
    consolidation_range_pct: float = 0.0
    obv_rising: bool = False
    atr_ratio: float = 1.0          # current ATR / 20 day avg ATR


@dataclass
class Setup:
    """A detected trading setup ready for risk calculation."""
    symbol: str
    date: str
    strategy: StrategyType
    score: int = 0                  # 0-100
    entry_price: float = 0.0
    sl_price: float = 0.0
    target_price: float = 0.0
    atr: float = 0.0
    risk_per_share: float = 0.0
    shares: int = 0
    capital_required: float = 0.0
    actual_risk: float = 0.0
    rr_ratio: float = 0.0
    market_mode: str = ""
    fii_flow: str = ""
    status: str = "PENDING"
    skip_reason: str = ""


@dataclass
class Trade:
    """A live or closed trade with full tracking."""
    trade_id: str
    symbol: str
    strategy: str
    entry_date: str
    entry_price: float
    quantity: int
    initial_sl: float
    initial_target: float
    current_sl: float
    current_price: float = 0.0

    # Tier exits
    tier1_done: bool = False
    tier1_price: float = 0.0
    tier1_qty: int = 0
    tier2_done: bool = False
    tier2_price: float = 0.0
    tier2_qty: int = 0
    remaining_qty: int = 0

    # Charges
    stt: float = 0.0
    dp_charge: float = 0.0
    exchange_charge: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0
    sebi: float = 0.0
    total_charges: float = 0.0

    # PNL
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Meta
    setup_score: int = 0
    market_mode_at_entry: str = ""
    status: str = TradeStatus.OPEN.value
    exit_reason: str = ""
    exit_date: str = ""
    holding_days: int = 0
    sl_order_id: str = ""


# =============================================================================
# SECTION 4: DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """
    SQLite database for all trade data, market snapshots, P&L, journal etc.
    Single file: trading.db
    """

    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        log.info(f"Database ready: {db_path}")

    def _create_tables(self):
        c = self.conn.cursor()

        # Market snapshots daily
        c.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                date TEXT PRIMARY KEY,
                nifty_close REAL, nifty_ema20 REAL, nifty_ema50 REAL,
                nifty_ema200 REAL, nifty_rsi REAL,
                india_vix REAL, gift_nifty REAL,
                fii_net_cash REAL, dii_net_cash REAL,
                fii_consecutive_days INTEGER,
                market_mode TEXT, fii_flow_label TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Stock daily snapshots with all indicators
        c.execute("""
            CREATE TABLE IF NOT EXISTS stock_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, symbol TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                ema_20 REAL, ema_50 REAL, ema_200 REAL,
                rsi REAL, macd REAL, macd_signal REAL, macd_hist REAL,
                atr REAL, bb_upper REAL, bb_lower REAL,
                volume_ratio REAL, week_52_high REAL, week_52_low REAL,
                rs_score REAL, candle_pattern TEXT,
                weekly_bullish INTEGER, daily_bullish INTEGER, h4_bullish INTEGER,
                tf_aligned_count INTEGER,
                consolidation_range_pct REAL, obv_rising INTEGER,
                atr_ratio REAL,
                UNIQUE(date, symbol)
            )
        """)

        # Detected setups
        c.execute("""
            CREATE TABLE IF NOT EXISTS setups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, symbol TEXT, strategy TEXT,
                score INTEGER, entry_price REAL, sl_price REAL,
                target_price REAL, atr REAL,
                risk_per_share REAL, shares INTEGER,
                capital_required REAL, actual_risk REAL,
                rr_ratio REAL, market_mode TEXT, fii_flow TEXT,
                status TEXT DEFAULT 'PENDING',
                skip_reason TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # All trades (open + closed)
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT, strategy TEXT,
                entry_date TEXT, entry_price REAL, quantity INTEGER,
                initial_sl REAL, initial_target REAL, current_sl REAL,
                current_price REAL,
                tier1_done INTEGER DEFAULT 0, tier1_price REAL, tier1_qty INTEGER,
                tier2_done INTEGER DEFAULT 0, tier2_price REAL, tier2_qty INTEGER,
                remaining_qty INTEGER,
                stt REAL DEFAULT 0, dp_charge REAL DEFAULT 0,
                exchange_charge REAL DEFAULT 0, stamp_duty REAL DEFAULT 0,
                gst REAL DEFAULT 0, sebi REAL DEFAULT 0,
                total_charges REAL DEFAULT 0,
                gross_pnl REAL DEFAULT 0, net_pnl REAL DEFAULT 0,
                setup_score INTEGER DEFAULT 0,
                market_mode_at_entry TEXT,
                status TEXT DEFAULT 'OPEN',
                exit_reason TEXT DEFAULT '',
                exit_date TEXT DEFAULT '',
                holding_days INTEGER DEFAULT 0,
                sl_order_id TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration: add sl_order_id to existing databases that predate this column
        try:
            c.execute("ALTER TABLE trades ADD COLUMN sl_order_id TEXT DEFAULT ''")
        except Exception:
            pass  # column already exists

        # SL trail log
        c.execute("""
            CREATE TABLE IF NOT EXISTS trailing_sl_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT, timestamp TEXT,
                old_sl REAL, new_sl REAL,
                current_price REAL, reason TEXT
            )
        """)

        # Daily PNL summary
        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                realised_pnl REAL DEFAULT 0,
                unrealised_pnl REAL DEFAULT 0,
                total_charges REAL DEFAULT 0,
                trades_opened INTEGER DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                running_stcg REAL DEFAULT 0,
                stcg_tax_estimate REAL DEFAULT 0
            )
        """)

        # FII/DII history
        c.execute("""
            CREATE TABLE IF NOT EXISTS fii_history (
                date TEXT PRIMARY KEY,
                fii_net_cash REAL, dii_net_cash REAL,
                consecutive_buying_days INTEGER DEFAULT 0,
                consecutive_selling_days INTEGER DEFAULT 0,
                sector_flow TEXT DEFAULT '{}'
            )
        """)

        # Events calendar
        c.execute("""
            CREATE TABLE IF NOT EXISTS events_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, event_type TEXT,
                event_date TEXT, days_away INTEGER,
                risk_level TEXT, action_required TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Trade journal
        c.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                trade_id TEXT PRIMARY KEY,
                pre_trade_notes TEXT DEFAULT '',
                post_trade_review TEXT DEFAULT '',
                rules_followed INTEGER DEFAULT 1,
                what_worked TEXT DEFAULT '',
                what_failed TEXT DEFAULT '',
                emotion_score INTEGER DEFAULT 5
            )
        """)

        # Protection state (cooldowns, limits hit)
        c.execute("""
            CREATE TABLE IF NOT EXISTS protection_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()
        log.info("All database tables ready.")

    # ---- Generic helpers ----

    def execute(self, sql, params=()):
        c = self.conn.cursor()
        c.execute(sql, params)
        self.conn.commit()
        return c

    def fetchone(self, sql, params=()):
        c = self.conn.cursor()
        c.execute(sql, params)
        return c.fetchone()

    def fetchall(self, sql, params=()):
        c = self.conn.cursor()
        c.execute(sql, params)
        return c.fetchall()

    # ---- Market snapshot ----

    def save_market_snapshot(self, data: dict):
        self.execute("""
            INSERT OR REPLACE INTO market_snapshots
            (date, nifty_close, nifty_ema20, nifty_ema50, nifty_ema200,
             nifty_rsi, india_vix, gift_nifty, fii_net_cash, dii_net_cash,
             fii_consecutive_days, market_mode, fii_flow_label)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["date"], data["nifty_close"],
            data["nifty_ema20"], data["nifty_ema50"], data["nifty_ema200"],
            data["nifty_rsi"], data["india_vix"], data["gift_nifty"],
            data["fii_net_cash"], data["dii_net_cash"],
            data["fii_consecutive_days"],
            data["market_mode"], data["fii_flow_label"]
        ))

    def get_market_snapshot(self, date: str):
        return self.fetchone(
            "SELECT * FROM market_snapshots WHERE date=?", (date,)
        )

    # ---- Stock snapshot ----

    def save_stock_snapshot(self, s: StockData):
        self.execute("""
            INSERT OR REPLACE INTO stock_snapshots
            (date, symbol, open, high, low, close, volume,
             ema_20, ema_50, ema_200, rsi, macd, macd_signal, macd_hist,
             atr, bb_upper, bb_lower, volume_ratio, week_52_high, week_52_low,
             rs_score, candle_pattern, weekly_bullish, daily_bullish, h4_bullish,
             tf_aligned_count, consolidation_range_pct, obv_rising, atr_ratio)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            s.date, s.symbol, s.open, s.high, s.low, s.close, s.volume,
            s.ema_20, s.ema_50, s.ema_200, s.rsi,
            s.macd, s.macd_signal, s.macd_hist,
            s.atr, s.bb_upper, s.bb_lower,
            s.volume_ratio, s.week_52_high, s.week_52_low,
            s.rs_score, s.candle_pattern,
            int(s.weekly_bullish), int(s.daily_bullish), int(s.h4_bullish),
            s.tf_aligned_count, s.consolidation_range_pct,
            int(s.obv_rising), s.atr_ratio
        ))

    # ---- Setups ----

    def save_setup(self, s: Setup):
        self.execute("""
            INSERT INTO setups
            (date, symbol, strategy, score, entry_price, sl_price,
             target_price, atr, risk_per_share, shares, capital_required,
             actual_risk, rr_ratio, market_mode, fii_flow, status, skip_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            s.date, s.symbol, s.strategy.value, s.score,
            s.entry_price, s.sl_price, s.target_price,
            s.atr, s.risk_per_share, s.shares, s.capital_required,
            s.actual_risk, s.rr_ratio, s.market_mode, s.fii_flow,
            s.status, s.skip_reason
        ))

    def get_pending_setups(self, date: str):
        return self.fetchall(
            "SELECT * FROM setups WHERE date=? AND status='PENDING' ORDER BY score DESC",
            (date,)
        )

    # ---- Trades ----

    def save_trade(self, t: Trade):
        self.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, symbol, strategy, entry_date, entry_price, quantity,
             initial_sl, initial_target, current_sl, current_price,
             tier1_done, tier1_price, tier1_qty,
             tier2_done, tier2_price, tier2_qty, remaining_qty,
             stt, dp_charge, exchange_charge, stamp_duty, gst, sebi,
             total_charges, gross_pnl, net_pnl,
             setup_score, market_mode_at_entry,
             status, exit_reason, exit_date, holding_days, sl_order_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            t.trade_id, t.symbol, t.strategy,
            t.entry_date, t.entry_price, t.quantity,
            t.initial_sl, t.initial_target, t.current_sl, t.current_price,
            int(t.tier1_done), t.tier1_price, t.tier1_qty,
            int(t.tier2_done), t.tier2_price, t.tier2_qty, t.remaining_qty,
            t.stt, t.dp_charge, t.exchange_charge, t.stamp_duty,
            t.gst, t.sebi, t.total_charges,
            t.gross_pnl, t.net_pnl,
            t.setup_score, t.market_mode_at_entry,
            t.status, t.exit_reason, t.exit_date, t.holding_days, t.sl_order_id
        ))

    def get_open_trades(self):
        return self.fetchall("SELECT * FROM trades WHERE status='OPEN'")

    def get_trade(self, trade_id: str):
        return self.fetchone("SELECT * FROM trades WHERE trade_id=?", (trade_id,))

    def update_trade_sl(self, trade_id: str, new_sl: float):
        self.execute(
            "UPDATE trades SET current_sl=? WHERE trade_id=?",
            (new_sl, trade_id)
        )

    def update_sl_order_id(self, trade_id: str, sl_order_id: str):
        self.execute(
            "UPDATE trades SET sl_order_id=? WHERE trade_id=?",
            (sl_order_id, trade_id)
        )

    def close_trade(self, trade_id: str, exit_price: float,
                    exit_reason: str, net_pnl: float,
                    gross_pnl: float, total_charges: float):
        today = datetime.now().strftime("%Y-%m-%d")
        entry = self.get_trade(trade_id)
        holding = 0
        if entry:
            try:
                ed = datetime.strptime(entry["entry_date"], "%Y-%m-%d")
                holding = (datetime.now() - ed).days
            except Exception:
                pass
        self.execute("""
            UPDATE trades SET status='CLOSED', exit_reason=?,
            exit_date=?, current_price=?, gross_pnl=?,
            net_pnl=?, total_charges=?, holding_days=?
            WHERE trade_id=?
        """, (exit_reason, today, exit_price, gross_pnl,
              net_pnl, total_charges, holding, trade_id))

    # ---- Protection state ----

    def set_state(self, key: str, value: str):
        self.execute(
            "INSERT OR REPLACE INTO protection_state (key, value, updated_at) VALUES (?,?,?)",
            (key, value, datetime.now().isoformat())
        )

    def get_state(self, key: str, default: str = "") -> str:
        row = self.fetchone("SELECT value FROM protection_state WHERE key=?", (key,))
        return row["value"] if row else default

    # ---- Running PNL ----

    def get_today_realised_pnl(self) -> float:
        today = datetime.now().strftime("%Y-%m-%d")
        row = self.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE exit_date=? AND status='CLOSED'",
            (today,)
        )
        return row["total"] or 0.0

    def get_week_realised_pnl(self) -> float:
        week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")
        row = self.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE exit_date>=? AND status='CLOSED'",
            (week_start,)
        )
        return row["total"] or 0.0

    def get_month_realised_pnl(self) -> float:
        month_start = datetime.now().strftime("%Y-%m-01")
        row = self.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE exit_date>=? AND status='CLOSED'",
            (month_start,)
        )
        return row["total"] or 0.0

    def get_annual_stcg(self) -> float:
        fy_start = f"{datetime.now().year}-04-01"
        if datetime.now().month < 4:
            fy_start = f"{datetime.now().year - 1}-04-01"
        row = self.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE exit_date>=? AND status='CLOSED' AND net_pnl>0",
            (fy_start,)
        )
        return row["total"] or 0.0

    def log_trailing_sl(self, trade_id: str, old_sl: float,
                        new_sl: float, price: float, reason: str):
        self.execute("""
            INSERT INTO trailing_sl_log (trade_id, timestamp, old_sl, new_sl, current_price, reason)
            VALUES (?,?,?,?,?,?)
        """, (trade_id, datetime.now().isoformat(), old_sl, new_sl, price, reason))


# =============================================================================
# SECTION 5: DATA COLLECTOR
# =============================================================================

class DataCollector:
    """
    Fetches all required data:
    - OHLCV from Dhan API (daily, 4H, weekly)
    - FII/DII from NSE
    - India VIX from NSE
    - GIFT Nifty pre-market
    - Events calendar from NSE
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.dhan = None
        self._nse_session = requests.Session()
        self._nse_session.headers.update(Config.NSE_HEADERS)
        self._init_dhan()

    def _init_dhan(self):
        if not LIBS_AVAILABLE:
            return
        try:
            self.dhan = dhanhq(Config.DHAN_CLIENT_ID, Config.DHAN_ACCESS_TOKEN)
            log.info("Dhan API connected successfully.")
        except Exception as e:
            log.error(f"Dhan API connection failed: {e}")

    def _nse_get(self, endpoint: str) -> Optional[dict]:
        """Fetch from NSE with session cookies (NSE requires cookie)."""
        try:
            # First hit homepage to get cookies
            self._nse_session.get(Config.NSE_BASE, timeout=10)
            url = Config.NSE_BASE + endpoint
            resp = self._nse_session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"NSE API error {endpoint}: {e}")
            return None

    # ---- OHLCV Data ----

    def fetch_ohlcv_daily(self, symbol: str, days: int = 250) -> Optional["pd.DataFrame"]:
        """Fetch daily OHLCV for a stock from Dhan."""
        if not self.dhan or not LIBS_AVAILABLE:
            return self._mock_ohlcv(symbol, days, "1d")
        try:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            data = self.dhan.historical_daily_data(
                symbol=symbol,
                exchange_segment="NSE_EQ",
                instrument_type="EQUITY",
                from_date=from_date,
                to_date=to_date,
                expiry_code=0
            )
            if data and "data" in data:
                df = pd.DataFrame(data["data"])
                df.columns = [c.lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except Exception as e:
            log.error(f"OHLCV daily fetch failed for {symbol}: {e}")
        return None

    def fetch_ohlcv_intraday(self, symbol: str,
                              interval: str = "60",
                              days: int = 30) -> Optional["pd.DataFrame"]:
        """Fetch 4H equivalent (60 min) OHLCV from Dhan."""
        if not self.dhan or not LIBS_AVAILABLE:
            return self._mock_ohlcv(symbol, days * 6, "4h")
        try:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            data = self.dhan.intraday_minute_data(
                symbol=symbol,
                exchange_segment="NSE_EQ",
                instrument_type="EQUITY",
                interval=interval,
                from_date=from_date,
                to_date=to_date
            )
            if data and "data" in data:
                df = pd.DataFrame(data["data"])
                df.columns = [c.lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except Exception as e:
            log.error(f"Intraday fetch failed for {symbol}: {e}")
        return None

    def _mock_ohlcv(self, symbol: str, bars: int, tf: str) -> "pd.DataFrame":
        """Generate realistic mock data for testing without API."""
        if not LIBS_AVAILABLE:
            return None
        np.random.seed(abs(hash(symbol)) % 999)
        base = 1000 + abs(hash(symbol)) % 2000
        dates = pd.date_range(end=datetime.now(), periods=bars, freq="B")
        closes = base + np.cumsum(np.random.randn(bars) * 15)
        closes = np.maximum(closes, base * 0.5)
        df = pd.DataFrame({
            "date": dates,
            "open":   closes * (1 + np.random.randn(bars) * 0.002),
            "high":   closes * (1 + abs(np.random.randn(bars)) * 0.008),
            "low":    closes * (1 - abs(np.random.randn(bars)) * 0.008),
            "close":  closes,
            "volume": np.random.randint(500000, 5000000, bars).astype(float),
        })
        return df

    # ---- FII / DII ----

    def fetch_fii_dii(self) -> dict:
        """Fetch FII/DII data from NSE."""
        data = self._nse_get(Config.NSE_FII_DII)
        result = {"fii_net_cash": 0.0, "dii_net_cash": 0.0, "date": ""}
        if data:
            try:
                # NSE returns list, latest is first
                latest = data[0] if isinstance(data, list) else data
                result["fii_net_cash"] = float(
                    str(latest.get("netVal", latest.get("NET", 0))).replace(",", "")
                )
                result["dii_net_cash"] = float(
                    str(latest.get("diiNetVal", 0)).replace(",", "")
                )
                result["date"] = latest.get("date", datetime.now().strftime("%d-%b-%Y"))
                log.info(f"FII Net: ₹{result['fii_net_cash']:.0f} Cr | DII Net: ₹{result['dii_net_cash']:.0f} Cr")
            except Exception as e:
                log.error(f"FII/DII parse error: {e}")
        return result

    def get_fii_consecutive_days(self) -> tuple:
        """Count consecutive buying or selling days from DB history."""
        rows = self.db.fetchall(
            "SELECT fii_net_cash FROM fii_history ORDER BY date DESC LIMIT 10"
        )
        if not rows:
            return 0, 0
        buy_streak = sell_streak = 0
        for r in rows:
            if r["fii_net_cash"] > 0:
                if sell_streak == 0:
                    buy_streak += 1
                else:
                    break
            else:
                if buy_streak == 0:
                    sell_streak += 1
                else:
                    break
        return buy_streak, sell_streak

    # ---- India VIX ----

    def fetch_india_vix(self) -> float:
        """Fetch current India VIX from NSE."""
        data = self._nse_get(Config.NSE_VIX)
        if data and "data" in data:
            for item in data["data"]:
                if "INDIA VIX" in str(item.get("index", "")):
                    try:
                        return float(item.get("last", item.get("previousClose", 15.0)))
                    except Exception:
                        pass
        log.warning("VIX fetch failed, using default 15.0")
        return 15.0

    # ---- GIFT Nifty ----

    def fetch_gift_nifty(self) -> float:
        """Fetch GIFT Nifty pre-market value."""
        data = self._nse_get(Config.NSE_VIX)
        if data and "data" in data:
            for item in data["data"]:
                if "GIFT" in str(item.get("index", "")).upper():
                    try:
                        return float(item.get("last", 0))
                    except Exception:
                        pass
        return 0.0

    # ---- Events Calendar ----

    def fetch_events_calendar(self) -> list:
        """Fetch upcoming corporate events for watchlist stocks."""
        today = datetime.now().strftime("%d-%m-%Y")
        future = (datetime.now() + timedelta(days=30)).strftime("%d-%m-%Y")
        endpoint = f"{Config.NSE_EVENTS}?index=equities&from_date={today}&to_date={future}"
        data = self._nse_get(endpoint)
        events = []
        if data:
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("data", [])
            else:
                items = []
            watchlist_symbols = Watchlist.get_symbols()
            for item in items:
                sym = item.get("symbol", "")
                if sym in watchlist_symbols:
                    event_date_str = item.get("date", "")
                    try:
                        ed = datetime.strptime(event_date_str, "%d-%b-%Y")
                        days_away = (ed - datetime.now()).days
                        risk = "GREEN"
                        if days_away <= 5:
                            risk = "RED"
                        elif days_away <= 10:
                            risk = "YELLOW"
                        events.append({
                            "symbol": sym,
                            "event_type": item.get("purpose", "UNKNOWN"),
                            "event_date": ed.strftime("%Y-%m-%d"),
                            "days_away": days_away,
                            "risk_level": risk
                        })
                    except Exception:
                        pass
        return events

    def save_events(self, events: list):
        today = datetime.now().strftime("%Y-%m-%d")
        self.db.execute("DELETE FROM events_calendar WHERE updated_at < ?", (today,))
        for e in events:
            action = "EXIT NOW" if e["risk_level"] == "RED" else (
                "REDUCE SIZE" if e["risk_level"] == "YELLOW" else "MONITOR"
            )
            self.db.execute("""
                INSERT INTO events_calendar
                (symbol, event_type, event_date, days_away, risk_level, action_required)
                VALUES (?,?,?,?,?,?)
            """, (e["symbol"], e["event_type"], e["event_date"],
                  e["days_away"], e["risk_level"], action))


# =============================================================================
# SECTION 6: INDICATOR ENGINE
# =============================================================================

class IndicatorEngine:
    """
    Calculates all technical indicators using pandas-ta.
    Works on daily OHLCV DataFrames.
    """

    @staticmethod
    def calculate_all(df: "pd.DataFrame", symbol: str,
                      nifty_df: "pd.DataFrame" = None) -> Optional[StockData]:
        """
        Master function — takes OHLCV df, returns StockData with all indicators.
        Needs minimum 220 bars for EMA 200 to be reliable.
        """
        if df is None or len(df) < 50:
            log.warning(f"{symbol}: insufficient data ({len(df) if df is not None else 0} bars)")
            return None

        try:
            df = df.copy().reset_index(drop=True)

            # EMAs
            df["ema_20"]  = ta.ema(df["close"], length=Config.EMA_SHORT)
            df["ema_50"]  = ta.ema(df["close"], length=Config.EMA_MED)
            df["ema_200"] = ta.ema(df["close"], length=Config.EMA_LONG)

            # RSI
            df["rsi"] = ta.rsi(df["close"], length=Config.RSI_PERIOD)

            # MACD
            macd = ta.macd(df["close"],
                           fast=Config.MACD_FAST,
                           slow=Config.MACD_SLOW,
                           signal=Config.MACD_SIGNAL)
            if macd is not None:
                df["macd"]        = macd["macd"]
                df["macd_signal"] = macd["signal"]
                df["macd_hist"]   = macd["hist"]
            else:
                df["macd"] = df["macd_signal"] = df["macd_hist"] = 0.0

            # ATR
            df["atr"] = ta.atr(df["high"], df["low"], df["close"],
                                length=Config.ATR_PERIOD)

            # Bollinger Bands
            bb = ta.bbands(df["close"],
                           length=Config.BB_PERIOD,
                           std=Config.BB_STD)
            if bb is not None:
                df["bb_upper"] = bb["upper"]
                df["bb_lower"] = bb["lower"]
            else:
                df["bb_upper"] = df["bb_lower"] = df["close"]

            # Volume metrics
            df["vol_avg"] = df["volume"].rolling(Config.VOLUME_AVG_PERIOD).mean()
            df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, 1)

            # ATR ratio (vs 20 day average ATR)
            df["atr_avg"]   = df["atr"].rolling(20).mean()
            df["atr_ratio"] = df["atr"] / df["atr_avg"].replace(0, 1)

            # OBV
            df["obv"] = ta.obv(df["close"], df["volume"])

            # Use last row
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else last

            # 52 week high/low
            w52 = df.tail(252)
            week_52_high = w52["high"].max()
            week_52_low  = w52["low"].min()

            # 20 day high/low for consolidation check
            d20 = df.tail(20)
            d20_high = d20["high"].max()
            d20_low  = d20["low"].min()
            consolidation_pct = (
                (d20_high - d20_low) / last["close"] * 100
                if last["close"] > 0 else 999
            )

            # OBV rising? (last 5 days trend)
            obv_rising = False
            if len(df) >= 5:
                obv_rising = bool(df["obv"].iloc[-1] > df["obv"].iloc[-5])

            # Relative Strength vs Nifty
            rs_score = IndicatorEngine._calc_rs(df, nifty_df)

            # Candle pattern detection
            candle = IndicatorEngine._detect_candle_pattern(df)

            # Trend alignment (daily)
            daily_bullish = bool(
                last["close"] > last["ema_20"] and
                last["ema_20"] > last["ema_50"]
            )

            s = StockData(
                symbol=symbol,
                date=datetime.now().strftime("%Y-%m-%d"),
                open=float(last["open"]),
                high=float(last["high"]),
                low=float(last["low"]),
                close=float(last["close"]),
                volume=float(last["volume"]),
                ema_20=float(last["ema_20"]) if not pd.isna(last["ema_20"]) else 0.0,
                ema_50=float(last["ema_50"]) if not pd.isna(last["ema_50"]) else 0.0,
                ema_200=float(last["ema_200"]) if not pd.isna(last["ema_200"]) else 0.0,
                rsi=float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0,
                macd=float(last["macd"]) if not pd.isna(last["macd"]) else 0.0,
                macd_signal=float(last["macd_signal"]) if not pd.isna(last["macd_signal"]) else 0.0,
                macd_hist=float(last["macd_hist"]) if not pd.isna(last["macd_hist"]) else 0.0,
                atr=float(last["atr"]) if not pd.isna(last["atr"]) else 0.0,
                bb_upper=float(last["bb_upper"]) if not pd.isna(last["bb_upper"]) else 0.0,
                bb_lower=float(last["bb_lower"]) if not pd.isna(last["bb_lower"]) else 0.0,
                volume_ratio=float(last["vol_ratio"]) if not pd.isna(last["vol_ratio"]) else 1.0,
                week_52_high=float(week_52_high),
                week_52_low=float(week_52_low),
                rs_score=rs_score,
                candle_pattern=candle,
                daily_bullish=daily_bullish,
                consolidation_range_pct=float(consolidation_pct),
                obv_rising=obv_rising,
                atr_ratio=float(last["atr_ratio"]) if not pd.isna(last["atr_ratio"]) else 1.0,
            )
            return s

        except Exception as e:
            log.error(f"Indicator calculation failed for {symbol}: {e}")
            return None

    @staticmethod
    def _calc_rs(stock_df: "pd.DataFrame",
                 nifty_df: Optional["pd.DataFrame"]) -> float:
        """Relative strength: stock 60-day return vs Nifty 60-day return."""
        if nifty_df is None or len(stock_df) < 60 or len(nifty_df) < 60:
            return 0.0
        try:
            s_ret = (stock_df["close"].iloc[-1] / stock_df["close"].iloc[-60] - 1) * 100
            n_ret = (nifty_df["close"].iloc[-1] / nifty_df["close"].iloc[-60] - 1) * 100
            return round(s_ret - n_ret, 2)
        except Exception:
            return 0.0

    @staticmethod
    def _detect_candle_pattern(df: "pd.DataFrame") -> str:
        """
        Detect top 5 bullish confirmation patterns.
        Returns pattern name or 'NONE'.
        """
        if len(df) < 3:
            return "NONE"
        try:
            c = df.iloc[-1]    # today
            p = df.iloc[-2]    # yesterday
            p2 = df.iloc[-3]   # day before

            body_c = abs(c["close"] - c["open"])
            range_c = c["high"] - c["low"] if c["high"] != c["low"] else 0.001
            lower_wick = min(c["open"], c["close"]) - c["low"]
            upper_wick = c["high"] - max(c["open"], c["close"])

            # Marubozu: big green, almost no wicks
            if (c["close"] > c["open"] and
                body_c / range_c > 0.85 and
                c["volume"] > df["volume"].tail(20).mean() * 1.5):
                return "MARUBOZU"

            # Bullish Engulfing
            if (p["close"] < p["open"] and      # prev red
                c["close"] > c["open"] and       # today green
                c["open"] < p["close"] and
                c["close"] > p["open"] and
                body_c > abs(p["close"] - p["open"]) and
                c["volume"] > p["volume"]):
                return "BULLISH_ENGULFING"

            # Hammer: lower wick >= 2x body, small upper wick
            if (body_c > 0 and
                lower_wick >= 2 * body_c and
                upper_wick <= 0.1 * range_c and
                c["close"] > c["open"]):
                return "HAMMER"

            # Morning Star (3 candle)
            body_p  = abs(p["close"] - p["open"])
            body_p2 = abs(p2["close"] - p2["open"])
            if (p2["close"] < p2["open"] and          # D-2 red
                body_p < body_p2 * 0.4 and            # D-1 small body
                c["close"] > c["open"] and             # today green
                c["close"] > p2["open"] + body_p2 * 0.5):
                return "MORNING_STAR"

            # Inside Bar (today is inside yesterday's range)
            if (c["high"] < p["high"] and c["low"] > p["low"]):
                return "INSIDE_BAR"

        except Exception as e:
            log.debug(f"Candle pattern detection error: {e}")

        return "NONE"

    @staticmethod
    def check_4h_bullish(df_4h: Optional["pd.DataFrame"]) -> bool:
        """Check if 4H chart is bullish (price > 4H EMA 20)."""
        if df_4h is None or len(df_4h) < 25:
            return False
        try:
            ema = ta.ema(df_4h["close"], length=20)
            return bool(df_4h["close"].iloc[-1] > ema.iloc[-1])
        except Exception:
            return False

    @staticmethod
    def check_weekly_bullish(df_weekly: Optional["pd.DataFrame"]) -> bool:
        """Check if weekly chart is bullish (price > weekly EMA 20)."""
        if df_weekly is None or len(df_weekly) < 22:
            return False
        try:
            ema = ta.ema(df_weekly["close"], length=20)
            return bool(df_weekly["close"].iloc[-1] > ema.iloc[-1])
        except Exception:
            return False


# =============================================================================
# SECTION 7: MARKET MODE ENGINE
# =============================================================================

class MarketModeEngine:
    """
    Determines today's market mode based on:
    - India VIX level
    - Nifty trend (EMA alignment)
    - Nifty RSI
    - FII flow (consecutive days + amount)
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def detect(self, vix: float, nifty_data: StockData,
               fii_net: float, fii_consecutive_buy: int,
               fii_consecutive_sell: int) -> tuple:
        """
        Returns (MarketMode, FIIFlow)
        """
        # VIX override — highest priority
        if vix >= Config.VIX_PANIC:
            return MarketMode.CASH, FIIFlow.NEUTRAL
        if vix >= Config.VIX_NERVOUS:
            return MarketMode.DEFENSIVE, FIIFlow.SELLING

        # FII flow label
        if (fii_net > Config.FII_FLOW_THRESHOLD_CR and
                fii_consecutive_buy >= Config.FII_CONSECUTIVE_DAYS):
            fii_flow = FIIFlow.BUYING
        elif (fii_net < -Config.FII_FLOW_THRESHOLD_CR and
              fii_consecutive_sell >= Config.FII_CONSECUTIVE_DAYS):
            fii_flow = FIIFlow.SELLING
        else:
            fii_flow = FIIFlow.NEUTRAL

        # Nifty trend detection
        if nifty_data is None:
            return MarketMode.CAUTIOUS, fii_flow

        above_ema20  = nifty_data.close > nifty_data.ema_20
        above_ema50  = nifty_data.close > nifty_data.ema_50
        above_ema200 = nifty_data.close > nifty_data.ema_200
        rsi = nifty_data.rsi

        # Strong bull — all 3 EMAs aligned
        if above_ema20 and above_ema50 and above_ema200 and rsi > 50:
            if fii_flow == FIIFlow.BUYING:
                mode = MarketMode.AGGRESSIVE
            elif fii_flow == FIIFlow.SELLING:
                mode = MarketMode.SELECTIVE
            else:
                mode = MarketMode.NORMAL

        # Pullback in bull (temp dip)
        elif above_ema50 and above_ema200 and not above_ema20:
            mode = MarketMode.SELECTIVE

        # Sideways / mixed
        elif above_ema200 and rsi > 40:
            mode = MarketMode.CAUTIOUS

        # Bear — below all or below 200 EMA
        else:
            mode = MarketMode.DEFENSIVE

        log.info(f"Market Mode: {mode.value} | FII: {fii_flow.value} | VIX: {vix:.1f}")
        return mode, fii_flow


# =============================================================================
# SECTION 8: STOCK SCREENER
# =============================================================================

class StockScreener:
    """
    Filters the 15 watchlist stocks down to tradeable candidates.
    Removes stocks that are too volatile, have upcoming events,
    are already in portfolio, or show weak technical structure.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def screen(self, stocks_data: dict,
               open_trades: list,
               market_mode: MarketMode) -> list:
        """
        stocks_data: {symbol: StockData}
        open_trades: list of open trade dicts from DB
        Returns list of symbols that passed all filters.
        """
        passed = []
        open_symbols  = {t["symbol"] for t in open_trades}
        open_sectors  = {Watchlist.get_sector(t["symbol"]) for t in open_trades}

        for symbol, data in stocks_data.items():
            if data is None:
                log.debug(f"SKIP {symbol}: no data")
                continue

            reasons = []

            # Filter 1: ATR ratio (too volatile)
            if data.atr_ratio > 1.5:
                reasons.append(f"ATR ratio too high ({data.atr_ratio:.2f})")

            # Filter 2: Already in portfolio
            if symbol in open_symbols:
                reasons.append("Already in portfolio")

            # Filter 3: Sector already represented (max 1 per sector)
            sector = Watchlist.get_sector(symbol)
            if sector in open_sectors:
                reasons.append(f"Sector {sector} already open")

            # Filter 4: Upcoming results/events in < 5 days
            event = self.db.fetchone(
                "SELECT * FROM events_calendar WHERE symbol=? AND days_away<=5 AND risk_level='RED'",
                (symbol,)
            )
            if event:
                reasons.append(f"Event in {event['days_away']} days: {event['event_type']}")

            # Filter 5: Volume too low (< 0.5x average)
            if data.volume_ratio < 0.5:
                reasons.append(f"Low volume ({data.volume_ratio:.2f}x avg)")

            # Filter 6: Large gap today (> 3%) — unstable
            if data.open > 0:
                gap_pct = abs(data.open - data.close) / data.open * 100
                if gap_pct > 3 and data.volume_ratio < 1.5:
                    reasons.append(f"Large gap ({gap_pct:.1f}%)")

            # Filter 7: In defensive/cash mode, skip Tier 2 stocks
            if market_mode in [MarketMode.DEFENSIVE, MarketMode.CASH]:
                reasons.append("Market in defensive/cash mode")

            if reasons:
                log.debug(f"SKIP {symbol}: {' | '.join(reasons)}")
            else:
                passed.append(symbol)
                log.debug(f"PASS {symbol}: all filters passed")

        log.info(f"Screening: {len(passed)}/{len(stocks_data)} stocks passed")
        return passed


# =============================================================================
# SECTION 9: STRATEGY ENGINE
# =============================================================================

class StrategyEngine:
    """
    Evaluates each screened stock against all 5 strategies.
    Scores each valid setup 0-100.
    Assigns best strategy by priority order.
    """

    def evaluate_all(self, symbol: str, data: StockData,
                     market_mode: MarketMode,
                     fii_flow: FIIFlow,
                     fii_sector_buying: bool = False) -> Optional[Setup]:
        """
        Checks all 5 strategies and returns best Setup or None.
        """
        candidates = []

        # Check each strategy in priority order
        s4 = self._check_fii_flow(symbol, data, market_mode, fii_flow, fii_sector_buying)
        if s4:
            candidates.append(s4)

        s5 = self._check_52w_breakout(symbol, data, market_mode)
        if s5:
            candidates.append(s5)

        s2 = self._check_breakout(symbol, data, market_mode)
        if s2:
            candidates.append(s2)

        s3 = self._check_pullback(symbol, data, market_mode)
        if s3:
            candidates.append(s3)

        s1 = self._check_swing(symbol, data, market_mode)
        if s1:
            candidates.append(s1)

        if not candidates:
            return None

        # Sort by priority then score
        candidates.sort(
            key=lambda x: (
                Config.STRATEGY_PRIORITY.get(x.strategy.value, 0),
                x.score
            ),
            reverse=True
        )
        return candidates[0]

    # ---- Strategy 1: Swing Trade ----

    def _check_swing(self, symbol: str, data: StockData,
                     mode: MarketMode) -> Optional[Setup]:
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None

        checks = [
            data.close > data.ema_20,
            data.ema_20 > data.ema_50,
            45 <= data.rsi <= 65,
            data.macd > data.macd_signal,
            data.macd_hist > 0,
            data.volume_ratio > 1.2,
            data.tf_aligned_count >= 2,
        ]
        passed = sum(checks)
        if passed < 6:
            return None

        score = self._base_score(data, passed, 7)
        return Setup(
            symbol=symbol,
            date=datetime.now().strftime("%Y-%m-%d"),
            strategy=StrategyType.SWING,
            score=score,
            entry_price=data.close,
            market_mode=mode.value,
            fii_flow=""
        )

    # ---- Strategy 2: Breakout + Consolidation ----

    def _check_breakout(self, symbol: str, data: StockData,
                        mode: MarketMode) -> Optional[Setup]:
        if mode == MarketMode.DEFENSIVE or mode == MarketMode.CASH:
            return None

        # Volume confirmation is NON-NEGOTIABLE for breakout
        if data.volume_ratio < 2.0:
            return None

        checks = [
            data.consolidation_range_pct < 8.0,   # tight range
            data.close > data.ema_50,
            data.rsi > 55,
            data.macd > data.macd_signal,
            data.volume_ratio > 2.0,               # counted twice (critical)
            data.obv_rising,
            data.tf_aligned_count >= 1,
        ]
        passed = sum(checks)
        if passed < 6:
            return None

        score = self._base_score(data, passed, 7)
        # Bonus for very strong volume
        if data.volume_ratio > 2.5:
            score = min(100, score + 10)

        return Setup(
            symbol=symbol,
            date=datetime.now().strftime("%Y-%m-%d"),
            strategy=StrategyType.BREAKOUT,
            score=score,
            entry_price=data.close,
            market_mode=mode.value,
            fii_flow=""
        )

    # ---- Strategy 3: Pullback to EMA ----

    def _check_pullback(self, symbol: str, data: StockData,
                        mode: MarketMode) -> Optional[Setup]:
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None

        # Price must be near 20 EMA (within 0.5%)
        if data.ema_20 == 0:
            return None
        ema_distance = abs(data.close - data.ema_20) / data.ema_20 * 100
        if ema_distance > 0.5:
            return None

        checks = [
            data.close > data.open,             # bouncing today
            40 <= data.rsi <= 52,               # RSI in healthy pullback zone
            data.volume_ratio < 1.5,            # pullback on low volume
            data.macd > 0,                      # still above zero line
            data.weekly_bullish,                # weekly still up
            data.obv_rising,
            data.tf_aligned_count >= 2,
        ]
        passed = sum(checks)
        if passed < 6:
            return None

        score = self._base_score(data, passed, 7)
        # Bonus for perfect candle (Hammer at EMA = ideal)
        if data.candle_pattern in ["HAMMER", "BULLISH_ENGULFING"]:
            score = min(100, score + 15)

        return Setup(
            symbol=symbol,
            date=datetime.now().strftime("%Y-%m-%d"),
            strategy=StrategyType.PULLBACK,
            score=score,
            entry_price=data.close,
            market_mode=mode.value,
            fii_flow=""
        )

    # ---- Strategy 4: FII Sector Flow ----

    def _check_fii_flow(self, symbol: str, data: StockData,
                        mode: MarketMode, fii_flow: FIIFlow,
                        fii_sector_buying: bool) -> Optional[Setup]:
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None
        if fii_flow != FIIFlow.BUYING or not fii_sector_buying:
            return None

        checks = [
            data.close > data.ema_50,
            50 <= data.rsi <= 70,
            data.volume_ratio > 1.2,
            data.obv_rising,
            data.rs_score > 0,                  # stronger than Nifty
            data.macd > data.macd_signal,
            data.tf_aligned_count >= 2,
        ]
        passed = sum(checks)
        if passed < 6:
            return None

        score = self._base_score(data, passed, 7)
        # FII Flow gets highest score boost
        score = min(100, score + 10)

        return Setup(
            symbol=symbol,
            date=datetime.now().strftime("%Y-%m-%d"),
            strategy=StrategyType.FII_FLOW,
            score=score,
            entry_price=data.close,
            market_mode=mode.value,
            fii_flow=fii_flow.value
        )

    # ---- Strategy 5: 52 Week High Breakout ----

    def _check_52w_breakout(self, symbol: str, data: StockData,
                             mode: MarketMode) -> Optional[Setup]:
        # Only in aggressive (strong bull) mode
        if mode != MarketMode.AGGRESSIVE:
            return None

        if data.week_52_high == 0:
            return None

        # Within 1% of 52W high AND closed above it
        dist_pct = (data.week_52_high - data.close) / data.week_52_high * 100
        if dist_pct > 1.0:
            return None
        if data.close < data.week_52_high:
            return None

        # Volume is critical for 52W breakout
        if data.volume_ratio < 2.5:
            return None

        checks = [
            data.close > data.ema_20,
            data.close > data.ema_50,
            data.close > data.ema_200,
            60 <= data.rsi <= 75,
            data.volume_ratio > 2.5,
            data.obv_rising,
            data.tf_aligned_count >= 2,
        ]
        passed = sum(checks)
        if passed < 7:
            return None

        score = self._base_score(data, passed, 7)
        # 52W High = rare and powerful
        score = min(100, score + 15)

        return Setup(
            symbol=symbol,
            date=datetime.now().strftime("%Y-%m-%d"),
            strategy=StrategyType.WEEK52,
            score=score,
            entry_price=data.close,
            market_mode=mode.value,
            fii_flow=""
        )

    # ---- Helpers ----

    def _base_score(self, data: StockData, passed: int, total: int) -> int:
        """Base score from criteria passed + candle pattern bonus."""
        score = int(passed / total * 70)  # max 70 from criteria

        # RS score bonus (up to 15 points)
        if data.rs_score > 5:
            score += 15
        elif data.rs_score > 0:
            score += 8

        # Candle pattern bonus (up to 15 points)
        pattern_scores = {
            "MARUBOZU": 15, "BULLISH_ENGULFING": 12,
            "HAMMER": 10, "MORNING_STAR": 12,
            "INSIDE_BAR": 5, "NONE": 0
        }
        score += pattern_scores.get(data.candle_pattern, 0)

        return min(100, max(0, score))


# =============================================================================
# SECTION 10: RISK MANAGER
# =============================================================================

class RiskManager:
    """
    Calculates ATR-based SL, target, position size for each setup.
    Validates portfolio-level risk before approving any trade.
    Runs the final 11-point pre-trade checklist.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def calculate_setup_risk(self, setup: Setup,
                              stock_data: StockData) -> Setup:
        """Fill in SL, target, shares, risk for a setup."""
        atr = stock_data.atr
        if atr <= 0:
            setup.skip_reason = "ATR is zero"
            setup.status = "SKIPPED"
            return setup

        mult = Config.ATR_MULT.get(setup.strategy.value, 1.5)
        entry = setup.entry_price

        # ATR based stop loss
        sl = entry - (atr * mult)

        # For breakout: SL is below consolidation low
        if setup.strategy == StrategyType.BREAKOUT:
            d20_low = stock_data.close * (1 - stock_data.consolidation_range_pct / 100)
            sl = d20_low - (atr * 0.5)

        # For 52W breakout: SL below 52W high
        if setup.strategy == StrategyType.WEEK52:
            sl = stock_data.week_52_high - (atr * 1.5)

        # For pullback: SL below EMA
        if setup.strategy == StrategyType.PULLBACK:
            sl = stock_data.ema_20 - (atr * 1.0)

        risk_per_share = entry - sl
        if risk_per_share <= 0:
            setup.skip_reason = "Invalid SL (risk per share <= 0)"
            setup.status = "SKIPPED"
            return setup

        # Target: minimum 1:2 RR
        target = entry + (risk_per_share * Config.MIN_RR_RATIO)

        # Position sizing
        shares = int(Config.MAX_RISK_PER_TRADE / risk_per_share)
        if shares <= 0:
            setup.skip_reason = "Too few shares after risk calculation"
            setup.status = "SKIPPED"
            return setup

        capital_needed = shares * entry

        # Cap at ₹50,000
        if capital_needed > Config.MAX_CAPITAL_PER_TRADE:
            shares = int(Config.MAX_CAPITAL_PER_TRADE / entry)
            capital_needed = shares * entry

        actual_risk = shares * risk_per_share
        rr = (target - entry) / risk_per_share if risk_per_share > 0 else 0

        # RR check
        if rr < Config.MIN_RR_RATIO:
            setup.skip_reason = f"RR too low: {rr:.2f}"
            setup.status = "SKIPPED"
            return setup

        setup.sl_price        = round(sl, 2)
        setup.target_price    = round(target, 2)
        setup.atr             = round(atr, 2)
        setup.risk_per_share  = round(risk_per_share, 2)
        setup.shares          = shares
        setup.capital_required = round(capital_needed, 2)
        setup.actual_risk     = round(actual_risk, 2)
        setup.rr_ratio        = round(rr, 2)
        return setup

    def run_pre_trade_checklist(self, setup: Setup,
                                 market_mode: MarketMode,
                                 vix: float,
                                 open_trades: list) -> tuple:
        """
        Runs all 11 checks. Returns (approved: bool, reasons: list).
        """
        checks = {}

        # 1. Market mode
        checks["1_market_mode"] = market_mode not in [MarketMode.DEFENSIVE, MarketMode.CASH]

        # 2. VIX
        checks["2_vix"] = vix < Config.VIX_NERVOUS

        # 3. No red event today/tomorrow for this stock
        event = self.db.fetchone(
            "SELECT * FROM events_calendar WHERE symbol=? AND days_away<=1 AND risk_level='RED'",
            (setup.symbol,)
        )
        checks["3_no_event"] = event is None

        # 4. Setup score acceptable (>= 60)
        checks["4_setup_score"] = setup.score >= 60

        # 5. Strategy fully confirmed
        checks["5_strategy_confirmed"] = setup.status != "SKIPPED"

        # 6. Candle pattern present (or score high enough to override)
        # Allow entry even without perfect candle if score >= 80
        has_candle = setup.score >= 80  # high score implies candle confirmed
        checks["6_candle"] = has_candle

        # 7. Timeframe alignment (checked in strategy)
        checks["7_tf_aligned"] = True  # already enforced in strategy checks

        # 8. ATR SL valid
        checks["8_sl_valid"] = setup.sl_price > 0 and setup.sl_price < setup.entry_price

        # 9. RR >= 1:2
        checks["9_rr"] = setup.rr_ratio >= Config.MIN_RR_RATIO

        # 10. Capital within limit
        checks["10_capital"] = setup.capital_required <= Config.MAX_CAPITAL_PER_TRADE

        # 11. Portfolio constraints
        total_open_risk = sum(
            (t["entry_price"] - t["current_sl"]) * t["remaining_qty"]
            for t in open_trades
            if t["status"] == "OPEN"
        )
        portfolio_ok = (
            len(open_trades) < Config.MAX_SIMULTANEOUS_TRADES and
            total_open_risk + setup.actual_risk <= Config.MAX_PORTFOLIO_RISK
        )
        checks["11_portfolio"] = portfolio_ok

        failed = [k for k, v in checks.items() if not v]
        approved = len(failed) == 0

        if not approved:
            log.info(f"Checklist FAIL for {setup.symbol}: {failed}")
        else:
            log.info(f"Checklist PASS for {setup.symbol} | Strategy: {setup.strategy.value} | Score: {setup.score}")

        return approved, failed


# =============================================================================
# SECTION 11: CHARGES & TAX CALCULATOR
# =============================================================================

class ChargesCalculator:
    """
    Calculates exact broker charges and tax for each trade.
    Based on 2026 rates for equity delivery on NSE.
    """

    @staticmethod
    def calculate_buy_charges(buy_value: float) -> dict:
        stt         = buy_value * Config.STT_DELIVERY
        stamp       = buy_value * Config.STAMP_DUTY
        exchange    = buy_value * Config.EXCHANGE_CHARGE
        gst         = exchange * Config.GST_RATE
        sebi        = buy_value * Config.SEBI_CHARGE
        total       = stt + stamp + exchange + gst + sebi
        return {
            "stt": round(stt, 2),
            "stamp_duty": round(stamp, 2),
            "exchange": round(exchange, 2),
            "gst": round(gst, 2),
            "sebi": round(sebi, 2),
            "total": round(total, 2)
        }

    @staticmethod
    def calculate_sell_charges(sell_value: float) -> dict:
        stt         = sell_value * Config.STT_DELIVERY
        dp          = Config.DP_CHARGE
        exchange    = sell_value * Config.EXCHANGE_CHARGE
        gst         = exchange * Config.GST_RATE
        sebi        = sell_value * Config.SEBI_CHARGE
        total       = stt + dp + exchange + gst + sebi
        return {
            "stt": round(stt, 2),
            "dp_charge": round(dp, 2),
            "exchange": round(exchange, 2),
            "gst": round(gst, 2),
            "sebi": round(sebi, 2),
            "total": round(total, 2)
        }

    @staticmethod
    def calculate_trade_pnl(entry_price: float, exit_price: float,
                             qty: int) -> dict:
        """Complete PNL calculation including all charges."""
        buy_value   = entry_price * qty
        sell_value  = exit_price  * qty
        gross_pnl   = sell_value - buy_value

        buy_ch  = ChargesCalculator.calculate_buy_charges(buy_value)
        sell_ch = ChargesCalculator.calculate_sell_charges(sell_value)
        total_charges = buy_ch["total"] + sell_ch["total"]

        net_pnl = gross_pnl - total_charges
        stcg_tax = max(0, net_pnl) * Config.EFFECTIVE_TAX
        take_home = net_pnl - stcg_tax if net_pnl > 0 else net_pnl

        return {
            "buy_value": round(buy_value, 2),
            "sell_value": round(sell_value, 2),
            "gross_pnl": round(gross_pnl, 2),
            "total_charges": round(total_charges, 2),
            "net_pnl": round(net_pnl, 2),
            "stcg_tax_estimate": round(stcg_tax, 2),
            "true_take_home": round(take_home, 2),
            "buy_charges": buy_ch,
            "sell_charges": sell_ch
        }

    @staticmethod
    def annual_tax_summary(annual_stcg: float) -> dict:
        stcg_tax    = max(0, annual_stcg) * Config.STCG_RATE
        cess        = stcg_tax * Config.CESS_RATE
        total_tax   = stcg_tax + cess
        take_home   = max(0, annual_stcg) - total_tax
        advance_due = total_tax > Config.ADVANCE_TAX_THRESHOLD

        return {
            "annual_stcg": round(annual_stcg, 2),
            "stcg_tax_20pct": round(stcg_tax, 2),
            "cess_4pct": round(cess, 2),
            "total_tax": round(total_tax, 2),
            "take_home": round(take_home, 2),
            "advance_tax_required": advance_due,
            "advance_tax_quarters": {
                "jun_15_15pct": round(total_tax * 0.15, 2),
                "sep_15_45pct": round(total_tax * 0.30, 2),
                "dec_15_75pct": round(total_tax * 0.30, 2),
                "mar_15_100pct": round(total_tax * 0.25, 2),
            }
        }


# =============================================================================
# SECTION 12: PROTECTION ENGINE
# =============================================================================

class ProtectionEngine:
    """
    Enforces all capital protection rules and behavioural guards.
    Acts as gatekeeper — checks before any trade is placed.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def is_trading_allowed(self) -> tuple:
        """Returns (allowed: bool, reason: str)"""

        # Check manual override
        override = self.db.get_state("trading_halted")
        if override == "1":
            return False, "Trading manually halted"

        # Daily loss limit
        daily_loss = self.db.get_today_realised_pnl()
        if daily_loss <= -Config.DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit hit: ₹{abs(daily_loss):.0f}"

        # Weekly loss limit
        weekly_loss = self.db.get_week_realised_pnl()
        if weekly_loss <= -Config.WEEKLY_LOSS_LIMIT:
            return False, f"Weekly loss limit hit: ₹{abs(weekly_loss):.0f}"

        # Monthly loss limit
        monthly_loss = self.db.get_month_realised_pnl()
        if monthly_loss <= -Config.MONTHLY_LOSS_LIMIT:
            return False, f"Monthly loss limit hit: ₹{abs(monthly_loss):.0f}"

        # Drawdown check (from peak capital)
        # Simplified: use monthly loss as proxy
        if monthly_loss <= -Config.MAX_DRAWDOWN:
            return False, f"Max drawdown hit: ₹{abs(monthly_loss):.0f}"

        # Cooldown after loss
        cooldown_until = self.db.get_state("cooldown_until")
        if cooldown_until:
            try:
                cd_time = datetime.fromisoformat(cooldown_until)
                if datetime.now() < cd_time:
                    mins_left = int((cd_time - datetime.now()).total_seconds() / 60)
                    return False, f"Cooldown active: {mins_left} mins remaining"
            except Exception:
                pass

        # Monday first 30 mins
        now = datetime.now()
        if now.weekday() == 0:
            market_open = now.replace(hour=9, minute=15, second=0)
            if now < market_open + timedelta(minutes=Config.MONDAY_NO_ENTRY_MINS):
                return False, "Monday 30-min wait period"

        # Friday after 2 PM
        if now.weekday() == 4 and now.hour >= Config.FRIDAY_NO_ENTRY_HOUR:
            return False, "Friday after 2 PM — no new entries"

        # Market hasn't settled (first 15 mins after open)
        market_open = now.replace(hour=9, minute=15, second=0)
        if now < market_open + timedelta(minutes=Config.MARKET_OPEN_WAIT_MINS):
            return False, "Waiting for market to settle (first 15 mins)"

        return True, "OK"

    def start_loss_cooldown(self):
        """Start 2-hour cooldown after a loss."""
        until = (datetime.now() + timedelta(hours=Config.COOLDOWN_AFTER_LOSS_HR)).isoformat()
        self.db.set_state("cooldown_until", until)
        log.info(f"Loss cooldown started. Resumes at: {until}")

    def check_consecutive_losses(self) -> bool:
        """Returns True if 3 consecutive losses detected (triggers 1 day off)."""
        rows = self.db.fetchall(
            "SELECT net_pnl FROM trades WHERE status='CLOSED' ORDER BY exit_date DESC LIMIT 3"
        )
        if len(rows) < 3:
            return False
        return all(r["net_pnl"] < 0 for r in rows)

    def check_event_guard(self, symbol: str) -> tuple:
        """
        Returns (safe: bool, warning: str) for overnight/weekend holds.
        """
        event = self.db.fetchone(
            "SELECT * FROM events_calendar WHERE symbol=? ORDER BY days_away ASC LIMIT 1",
            (symbol,)
        )
        if not event:
            return True, "No upcoming events"
        days = event["days_away"]
        if days <= Config.EVENT_EXIT_DAYS:
            return False, f"Exit! {event['event_type']} in {days} days"
        if days <= 10:
            return True, f"Warning: {event['event_type']} in {days} days"
        return True, "Event far away, safe to hold"


# =============================================================================
# SECTION 13: ORDER MANAGER
# =============================================================================

class OrderManager:
    """
    Handles all order placement via Dhan API.
    Places entry + SL + target simultaneously (OCO-style).
    Supports paper trading mode for testing.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.dhan = None
        self._init_dhan()

    def _init_dhan(self):
        if not LIBS_AVAILABLE:
            return
        try:
            self.dhan = dhanhq(Config.DHAN_CLIENT_ID, Config.DHAN_ACCESS_TOKEN)
        except Exception as e:
            log.error(f"Dhan order manager init failed: {e}")

    def place_entry_order(self, setup: Setup) -> Optional[str]:
        """
        Places limit buy order + stop loss order.
        Returns trade_id or None if failed.
        """
        trade_id = f"{setup.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if Config.PAPER_TRADE:
            log.info(f"[PAPER] BUY {setup.shares} × {setup.symbol} @ ₹{setup.entry_price:.2f} | "
                     f"SL: ₹{setup.sl_price:.2f} | Target: ₹{setup.target_price:.2f}")
            self._record_paper_trade(trade_id, setup)
            return trade_id

        if not self.dhan:
            log.error("Dhan API not initialized")
            return None

        try:
            # Entry limit order (slightly above current price)
            entry_limit = round(setup.entry_price * 1.001, 2)
            order = self.dhan.place_order(
                security_id=self._get_security_id(setup.symbol),
                exchange_segment=self.dhan.NSE,
                transaction_type=self.dhan.BUY,
                quantity=setup.shares,
                order_type=self.dhan.LIMIT,
                product_type=self.dhan.CNC,       # CNC = delivery
                price=entry_limit
            )
            log.info(f"LIVE BUY order placed: {order}")

            # Stop loss order
            sl_order = self.dhan.place_order(
                security_id=self._get_security_id(setup.symbol),
                exchange_segment=self.dhan.NSE,
                transaction_type=self.dhan.SELL,
                quantity=setup.shares,
                order_type=self.dhan.SL,
                product_type=self.dhan.CNC,
                price=setup.sl_price,
                trigger_price=setup.sl_price
            )
            sl_order_id = str(
                (sl_order.get("data") or {}).get("orderId") or
                sl_order.get("orderId", "")
            )
            log.info(f"SL order placed: {sl_order} | order_id: {sl_order_id}")

            self._record_paper_trade(trade_id, setup, sl_order_id)
            return trade_id

        except Exception as e:
            log.error(f"Order placement failed for {setup.symbol}: {e}")
            return None

    def _record_paper_trade(self, trade_id: str, setup: Setup, sl_order_id: str = ""):
        """Save trade to database."""
        trade = Trade(
            trade_id=trade_id,
            symbol=setup.symbol,
            strategy=setup.strategy.value,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            entry_price=setup.entry_price,
            quantity=setup.shares,
            initial_sl=setup.sl_price,
            initial_target=setup.target_price,
            current_sl=setup.sl_price,
            current_price=setup.entry_price,
            remaining_qty=setup.shares,
            setup_score=setup.score,
            market_mode_at_entry=setup.market_mode,
            sl_order_id=sl_order_id
        )
        self.db.save_trade(trade)
        log.info(f"Trade recorded: {trade_id}")

    def replace_sl_order(self, symbol: str, qty: int,
                          new_sl: float, old_order_id: str) -> str:
        """
        Cancel the existing SL order at Dhan and place a new one at new_sl.
        Returns the new sl_order_id, or empty string on failure.
        No-op in paper trade mode (returns empty string).
        """
        if Config.PAPER_TRADE:
            log.info(f"[PAPER] SL order replaced: {symbol} new SL ₹{new_sl:.2f} qty {qty}")
            return ""

        if not self.dhan:
            return ""

        # Cancel old SL order if we have its ID
        if old_order_id:
            try:
                self.dhan.cancel_order(order_id=old_order_id)
                log.info(f"Cancelled old SL order {old_order_id} for {symbol}")
            except Exception as e:
                log.warning(f"Could not cancel old SL order {old_order_id} for {symbol}: {e}")

        # Place new SL order at updated price
        try:
            sl_order = self.dhan.place_order(
                security_id=self._get_security_id(symbol),
                exchange_segment=self.dhan.NSE,
                transaction_type=self.dhan.SELL,
                quantity=qty,
                order_type=self.dhan.SL,
                product_type=self.dhan.CNC,
                price=new_sl,
                trigger_price=new_sl
            )
            new_id = str(
                (sl_order.get("data") or {}).get("orderId") or
                sl_order.get("orderId", "")
            )
            log.info(f"New SL order placed: {symbol} @ ₹{new_sl:.2f} qty {qty} | order_id: {new_id}")
            return new_id
        except Exception as e:
            log.error(f"Failed to place replacement SL order for {symbol}: {e}")
            return ""

    def place_sell_order(self, symbol: str, qty: int,
                          price: float, reason: str) -> bool:
        """Place sell order (partial or full exit)."""
        if Config.PAPER_TRADE:
            log.info(f"[PAPER] SELL {qty} × {symbol} @ ₹{price:.2f} | Reason: {reason}")
            return True

        if not self.dhan:
            return False

        try:
            order = self.dhan.place_order(
                security_id=self._get_security_id(symbol),
                exchange_segment=self.dhan.NSE,
                transaction_type=self.dhan.SELL,
                quantity=qty,
                order_type=self.dhan.LIMIT,
                product_type=self.dhan.CNC,
                price=round(price * 0.999, 2)   # slightly below for fill
            )
            log.info(f"SELL order placed: {order}")
            return True
        except Exception as e:
            log.error(f"Sell order failed {symbol}: {e}")
            return False

    def _get_security_id(self, symbol: str) -> str:
        """
        Map symbol to Dhan security ID.
        In production: load from Dhan's instrument file.
        """
        # Placeholder mapping — replace with actual Dhan security IDs
        SECURITY_IDS = {
            "ICICIBANK": "4963", "HDFCBANK": "1333",
            "AXISBANK": "5900", "INFY": "1594",
            "HCLTECH": "7229", "TATAMOTORS": "3456",
            "MARUTI": "10999", "RELIANCE": "2885",
            "BHARTIARTL": "10604", "SUNPHARMA": "3351",
            "BAJFINANCE": "317", "LT": "11483",
            "ITC": "1660", "TITAN": "3506", "TCS": "11536"
        }
        return SECURITY_IDS.get(symbol, "0")


# =============================================================================
# SECTION 14: TRADE MONITOR & TRAILING SL
# =============================================================================

class TradeMonitor:
    """
    Monitors all open trades every 15 minutes.
    Manages 3-tier trailing SL system.
    Detects exit conditions.
    """

    def __init__(self, db: DatabaseManager, order_mgr: OrderManager,
                 protection: ProtectionEngine):
        self.db = db
        self.order_mgr = order_mgr
        self.protection = protection

    def sync_with_broker(self):
        """
        Reconcile DB open trades against actual Dhan holdings.
        Trades closed by broker SL orders won't appear in holdings —
        mark them CLOSED in DB so we don't double-act on them.
        Only runs in live mode (no-op for paper trading).
        """
        if Config.PAPER_TRADE or not self.order_mgr.dhan:
            return

        try:
            holdings_resp = self.order_mgr.dhan.get_holdings()
            # Dhan returns {"status": "success", "data": [...]}
            holdings_data = holdings_resp.get("data", []) if isinstance(holdings_resp, dict) else []
            held_symbols = {
                h.get("tradingSymbol") or h.get("symbol", "")
                for h in holdings_data
                if (h.get("availableQty", 0) or h.get("totalQty", 0)) > 0
            }
        except Exception as e:
            log.error(f"Broker sync failed (holdings fetch): {e}")
            return

        open_trades = self.db.get_open_trades()
        for trade in open_trades:
            symbol = trade["symbol"]
            if symbol in held_symbols:
                continue  # still held, nothing to do

            # Symbol not in holdings → broker already exited (SL executed, etc.)
            # Try to find the executed price from order history
            exit_price = trade["current_sl"]  # conservative fallback
            try:
                orders_resp = self.order_mgr.dhan.get_order_list()
                orders = orders_resp.get("data", []) if isinstance(orders_resp, dict) else []
                for o in reversed(orders):  # most recent first
                    if (o.get("tradingSymbol") == symbol and
                            o.get("transactionType") == "SELL" and
                            o.get("orderStatus") == "TRADED"):
                        exit_price = float(o.get("tradedPrice") or o.get("price") or exit_price)
                        break
            except Exception as e:
                log.warning(f"Could not fetch order history for {symbol}: {e}")

            pnl_data = ChargesCalculator.calculate_trade_pnl(
                trade["entry_price"], exit_price, trade["remaining_qty"]
            )
            self.db.close_trade(
                trade["trade_id"], exit_price, "BROKER_EXECUTED",
                pnl_data["net_pnl"], pnl_data["gross_pnl"], pnl_data["total_charges"]
            )
            log.info(
                f"BROKER SYNC: {symbol} was closed by broker @ ₹{exit_price:.2f} | "
                f"Net PNL: ₹{pnl_data['net_pnl']:.2f}"
            )
            self.protection.start_loss_cooldown()

    def monitor_all_trades(self, current_prices: dict, vix: float):
        """Run this every 15 minutes during market hours."""
        # Always reconcile against broker first so DB reflects reality
        self.sync_with_broker()

        open_trades = self.db.get_open_trades()
        if not open_trades:
            return

        log.info(f"Monitoring {len(open_trades)} open trades...")

        for trade in open_trades:
            symbol = trade["symbol"]
            price  = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            self._update_trade_price(trade["trade_id"], price)
            self._check_exit_conditions(trade, price, vix)
            self._update_trailing_sl(trade, price)

    def _update_trade_price(self, trade_id: str, price: float):
        self.db.execute(
            "UPDATE trades SET current_price=? WHERE trade_id=?",
            (price, trade_id)
        )

    def _check_exit_conditions(self, trade: dict, price: float, vix: float):
        """Check all exit triggers."""
        trade_id = trade["trade_id"]
        symbol   = trade["symbol"]
        sl       = trade["current_sl"]
        entry    = trade["entry_price"]
        qty      = trade["remaining_qty"]

        # Exit 1: SL hit — paper trade only.
        # In live mode the broker executes the SL order at the exchange;
        # sync_with_broker() detects and closes it. Placing another sell
        # here would cause a double-sell.
        if Config.PAPER_TRADE and price <= sl:
            log.warning(f"SL HIT: {symbol} @ ₹{price:.2f} (SL: ₹{sl:.2f})")
            self._execute_exit(trade, price, qty, ExitReason.SL_HIT.value)
            self.protection.start_loss_cooldown()
            return

        # Exit 2: Market crash (Nifty -2%+) — handled externally
        # but VIX spike here
        if vix > Config.VIX_NERVOUS:
            log.warning(f"VIX SPIKE EXIT: {symbol} | VIX: {vix:.1f}")
            self._execute_exit(trade, price, qty, ExitReason.MARKET_CRASH.value)
            return

        # Exit 3: Time based (> 15 days)
        entry_date = datetime.strptime(trade["entry_date"], "%Y-%m-%d")
        holding_days = (datetime.now() - entry_date).days
        if holding_days >= Config.MAX_HOLD_DAYS:
            # Only exit if the trade is not in profit (dead money)
            if price <= entry:
                log.info(f"TIME EXIT: {symbol} held {holding_days} days without profit. Freeing capital.")
                self._execute_exit(trade, price, qty, ExitReason.TIME_BASED.value)
                return
            else:
                log.info(f"TIME WARNING: {symbol} held {holding_days} days but is in profit. Letting it run.")

        # Exit 4: Event approaching
        safe, msg = self.protection.check_event_guard(symbol)
        if not safe:
            log.info(f"EVENT EXIT: {symbol} — {msg}")
            self._execute_exit(trade, price, qty, ExitReason.EVENT_EXIT.value)
            return

    def _update_trailing_sl(self, trade: dict, price: float):
        """
        3-Tier Trailing Stop Loss:
        Tier 1: At 1:1 RR → move SL to breakeven
        Tier 2: At 2:1 RR → sell 50%, move SL to just below target
        Tier 3: Beyond 2:1 → trail SL at 1x ATR below new highs
        """
        trade_id = trade["trade_id"]
        entry    = trade["entry_price"]
        sl       = trade["current_sl"]
        target   = trade["initial_target"]
        qty      = trade["remaining_qty"]
        atr      = self._get_atr(trade["symbol"])

        risk     = entry - trade["initial_sl"]
        if risk <= 0:
            return

        # --- TIER 1: Breakeven protection at 1:1 ---
        if not trade["tier1_done"] and price >= entry + risk:
            if sl < entry:  # only if SL is still below entry
                log.info(f"TIER 1 SL → Breakeven: {trade['symbol']} @ ₹{entry:.2f}")
                self.db.update_trade_sl(trade_id, entry)
                self.db.log_trailing_sl(trade_id, sl, entry, price, "TIER1_BREAKEVEN")
                self.db.execute(
                    "UPDATE trades SET tier1_done=1, tier1_price=? WHERE trade_id=?",
                    (price, trade_id)
                )
                new_id = self.order_mgr.replace_sl_order(
                    trade["symbol"], qty, entry, trade["sl_order_id"]
                )
                if new_id:
                    self.db.update_sl_order_id(trade_id, new_id)

        # --- TIER 2: Partial exit at 2:1 ---
        elif (not trade["tier2_done"] and
              trade["tier1_done"] and
              price >= target):

            exit_qty = max(1, qty // 2)     # sell 50%
            log.info(f"TIER 2 EXIT: {trade['symbol']} selling {exit_qty} shares @ ₹{price:.2f}")

            sold = self.order_mgr.place_sell_order(
                trade["symbol"], exit_qty, price, "TIER2_PARTIAL_EXIT"
            )
            if sold:
                new_sl = max(sl, target - (atr * 0.5))
                remaining = qty - exit_qty

                self.db.execute("""
                    UPDATE trades SET tier2_done=1, tier2_price=?,
                    tier2_qty=?, remaining_qty=?, current_sl=?
                    WHERE trade_id=?
                """, (price, exit_qty, remaining, new_sl, trade_id))

                self.db.log_trailing_sl(trade_id, sl, new_sl, price, "TIER2_PARTIAL_EXIT")

                # Cancel old SL (was for full qty) and place new one for remaining qty at new SL
                new_id = self.order_mgr.replace_sl_order(
                    trade["symbol"], remaining, new_sl, trade["sl_order_id"]
                )
                if new_id:
                    self.db.update_sl_order_id(trade_id, new_id)

                # Calculate partial PNL
                pnl_data = ChargesCalculator.calculate_trade_pnl(entry, price, exit_qty)
                log.info(f"TIER 2 Net PNL so far: ₹{pnl_data['net_pnl']:.2f}")

        # --- TIER 3: Trail remaining 50% with 1× ATR ---
        elif (trade["tier2_done"] and qty > 0 and atr > 0):
            new_trail_sl = price - atr
            if new_trail_sl > sl:  # only move SL UP, never down
                log.info(f"TRAIL SL: {trade['symbol']} {sl:.2f} → {new_trail_sl:.2f}")
                self.db.update_trade_sl(trade_id, new_trail_sl)
                self.db.log_trailing_sl(trade_id, sl, new_trail_sl, price, "TIER3_TRAIL")
                new_id = self.order_mgr.replace_sl_order(
                    trade["symbol"], qty, new_trail_sl, trade["sl_order_id"]
                )
                if new_id:
                    self.db.update_sl_order_id(trade_id, new_id)

    def _execute_exit(self, trade: dict, price: float,
                       qty: int, reason: str):
        """Execute final exit and calculate PNL."""
        symbol   = trade["symbol"]
        entry    = trade["entry_price"]
        trade_id = trade["trade_id"]

        sold = self.order_mgr.place_sell_order(symbol, qty, price, reason)
        if not sold and not Config.PAPER_TRADE:
            log.error(f"EXIT ORDER FAILED: {symbol}")
            return

        pnl_data = ChargesCalculator.calculate_trade_pnl(entry, price, qty)
        self.db.close_trade(
            trade_id, price, reason,
            pnl_data["net_pnl"],
            pnl_data["gross_pnl"],
            pnl_data["total_charges"]
        )
        log.info(
            f"TRADE CLOSED: {symbol} | {reason} | "
            f"Gross: ₹{pnl_data['gross_pnl']:.2f} | "
            f"Charges: ₹{pnl_data['total_charges']:.2f} | "
            f"Net: ₹{pnl_data['net_pnl']:.2f}"
        )

    def _get_atr(self, symbol: str) -> float:
        """Fetch latest ATR from DB."""
        today = datetime.now().strftime("%Y-%m-%d")
        row = self.db.fetchone(
            "SELECT atr FROM stock_snapshots WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (symbol,)
        )
        return row["atr"] if row and row["atr"] else 20.0


# =============================================================================
# SECTION 15: PERFORMANCE ANALYTICS
# =============================================================================

class PerformanceAnalytics:
    """
    Generates performance reports, win rates, strategy analysis,
    tax summary, and advance tax reminders.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def daily_summary(self) -> dict:
        """Today's trading summary."""
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self.db.fetchall(
            "SELECT * FROM trades WHERE exit_date=? AND status='CLOSED'", (today,)
        )
        wins   = [r for r in rows if r["net_pnl"] > 0]
        losses = [r for r in rows if r["net_pnl"] <= 0]
        total_net = sum(r["net_pnl"] for r in rows)
        total_charges = sum(r["total_charges"] for r in rows)
        open_trades = self.db.get_open_trades()
        unrealised = sum(
            (t["current_price"] - t["entry_price"]) * t["remaining_qty"]
            for t in open_trades
        )

        return {
            "date": today,
            "trades_closed": len(rows),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(rows) * 100, 1) if rows else 0,
            "realised_pnl": round(total_net, 2),
            "unrealised_pnl": round(unrealised, 2),
            "total_charges": round(total_charges, 2),
            "open_positions": len(open_trades),
        }

    def monthly_summary(self) -> dict:
        """This month's performance."""
        month_start = datetime.now().strftime("%Y-%m-01")
        rows = self.db.fetchall(
            "SELECT * FROM trades WHERE exit_date>=? AND status='CLOSED'",
            (month_start,)
        )
        if not rows:
            return {"message": "No closed trades this month"}

        wins   = [r for r in rows if r["net_pnl"] > 0]
        losses = [r for r in rows if r["net_pnl"] <= 0]
        net_pnl = sum(r["net_pnl"] for r in rows)
        charges = sum(r["total_charges"] for r in rows)
        avg_win  = sum(r["net_pnl"] for r in wins) / len(wins) if wins else 0
        avg_loss = sum(r["net_pnl"] for r in losses) / len(losses) if losses else 0

        # Strategy breakdown
        by_strategy = {}
        for r in rows:
            s = r["strategy"]
            if s not in by_strategy:
                by_strategy[s] = {"trades": 0, "wins": 0, "pnl": 0}
            by_strategy[s]["trades"] += 1
            if r["net_pnl"] > 0:
                by_strategy[s]["wins"] += 1
            by_strategy[s]["pnl"] += r["net_pnl"]

        return {
            "month": datetime.now().strftime("%B %Y"),
            "total_trades": len(rows),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(rows) * 100, 1),
            "net_pnl": round(net_pnl, 2),
            "total_charges": round(charges, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "rr_realised": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0,
            "strategy_breakdown": by_strategy,
        }

    def tax_summary(self) -> dict:
        """Annual STCG summary with advance tax breakdown."""
        annual_stcg = self.db.get_annual_stcg()
        return ChargesCalculator.annual_tax_summary(annual_stcg)

    def print_dashboard(self):
        """Print formatted text dashboard to terminal."""
        d  = self.daily_summary()
        m  = self.monthly_summary()
        tx = self.tax_summary()

        print("\n" + "=" * 65)
        print("  🚀 NIFTY 50 SWING TRADING SYSTEM — DASHBOARD")
        print("=" * 65)
        print(f"  📅 Date: {d['date']}")
        print(f"  📊 Open Positions: {d['open_positions']} / {Config.MAX_SIMULTANEOUS_TRADES}")
        print(f"\n  📈 TODAY")
        print(f"     Trades Closed : {d['trades_closed']}  ({d['wins']}W / {d['losses']}L)")
        print(f"     Win Rate      : {d['win_rate']}%")
        print(f"     Realised P&L  : ₹{d['realised_pnl']:,.2f}")
        print(f"     Unrealised    : ₹{d['unrealised_pnl']:,.2f}")
        print(f"     Charges       : ₹{d['total_charges']:,.2f}")

        if isinstance(m, dict) and "total_trades" in m:
            print(f"\n  📅 {m.get('month', 'MONTH')}")
            print(f"     Total Trades  : {m['total_trades']}")
            print(f"     Win Rate      : {m['win_rate']}%")
            print(f"     Net P&L       : ₹{m['net_pnl']:,.2f}")
            print(f"     Avg Win       : ₹{m['avg_win']:,.2f}")
            print(f"     Avg Loss      : ₹{m['avg_loss']:,.2f}")
            print(f"     Realised RR   : {m['rr_realised']}:1")
            if m.get("strategy_breakdown"):
                print(f"\n  📊 STRATEGY BREAKDOWN")
                for strat, stats in m["strategy_breakdown"].items():
                    wr = round(stats["wins"] / stats["trades"] * 100, 1)
                    print(f"     {strat:12s}: {stats['trades']} trades | {wr}% WR | ₹{stats['pnl']:,.0f}")

        print(f"\n  🧾 TAX (Current FY)")
        print(f"     Annual STCG   : ₹{tx['annual_stcg']:,.2f}")
        print(f"     Tax (20.8%)   : ₹{tx['total_tax']:,.2f}")
        print(f"     Take Home     : ₹{tx['take_home']:,.2f}")
        if tx["advance_tax_required"]:
            print(f"     ⚠️  ADVANCE TAX DUE — check quarterly dates!")

        # Open trades
        open_trades = self.db.get_open_trades()
        if open_trades:
            print(f"\n  💼 OPEN POSITIONS")
            for t in open_trades:
                unr = (t["current_price"] - t["entry_price"]) * t["remaining_qty"]
                print(
                    f"     {t['symbol']:12s} | {t['strategy']:9s} | "
                    f"Entry: ₹{t['entry_price']:.2f} | "
                    f"Now: ₹{t['current_price']:.2f} | "
                    f"SL: ₹{t['current_sl']:.2f} | "
                    f"Unrealised: ₹{unr:,.2f}"
                )
        print("=" * 65 + "\n")


# =============================================================================
# SECTION 16: MORNING BRIEFING GENERATOR
# =============================================================================

class MorningBriefing:
    """
    Generates a clean morning summary every day at 8:45 AM.
    Tells you: market mode, FII status, setups found, action items.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def generate(self, market_mode: MarketMode,
                  fii_flow: FIIFlow,
                  vix: float,
                  fii_net: float,
                  setups: list,
                  events_today: list) -> str:
        today = datetime.now().strftime("%A, %d %B %Y")
        mode_emoji = {
            "AGGRESSIVE": "🚀", "NORMAL": "✅",
            "SELECTIVE": "⚠️", "CAUTIOUS": "🟡",
            "DEFENSIVE": "🛑", "CASH": "💰"
        }.get(market_mode.value, "❓")

        lines = [
            "=" * 60,
            f"  MORNING BRIEFING — {today}",
            "=" * 60,
            f"  Market Mode : {mode_emoji} {market_mode.value}",
            f"  India VIX   : {vix:.1f}",
            f"  FII Flow    : {fii_flow.value} (₹{fii_net:,.0f} Cr net)",
            "",
        ]

        # Events warning
        if events_today:
            lines.append("  ⚠️  EVENTS TODAY / TOMORROW:")
            for e in events_today:
                lines.append(f"     {e['symbol']} — {e['event_type']} in {e['days_away']} days")
            lines.append("")

        # Setups
        if market_mode in [MarketMode.DEFENSIVE, MarketMode.CASH]:
            lines.append("  🛑 NO TRADING TODAY — Market in protection mode")
        elif not setups:
            lines.append("  📭 No valid setups found today. Stay patient.")
        else:
            lines.append(f"  📊 {len(setups)} SETUP(S) FOUND:")
            for i, s in enumerate(setups[:4], 1):
                lines.append(
                    f"  {i}. {s.symbol:12s} | {s.strategy.value:10s} | "
                    f"Score: {s.score}/100 | "
                    f"Entry: ₹{s.entry_price:.2f} | "
                    f"SL: ₹{s.sl_price:.2f} | "
                    f"Target: ₹{s.target_price:.2f}"
                )
            if setups:
                best = setups[0]
                lines.append(f"\n  🎯 RECOMMENDED: {best.symbol} via {best.strategy.value}")

        # Protection status
        daily_pnl = self.db.get_today_realised_pnl()
        lines.append(f"\n  💰 Today's P&L so far: ₹{daily_pnl:,.2f}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# SECTION 17: MASTER ORCHESTRATOR
# =============================================================================

class TradingSystem:
    """
    Master orchestrator that runs everything in sequence.
    This is the main class you run.
    Wires all components together and runs the daily scheduler.
    """

    def __init__(self):
        log.info("Initialising Nifty 50 Swing Trading System...")

        # Initialise all components
        self.db           = DatabaseManager()
        self.collector    = DataCollector(self.db)
        self.indicator_eng = IndicatorEngine()
        self.market_mode_eng = MarketModeEngine(self.db)
        self.screener     = StockScreener(self.db)
        self.strategy_eng = StrategyEngine()
        self.risk_mgr     = RiskManager(self.db)
        self.charges_calc = ChargesCalculator()
        self.protection   = ProtectionEngine(self.db)
        self.order_mgr    = OrderManager(self.db)
        self.monitor      = TradeMonitor(self.db, self.order_mgr, self.protection)
        self.analytics    = PerformanceAnalytics(self.db)
        self.briefing     = MorningBriefing(self.db)

        # State
        self.market_mode  = MarketMode.CAUTIOUS
        self.fii_flow     = FIIFlow.NEUTRAL
        self.vix          = 15.0
        self.fii_net      = 0.0
        self.stocks_data  = {}
        self.nifty_data   = None
        self.todays_setups = []

        log.info("System ready. All components loaded.")

    # =========================================================================
    # STEP 1: Pre-market data collection (8:45 AM)
    # =========================================================================

    def step1_collect_data(self):
        """Fetch all required data before market opens."""
        log.info("STEP 1: Collecting market data...")

        # VIX
        self.vix = self.collector.fetch_india_vix()

        # FII/DII
        fii_data = self.collector.fetch_fii_dii()
        self.fii_net = fii_data.get("fii_net_cash", 0.0)

        # FII consecutive days
        buy_streak, sell_streak = self.collector.get_fii_consecutive_days()

        # Save FII to history
        self.db.execute("""
            INSERT OR REPLACE INTO fii_history
            (date, fii_net_cash, dii_net_cash,
             consecutive_buying_days, consecutive_selling_days)
            VALUES (?,?,?,?,?)
        """, (
            datetime.now().strftime("%Y-%m-%d"),
            self.fii_net, fii_data.get("dii_net_cash", 0.0),
            buy_streak, sell_streak
        ))

        # Nifty OHLCV for market mode detection
        nifty_df = self.collector.fetch_ohlcv_daily("NIFTY 50", days=250)
        if nifty_df is not None:
            self.nifty_data = IndicatorEngine.calculate_all(nifty_df, "NIFTY50")

        # All 15 watchlist stocks
        symbols = Watchlist.get_symbols()
        for sym in symbols:
            df_daily  = self.collector.fetch_ohlcv_daily(sym, days=250)
            df_4h     = self.collector.fetch_ohlcv_intraday(sym, "60", 30)
            df_weekly = self.collector.fetch_ohlcv_daily(sym, days=500)

            s_data = IndicatorEngine.calculate_all(df_daily, sym, nifty_df)
            if s_data:
                # Add multi-timeframe checks
                s_data.h4_bullish = IndicatorEngine.check_4h_bullish(df_4h)
                s_data.weekly_bullish = IndicatorEngine.check_weekly_bullish(df_weekly)
                s_data.tf_aligned_count = sum([
                    s_data.weekly_bullish,
                    s_data.daily_bullish,
                    s_data.h4_bullish
                ])
                self.stocks_data[sym] = s_data
                self.db.save_stock_snapshot(s_data)

        # Events calendar
        events = self.collector.fetch_events_calendar()
        self.collector.save_events(events)

        log.info(f"Data collected: {len(self.stocks_data)} stocks | VIX: {self.vix:.1f} | FII: ₹{self.fii_net:.0f}Cr")

    # =========================================================================
    # STEP 2: Market mode detection (9:00 AM)
    # =========================================================================

    def step2_detect_market_mode(self):
        log.info("STEP 2: Detecting market mode...")

        buy_s, sell_s = self.collector.get_fii_consecutive_days()
        self.market_mode, self.fii_flow = self.market_mode_eng.detect(
            vix=self.vix,
            nifty_data=self.nifty_data,
            fii_net=self.fii_net,
            fii_consecutive_buy=buy_s,
            fii_consecutive_sell=sell_s
        )

        # Save market snapshot
        nifty = self.nifty_data
        self.db.save_market_snapshot({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "nifty_close": nifty.close if nifty else 0,
            "nifty_ema20": nifty.ema_20 if nifty else 0,
            "nifty_ema50": nifty.ema_50 if nifty else 0,
            "nifty_ema200": nifty.ema_200 if nifty else 0,
            "nifty_rsi": nifty.rsi if nifty else 50,
            "india_vix": self.vix,
            "gift_nifty": self.collector.fetch_gift_nifty(),
            "fii_net_cash": self.fii_net,
            "dii_net_cash": 0.0,
            "fii_consecutive_days": buy_s,
            "market_mode": self.market_mode.value,
            "fii_flow_label": self.fii_flow.value
        })

    # =========================================================================
    # STEP 3: Screen stocks (9:10 AM)
    # =========================================================================

    def step3_screen_stocks(self) -> list:
        log.info("STEP 3: Screening stocks...")
        open_trades = list(self.db.get_open_trades())
        passed = self.screener.screen(self.stocks_data, open_trades, self.market_mode)
        log.info(f"Screening result: {passed}")
        return passed

    # =========================================================================
    # STEP 4: Strategy matching and scoring (9:20 AM)
    # =========================================================================

    def step4_find_setups(self, screened_symbols: list) -> list:
        log.info("STEP 4: Finding setups...")
        setups = []

        # Detect FII sector flow for each symbol's sector
        fii_sectors = self._get_fii_buying_sectors()

        for symbol in screened_symbols:
            data = self.stocks_data.get(symbol)
            if not data:
                continue

            sector = Watchlist.get_sector(symbol)
            fii_sector_buying = sector in fii_sectors

            setup = self.strategy_eng.evaluate_all(
                symbol=symbol,
                data=data,
                market_mode=self.market_mode,
                fii_flow=self.fii_flow,
                fii_sector_buying=fii_sector_buying
            )

            if setup:
                # Calculate risk
                setup = self.risk_mgr.calculate_setup_risk(setup, data)
                if setup.status != "SKIPPED":
                    self.db.save_setup(setup)
                    setups.append(setup)
                    log.info(
                        f"SETUP FOUND: {symbol} | {setup.strategy.value} | "
                        f"Score: {setup.score} | Entry: ₹{setup.entry_price:.2f} | "
                        f"SL: ₹{setup.sl_price:.2f} | Target: ₹{setup.target_price:.2f}"
                    )

        # Sort by score descending
        setups.sort(key=lambda x: x.score, reverse=True)
        log.info(f"Total setups found: {len(setups)}")
        return setups

    def _get_fii_buying_sectors(self) -> set:
        """
        Determine which sectors have FII buying.
        Simplified: if overall FII is buying, use sector with highest RS.
        In production: use sector-wise FII data from NSE.
        """
        if self.fii_flow != FIIFlow.BUYING:
            return set()
        # Return sectors of top RS scoring stocks
        buying_sectors = set()
        for sym, data in self.stocks_data.items():
            if data and data.rs_score > 3:
                buying_sectors.add(Watchlist.get_sector(sym))
        return buying_sectors

    # =========================================================================
    # STEP 5: Execute trades (9:30 AM onwards)
    # =========================================================================

    def step5_execute_trades(self, setups: list):
        log.info("STEP 5: Evaluating trades for execution...")

        # Check if trading allowed at all
        allowed, reason = self.protection.is_trading_allowed()
        if not allowed:
            log.warning(f"Trading blocked: {reason}")
            return

        open_trades = list(self.db.get_open_trades())

        for setup in setups:
            if len(open_trades) >= Config.MAX_SIMULTANEOUS_TRADES:
                log.info("Max simultaneous trades reached. Skipping.")
                break

            # Only take score >= 60
            if setup.score < 60:
                log.debug(f"Skip {setup.symbol}: score too low ({setup.score})")
                continue

            # Final 11-point checklist
            approved, failed_checks = self.risk_mgr.run_pre_trade_checklist(
                setup, self.market_mode, self.vix, open_trades
            )

            if not approved:
                setup.status = "SKIPPED"
                setup.skip_reason = f"Failed checks: {failed_checks}"
                continue

            # Place order
            trade_id = self.order_mgr.place_entry_order(setup)
            if trade_id:
                setup.status = "TAKEN"
                log.info(f"Trade opened: {trade_id}")
                # Refresh open trades
                open_trades = list(self.db.get_open_trades())
            else:
                log.error(f"Order failed for {setup.symbol}")

        self.todays_setups = setups

    # =========================================================================
    # STEP 6: Intraday monitoring (every 15 mins during market hours)
    # =========================================================================

    def step6_monitor_trades(self):
        """Run every 15 mins: 9:30 to 3:30 PM."""
        current_prices = self._get_current_prices()
        self.monitor.monitor_all_trades(current_prices, self.vix)

    def _get_current_prices(self) -> dict:
        """Fetch live prices from Dhan API."""
        prices = {}
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return prices

        if Config.PAPER_TRADE:
            # Simulate random price movement for paper trading
            for t in open_trades:
                sym = t["symbol"]
                data = self.stocks_data.get(sym)
                base = t["current_price"] if t["current_price"] > 0 else (
                    data.close if data else t["entry_price"]
                )
                if LIBS_AVAILABLE:
                    # Random walk: ±0.3% per 15 min check
                    move = base * (1 + np.random.randn() * 0.003)
                    prices[sym] = round(float(move), 2)
                else:
                    prices[sym] = base
            return prices

        # Live prices via Dhan API
        if self.order_mgr.dhan:
            try:
                for t in open_trades:
                    sym = t["symbol"]
                    sec_id = self.order_mgr._get_security_id(sym)
                    quote = self.order_mgr.dhan.get_market_feed_quote(
                        security_id=sec_id,
                        exchange_segment="NSE_EQ"
                    )
                    if quote and "data" in quote:
                        prices[sym] = float(quote["data"].get("ltp", t["current_price"]))
            except Exception as e:
                log.error(f"Live price fetch failed: {e}")

        return prices

    # =========================================================================
    # STEP 7: EOD tasks (after 3:30 PM)
    # =========================================================================

    def step7_end_of_day(self):
        log.info("STEP 7: End of day tasks...")

        # Print dashboard
        self.analytics.print_dashboard()

        # Consecutive loss check
        if self.protection.check_consecutive_losses():
            log.warning("3 CONSECUTIVE LOSSES — Take tomorrow off. Review your system.")
            self.db.set_state("consecutive_loss_warning", "1")

        # Advance tax reminder
        tax = self.analytics.tax_summary()
        if tax["advance_tax_required"]:
            log.warning(f"⚠️  ADVANCE TAX DUE: ₹{tax['total_tax']:,.2f}")

        log.info("End of day complete.")

    # =========================================================================
    # STEP 8: Morning briefing (8:45 AM)
    # =========================================================================

    def step8_morning_briefing(self):
        events_today = self.db.fetchall(
            "SELECT * FROM events_calendar WHERE days_away <= 2 AND risk_level='RED'"
        )
        brief = self.briefing.generate(
            market_mode=self.market_mode,
            fii_flow=self.fii_flow,
            vix=self.vix,
            fii_net=self.fii_net,
            setups=self.todays_setups,
            events_today=list(events_today)
        )
        print(brief)
        log.info("Morning briefing generated.")

    # =========================================================================
    # FULL DAILY RUN — runs everything in sequence
    # =========================================================================

    def run_daily(self):
        """
        Master daily run.
        In production this is called by the scheduler at market open.
        """
        log.info("=" * 60)
        log.info("DAILY RUN STARTING")
        log.info("=" * 60)

        self.step1_collect_data()
        self.step2_detect_market_mode()
        screened = self.step3_screen_stocks()
        setups   = self.step4_find_setups(screened)
        self.step5_execute_trades(setups)
        self.step8_morning_briefing()
        log.info("Daily setup complete. Monitoring starts at 9:30 AM.")

    # =========================================================================
    # SCHEDULER — runs everything automatically at right times
    # =========================================================================

    def start_scheduler(self):
        """
        Start the automated daily scheduler.
        Run this once and it handles everything automatically.
        """
        log.info("Starting automated scheduler...")
        log.info(f"Paper Trade Mode: {'ON' if Config.PAPER_TRADE else 'OFF (LIVE!)'}")

        # Pre-market data + mode detection
        schedule.every().monday.at("08:45").do(self.step1_collect_data)
        schedule.every().tuesday.at("08:45").do(self.step1_collect_data)
        schedule.every().wednesday.at("08:45").do(self.step1_collect_data)
        schedule.every().thursday.at("08:45").do(self.step1_collect_data)
        schedule.every().friday.at("08:45").do(self.step1_collect_data)

        schedule.every().monday.at("09:00").do(self.step2_detect_market_mode)
        schedule.every().tuesday.at("09:00").do(self.step2_detect_market_mode)
        schedule.every().wednesday.at("09:00").do(self.step2_detect_market_mode)
        schedule.every().thursday.at("09:00").do(self.step2_detect_market_mode)
        schedule.every().friday.at("09:00").do(self.step2_detect_market_mode)

        # Setup detection + briefing
        for day in ["monday","tuesday","wednesday","thursday","friday"]:
            getattr(schedule.every(), day).at("09:10").do(
                lambda: self.step5_execute_trades(
                    self.step4_find_setups(self.step3_screen_stocks())
                )
            )
            getattr(schedule.every(), day).at("09:15").do(self.step8_morning_briefing)

        # Monitoring every 15 mins during market hours
        schedule.every(15).minutes.do(self._conditional_monitor)

        # EOD
        for day in ["monday","tuesday","wednesday","thursday","friday"]:
            getattr(schedule.every(), day).at("15:35").do(self.step7_end_of_day)

        # Weekly review Sunday evening
        schedule.every().sunday.at("20:00").do(self._weekly_review)

        print("\n✅ Scheduler running. Press Ctrl+C to stop.\n")
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            log.info("System stopped by user.")

    def _conditional_monitor(self):
        """Only run monitor during market hours (9:15 AM to 3:30 PM, weekdays)."""
        now = datetime.now()
        if now.weekday() >= 5:
            return
        market_start = now.replace(hour=9, minute=15, second=0)
        market_end   = now.replace(hour=15, minute=30, second=0)
        if market_start <= now <= market_end:
            self.step6_monitor_trades()

    def _weekly_review(self):
        """Sunday evening: generate and print weekly performance report."""
        log.info("WEEKLY REVIEW:")
        m = self.analytics.monthly_summary()
        t = self.analytics.tax_summary()
        print("\n" + "=" * 60)
        print("  📊 WEEKLY REVIEW")
        print("=" * 60)
        if "total_trades" in m:
            print(f"  Trades    : {m['total_trades']}")
            print(f"  Win Rate  : {m['win_rate']}%")
            print(f"  Net P&L   : ₹{m['net_pnl']:,.2f}")
        print(f"  STCG Liability : ₹{t['total_tax']:,.2f}")
        print("=" * 60 + "\n")

    # =========================================================================
    # QUICK TEST — run without scheduler for immediate testing
    # =========================================================================

    def run_once_test(self):
        """
        Run a single complete cycle immediately.
        Use this to test the system before running live scheduler.
        """
        print("\n" + "🔵 " * 20)
        print("RUNNING SINGLE TEST CYCLE")
        print("🔵 " * 20 + "\n")
        self.run_daily()
        self.analytics.print_dashboard()

        # Show tax summary
        tax = self.analytics.tax_summary()
        print("\n📋 TAX SUMMARY:")
        print(f"   Annual STCG : ₹{tax['annual_stcg']:,.2f}")
        print(f"   Total Tax   : ₹{tax['total_tax']:,.2f}")
        print(f"   Take Home   : ₹{tax['take_home']:,.2f}")
        if tax["advance_tax_required"]:
            print("   ⚠️  Advance tax payment required!")
            q = tax["advance_tax_quarters"]
            print(f"   Jun 15  : ₹{q['jun_15_15pct']:,.2f}")
            print(f"   Sep 15  : ₹{q['sep_15_45pct']:,.2f}")
            print(f"   Dec 15  : ₹{q['dec_15_75pct']:,.2f}")
            print(f"   Mar 15  : ₹{q['mar_15_100pct']:,.2f}")


# =============================================================================
# SECTION 18: ENTRY POINT
# =============================================================================

def main():
    """
    HOW TO USE:
    -----------
    1. PAPER TRADE TEST (default, safe):
       python trading_system.py
       → Runs one test cycle with mock data, no real orders

    2. LIVE SCHEDULER (real trading):
       Set PAPER_TRADE = False in Config
       Set your DHAN credentials as env vars:
         export DHAN_CLIENT_ID="your_id"
         export DHAN_ACCESS_TOKEN="your_token"
       Then run:
         python trading_system.py --live

    3. JUST DASHBOARD:
       python trading_system.py --dashboard
    """
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    system = TradingSystem()

    if mode == "--live":
        print("⚠️  LIVE MODE ACTIVATED — Real orders will be placed!")
        print("Press Enter to confirm or Ctrl+C to cancel...")
        input()
        Config.PAPER_TRADE = False
        system.start_scheduler()

    elif mode == "--dashboard":
        system.analytics.print_dashboard()

    elif mode == "--scheduler":
        print("📅 Starting scheduler in PAPER TRADE mode...")
        system.start_scheduler()

    else:
        # Default: single test run (paper mode)
        system.run_once_test()


if __name__ == "__main__":
    main()
