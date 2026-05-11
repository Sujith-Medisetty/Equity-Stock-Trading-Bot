"""
models.py — All data structures used across the system.

This file defines every enum and dataclass the system works with.
Nothing is calculated here — it's purely shape definitions.
Every other file imports from here. If you want to understand
what data flows through the system, start here.
"""

from dataclasses import dataclass
from enum import Enum


# -----------------------------------------------------------------------------
# ENUMS
# Enums are used instead of plain strings so typos get caught at import time.
# e.g. MarketMode.AGGRESSIVE is safer than passing "AGGRESSIVE" around.
# -----------------------------------------------------------------------------

class MarketMode(Enum):
    """
    Overall market regime detected each morning.
    Controls which strategies are allowed and how aggressively we trade.

    AGGRESSIVE → Nifty above all EMAs + FII buying → full 4 positions allowed
    NORMAL     → Nifty above all EMAs, FII neutral  → normal trading
    SELECTIVE  → Nifty in pullback or FII selling    → only highest-score setups
    CAUTIOUS   → Nifty sideways, above 200 EMA only → 1-2 positions max
    DEFENSIVE  → Nifty below 200 EMA                → no new entries
    CASH       → VIX panic (>28)                     → exit everything, no trades
    """
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL     = "NORMAL"
    SELECTIVE  = "SELECTIVE"
    CAUTIOUS   = "CAUTIOUS"
    DEFENSIVE  = "DEFENSIVE"
    CASH       = "CASH"


class FIIFlow(Enum):
    """
    Foreign Institutional Investor activity direction.
    FIIs move large enough amounts to influence index direction.
    We use this as a tailwind/headwind signal — never trade against heavy FII selling.

    BUYING  → FII net positive for 3+ consecutive days above ₹2000 Cr threshold
    SELLING → FII net negative for 3+ consecutive days
    NEUTRAL → No clear trend
    """
    BUYING  = "BUYING"
    SELLING = "SELLING"
    NEUTRAL = "NEUTRAL"


class StrategyType(Enum):
    """
    The 4 entry strategies, in priority order (highest first):
    WEEK52    → Stock breaking out of 52-week high (only in AGGRESSIVE mode)
    BREAKOUT  → Stock breaking out of consolidation with 2x+ volume
    PULLBACK  → Stock pulling back to 20 EMA in an uptrend
    SWING     → General trend-following on multi-timeframe alignment

    FII sector buying is no longer a standalone strategy — it is a score
    modifier (+15) applied on top of PULLBACK and BREAKOUT when FII is
    actively buying in the stock's sector.
    """
    SWING    = "SWING"
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    WEEK52   = "WEEK52"


class TradeStatus(Enum):
    """Lifecycle state of a trade in the database."""
    OPEN   = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(Enum):
    """
    Why a trade was closed. Stored in DB for post-trade analysis.
    SL_HIT          → Price hit stop loss (paper mode only — live mode broker fires it)
    TARGET_HIT      → Price reached initial target (not currently auto-triggered)
    TRAILING_SL     → Tier 3 trail SL was hit
    TIME_BASED      → Held > 15 days without profit (dead money, free up capital)
    MARKET_CRASH    → VIX spiked above 22, exit everything
    EVENT_EXIT      → Earnings/results within 5 days, reduce risk
    MANUAL          → User manually closed
    BROKER_EXECUTED → Broker fired the SL order at exchange, detected via get_holdings()
    """
    SL_HIT          = "SL_HIT"
    TARGET_HIT      = "TARGET_HIT"
    TRAILING_SL     = "TRAILING_SL"
    TIME_BASED      = "TIME_BASED"
    MARKET_CRASH    = "MARKET_CRASH"
    EVENT_EXIT      = "EVENT_EXIT"
    MANUAL          = "MANUAL"
    BROKER_EXECUTED = "BROKER_EXECUTED"


# -----------------------------------------------------------------------------
# DATACLASSES
# These are plain data containers — no logic, just fields.
# -----------------------------------------------------------------------------

@dataclass
class StockData:
    """
    All technical indicator values for a single stock on a single day.
    Produced by IndicatorEngine.calculate_all() from raw OHLCV data.
    Used by StockScreener and StrategyEngine to make decisions.

    Key fields:
    - ema_20/50/200    → trend direction and support levels
    - rsi              → momentum (45-65 = healthy uptrend)
    - macd/signal/hist → momentum confirmation
    - atr              → daily price range — used for SL sizing
    - volume_ratio     → today's volume vs 20-day avg (>2 = breakout confirmation)
    - rs_score         → stock's 60-day return minus Nifty's — positive = outperforming
    - tf_aligned_count → how many timeframes (weekly/daily/4H) are bullish (0-3)
    - consolidation_range_pct → how tight the last 20-day range is (< 8% = tight box)
    - candle_pattern   → last candle: MARUBOZU, HAMMER, BULLISH_ENGULFING, etc.
    """
    symbol: str
    date:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float

    ema_20:       float = 0.0
    ema_50:       float = 0.0
    ema_200:      float = 0.0
    rsi:          float = 50.0
    macd:         float = 0.0
    macd_signal:  float = 0.0
    macd_hist:    float = 0.0
    atr:          float = 0.0

    volume_ratio:            float = 1.0
    week_52_high:            float = 0.0
    rs_score:                float = 0.0
    candle_pattern:          str   = "NONE"
    weekly_bullish:          bool  = False
    daily_bullish:           bool  = False
    h4_bullish:              bool  = False
    tf_aligned_count:        int   = 0
    consolidation_range_pct: float = 999.0
    obv_rising:              bool  = False
    atr_ratio:               float = 1.0

    # FVG (Fair Value Gap) fields — computed by IndicatorEngine, used by screener + strategy + risk
    in_fvg_zone:  bool  = False  # price inside a RECENT unfilled bullish FVG (≤10 days old) → screener blocks entry
    fvg_pullback: bool  = False  # price inside ANY unfilled bullish FVG → PULLBACK score +10
    fvg_target:   float = 0.0   # bottom of nearest unfilled bullish FVG above price (within 8%) → target override
    # Raw zone lists stored so FVG flags can be re-evaluated against live price in midday scan
    # without re-running detection on the full DataFrame. Not persisted to DB — memory only.
    fvg_zones:    object = None  # daily FVG zones: list of {"bottom", "top", "age"} dicts
    fvg_zones_4h: object = None  # 4H (60-min) FVG zones: same structure, finer-grained zones


@dataclass
class Setup:
    """
    A trade opportunity identified by StrategyEngine and sized by RiskManager.

    Lifecycle:
    1. StrategyEngine creates it with entry_price and score
    2. RiskManager fills in sl_price, target_price, shares, rr_ratio
    3. RiskManager.run_pre_trade_checklist() approves or rejects it
    4. If approved → OrderManager places the entry order
    5. status moves from PENDING → TAKEN or SKIPPED

    score (0-100):
    - 70% from how many strategy criteria passed
    - 15% from relative strength vs Nifty
    - 15% from candle pattern quality
    Minimum score to enter: 60. High confidence: 80+.
    """
    symbol:       str
    date:         str
    strategy:     StrategyType
    score:        int
    entry_price:  float
    market_mode:  str
    fii_flow:     str

    sl_price:         float = 0.0
    target_price:     float = 0.0
    atr:              float = 0.0
    risk_per_share:   float = 0.0   # entry - sl
    shares:           int   = 0     # sized so max loss = ₹1500
    capital_required: float = 0.0
    actual_risk:      float = 0.0   # shares × risk_per_share
    rr_ratio:         float = 0.0   # must be >= 2.0
    status:           str   = "PENDING"
    skip_reason:      str   = ""


@dataclass
class Trade:
    """
    A live or closed trade with full tracking.

    Created by OrderManager when an entry order is placed.
    Updated by TradeMonitor every 15 mins (current_price, SL levels, tier progress).
    Closed by TradeMonitor when an exit condition is met.

    3-Tier trailing SL system:
    - tier1: at 1:1 RR → SL moves to breakeven (no-loss trade guaranteed)
    - tier2: at 2:1 RR → sell 50% shares, SL tightens near target
    - tier3: beyond 2:1 → trail remaining 50% at 1×ATR below new highs

    sl_order_id: the Upstox order ID of the SL order sitting at the exchange.
    Needed so we can cancel + replace it when the trailing SL moves up.
    """
    trade_id:       str
    symbol:         str
    strategy:       str
    entry_date:     str
    entry_price:    float
    quantity:       int
    initial_sl:     float
    initial_target: float
    current_sl:     float
    current_price:  float = 0.0

    # Tier exit tracking
    tier1_done:    bool  = False
    tier1_price:   float = 0.0
    tier1_qty:     int   = 0
    tier2_done:    bool  = False
    tier2_price:   float = 0.0
    tier2_qty:     int   = 0
    remaining_qty: int   = 0

    # Broker charges (filled at close time)
    stt:             float = 0.0
    dp_charge:       float = 0.0
    exchange_charge: float = 0.0
    stamp_duty:      float = 0.0
    gst:             float = 0.0
    sebi:            float = 0.0
    total_charges:   float = 0.0

    gross_pnl: float = 0.0  # before charges
    net_pnl:   float = 0.0  # after charges

    setup_score:          int = 0
    market_mode_at_entry: str = ""
    status:               str = TradeStatus.OPEN.value
    exit_reason:          str = ""
    exit_date:            str = ""
    holding_days:         int = 0
    sl_order_id:          str = ""  # Upstox order ID — needed for cancel+replace on SL update
