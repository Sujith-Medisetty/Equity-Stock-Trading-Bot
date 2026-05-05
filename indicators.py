"""
indicators.py — Technical indicator calculations.

Two classes live here:

1. ta — a pure pandas/numpy implementation of every indicator we need.
   No external TA library required. This keeps dependencies minimal and
   makes the math transparent — you can read exactly what each indicator does.

2. IndicatorEngine — takes raw OHLCV DataFrames from DataCollector and
   produces a fully populated StockData object with all indicators calculated.
   This is the bridge between raw price data and tradeable signals.

Flow:
  DataCollector.fetch_ohlcv_daily()  →  raw DataFrame
  IndicatorEngine.calculate_all()    →  StockData (with all indicators)
  StrategyEngine.evaluate_all()      →  uses StockData to find setups
"""

from typing import Optional
from datetime import datetime

from config import Config, log, LIBS_AVAILABLE
from models import StockData

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass


class ta:
    """
    Pure pandas/numpy technical indicator library.
    All methods are static — just call ta.ema(), ta.rsi(), etc.

    Why not use pandas_ta or TA-Lib?
    - Fewer dependencies, easier to install anywhere
    - Transparent math — no black box
    - These 6 indicators are all we need
    """

    @staticmethod
    def ema(series: "pd.Series", length: int) -> "pd.Series":
        """Exponential Moving Average — reacts faster to recent price than SMA."""
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rsi(series: "pd.Series", length: int = 14) -> "pd.Series":
        """
        Relative Strength Index (0-100).
        < 40 = oversold/weak, 40-60 = healthy trend, > 70 = overbought.
        We look for 45-65 range for clean swing entries.
        """
        delta    = series.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=length - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=length - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: "pd.Series", fast=12, slow=26, signal=9) -> Optional["pd.DataFrame"]:
        """
        MACD = fast EMA - slow EMA. Signal = EMA of MACD. Histogram = MACD - Signal.
        We use: macd > signal (momentum positive) and histogram > 0 (accelerating).
        """
        if not LIBS_AVAILABLE:
            return None
        ema_fast    = ta.ema(series, fast)
        ema_slow    = ta.ema(series, slow)
        macd_line   = ema_fast - ema_slow
        signal_line = ta.ema(macd_line, signal)
        return pd.DataFrame({
            "macd":   macd_line,
            "signal": signal_line,
            "hist":   macd_line - signal_line
        })

    @staticmethod
    def atr(high: "pd.Series", low: "pd.Series",
            close: "pd.Series", length: int = 14) -> "pd.Series":
        """
        Average True Range — measures daily price volatility.
        Critical for stop loss placement: SL = entry - (ATR × multiplier).
        A stock with ATR=20 needs at least ₹20 breathing room before SL.
        """
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
        """
        Bollinger Bands — upper/lower bands at 2 std deviations from 20-day SMA.
        Stored in StockData but not currently used in strategy criteria.
        Useful for future volatility-based entries.
        """
        if not LIBS_AVAILABLE:
            return None
        mid   = series.rolling(length).mean()
        sigma = series.rolling(length).std()
        return pd.DataFrame({
            "lower": mid - std * sigma,
            "mid":   mid,
            "upper": mid + std * sigma
        })

    @staticmethod
    def obv(close: "pd.Series", volume: "pd.Series") -> "pd.Series":
        """
        On-Balance Volume — cumulative volume in the direction of price.
        Rising OBV = institutional accumulation = bullish confirmation.
        We use: obv_rising = OBV today > OBV 5 days ago.
        """
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return (direction * volume).cumsum()


class IndicatorEngine:
    """
    Converts raw OHLCV DataFrames into fully calculated StockData objects.

    Called once per stock per day during step1_collect_data().
    Results are saved to the stock_snapshots table in the DB,
    but also held in memory (self.stocks_data dict in TradingSystem)
    for immediate use during the same session.

    Also handles multi-timeframe checks:
    - check_4h_bullish()     → is the 60-min chart in an uptrend?
    - check_weekly_bullish() → is the weekly chart in an uptrend?
    These feed into tf_aligned_count — strategies require 2+ timeframes aligned.
    """

    @staticmethod
    def calculate_all(df: "pd.DataFrame", symbol: str,
                      nifty_df: "pd.DataFrame" = None) -> Optional[StockData]:
        """
        Master function. Takes a daily OHLCV DataFrame, returns StockData.
        Needs minimum 50 bars. Ideally 250 bars for EMA-200 to be reliable.
        nifty_df is passed to calculate relative strength vs the index.
        """
        if df is None or len(df) < 50:
            log.warning(f"{symbol}: insufficient data ({len(df) if df is not None else 0} bars)")
            return None
        try:
            df = df.copy().reset_index(drop=True)

            # Trend indicators
            df["ema_20"]  = ta.ema(df["close"], length=Config.EMA_SHORT)
            df["ema_50"]  = ta.ema(df["close"], length=Config.EMA_MED)
            df["ema_200"] = ta.ema(df["close"], length=Config.EMA_LONG)
            df["rsi"]     = ta.rsi(df["close"], length=Config.RSI_PERIOD)

            macd = ta.macd(df["close"], fast=Config.MACD_FAST,
                           slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
            if macd is not None:
                df["macd"]        = macd["macd"]
                df["macd_signal"] = macd["signal"]
                df["macd_hist"]   = macd["hist"]
            else:
                df["macd"] = df["macd_signal"] = df["macd_hist"] = 0.0

            # Volatility
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=Config.ATR_PERIOD)
            bb = ta.bbands(df["close"], length=Config.BB_PERIOD, std=Config.BB_STD)
            if bb is not None:
                df["bb_upper"] = bb["upper"]
                df["bb_lower"] = bb["lower"]
            else:
                df["bb_upper"] = df["bb_lower"] = df["close"]

            # Volume analysis
            df["vol_avg"]   = df["volume"].rolling(Config.VOLUME_AVG_PERIOD).mean()
            df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, 1)

            # ATR ratio: today's ATR vs its own 20-day average.
            # > 1.5 = unusually volatile day → screener filters these out
            df["atr_avg"]   = df["atr"].rolling(20).mean()
            df["atr_ratio"] = df["atr"] / df["atr_avg"].replace(0, 1)

            # OBV — directional volume accumulation
            df["obv"] = ta.obv(df["close"], df["volume"])

            last = df.iloc[-1]

            # 52-week range (used for WEEK52 strategy)
            w52          = df.tail(252)
            week_52_high = w52["high"].max()
            week_52_low  = w52["low"].min()

            # 20-day consolidation tightness (used for BREAKOUT strategy)
            # Tight range < 8% = stock coiling before a breakout
            d20  = df.tail(20)
            consolidation_pct = (
                (d20["high"].max() - d20["low"].min()) / last["close"] * 100
                if last["close"] > 0 else 999
            )

            # OBV trend: is today's OBV higher than 5 days ago?
            obv_rising = len(df) >= 5 and bool(df["obv"].iloc[-1] > df["obv"].iloc[-5])

            return StockData(
                symbol=symbol,
                date=datetime.now().strftime("%Y-%m-%d"),
                open=float(last["open"]),
                high=float(last["high"]),
                low=float(last["low"]),
                close=float(last["close"]),
                volume=float(last["volume"]),
                ema_20=float(last["ema_20"])       if not pd.isna(last["ema_20"])       else 0.0,
                ema_50=float(last["ema_50"])       if not pd.isna(last["ema_50"])       else 0.0,
                ema_200=float(last["ema_200"])     if not pd.isna(last["ema_200"])      else 0.0,
                rsi=float(last["rsi"])             if not pd.isna(last["rsi"])          else 50.0,
                macd=float(last["macd"])           if not pd.isna(last["macd"])         else 0.0,
                macd_signal=float(last["macd_signal"]) if not pd.isna(last["macd_signal"]) else 0.0,
                macd_hist=float(last["macd_hist"]) if not pd.isna(last["macd_hist"])   else 0.0,
                atr=float(last["atr"])             if not pd.isna(last["atr"])          else 0.0,
                bb_upper=float(last["bb_upper"])   if not pd.isna(last["bb_upper"])    else 0.0,
                bb_lower=float(last["bb_lower"])   if not pd.isna(last["bb_lower"])    else 0.0,
                volume_ratio=float(last["vol_ratio"]) if not pd.isna(last["vol_ratio"]) else 1.0,
                week_52_high=float(week_52_high),
                week_52_low=float(week_52_low),
                rs_score=IndicatorEngine._calc_rs(df, nifty_df),
                candle_pattern=IndicatorEngine._detect_candle_pattern(df),
                daily_bullish=bool(last["close"] > last["ema_20"] and
                                   last["ema_20"] > last["ema_50"]),
                consolidation_range_pct=float(consolidation_pct),
                obv_rising=obv_rising,
                atr_ratio=float(last["atr_ratio"]) if not pd.isna(last["atr_ratio"]) else 1.0,
            )
        except Exception as e:
            log.error(f"Indicator calculation failed for {symbol}: {e}")
            return None

    @staticmethod
    def _calc_rs(stock_df: "pd.DataFrame", nifty_df: Optional["pd.DataFrame"]) -> float:
        """
        Relative Strength score = stock's 60-day return minus Nifty's 60-day return.
        Positive = stock outperforming the index = smart money may be accumulating.
        > 5 = strong outperformer (gets +15 score bonus in StrategyEngine)
        > 0 = slight outperformer (gets +8 score bonus)
        """
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
        Detects the 5 most reliable bullish confirmation patterns on the last candle.
        A strong candle pattern adds 10-15 points to the setup score.

        MARUBOZU        → big green candle, almost no wicks, high volume = strong buyers
        BULLISH_ENGULFING → today's green candle fully wraps yesterday's red = reversal
        HAMMER          → long lower wick (buyers rejected the low) = support found
        MORNING_STAR    → 3-candle reversal: red → small → green = momentum shift
        INSIDE_BAR      → today's range inside yesterday's = coiling, breakout coming
        """
        if len(df) < 3:
            return "NONE"
        try:
            c  = df.iloc[-1]
            p  = df.iloc[-2]
            p2 = df.iloc[-3]

            body_c     = abs(c["close"] - c["open"])
            range_c    = c["high"] - c["low"] if c["high"] != c["low"] else 0.001
            lower_wick = min(c["open"], c["close"]) - c["low"]
            upper_wick = c["high"] - max(c["open"], c["close"])

            if (c["close"] > c["open"] and body_c / range_c > 0.85 and
                    c["volume"] > df["volume"].tail(20).mean() * 1.5):
                return "MARUBOZU"

            if (p["close"] < p["open"] and c["close"] > c["open"] and
                    c["open"] < p["close"] and c["close"] > p["open"] and
                    body_c > abs(p["close"] - p["open"]) and c["volume"] > p["volume"]):
                return "BULLISH_ENGULFING"

            if (body_c > 0 and lower_wick >= 2 * body_c and
                    upper_wick <= 0.1 * range_c and c["close"] > c["open"]):
                return "HAMMER"

            body_p  = abs(p["close"]  - p["open"])
            body_p2 = abs(p2["close"] - p2["open"])
            if (p2["close"] < p2["open"] and body_p < body_p2 * 0.4 and
                    c["close"] > c["open"] and
                    c["close"] > p2["open"] + body_p2 * 0.5):
                return "MORNING_STAR"

            if c["high"] < p["high"] and c["low"] > p["low"]:
                return "INSIDE_BAR"

        except Exception as e:
            log.debug(f"Candle pattern error: {e}")
        return "NONE"

    @staticmethod
    def check_4h_bullish(df_4h: Optional["pd.DataFrame"]) -> bool:
        """
        4H timeframe bullish check: is price above the 20-period EMA on the 60-min chart?
        Used to confirm the daily signal has intraday momentum support.
        """
        if df_4h is None or len(df_4h) < 25:
            return False
        try:
            ema = ta.ema(df_4h["close"], length=20)
            return bool(df_4h["close"].iloc[-1] > ema.iloc[-1])
        except Exception:
            return False

    @staticmethod
    def check_weekly_bullish(df_weekly: Optional["pd.DataFrame"]) -> bool:
        """
        Weekly timeframe bullish check: is price above the 20-week EMA?
        The highest timeframe we check — if weekly is bearish, all daily setups are suspect.
        """
        if df_weekly is None or len(df_weekly) < 22:
            return False
        try:
            ema = ta.ema(df_weekly["close"], length=20)
            return bool(df_weekly["close"].iloc[-1] > ema.iloc[-1])
        except Exception:
            return False
