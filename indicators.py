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
        """Exponential Moving Average — reacts faster to recent price than SMA.
        ewm(span=length) gives the standard EMA formula.
        adjust=False means each value uses the recursive formula: EMA = price×α + prev_EMA×(1-α)
        where α = 2/(span+1). This matches how most charting platforms calculate EMA."""
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rsi(series: "pd.Series", length: int = 14) -> "pd.Series":
        """
        Relative Strength Index (0-100).
        < 40 = oversold/weak, 40-60 = healthy trend, > 70 = overbought.
        We look for 45-65 range for clean swing entries.

        Formula:
          delta = daily price change
          gain  = positive changes only (losses set to 0)
          loss  = negative changes only (gains set to 0, taken as positive)
          RS    = EMA(gain) / EMA(loss) over the period
          RSI   = 100 - (100 / (1 + RS))
        """
        delta    = series.diff()                            # day-over-day price change
        gain     = delta.clip(lower=0)                     # keep only positive changes (up days)
        loss     = -delta.clip(upper=0)                    # keep only negative changes (as positive numbers)
        avg_gain = gain.ewm(com=length - 1, adjust=False).mean()   # smoothed average of gains
        avg_loss = loss.ewm(com=length - 1, adjust=False).mean()   # smoothed average of losses
        rs = avg_gain / avg_loss.replace(0, 1e-9)          # avoid division by zero with tiny epsilon
        return 100 - (100 / (1 + rs))                      # convert RS ratio to 0-100 scale

    @staticmethod
    def macd(series: "pd.Series", fast=12, slow=26, signal=9) -> Optional["pd.DataFrame"]:
        """
        MACD = fast EMA - slow EMA. Signal = EMA of MACD. Histogram = MACD - Signal.
        We use: macd > signal (momentum positive) and histogram > 0 (accelerating).

        Positive MACD: short-term momentum above long-term = upward trend in short term.
        MACD > signal: the trend is accelerating (getting stronger).
        Histogram > 0: same as above, just visualised as a bar chart value.
        """
        if not LIBS_AVAILABLE:
            return None
        ema_fast    = ta.ema(series, fast)           # 12-day EMA — fast-reacting
        ema_slow    = ta.ema(series, slow)           # 26-day EMA — slow-reacting
        macd_line   = ema_fast - ema_slow            # positive = short-term momentum above long-term
        signal_line = ta.ema(macd_line, signal)      # 9-day EMA of MACD line — smooths out noise
        return pd.DataFrame({
            "macd":   macd_line,
            "signal": signal_line,
            "hist":   macd_line - signal_line        # positive histogram = MACD above signal = accelerating up
        })

    @staticmethod
    def atr(high: "pd.Series", low: "pd.Series",
            close: "pd.Series", length: int = 14) -> "pd.Series":
        """
        Average True Range — measures daily price volatility.
        Critical for stop loss placement: SL = entry - (ATR × multiplier).
        A stock with ATR=20 needs at least ₹20 breathing room before SL.

        True Range = max of these 3:
          1. high - low          (today's range)
          2. |high - prev_close| (gap up + today's range if gapped higher)
          3. |low - prev_close|  (gap down + today's range if gapped lower)
        This handles overnight gaps that would be missed by simple high-low range.
        ATR = EWM average of True Range over the period.
        """
        prev_close = close.shift(1)             # yesterday's close (shifted forward by 1 row)
        tr = pd.concat([
            high - low,                         # intraday range
            (high - prev_close).abs(),          # gap up size (or 0 if no gap)
            (low  - prev_close).abs()           # gap down size (or 0 if no gap)
        ], axis=1).max(axis=1)                  # take the largest of the 3 for each day
        return tr.ewm(com=length - 1, adjust=False).mean()   # smooth with EWM over 14 days

    @staticmethod
    def bbands(series: "pd.Series", length: int = 20,
               std: float = 2.0) -> Optional["pd.DataFrame"]:
        """
        Bollinger Bands — upper/lower bands at 2 std deviations from 20-day SMA.
        Stored in StockData but not currently used in strategy criteria.
        Useful for future volatility-based entries.

        mid   = 20-day simple moving average (the middle band)
        sigma = 20-day rolling standard deviation (measures volatility)
        upper = mid + 2×sigma  (price above here = extended/overbought)
        lower = mid - 2×sigma  (price below here = extended/oversold)
        """
        if not LIBS_AVAILABLE:
            return None
        mid   = series.rolling(length).mean()    # 20-day SMA
        sigma = series.rolling(length).std()     # 20-day standard deviation
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

        Logic:
          If price went up today → add today's volume to running total (buying pressure)
          If price went down today → subtract today's volume (selling pressure)
          If price unchanged → add 0
        Rising OBV while price consolidates = institutions quietly accumulating.
        """
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return (direction * volume).cumsum()    # running sum of signed volume


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
            df = df.copy().reset_index(drop=True)   # work on a copy so we don't mutate the original

            # --- Trend indicators ---
            df["ema_20"]  = ta.ema(df["close"], length=Config.EMA_SHORT)    # 20-day EMA — short-term trend
            df["ema_50"]  = ta.ema(df["close"], length=Config.EMA_MED)      # 50-day EMA — medium-term trend
            df["ema_200"] = ta.ema(df["close"], length=Config.EMA_LONG)     # 200-day EMA — long-term trend / bull/bear line

            df["rsi"]     = ta.rsi(df["close"], length=Config.RSI_PERIOD)   # 14-period RSI — momentum

            # MACD returns a DataFrame with 3 columns: macd, signal, hist
            macd = ta.macd(df["close"], fast=Config.MACD_FAST,
                           slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
            if macd is not None:
                df["macd"]        = macd["macd"]
                df["macd_signal"] = macd["signal"]
                df["macd_hist"]   = macd["hist"]
            else:
                df["macd"] = df["macd_signal"] = df["macd_hist"] = 0.0   # fallback if pandas not available

            # --- Volatility indicators ---
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=Config.ATR_PERIOD)  # 14-period ATR

            # Bollinger Bands — stored for reference but not used in strategy conditions currently
            bb = ta.bbands(df["close"], length=Config.BB_PERIOD, std=Config.BB_STD)
            if bb is not None:
                df["bb_upper"] = bb["upper"]
                df["bb_lower"] = bb["lower"]
            else:
                df["bb_upper"] = df["bb_lower"] = df["close"]  # fallback: bands = close price

            # --- Volume analysis ---
            # 20-day rolling average volume — baseline for comparison
            df["vol_avg"]   = df["volume"].rolling(Config.VOLUME_AVG_PERIOD).mean()
            # volume_ratio = today's volume / 20-day avg. >2.0 = breakout confirmation, <0.5 = thin day
            df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, 1)   # replace 0 avg to avoid div-by-zero

            # ATR ratio: today's ATR vs its own 20-day average.
            # > 1.5 = unusually volatile day → screener filters these out
            df["atr_avg"]   = df["atr"].rolling(20).mean()
            df["atr_ratio"] = df["atr"] / df["atr_avg"].replace(0, 1)   # replace 0 avg to avoid div-by-zero

            # OBV — directional volume accumulation
            df["obv"] = ta.obv(df["close"], df["volume"])

            # Get the last row — this is "today's" data (most recent candle)
            last = df.iloc[-1]

            # --- 52-week high/low ---
            # Use the last 252 trading days (~1 year) to find the range
            w52          = df.tail(252)
            week_52_high = w52["high"].max()    # highest point in the last year — key resistance level
            week_52_low  = w52["low"].min()     # lowest point in the last year — key support level

            # --- 20-day consolidation tightness ---
            # Used for BREAKOUT strategy: a range < 8% means the stock was coiling (tight box)
            # (max high - min low over last 20 days) / close × 100 = range as % of price
            d20  = df.tail(20)
            consolidation_pct = (
                (d20["high"].max() - d20["low"].min()) / last["close"] * 100
                if last["close"] > 0 else 999   # 999 = invalid/unknown — will fail the < 8% check
            )

            # OBV trend: is today's OBV higher than 5 days ago?
            # Rising OBV = net institutional accumulation over the past week
            obv_rising = len(df) >= 5 and bool(df["obv"].iloc[-1] > df["obv"].iloc[-5])

            # --- FVG (Fair Value Gap) zones ---
            current_close = float(last["close"])
            fvg_zones     = IndicatorEngine._detect_fvg_zones(df)

            # Use Case 1: price inside a RECENT FVG (≤10 candles old) = uncertain zone
            # The market is actively filling a fresh imbalance → screener blocks entry
            in_fvg_zone = any(
                z["bottom"] <= current_close <= z["top"] and z["age"] <= 10
                for z in fvg_zones
            )

            # Use Case 2: price inside ANY unfilled bullish FVG (any age)
            # When PULLBACK fires here, it's extra confluence → score +10 in strategy.py
            fvg_pullback = any(
                z["bottom"] <= current_close <= z["top"]
                for z in fvg_zones
            )

            # Use Case 3: nearest unfilled bullish FVG above price, within 8% of current price
            # Its bottom is a natural "magnet" → used as target override in risk.py
            fvg_target = 0.0
            for z in sorted(fvg_zones, key=lambda x: x["bottom"]):
                if z["bottom"] > current_close and z["bottom"] <= current_close * 1.08:
                    fvg_target = z["bottom"]
                    break

            # Build and return the StockData object with all computed values
            # pd.isna() checks are needed because early bars (before enough data for EMA-200 etc.) are NaN
            return StockData(
                symbol=symbol,
                date=datetime.now().strftime("%Y-%m-%d"),
                open=float(last["open"]),
                high=float(last["high"]),
                low=float(last["low"]),
                close=float(last["close"]),    # yesterday's closing price (8:45 AM data, market not open yet)
                volume=float(last["volume"]),
                ema_20=float(last["ema_20"])       if not pd.isna(last["ema_20"])       else 0.0,
                ema_50=float(last["ema_50"])       if not pd.isna(last["ema_50"])       else 0.0,
                ema_200=float(last["ema_200"])     if not pd.isna(last["ema_200"])      else 0.0,
                rsi=float(last["rsi"])             if not pd.isna(last["rsi"])          else 50.0,  # default 50 = neutral
                macd=float(last["macd"])           if not pd.isna(last["macd"])         else 0.0,
                macd_signal=float(last["macd_signal"]) if not pd.isna(last["macd_signal"]) else 0.0,
                macd_hist=float(last["macd_hist"]) if not pd.isna(last["macd_hist"])   else 0.0,
                atr=float(last["atr"])             if not pd.isna(last["atr"])          else 0.0,
                bb_upper=float(last["bb_upper"])   if not pd.isna(last["bb_upper"])    else 0.0,
                bb_lower=float(last["bb_lower"])   if not pd.isna(last["bb_lower"])    else 0.0,
                volume_ratio=float(last["vol_ratio"]) if not pd.isna(last["vol_ratio"]) else 1.0,  # default 1.0 = average
                week_52_high=float(week_52_high),
                week_52_low=float(week_52_low),
                rs_score=IndicatorEngine._calc_rs(df, nifty_df),             # relative strength vs Nifty
                candle_pattern=IndicatorEngine._detect_candle_pattern(df),   # MARUBOZU, HAMMER, etc.
                daily_bullish=bool(last["close"] > last["ema_20"] and
                                   last["ema_20"] > last["ema_50"]),          # close > EMA20 > EMA50 = trend intact
                consolidation_range_pct=float(consolidation_pct),
                obv_rising=obv_rising,
                atr_ratio=float(last["atr_ratio"]) if not pd.isna(last["atr_ratio"]) else 1.0,
                in_fvg_zone=in_fvg_zone,
                fvg_pullback=fvg_pullback,
                fvg_target=round(fvg_target, 2),
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

        Formula:
          stock_60d_return = (close_today / close_60days_ago - 1) × 100
          nifty_60d_return = same for Nifty
          rs_score = stock_return - nifty_return
          e.g. +8.5 means the stock returned 8.5% more than Nifty over 60 days
        """
        if nifty_df is None or len(stock_df) < 60 or len(nifty_df) < 60:
            return 0.0  # can't calculate without sufficient history — neutral score
        try:
            s_ret = (stock_df["close"].iloc[-1] / stock_df["close"].iloc[-60] - 1) * 100  # stock's 60-day % return
            n_ret = (nifty_df["close"].iloc[-1] / nifty_df["close"].iloc[-60] - 1) * 100  # Nifty's 60-day % return
            return round(s_ret - n_ret, 2)  # positive = outperforming, negative = underperforming
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
            return "NONE"   # need at least 3 candles for morning star pattern
        try:
            c  = df.iloc[-1]   # today (most recent candle)
            p  = df.iloc[-2]   # yesterday
            p2 = df.iloc[-3]   # day before yesterday

            body_c     = abs(c["close"] - c["open"])                        # size of today's candle body
            range_c    = c["high"] - c["low"] if c["high"] != c["low"] else 0.001  # total range (avoid div-by-zero)
            lower_wick = min(c["open"], c["close"]) - c["low"]             # lower tail length
            upper_wick = c["high"] - max(c["open"], c["close"])            # upper tail length

            # MARUBOZU: body is 85%+ of the full range, high volume = pure buying power, no indecision
            if (c["close"] > c["open"] and body_c / range_c > 0.85 and
                    c["volume"] > df["volume"].tail(20).mean() * 1.5):
                return "MARUBOZU"

            # BULLISH ENGULFING: yesterday was red, today is green AND today's body fully contains yesterday's body
            # Also requires today's volume > yesterday's = more conviction on the reversal
            if (p["close"] < p["open"] and c["close"] > c["open"] and
                    c["open"] < p["close"] and c["close"] > p["open"] and
                    body_c > abs(p["close"] - p["open"]) and c["volume"] > p["volume"]):
                return "BULLISH_ENGULFING"

            # HAMMER: small body at the TOP of the range, long lower wick (>2× body), tiny upper wick
            # Interpretation: sellers pushed price down hard but buyers recovered almost all of it = support
            if (body_c > 0 and lower_wick >= 2 * body_c and
                    upper_wick <= 0.1 * range_c and c["close"] > c["open"]):
                return "HAMMER"

            # MORNING STAR (3-candle): big red → small body (indecision) → big green crossing midpoint of red
            # This is a classic 3-candle reversal pattern at a support level
            body_p  = abs(p["close"]  - p["open"])
            body_p2 = abs(p2["close"] - p2["open"])
            if (p2["close"] < p2["open"] and            # 2 days ago = red candle (sellers in control)
                    body_p < body_p2 * 0.4 and          # yesterday = small body (indecision, spinning top)
                    c["close"] > c["open"] and           # today = green candle (buyers taking over)
                    c["close"] > p2["open"] + body_p2 * 0.5):  # today's close above midpoint of the original red candle
                return "MORNING_STAR"

            # INSIDE BAR: today's high < yesterday's high AND today's low > yesterday's low
            # The range is "inside" the previous candle = market coiling, about to break out
            if c["high"] < p["high"] and c["low"] > p["low"]:
                return "INSIDE_BAR"

        except Exception as e:
            log.debug(f"Candle pattern error: {e}")
        return "NONE"   # no pattern detected

    @staticmethod
    def _detect_fvg_zones(df: "pd.DataFrame", lookback: int = 50) -> list:
        """
        Scans the last `lookback` daily candles for unfilled bullish FVGs.

        Bullish FVG (3-candle pattern):
          Candle 1: any candle         → use its HIGH as FVG bottom
          Candle 2: big impulse move   → price jumped through a range
          Candle 3: candle after move  → use its LOW as FVG top

        Valid FVG: candle_3.low > candle_1.high  (actual gap exists between them)
        Filled: any subsequent candle's low traded down to or below the FVG bottom.

        Returns list of {"bottom": float, "top": float, "age": int}
          age = trading days since the FVG formed (0 = formed yesterday, 10 = 10 days ago)
        Sorted newest first (age ascending).
        """
        if not LIBS_AVAILABLE or df is None or len(df) < 3:
            return []
        df_w = df.tail(lookback).reset_index(drop=True)
        n    = len(df_w)
        zones = []
        for i in range(n - 2):
            c1_high = float(df_w.iloc[i]["high"])
            c3_low  = float(df_w.iloc[i + 2]["low"])
            if c3_low <= c1_high:
                continue   # no gap between candle 1 high and candle 3 low
            # Check if any candle AFTER candle 3 has already filled the gap
            # (filled = a candle's low traded back down to/below the FVG bottom)
            filled = any(
                float(df_w.iloc[j]["low"]) <= c1_high
                for j in range(i + 3, n)
            )
            if not filled:
                age = (n - 1) - (i + 2)   # candles elapsed since candle 3 (0 = most recent)
                zones.append({"bottom": round(c1_high, 2), "top": round(c3_low, 2), "age": age})
        zones.sort(key=lambda z: z["age"])   # newest first
        return zones

    @staticmethod
    def check_4h_bullish(df_4h: Optional["pd.DataFrame"]) -> bool:
        """
        4H timeframe bullish check: is price above the 20-period EMA on the 60-min chart?
        Used to confirm the daily signal has intraday momentum support.
        Returns False if data is insufficient (< 25 hourly bars = 25 hours of data)
        """
        if df_4h is None or len(df_4h) < 25:
            return False
        try:
            ema = ta.ema(df_4h["close"], length=20)   # 20-period EMA on 60-min chart
            return bool(df_4h["close"].iloc[-1] > ema.iloc[-1])   # is current price above EMA?
        except Exception:
            return False

    @staticmethod
    def check_weekly_bullish(df_weekly: Optional["pd.DataFrame"]) -> bool:
        """
        Weekly timeframe bullish check: is price above the 20-week EMA?
        The highest timeframe we check — if weekly is bearish, all daily setups are suspect.
        Needs 22+ bars (22 weeks ≈ 5 months) for the EMA to be meaningful.
        """
        if df_weekly is None or len(df_weekly) < 22:
            return False
        try:
            ema = ta.ema(df_weekly["close"], length=20)   # 20-period EMA on weekly bars
            return bool(df_weekly["close"].iloc[-1] > ema.iloc[-1])  # is current price above weekly EMA?
        except Exception:
            return False
