"""
strategy.py — The 5 entry strategies and their scoring system.

Each strategy is a set of conditions that must ALL (or nearly all) be true
before a Setup is created. After that, RiskManager sizes the position and
validates the risk:reward ratio. Only setups that clear both layers get traded.

The 5 strategies in priority order (highest to lowest):

FII_FLOW (priority 5)
  Strongest edge. Requires FII consistently buying in this stock's sector AND
  the stock itself is outperforming Nifty. When institutions put thousands of crores
  into a sector, individual stocks ride the wave regardless of technicals.
  Only works in NORMAL or AGGRESSIVE market mode.

WEEK52 (priority 4)
  52-week high breakout. Rare but powerful — when a stock hits a new 52W high
  with 2.5x volume, it means the last overhead resistance has been cleared.
  All sellers from the past year are now in profit and not pressing.
  AGGRESSIVE mode only (bull market condition required for this to work).

BREAKOUT (priority 3)
  Price breaking out of a tight consolidation box (< 8% range over 20 days)
  with 2x+ volume. The tight range shows supply/demand in balance; the volume
  spike on the break shows buyers winning the standoff. Works in most modes
  except DEFENSIVE and CASH.

PULLBACK (priority 2)
  Price pulling back to the 20 EMA in an established uptrend and bouncing.
  The pullback is the second-best risk:reward entry in a trend — you're buying
  closer to support (the EMA) with the trend already confirmed above you.
  NORMAL and AGGRESSIVE mode only.

SWING (priority 1)
  General trend-following when multiple timeframes align. All EMAs stacked
  (close > EMA20 > EMA50), RSI in the healthy 45-65 zone, MACD positive,
  and at least 2 of 3 timeframes (weekly/daily/4H) bullish.
  The baseline strategy — works whenever markets are not in a downtrend.

Scoring (0-100):
  70% from how many of the strategy's criteria passed
  15% from relative strength (stock outperforming Nifty)
  15% from candle pattern quality (MARUBOZU, BULLISH_ENGULFING, etc.)
  Minimum to enter: 60. High confidence: 80+.

evaluate_all() runs all 5 strategies on a stock and picks the highest-priority
one with the highest score. One setup per stock per day maximum.
"""

from datetime import datetime
from typing import Optional

from config import Config, log
from models import MarketMode, FIIFlow, StrategyType, StockData, Setup


class StrategyEngine:
    """
    Evaluates all 5 strategies against a stock's technical data.
    Called once per screened stock in step4_find_setups().
    Returns the single best Setup or None if no strategy qualifies.
    """

    def evaluate_all(self, symbol: str, data: StockData,
                     market_mode: MarketMode, fii_flow: FIIFlow,
                     fii_sector_buying: bool = False) -> Optional[Setup]:
        """
        Tries all 5 strategies and returns the best one (by priority, then score).
        A stock could qualify for multiple strategies simultaneously
        (e.g., BREAKOUT + FII_FLOW both pass). We always take the higher-priority one
        because FII backing makes the trade more reliable than a standalone breakout.
        """
        candidates = []

        for check in [
            self._check_fii_flow(symbol, data, market_mode, fii_flow, fii_sector_buying),
            self._check_52w_breakout(symbol, data, market_mode),
            self._check_breakout(symbol, data, market_mode),
            self._check_pullback(symbol, data, market_mode),
            self._check_swing(symbol, data, market_mode),
        ]:
            if check:
                candidates.append(check)

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (Config.STRATEGY_PRIORITY.get(x.strategy.value, 0), x.score),
            reverse=True
        )
        return candidates[0]

    def _check_swing(self, symbol, data, mode):
        """
        Trend-following entry. Requires ALL 7 conditions to be true
        (sum(checks) >= 6 with 7 total is 6 out of 7 minimum — one soft failure allowed).
        Close > EMA20 > EMA50 confirms the trend hierarchy is intact.
        RSI 45-65: above 50 confirms momentum, below 65 means not overbought yet.
        MACD > signal + histogram > 0: momentum is positive and accelerating.
        Volume 1.2x: at least modest volume confirms real buyers, not just drift.
        tf_aligned >= 2: daily signal is confirmed on at least one other timeframe.
        """
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
        if sum(checks) < 6:
            return None
        return Setup(symbol=symbol, date=datetime.now().strftime("%Y-%m-%d"),
                     strategy=StrategyType.SWING, score=self._score(data, checks),
                     entry_price=data.close, market_mode=mode.value, fii_flow="")

    def _check_breakout(self, symbol, data, mode):
        """
        Breakout from tight consolidation. Hard gate: volume_ratio < 2.0 → skip
        immediately (no point scoring a breakout without the volume confirmation).
        consolidation_range_pct < 8%: the 20-day high-low range is < 8% of price.
        Tight coiling before the break is the setup. A 2.5x volume bonus adds 10
        extra score points — explosive volume on a breakout is a very strong signal.
        """
        if mode in [MarketMode.DEFENSIVE, MarketMode.CASH]:
            return None
        if data.volume_ratio < 2.0:
            return None
        checks = [
            data.consolidation_range_pct < 8.0,
            data.close > data.ema_50,
            data.rsi > 55,
            data.macd > data.macd_signal,
            data.volume_ratio > 2.0,
            data.obv_rising,
            data.tf_aligned_count >= 1,
        ]
        if sum(checks) < 6:
            return None
        score = self._score(data, checks)
        if data.volume_ratio > 2.5:
            score = min(100, score + 10)
        return Setup(symbol=symbol, date=datetime.now().strftime("%Y-%m-%d"),
                     strategy=StrategyType.BREAKOUT, score=score,
                     entry_price=data.close, market_mode=mode.value, fii_flow="")

    def _check_pullback(self, symbol, data, mode):
        """
        Pullback to 20 EMA in an uptrend. Hard gate: price must be within 0.5% of
        EMA20 — any further away and it's not actually a clean pullback touch.
        RSI 40-52: momentum has cooled but not broken (< 40 = actual breakdown).
        Low volume (< 1.5x avg) on the pullback confirms it's consolidation, not
        distribution. A HAMMER or BULLISH_ENGULFING candle at this level is a
        strong reversal signal → +15 bonus score.
        """
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None
        if data.ema_20 == 0:
            return None
        if abs(data.close - data.ema_20) / data.ema_20 * 100 > 0.5:
            return None
        checks = [
            data.close > data.open,
            40 <= data.rsi <= 52,
            data.volume_ratio < 1.5,
            data.macd > 0,
            data.weekly_bullish,
            data.obv_rising,
            data.tf_aligned_count >= 2,
        ]
        if sum(checks) < 6:
            return None
        score = self._score(data, checks)
        if data.candle_pattern in ["HAMMER", "BULLISH_ENGULFING"]:
            score = min(100, score + 15)
        return Setup(symbol=symbol, date=datetime.now().strftime("%Y-%m-%d"),
                     strategy=StrategyType.PULLBACK, score=score,
                     entry_price=data.close, market_mode=mode.value, fii_flow="")

    def _check_fii_flow(self, symbol, data, mode, fii_flow, fii_sector_buying):
        """
        FII tailwind play. Requires two conditions before any technical checks:
        1. fii_flow == BUYING: FIIs have been buying for 3+ consecutive days above ₹2000 Cr.
        2. fii_sector_buying: this stock's sector is receiving FII inflows
           (determined by rs_score > 3 for stocks in that sector).
        RS score > 0 check: the stock itself must be outperforming Nifty — FII flow into
        a sector doesn't help if the specific stock is an underperformer within it.
        Gets a flat +10 bonus because institutional tailwind is the strongest edge we track.
        """
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None
        if fii_flow != FIIFlow.BUYING or not fii_sector_buying:
            return None
        checks = [
            data.close > data.ema_50,
            50 <= data.rsi <= 70,
            data.volume_ratio > 1.2,
            data.obv_rising,
            data.rs_score > 0,
            data.macd > data.macd_signal,
            data.tf_aligned_count >= 2,
        ]
        if sum(checks) < 6:
            return None
        score = min(100, self._score(data, checks) + 10)
        return Setup(symbol=symbol, date=datetime.now().strftime("%Y-%m-%d"),
                     strategy=StrategyType.FII_FLOW, score=score,
                     entry_price=data.close, market_mode=mode.value, fii_flow=fii_flow.value)

    def _check_52w_breakout(self, symbol, data, mode):
        """
        52-week high breakout. AGGRESSIVE mode only — this setup only works when
        the overall bull market is strong (FII buying + all EMAs aligned).
        Price must be AT or above the 52W high (within 1% tolerance for rounding).
        Volume must be 2.5x+ — a 52W breakout on light volume often fails.
        RSI 60-75: strong momentum but not yet extreme. All 3 EMAs must be aligned.
        Requires ALL 7 checks to pass (no soft failure allowed: sum(checks) < 7 → skip).
        +15 bonus because 52W breakouts in bull markets have a very high continuation rate.
        """
        if mode != MarketMode.AGGRESSIVE:
            return None
        if data.week_52_high == 0 or data.close < data.week_52_high:
            return None
        if (data.week_52_high - data.close) / data.week_52_high * 100 > 1.0:
            return None
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
        if sum(checks) < 7:
            return None
        score = min(100, self._score(data, checks) + 15)
        return Setup(symbol=symbol, date=datetime.now().strftime("%Y-%m-%d"),
                     strategy=StrategyType.WEEK52, score=score,
                     entry_price=data.close, market_mode=mode.value, fii_flow="")

    def _score(self, data: StockData, checks: list) -> int:
        """
        Scores a setup 0-100 across three components:
        1. Check pass rate (up to 70): what % of the strategy's criteria passed.
        2. Relative strength (8 or 15): stock outperforming Nifty signals smart money.
        3. Candle pattern (0-15): confirmation that today's price action agrees with setup.
        Minimum score to enter a trade is 60 (checked in step5_execute_trades).
        Score >= 80 is considered high-confidence (relevant to check 6 in pre-trade checklist).
        """
        score = int(sum(checks) / len(checks) * 70)
        if data.rs_score > 5:
            score += 15
        elif data.rs_score > 0:
            score += 8
        pattern_bonus = {
            "MARUBOZU": 15, "BULLISH_ENGULFING": 12,
            "HAMMER": 10, "MORNING_STAR": 12, "INSIDE_BAR": 5
        }
        score += pattern_bonus.get(data.candle_pattern, 0)
        return min(100, max(0, score))
