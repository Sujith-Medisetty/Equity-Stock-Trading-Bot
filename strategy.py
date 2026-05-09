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

        # Run all 5 checks. Each returns a Setup object if the strategy qualifies,
        # or None if the stock doesn't meet the criteria.
        for check in [
            self._check_fii_flow(symbol, data, market_mode, fii_flow, fii_sector_buying),
            self._check_52w_breakout(symbol, data, market_mode),
            self._check_breakout(symbol, data, market_mode),
            self._check_pullback(symbol, data, market_mode),
            self._check_swing(symbol, data, market_mode),
        ]:
            if check:
                candidates.append(check)  # only add non-None results

        if not candidates:
            return None  # no strategy qualifies for this stock today

        # Sort by (strategy priority, score) descending.
        # Priority is checked first — FII_FLOW (5) always beats BREAKOUT (3) even if BREAKOUT scored higher.
        # Within same priority, higher score wins.
        candidates.sort(
            key=lambda x: (Config.STRATEGY_PRIORITY.get(x.strategy.value, 0), x.score),
            reverse=True
        )
        return candidates[0]  # return the single best setup

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
        # SWING only works in uptrend market conditions — no use trend-following in a sideways/bear market
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None

        checks = [
            data.close > data.ema_20,          # price above short-term trend line
            data.ema_20 > data.ema_50,          # short-term EMA above medium-term — trend intact
            45 <= data.rsi <= 65,               # healthy momentum zone: not overbought (>65), not weak (<45)
            data.macd > data.macd_signal,       # MACD line above signal line → momentum positive
            data.macd_hist > 0,                 # histogram > 0 means momentum is accelerating upward
            data.volume_ratio > 1.2,            # volume 20%+ above average — real buying, not just drift
            data.tf_aligned_count >= 2,         # at least 2 of 3 timeframes (weekly/daily/4H) bullish
        ]

        # Need at least 6 out of 7 checks — allows one minor failure (e.g. volume slightly low)
        if sum(checks) < 6:
            return None

        # Create the Setup object. entry_price = yesterday's close (will be updated to live price in step5)
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
        # BREAKOUT doesn't work in DEFENSIVE/CASH — no upward momentum in those regimes
        if mode in [MarketMode.DEFENSIVE, MarketMode.CASH]:
            return None

        # Hard gate: without 2x volume, the "breakout" is likely a false move.
        # Return immediately — don't even evaluate the checks below.
        if data.volume_ratio < 2.0:
            return None

        checks = [
            data.consolidation_range_pct < 8.0,  # 20-day range < 8% of price = stock was coiling
            data.close > data.ema_50,             # price above medium-term trend — breakout has support
            data.rsi > 55,                        # momentum confirming the move (above neutral 50)
            data.macd > data.macd_signal,         # MACD says momentum is positive
            data.volume_ratio > 2.0,              # volume at least 2x average (redundant with hard gate, but scores it)
            data.obv_rising,                      # OBV rising = institutional accumulation behind the move
            data.tf_aligned_count >= 1,           # at least 1 other timeframe bullish (daily breakout has some backing)
        ]

        # Need at least 6 of 7
        if sum(checks) < 6:
            return None

        score = self._score(data, checks)

        # Explosive volume (2.5x+) on a breakout is a very strong signal — give bonus points.
        # The extra 10 points can push a 70-score setup to 80 (high-confidence tier).
        if data.volume_ratio > 2.5:
            score = min(100, score + 10)  # cap at 100

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
        # PULLBACK needs a confirmed uptrend to pull back in — not valid in SELECTIVE/CAUTIOUS/DEFENSIVE
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None

        # Guard: if EMA20 is zero, the indicator didn't calculate properly — skip
        if data.ema_20 == 0:
            return None

        # Hard gate: price must be very close to EMA20 (within 0.5%).
        # This is the "touching the EMA" condition. If price is 2% above EMA20, it's not a pullback.
        if abs(data.close - data.ema_20) / data.ema_20 * 100 > 0.5:
            return None

        checks = [
            data.close > data.open,         # green candle on the pullback = buyers stepped in today
            40 <= data.rsi <= 52,           # RSI cooled from uptrend highs but not broken below 40
            data.volume_ratio < 1.5,        # LOW volume on pullback = consolidation (not institutional selling)
            data.macd > 0,                  # MACD still positive = underlying trend intact
            data.weekly_bullish,            # weekly timeframe also bullish = higher-level trend supports entry
            data.obv_rising,               # OBV still rising despite the price pullback = accumulation
            data.tf_aligned_count >= 2,    # confirmation from 2+ timeframes
        ]

        if sum(checks) < 6:
            return None

        score = self._score(data, checks)

        # A reversal candle (HAMMER = long lower wick, buyers rejected the low)
        # or BULLISH_ENGULFING at the EMA20 level is textbook confluence — strong signal.
        if data.candle_pattern in ["HAMMER", "BULLISH_ENGULFING"]:
            score = min(100, score + 15)  # add 15 points, cap at 100

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
        # FII_FLOW only valid when market is in an uptrend — institutions don't create trends in bear markets
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None

        # Both conditions must be true: FII must be buying AND specifically into this stock's sector.
        # If FII is buying in banking but not IT, IT stocks don't qualify even if the stock looks good.
        if fii_flow != FIIFlow.BUYING or not fii_sector_buying:
            return None

        checks = [
            data.close > data.ema_50,          # price above medium-term trend — riding the institutional wave
            50 <= data.rsi <= 70,              # RSI in bullish zone: above neutral, not overbought
            data.volume_ratio > 1.2,           # volume confirming FII activity visible in the stock
            data.obv_rising,                   # OBV rising = net accumulation by institutions
            data.rs_score > 0,                 # THIS stock outperforming Nifty (not just its sector)
            data.macd > data.macd_signal,      # MACD momentum also agreeing
            data.tf_aligned_count >= 2,        # multiple timeframes aligned
        ]

        if sum(checks) < 6:
            return None

        # +10 flat bonus because institutional tailwind is the highest-conviction edge in this system.
        # A stock with FII behind it and all technicals aligned is the highest-probability setup.
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
        # WEEK52 only in full bull market (AGGRESSIVE) — in any other mode, new highs often reverse quickly
        if mode != MarketMode.AGGRESSIVE:
            return None

        # Hard gate 1: price must actually be at or above the 52W high.
        # week_52_high == 0 means data was insufficient to calculate it — skip.
        if data.week_52_high == 0 or data.close < data.week_52_high:
            return None

        # Hard gate 2: check within 1% tolerance.
        # If the 52W high is ₹1000 and price is ₹985, the breakout hasn't happened yet.
        # But if data rounding caused ₹1000 vs ₹1001, we still want to enter.
        if (data.week_52_high - data.close) / data.week_52_high * 100 > 1.0:
            return None

        # Hard gate 3: need strong volume confirmation — 52W breakouts on thin volume usually fail.
        if data.volume_ratio < 2.5:
            return None

        checks = [
            data.close > data.ema_20,     # short-term trend aligned
            data.close > data.ema_50,     # medium-term trend aligned
            data.close > data.ema_200,    # long-term trend aligned — all 3 EMAs stacked bullishly
            60 <= data.rsi <= 75,         # strong momentum (above 60) but not extreme (below 75)
            data.volume_ratio > 2.5,      # explosive volume confirms the breakout is real
            data.obv_rising,              # OBV rising = institutional accumulation behind the move
            data.tf_aligned_count >= 2,   # multiple timeframes bullish
        ]

        # WEEK52 requires ALL 7 checks — no soft failures allowed.
        # The setup is so rare and the expectation so high that any weakness is disqualifying.
        if sum(checks) < 7:
            return None

        # +15 bonus: 52W breakouts in bull markets have the highest continuation rate of all setups.
        # The reasoning: all sellers from the past year are now in profit — no overhead resistance.
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
        # Component 1: check pass rate scaled to 70.
        # e.g. 6/7 checks passed → 6/7 × 70 = 60. All 7 passed → 70.
        score = int(sum(checks) / len(checks) * 70)

        # Component 2: relative strength vs Nifty (60-day return differential).
        # rs_score > 5 = outperforming Nifty by 5%+ = strong institutional interest → +15
        # rs_score > 0 = slight outperformer → +8
        # rs_score <= 0 = underperformer → no bonus (don't buy underperformers)
        if data.rs_score > 5:
            score += 15
        elif data.rs_score > 0:
            score += 8

        # Component 3: candle pattern bonus.
        # A strong bullish candle on the entry day is confirmation that today's price
        # action is agreeing with the setup signal.
        pattern_bonus = {
            "MARUBOZU": 15,           # big green body, no wicks, high volume = dominant buyers
            "BULLISH_ENGULFING": 12,  # today's green candle fully wraps yesterday's red = reversal
            "HAMMER": 10,             # long lower wick = buyers strongly rejected the low
            "MORNING_STAR": 12,       # 3-candle reversal pattern = momentum shift confirmed
            "INSIDE_BAR": 5           # tight range inside previous bar = coiling for breakout
        }
        score += pattern_bonus.get(data.candle_pattern, 0)  # 0 if pattern is "NONE" or unrecognised

        # Ensure score stays within valid 0-100 range
        return min(100, max(0, score))
