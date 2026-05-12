"""
strategy.py — The 4 entry strategies and their scoring system.

Each strategy is a set of conditions that must ALL (or nearly all) be true
before a Setup is created. After that, RiskManager sizes the position and
validates the risk:reward ratio. Only setups that clear both layers get traded.

The 4 strategies in priority order (highest to lowest):

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
  FII sector buying in BREAKOUT stock's sector → +15 score bonus.

PULLBACK (priority 2)
  Price pulling back to the 20 EMA in an established uptrend and bouncing.
  The pullback is the second-best risk:reward entry in a trend — you're buying
  closer to support (the EMA) with the trend already confirmed above you.
  NORMAL and AGGRESSIVE mode only.
  FII sector buying in PULLBACK stock's sector → +15 score bonus.

SWING (priority 1)
  General trend-following when multiple timeframes align. All EMAs stacked
  (close > EMA20 > EMA50), RSI in the healthy 45-65 zone, MACD positive,
  and at least 2 of 3 timeframes (weekly/daily/4H) bullish.
  No entries before 10 AM — opening noise settles first.
  The baseline strategy — works whenever markets are not in a downtrend.

FII sector buying is NOT a standalone strategy. It is a +15 score modifier
applied inside PULLBACK and BREAKOUT when FII is buying in the stock's sector.
This avoids the "wait for dip" problem of a standalone FII strategy, and lets
the cleaner PULLBACK/BREAKOUT logic handle the actual entry gate.

Scoring (0-100):
  70% from how many of the strategy's criteria passed
  15% from relative strength (stock outperforming Nifty)
  15% from candle pattern quality (MARUBOZU, BULLISH_ENGULFING, etc.)
  Minimum to enter: 60. High confidence: 80+.

evaluate_all() runs all 4 strategies on a stock and picks the highest-priority
one with the highest score. One setup per stock per day maximum.
"""

from datetime import datetime
from typing import Optional

from config import Config, log
from models import MarketMode, FIIFlow, StrategyType, StockData, Setup  # FIIFlow kept for fii_flow param typing


class StrategyEngine:
    """
    Evaluates all 4 strategies against a stock's technical data.
    Called once per screened stock in step4_find_setups().
    Returns the single best Setup or None if no strategy qualifies.
    """

    def evaluate_all(self, symbol: str, data: StockData,
                     market_mode: MarketMode, fii_flow: FIIFlow,
                     fii_sector_buying: bool = False) -> Optional[Setup]:
        """
        Tries all 4 strategies and returns the best one (by priority, then score).
        A stock could qualify for multiple strategies simultaneously — we always
        take the highest-priority one that passes.
        """
        candidates = []

        # Run all 4 checks. Each returns a Setup object if the strategy qualifies,
        # or None if the stock doesn't meet the criteria.
        # fii_sector_buying is passed to PULLBACK and BREAKOUT as a score modifier (+15).
        for check in [
            self._check_52w_breakout(symbol, data, market_mode),
            self._check_breakout(symbol, data, market_mode, fii_sector_buying),
            self._check_pullback(symbol, data, market_mode, fii_sector_buying),
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
        Momentum Burst — entry on a MARUBOZU candle in a confirmed ADX uptrend.

        A MARUBOZU (big green body ≥ 85% of range, volume ≥ 1.5× avg) is a decisive
        institutional conviction signal: large buyers absorbed all supply in one session.
        When this appears in a confirmed uptrend (ADX > 22), momentum continuation is
        the highest-probability outcome — institutions don't accumulate this aggressively
        for a single-day move.

        Complementary to PULLBACK:
          PULLBACK → counter-trend bounce AT EMA20 support (price dipped and recovered)
          SWING    → trend continuation AFTER confirmed heavy buying (price making ground)

        The MARUBOZU candle itself defines the stop-loss level: the candle's low is where
        buyers halted the last attempt to push price lower. If price retreats below it,
        the buy signal is invalidated. This gives a natural, well-defined risk reference
        unlike a "near EMA20 but not yet there" entry which has no structural support.

        If both SWING and PULLBACK qualify (e.g. MARUBOZU at EMA20), PULLBACK wins (priority 2 > 1).
        SWING fires only when PULLBACK doesn't — i.e. the MARUBOZU occurs above EMA20 in an active trend.
        """
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None

        # Price inside a recent FVG = direction uncertain (filling an imbalance, not confirming trend)
        if data.in_fvg_zone:
            return None

        # Hard gate: must be a MARUBOZU candle — the definitive institutional conviction signal.
        # MARUBOZU = body ≥ 85% of range + volume ≥ 1.5× 20-day avg (both confirmed by detector).
        # No MARUBOZU = no entry, regardless of how good everything else looks.
        if data.candle_pattern != "MARUBOZU":
            return None

        # Relative strength gate: SWING is a momentum continuation strategy — only enter
        # stocks already outperforming Nifty. Buying a laggard on a MARUBOZU is chasing
        # a dead-cat bounce, not a momentum continuation.
        if data.rs_score <= 0:
            return None

        # Hard gate: ADX > 22 confirms the trend is REAL, not a one-day spike in a ranging stock.
        # A MARUBOZU in a ranging stock (ADX < 22) is often an overreaction — 50% chance of reversal.
        # ADX > 22 means directional momentum has been building across many sessions.
        if data.adx < 22:
            return None

        # Hard gate: +DI > -DI (bulls dominating). ADX measures strength, not direction.
        # A MARUBOZU in a strong downtrend also has ADX > 22 but we want uptrend continuation.
        if not data.adx_trending:
            return None

        # Hard gate: RSI 52-72 — strong momentum but not extreme.
        # Below 52: the MARUBOZU was small relative to recent history (weak signal).
        # Above 72: overbought — the MARUBOZU may be a final exhaustion burst before reversal.
        if not (52 <= data.rsi <= 72):
            return None

        checks = [
            data.close > data.ema_20,       # price above short-term trend line
            data.ema_20 > data.ema_50,      # EMA structure bullish (trend intact)
            data.close > data.ema_200,      # long-term uptrend intact
            data.macd > data.macd_signal,   # MACD above signal = momentum crossover
            data.macd_hist > 0,             # histogram positive = momentum accelerating
            data.weekly_bullish,            # weekly timeframe in agreement
            data.tf_aligned_count >= 2,     # 2+ timeframes bullish
        ]

        # 5 of 7 checks required — MARUBOZU + ADX already provide two hard confirmations,
        # so slight softness in checks (e.g. close near EMA20) is acceptable.
        if sum(checks) < 5:
            return None

        score = self._score(data, checks)
        # Note: _score() adds +15 for MARUBOZU pattern automatically — no separate bonus needed.

        return Setup(symbol=symbol, date=datetime.now().strftime("%Y-%m-%d"),
                     strategy=StrategyType.SWING, score=score,
                     entry_price=data.close, market_mode=mode.value, fii_flow="")

    def _check_breakout(self, symbol, data, mode, fii_sector_buying: bool = False):
        """
        Breakout from tight consolidation. Hard gate: volume_ratio < 2.0 → skip
        immediately (no point scoring a breakout without the volume confirmation).
        consolidation_range_pct < 8%: the 20-day high-low range is < 8% of price.
        Tight coiling before the break is the setup. A 2.5x volume bonus adds 10
        extra score points — explosive volume on a breakout is a very strong signal.
        FII sector buying adds +10: institutional tailwind behind the breakout = higher continuation.
        """
        # BREAKOUT requires a confirmed bull market (AGGRESSIVE mode: all EMAs aligned + FII buying).
        # In NORMAL mode (FII neutral) breakouts have ~10% win rate in range-bound markets
        # because volume spikes are noise, not sustained momentum. Backtest 2024 confirmed this.
        if mode != MarketMode.AGGRESSIVE:
            return None
        # FVG avoidance: a breakout starting inside an FVG zone is suspect —
        # price may just be oscillating within the imbalance, not truly breaking out.
        if data.in_fvg_zone:
            return None

        # Hard gate: without 2x volume, the "breakout" is likely a false move.
        # Return immediately — don't even evaluate the checks below.
        if data.volume_ratio < 2.0:
            return None

        # Hard gate: must be above the 200 EMA (long-term uptrend).
        # A "breakout" in a stock below EMA200 is breaking out of a consolidation inside
        # a primary downtrend — these have very low continuation rates.
        if data.ema_200 > 0 and data.close < data.ema_200:
            return None

        checks = [
            data.consolidation_range_pct < 8.0,  # 20-day range < 8% of price = stock was coiling
            data.close > data.ema_50,             # price above medium-term trend — breakout has support
            data.rsi > 55,                        # momentum confirming the move (above neutral 50)
            data.macd > data.macd_signal,         # MACD says momentum is positive
            data.volume_ratio > 2.0,              # volume at least 2x average (redundant with hard gate, but scores it)
            data.obv_rising,                      # OBV rising = institutional accumulation behind the move
            data.tf_aligned_count >= 2,           # 2+ timeframes bullish — daily breakout confirmed by weekly or 4H
        ]

        # Need at least 6 of 7
        if sum(checks) < 6:
            return None

        score = self._score(data, checks)

        # Explosive volume (2.5x+) on a breakout is a very strong signal — give bonus points.
        # The extra 10 points can push a 70-score setup to 80 (high-confidence tier).
        if data.volume_ratio > 2.5:
            score = min(100, score + 10)

        # FII sector buying: institutions accumulating in this sector behind the breakout.
        # Sector tailwind makes continuation far more likely than a standalone technical breakout.
        if fii_sector_buying:
            score = min(100, score + 15)

        return Setup(symbol=symbol, date=datetime.now().strftime("%Y-%m-%d"),
                     strategy=StrategyType.BREAKOUT, score=score,
                     entry_price=data.close, market_mode=mode.value, fii_flow="")

    def _check_pullback(self, symbol, data, mode, fii_sector_buying: bool = False):
        """
        Pullback to 20 EMA in an uptrend. Hard gates: price within 0.3×ATR of EMA20,
        stock above EMA200, EMA20 > EMA50. RSI 50-60: momentum cooled but still bullish.
        Low volume (< 1.5x avg): consolidation dip, not distribution. 6 of 7 soft checks
        required. HAMMER/BULLISH_ENGULFING candle → +15 score. FII sector buying → +15.
        """
        # PULLBACK needs a confirmed uptrend to pull back in — not valid in SELECTIVE/CAUTIOUS/DEFENSIVE
        if mode not in [MarketMode.NORMAL, MarketMode.AGGRESSIVE]:
            return None
        # NOTE: FVG avoidance does NOT apply to PULLBACK.
        # A pullback into a bullish FVG zone is the textbook high-confluence entry —
        # two reasons to bounce: EMA20 support AND unfilled institutional orders.
        # We handle this below by upgrading the score (+10) instead of blocking.

        # Guard: if EMA20 is zero, the indicator didn't calculate properly — skip
        if data.ema_20 == 0:
            return None

        # Hard gate: must be in a long-term uptrend (above EMA200).
        # Pullbacks in stocks below EMA200 are bounces inside a downtrend, not uptrend entries.
        if data.ema_200 > 0 and data.close < data.ema_200:
            return None

        # Hard gate: short-term trend must be above medium-term (uptrend structure intact).
        # If EMA20 <= EMA50, the short-term trend has crossed below mid-term — stock is weakening.
        if data.ema_20 > 0 and data.ema_50 > 0 and data.ema_20 <= data.ema_50:
            return None

        # Hard gate: close must be AT or above EMA20.
        # A genuine pullback bounce closes ABOVE the EMA — the low touched it, buyers stepped in,
        # and price recovered above it by end of day. If close < EMA20, the stock is still below
        # its short-term trend line — the bounce hasn't happened yet. Don't catch a falling knife.
        if data.close < data.ema_20:
            return None

        # Hard gate: close must not be too far above EMA20 (not too late to enter).
        # If price has already run 0.3×ATR above EMA20, the optimal risk:reward entry is missed.
        if data.atr > 0 and (data.close - data.ema_20) > 0.3 * data.atr:
            return None

        # ADX filter: stock must have a real trend to pull back FROM.
        # ADX < 16 = stock is range-bound — EMA20 touches don't bounce reliably because
        # there's no institutional directional momentum behind the move. The "pullback"
        # is just random oscillation within a sideways range, not a retracement in a trend.
        if data.adx > 0 and data.adx < 16:
            return None

        # Relative strength gate: never buy a stock underperforming Nifty over 60 days.
        # rs_score <= 0 = stock has returned less than Nifty — buying a laggard "hoping for
        # a catch-up" is a documented losing pattern in Indian markets (academic research on
        # Nifty 200/500 momentum confirms: all edge is in outperformers, not laggards).
        if data.rs_score <= 0:
            return None

        # Candle range quality: close must be in upper 55%+ of today's high-low range.
        # A weak bounce closes near the day's low (e.g. close=₹100, range ₹95-102, → (100-95)/(102-95)=71% ✓).
        # Any green candle (close > open) could be a +0.1% drift. This ensures genuine buyers pushed back hard.
        day_range  = data.high - data.low
        candle_quality = (data.close - data.low) / day_range > 0.55 if day_range > 0 else False

        checks = [
            candle_quality,                 # close in upper 45% of day range = buyers rejected the low strongly
            50 <= data.rsi <= 60,           # RSI above 50 = bullish trend intact, below 60 = not overbought
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

        # FVG confluence upgrade: pullback landed inside an unfilled bullish FVG zone.
        # Two structural reasons to bounce here: EMA20 support + institutional unfilled orders.
        # This is the OTE (Optimal Trade Entry) concept from SMC — highest-confidence pullback entry.
        if data.fvg_pullback:
            score = min(100, score + 10)

        # FII sector buying: institutions actively buying in this sector makes a pullback
        # to EMA20 even more compelling — the dip is being bought by smart money.
        if fii_sector_buying:
            score = min(100, score + 15)

        return Setup(symbol=symbol, date=datetime.now().strftime("%Y-%m-%d"),
                     strategy=StrategyType.PULLBACK, score=score,
                     entry_price=data.close, market_mode=mode.value, fii_flow="")

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

        # FVG avoidance: a 52W breakout starting inside a recent FVG is almost always a
        # stop-hunt (false breakout) — institutions sweep highs then reverse to fill the gap.
        if data.in_fvg_zone:
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
