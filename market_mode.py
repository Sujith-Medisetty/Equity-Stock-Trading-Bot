"""
market_mode.py — Detects today's overall market regime.

Called once every morning (step2_detect_market_mode) after data is collected.
The market mode is the master switch that controls everything downstream:
- Which strategies are allowed to run
- How many positions we take
- Whether we trade at all

Decision hierarchy (highest priority first):
1. VIX >= 28   → CASH. Markets panicking. Exit everything, no new trades.
2. VIX >= 22   → DEFENSIVE. High fear. No new entries, protect existing.
3. Nifty trend → AGGRESSIVE / NORMAL / SELECTIVE / CAUTIOUS / DEFENSIVE
   based on EMA alignment and RSI
4. FII flow    → modifies the above (FII buying upgrades, FII selling downgrades)

Why VIX matters:
VIX measures expected market volatility (fear index). When VIX spikes, stocks
move 2-5% in a day and stop losses get blown through. No point trading in chaos.

Why FII matters:
FIIs control ~25% of NSE market cap. When they buy consistently for 3+ days
with ₹2000+ Cr net, prices tend to rise regardless of technical setups.
When they sell, even good setups fail. We want FII as a tailwind, not headwind.
"""

from config import Config, log
from models import MarketMode, FIIFlow, StockData
from database import DatabaseManager


class MarketModeEngine:
    """
    Determines market regime from VIX, Nifty technicals, and FII flow.
    Used by TradingSystem every morning and passed to StrategyEngine
    and StockScreener to gate which setups are valid today.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def detect(self, vix: float, nifty_data: StockData,
               fii_net: float, fii_consecutive_buy: int,
               fii_consecutive_sell: int) -> tuple:
        """
        Returns (MarketMode, FIIFlow).

        VIX overrides everything — checked first.
        Then FII flow is determined independently.
        Then Nifty EMA alignment decides the base mode.
        FII flow can modify the final mode up or down.
        """

        # --- VIX override — market fear level takes priority over everything ---

        # VIX >= 28: full panic. Options are pricing in 2%+ daily moves.
        # SLs get blown through by gaps. Exit all positions, take no new trades.
        if vix >= Config.VIX_PANIC:
            return MarketMode.CASH, FIIFlow.NEUTRAL

        # VIX >= 22: nervous market. Existing positions may be at risk.
        # No new entries — protect what we have, let positions exit naturally or via SL.
        if vix >= Config.VIX_NERVOUS:
            return MarketMode.DEFENSIVE, FIIFlow.SELLING  # SELLING because fear usually accompanies selling

        # --- FII flow determination (independent of Nifty trend) ---

        # FII buying: net > ₹2000 Cr AND sustained for 3+ consecutive days.
        # A single day of buying could be noise. Three consecutive days confirms a trend.
        if (fii_net > Config.FII_FLOW_THRESHOLD_CR and
                fii_consecutive_buy >= Config.FII_CONSECUTIVE_DAYS):
            fii_flow = FIIFlow.BUYING

        # FII selling: net < -₹2000 Cr AND sustained for 3+ consecutive days.
        elif (fii_net < -Config.FII_FLOW_THRESHOLD_CR and
              fii_consecutive_sell >= Config.FII_CONSECUTIVE_DAYS):
            fii_flow = FIIFlow.SELLING

        else:
            fii_flow = FIIFlow.NEUTRAL  # no clear sustained direction

        # If Nifty data wasn't available (e.g. data fetch failed), default to CAUTIOUS.
        # This is conservative — we'd rather miss opportunities than trade without market context.
        if nifty_data is None:
            return MarketMode.CAUTIOUS, fii_flow

        # --- Nifty EMA alignment + RSI → base market mode ---

        # Precompute these booleans for readability in the conditions below
        above_ema20  = nifty_data.close > nifty_data.ema_20    # short-term trend bullish
        above_ema50  = nifty_data.close > nifty_data.ema_50    # medium-term trend bullish
        above_ema200 = nifty_data.close > nifty_data.ema_200   # long-term trend bullish
        rsi          = nifty_data.rsi                           # Nifty's RSI (above 50 = bullish momentum)

        # Strong bull: all 3 EMAs stacked (close > 20 > 50 > 200 implied) AND RSI above 50 (momentum positive)
        # This is the "everything aligned" condition.
        if above_ema20 and above_ema50 and above_ema200 and rsi > 50:
            if fii_flow == FIIFlow.BUYING:
                # Best possible condition: trend + momentum + institutions buying
                # → full aggression, all 4 strategies active, all 4 positions allowed
                mode = MarketMode.AGGRESSIVE
            elif fii_flow == FIIFlow.SELLING:
                # Trend looks fine but institutions are selling into strength — be careful.
                # Could be distribution (smart money exiting into retail buying).
                # Only take highest-score setups.
                mode = MarketMode.SELECTIVE
            else:
                # Trend intact, FII neutral — normal trading conditions
                mode = MarketMode.NORMAL

        # Temporary pullback in an ongoing bull market:
        # Nifty is above the 50-day and 200-day EMA but dipped below the 20-day EMA.
        # This is a normal healthy correction in an uptrend — don't go full defensive.
        # But be selective: only the best setups, since trend is slightly weakened.
        elif above_ema50 and above_ema200 and not above_ema20:
            mode = MarketMode.SELECTIVE  # only highest-score setups pass the strategy filter

        # Sideways / mixed: Nifty is above 200-day (long-term trend still up) but not the 50-day.
        # Also requires RSI > 40 to confirm it's not in a full breakdown.
        # This is a choppy, uncertain market — limit to 1-2 positions, very selective.
        elif above_ema200 and rsi > 40:
            mode = MarketMode.CAUTIOUS   # 1-2 positions max

        # Bear market / breakdown: Nifty below 200-day EMA, or RSI < 40 (momentum broken).
        # In a bear market, all swing trades are fighting the primary downtrend.
        # The few setups that form usually fail or reverse quickly.
        else:
            mode = MarketMode.DEFENSIVE  # no new entries allowed

        log.info(f"Market Mode: {mode.value} | FII: {fii_flow.value} | VIX: {vix:.1f}")
        return mode, fii_flow
