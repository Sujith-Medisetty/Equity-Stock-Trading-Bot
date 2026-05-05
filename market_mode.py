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

        # VIX override — market fear level takes priority over everything
        if vix >= Config.VIX_PANIC:
            return MarketMode.CASH, FIIFlow.NEUTRAL
        if vix >= Config.VIX_NERVOUS:
            return MarketMode.DEFENSIVE, FIIFlow.SELLING

        # FII flow: need sustained activity (3+ days) above threshold to count
        if (fii_net > Config.FII_FLOW_THRESHOLD_CR and
                fii_consecutive_buy >= Config.FII_CONSECUTIVE_DAYS):
            fii_flow = FIIFlow.BUYING
        elif (fii_net < -Config.FII_FLOW_THRESHOLD_CR and
              fii_consecutive_sell >= Config.FII_CONSECUTIVE_DAYS):
            fii_flow = FIIFlow.SELLING
        else:
            fii_flow = FIIFlow.NEUTRAL

        if nifty_data is None:
            return MarketMode.CAUTIOUS, fii_flow

        above_ema20  = nifty_data.close > nifty_data.ema_20
        above_ema50  = nifty_data.close > nifty_data.ema_50
        above_ema200 = nifty_data.close > nifty_data.ema_200
        rsi          = nifty_data.rsi

        # Strong bull: all 3 EMAs aligned (short > mid > long) + RSI above 50
        if above_ema20 and above_ema50 and above_ema200 and rsi > 50:
            if fii_flow == FIIFlow.BUYING:
                mode = MarketMode.AGGRESSIVE   # everything aligned → full aggression
            elif fii_flow == FIIFlow.SELLING:
                mode = MarketMode.SELECTIVE    # trend up but institutions selling → careful
            else:
                mode = MarketMode.NORMAL

        # Temporary pullback in bull (above 50 and 200 EMA, dipped below 20 EMA)
        elif above_ema50 and above_ema200 and not above_ema20:
            mode = MarketMode.SELECTIVE        # only highest-score setups

        # Sideways: above 200 EMA but not 50 (medium-term trend mixed)
        elif above_ema200 and rsi > 40:
            mode = MarketMode.CAUTIOUS         # 1-2 positions max, very selective

        # Bear: below 200 EMA or RSI < 40 — trend clearly down
        else:
            mode = MarketMode.DEFENSIVE        # no new entries

        log.info(f"Market Mode: {mode.value} | FII: {fii_flow.value} | VIX: {vix:.1f}")
        return mode, fii_flow
