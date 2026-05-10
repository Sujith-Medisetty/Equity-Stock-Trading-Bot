"""
screener.py — Hard filters that run before strategy evaluation.

The screener answers: "Is this stock even worth evaluating today?"
It is NOT looking for trade setups — that's the StrategyEngine's job.
The screener eliminates stocks that have structural reasons to skip,
so the StrategyEngine only sees clean candidates.

Filters applied (any failure = skip the stock):
1. ATR ratio > 1.5 — today's volatility is 50% above its own average.
   These stocks make SL placement unreliable. A wide-ranging day means
   the stock is in news-driven or panic mode — not suitable for swing trades.

2. Already in portfolio — we never add to an existing position. Each stock
   gets at most one open trade at a time.

3. Sector already open — max 1 open trade per sector. Holding ICICIBANK and
   HDFCBANK simultaneously doubles your banking exposure. This prevents that.

4. RED event within 5 days — earnings/results imminent. Prices can gap ±10%
   on results day. We exit before this, never enter into it.

5. Volume ratio < 0.5 — stock traded at less than half its normal volume today.
   Very low volume means the institutional desk isn't interested. Setups on
   thin days often fail or reverse immediately.

6. Market mode DEFENSIVE or CASH — no new entries allowed system-wide.

Note: the screener uses the market-mode-level check as a hard gate, but
individual strategies have their own mode checks too. Both layers apply.
"""

from config import Config, Watchlist, log
from models import MarketMode
from database import DatabaseManager


class StockScreener:
    """
    Runs the pre-screening pass over all stocks in stocks_data.
    Called in step3_screen_stocks() and returns a list of symbols
    that passed all filters. Only these symbols get evaluated by StrategyEngine.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def screen(self, stocks_data: dict, open_trades: list,
               market_mode: MarketMode) -> list:
        """
        Args:
          stocks_data  — dict of symbol → StockData, built in step1_collect_data()
          open_trades  — list of currently open Trade rows from DB
          market_mode  — current MarketMode detected this morning

        Returns:
          List of symbol strings that cleared all filters.
          Screener reasons are logged at DEBUG level so you can see why each was skipped.
        """
        passed       = []

        # Build lookup sets for faster checks: O(1) instead of O(n) for each stock
        open_symbols = {t["symbol"] for t in open_trades}    # symbols we already hold
        open_sectors = {Watchlist.get_sector(t["symbol"]) for t in open_trades}  # sectors already in portfolio

        for symbol, data in stocks_data.items():
            if data is None:
                continue  # IndicatorEngine returned None for this stock — skip

            reasons = []  # collect all reasons to skip — logged together at the end

            # --- Filter 1: ATR ratio ---
            # atr_ratio = today's ATR / 20-day average ATR.
            # > 1.5 means today is 50% more volatile than usual — news event or panic.
            # Exception: if volume is already 2x+ (potential breakout), raise limit to 2.5
            # because a real breakout day naturally has wider range than consolidation days.
            atr_limit = 2.5 if data.volume_ratio >= 2.0 else 1.5
            if data.atr_ratio > atr_limit:
                reasons.append(f"ATR ratio too high ({data.atr_ratio:.2f}, limit {atr_limit})")

            # --- Filter 2: Already holding this stock ---
            # We never add to an existing position — one trade per stock at a time.
            # Adding to a losing position is averaging down (bad risk management).
            # Adding to a winning position without a fresh setup is chasing.
            if symbol in open_symbols:
                reasons.append("Already in portfolio")

            # --- Filter 3: Sector already open ---
            # Max 1 open trade per sector. Prevents sector concentration risk:
            # e.g. if banks are hit by bad news, holding ICICIBANK + HDFCBANK = 2x the loss.
            sector = Watchlist.get_sector(symbol)
            if sector in open_sectors:
                reasons.append(f"Sector {sector} already open")

            # --- Filter 4: Upcoming RED event (earnings/results within 5 days) ---
            # Query events_calendar for this specific stock.
            # days_away <= 5 and risk_level='RED' = imminent event with high gap risk.
            # We never enter INTO an earnings event — the gap can be ±10% and SLs don't protect against overnight gaps.
            event = self.db.fetchone(
                "SELECT * FROM events_calendar WHERE symbol=? AND days_away<=5 AND risk_level='RED'",
                (symbol,)
            )
            if event:
                reasons.append(f"Event in {event['days_away']} days: {event['event_type']}")

            # --- Filter 5: Very low volume ---
            # volume_ratio = today's volume / 20-day average volume.
            # < 0.5 means trading at less than half the normal volume.
            # When institutions are absent, setups on thin days often fail or reverse immediately.
            if data.volume_ratio < 0.5:
                reasons.append(f"Low volume ({data.volume_ratio:.2f}x avg)")

            # --- Filter 6: Market mode gate ---
            # DEFENSIVE: Nifty below EMA200 — all upside setups are fighting the primary trend.
            # CASH: VIX >= 28 — market in full panic, no entries under any circumstances.
            if market_mode in [MarketMode.DEFENSIVE, MarketMode.CASH]:
                reasons.append("Market in defensive/cash mode")

            # --- Filter 7: FVG zone avoidance ---
            # Price inside a RECENT unfilled bullish FVG (formed in last 10 candles).
            # The market is actively filling an imbalance — direction uncertain for most strategies.
            # NOTE: PULLBACK inside an FVG is actually GOOD confluence (price pulled back to
            # institutional order zone). So we pass FVG stocks through here and let
            # strategy.py block non-PULLBACK strategies. The flag is logged but not a hard block.
            # This keeps the screener as a coarse pre-filter; strategy.py is the fine filter.
            if data.in_fvg_zone:
                log.debug(f"FVG NOTE {symbol}: price inside recent unfilled FVG — "
                          "only PULLBACK strategy valid here, others blocked in strategy.py")

            if reasons:
                # Log all reasons at DEBUG level — won't clutter normal output
                # but available if you run with --debug to understand why stocks were skipped
                log.debug(f"SKIP {symbol}: {' | '.join(reasons)}")
            else:
                passed.append(symbol)  # cleared all filters — send to strategy engine

        log.info(f"Screening: {len(passed)}/{len(stocks_data)} passed")
        return passed
