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
        open_symbols = {t["symbol"] for t in open_trades}
        open_sectors = {Watchlist.get_sector(t["symbol"]) for t in open_trades}

        for symbol, data in stocks_data.items():
            if data is None:
                continue

            reasons = []

            # ATR filter: reject if volatility is 50%+ above average.
            # Exception: if volume is 2x+ (potential breakout day), raise threshold to 2.5
            # because breakout days naturally have elevated ATR from the range expansion.
            atr_limit = 2.5 if data.volume_ratio >= 2.0 else 1.5
            if data.atr_ratio > atr_limit:
                reasons.append(f"ATR ratio too high ({data.atr_ratio:.2f}, limit {atr_limit})")

            if symbol in open_symbols:
                reasons.append("Already in portfolio")

            sector = Watchlist.get_sector(symbol)
            if sector in open_sectors:
                reasons.append(f"Sector {sector} already open")

            event = self.db.fetchone(
                "SELECT * FROM events_calendar WHERE symbol=? AND days_away<=5 AND risk_level='RED'",
                (symbol,)
            )
            if event:
                reasons.append(f"Event in {event['days_away']} days: {event['event_type']}")

            if data.volume_ratio < 0.5:
                reasons.append(f"Low volume ({data.volume_ratio:.2f}x avg)")

            if market_mode in [MarketMode.DEFENSIVE, MarketMode.CASH]:
                reasons.append("Market in defensive/cash mode")

            if reasons:
                log.debug(f"SKIP {symbol}: {' | '.join(reasons)}")
            else:
                passed.append(symbol)

        log.info(f"Screening: {len(passed)}/{len(stocks_data)} passed")
        return passed
