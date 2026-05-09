"""
data_collector.py — All inbound data fetching for the system.

Two external sources feed the system:

1. Dhan broker API (dhanhq SDK)
   - Historical OHLCV (daily) for indicator calculation
   - Intraday 60-min OHLCV for 4H timeframe check
   - Used for both paper and live mode — price data is real in both

2. NSE India public API (via pnsea — handles Akamai bot protection automatically)
   - India VIX (fear index) — nse.equity.find_index("INDIA VIX")
   - GIFT Nifty (pre-market global signal) — /api/allIndices
   - FII/DII daily net buying — /api/fiidiiTradeReact
   - Earnings/results calendar — /api/event-calendar

   pnsea bypasses NSE's Akamai WAF, so this works from cloud servers too.
   Install: pip install pnsea

Fallback:
If Dhan is not connected (no credentials, or import failed), _mock_ohlcv() generates
deterministic synthetic price data so the system can run end-to-end in test mode.
The mock uses a fixed seed derived from the symbol name, so the same symbol always
produces the same price history — this makes paper trade runs reproducible.
"""

from datetime import datetime, timedelta
from typing import Optional

from config import Config, Watchlist, log, LIBS_AVAILABLE
from database import DatabaseManager

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

try:
    from pnsea import NSE as PnseaNSE
    PNSEA_AVAILABLE = True
except ImportError:
    PNSEA_AVAILABLE = False
    log.warning("pnsea not installed — NSE data (VIX, FII, events) will be unavailable. pip install pnsea")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass


class DataCollector:
    """
    Fetches all market data needed by the system each morning.

    Called in step1_collect_data() for:
    - VIX, GIFT Nifty, FII/DII from NSE
    - Daily + intraday + weekly OHLCV for each of the 15 watchlist stocks
    - Events calendar (earnings dates) for all watchlist stocks

    The Dhan connection is initialised here AND separately in OrderManager.
    Two separate instances are intentional — DataCollector handles data,
    OrderManager handles orders. They don't share state.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.dhan = None                                       # Dhan API client for historical price data
        self._nse = PnseaNSE() if PNSEA_AVAILABLE else None   # NSE client for VIX, FII, events
        self._init_dhan()

    def _init_dhan(self):
        """Connect to Dhan using credentials from environment variables.
        If credentials are the defaults ("YOUR_CLIENT_ID"), Dhan will still initialise
        but API calls will return auth errors — the system falls back to mock data."""
        if not LIBS_AVAILABLE or dhanhq is None:
            return
        try:
            # DataCollector always uses LIVE credentials (not sandbox) for price data.
            # We want real historical OHLCV even in paper trade mode — only orders are sandboxed.
            self.dhan = dhanhq(Config.DHAN_CLIENT_ID, Config.DHAN_ACCESS_TOKEN)
            log.info("Dhan API connected.")
        except Exception as e:
            log.error(f"Dhan API connection failed: {e}")

    def _nse_get(self, url: str) -> Optional[dict]:
        """
        Fetches a NSE JSON endpoint using pnsea, which handles Akamai bot
        protection automatically — works from cloud servers unlike raw requests.
        Accepts a full URL or a path (prefixed with NSE_BASE if no scheme).
        Returns parsed JSON, or None on any failure.
        """
        if not self._nse:
            log.warning("pnsea not available — skipping NSE fetch")
            return None
        if not url.startswith("http"):
            url = Config.NSE_BASE + url   # prepend the NSE base URL if only path was given
        try:
            resp = self._nse.endpoint_tester(url)   # pnsea handles the Akamai cookie/header challenge
            return resp.json()
        except Exception as e:
            log.error(f"NSE API error {url}: {e}")
            return None

    # ---- OHLCV ----

    def fetch_ohlcv_daily(self, symbol: str, days: int = 250) -> Optional["pd.DataFrame"]:
        """
        Fetches daily candle data from Dhan for the given symbol.
        - 250 days = ~1 trading year — enough for EMA-200 to stabilise
        - 500 days = used for weekly chart (check_weekly_bullish needs 22 weekly bars)

        IMPORTANT: At 8:45 AM (before market opens), the most recent candle in this
        response is YESTERDAY's candle. The market hasn't opened yet, so today's candle
        doesn't exist. This means data.close = yesterday's closing price.
        The live price is fetched separately in step5 before placing any order.

        Returns a sorted DataFrame with lowercase columns: open, high, low, close, volume, date.
        Falls back to _mock_ohlcv() if Dhan is unavailable.
        """
        if not self.dhan or not LIBS_AVAILABLE:
            return self._mock_ohlcv(symbol, days)  # offline fallback — deterministic synthetic data
        try:
            to_date   = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            data = self.dhan.historical_daily_data(
                symbol=symbol, exchange_segment="NSE_EQ",
                instrument_type="EQUITY",
                from_date=from_date, to_date=to_date,
                expiry_code=0   # 0 = equity (not F&O)
            )
            if data and "data" in data:
                df = pd.DataFrame(data["data"])
                df.columns = [c.lower() for c in df.columns]   # normalise to lowercase column names
                df["date"] = pd.to_datetime(df["timestamp"])   # convert timestamp string to datetime
                return df.sort_values("date").reset_index(drop=True)   # ensure chronological order
        except Exception as e:
            log.error(f"OHLCV daily fetch failed for {symbol}: {e}")
        return None

    def fetch_ohlcv_intraday(self, symbol: str, interval: str = "60",
                              days: int = 30) -> Optional["pd.DataFrame"]:
        """
        Fetches 60-minute intraday bars for the 4H timeframe check.
        30 days of 60-min bars = ~180 candles → enough for a 20-period EMA.
        The 4H check (is price above 20-EMA on the 60-min chart?) confirms that
        daily momentum has intraday support before we enter a trade.
        interval="60" means 60-minute bars (1 bar per hour).
        """
        if not self.dhan or not LIBS_AVAILABLE:
            return self._mock_ohlcv(symbol, days * 6)   # 6 hourly bars per day
        try:
            to_date   = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            data = self.dhan.intraday_minute_data(
                symbol=symbol, exchange_segment="NSE_EQ",
                instrument_type="EQUITY", interval=interval,   # "60" = 60-minute bars
                from_date=from_date, to_date=to_date
            )
            if data and "data" in data:
                df = pd.DataFrame(data["data"])
                df.columns = [c.lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["timestamp"])
                return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            log.error(f"Intraday fetch failed for {symbol}: {e}")
        return None

    def _mock_ohlcv(self, symbol: str, bars: int) -> Optional["pd.DataFrame"]:
        """
        Synthetic OHLCV data for testing without a live Dhan connection.
        Uses a deterministic random seed (hash of symbol name) so the same symbol
        always generates the same fake price history — this makes paper trade
        runs reproducible and comparable across restarts.
        Base price varies by symbol so the dashboard shows different stocks at
        different price levels, which looks realistic.
        """
        if not LIBS_AVAILABLE:
            return None
        np.random.seed(abs(hash(symbol)) % 999)     # deterministic seed per symbol
        base   = 1000 + abs(hash(symbol)) % 2000    # base price between ₹1000 and ₹3000 per symbol
        dates  = pd.date_range(end=datetime.now(), periods=bars, freq="B")   # business days
        closes = base + np.cumsum(np.random.randn(bars) * 15)   # random walk from base price
        closes = np.maximum(closes, base * 0.5)                  # floor at 50% of base — prices don't go negative
        return pd.DataFrame({
            "date":   dates,
            "open":   closes * (1 + np.random.randn(bars) * 0.002),   # slight random offset from close
            "high":   closes * (1 + abs(np.random.randn(bars)) * 0.008),   # always above close
            "low":    closes * (1 - abs(np.random.randn(bars)) * 0.008),   # always below close
            "close":  closes,
            "volume": np.random.randint(500000, 5000000, bars).astype(float),  # realistic NSE volume range
        })

    # ---- FII/DII ----

    def fetch_fii_dii(self) -> dict:
        """
        Fetches today's FII and DII net cash buying/selling from NSE.
        Returns {"fii_net_cash": float, "dii_net_cash": float, "date": str}.
        Response is a list of 2 dicts — one per category (FII/FPI and DII).
        netValue is positive = buying, negative = selling (in ₹Cr).
        """
        data   = self._nse_get(Config.NSE_BASE + Config.NSE_FII_DII)
        result = {"fii_net_cash": 0.0, "dii_net_cash": 0.0, "date": ""}
        if not data or not isinstance(data, list):
            return result   # return zero defaults if fetch fails
        try:
            for entry in data:
                cat = entry.get("category", "")
                # netValue can be a string with commas (e.g. "2,345.67") — remove commas before converting
                val = float(str(entry.get("netValue", 0)).replace(",", ""))
                if "FII" in cat:
                    result["fii_net_cash"] = val    # positive = FII buying, negative = FII selling
                    result["date"] = entry.get("date", "")
                elif "DII" in cat:
                    result["dii_net_cash"] = val    # DII (domestic institutions) flow — stored but not used in mode detection
        except Exception as e:
            log.error(f"FII/DII parse error: {e}")
        return result

    def get_fii_consecutive_days(self) -> tuple:
        """
        Counts how many consecutive days FII has been buying or selling.
        Reads the last 10 days from fii_history and walks backwards until
        the direction changes. Returns (buy_streak, sell_streak).
        These are passed to MarketModeEngine.detect() which requires
        FII_CONSECUTIVE_DAYS (default 3) in the same direction before
        elevating or downgrading the market mode.

        Logic:
          Start from most recent day. Count consecutive days in same direction.
          Stop counting as soon as direction changes.
          e.g. [+3000, +2500, +1800, -500] → buy_streak=3, sell_streak=0
        """
        rows = self.db.fetchall(
            "SELECT fii_net_cash FROM fii_history ORDER BY date DESC LIMIT 10"
        )
        if not rows:
            return 0, 0   # no history yet — streaks are 0

        buy_streak = sell_streak = 0
        for r in rows:
            if r["fii_net_cash"] > 0:
                if sell_streak == 0:
                    buy_streak += 1         # still in buying streak — count it
                else:
                    break                   # buying after selling — streak broken
            else:
                if buy_streak == 0:
                    sell_streak += 1        # still in selling streak — count it
                else:
                    break                   # selling after buying — streak broken

        return buy_streak, sell_streak

    # ---- VIX / GIFT Nifty ----

    def fetch_india_vix(self) -> float:
        """
        Fetches India VIX using pnsea's native index lookup.
        VIX >= 28 → CASH mode (no trades). VIX >= 22 → DEFENSIVE mode.
        Defaults to 15.0 (calm market) if the fetch fails — intentionally
        conservative so a failed fetch doesn't accidentally block all trading.

        VIX = India's implied volatility index. It measures the expected
        daily move in Nifty over the next 30 days, derived from option prices.
        VIX 15 = market expects ~1% daily moves (calm)
        VIX 22 = market expects ~1.5% daily moves (nervous)
        VIX 28 = market expects 2%+ daily moves (panic)
        """
        if not self._nse:
            log.warning("VIX fetch unavailable (pnsea missing), using 15.0")
            return 15.0
        try:
            data = self._nse.equity.find_index("INDIA VIX")
            if data:
                # Different API versions return the value under different field names
                raw = data.get("last") or data.get("lastPrice") or data.get("last_price")
                if raw:
                    return float(str(raw).replace(",", ""))   # remove commas from number strings
        except Exception as e:
            log.error(f"VIX fetch error: {e}")
        log.warning("VIX fetch failed, using 15.0")
        return 15.0   # safe default: calm market assumption

    def fetch_gift_nifty(self) -> float:
        """
        GIFT Nifty pre-market signal. Fetched via the allIndices endpoint.
        Not used in trading logic — stored in market_snapshots for the morning briefing.
        A gap between GIFT Nifty and previous Nifty close indicates expected opening gap.
        """
        data = self._nse_get(Config.NSE_BASE + Config.NSE_VIX)
        if data and "data" in data:
            for item in data["data"]:
                if "GIFT" in str(item.get("index", "")).upper():
                    try:
                        return float(item.get("last", 0))
                    except Exception:
                        pass
        return 0.0   # return 0 if not found — not critical

    # ---- Events ----

    def fetch_events_calendar(self) -> list:
        """
        Fetches the NSE earnings/results calendar for the next 30 days,
        filtered to only the 15 watchlist stocks.
        Risk levels:
          RED    → event within 5 days: no new entries, exit open positions
          YELLOW → event within 10 days: reduce position size, monitor closely
          GREEN  → event is far enough away to ignore for now
        Events are saved to events_calendar and checked by StockScreener
        (blocks new entries) and TradeMonitor (triggers exits).
        """
        today  = datetime.now().strftime("%d-%m-%Y")
        future = (datetime.now() + timedelta(days=30)).strftime("%d-%m-%Y")
        # NSE API expects date format dd-mm-yyyy in the query params
        data   = self._nse_get(f"{Config.NSE_BASE}{Config.NSE_EVENTS}?index=equities&from_date={today}&to_date={future}")

        events = []
        if data:
            items = data if isinstance(data, list) else data.get("data", [])  # response could be list or dict with "data" key
            watchlist = Watchlist.get_symbols()
            for item in items:
                sym = item.get("symbol", "")
                if sym not in watchlist:
                    continue   # skip stocks we don't trade

                try:
                    # NSE returns event dates in dd-Mon-yyyy format (e.g. "09-May-2024")
                    ed = datetime.strptime(item.get("date", ""), "%d-%b-%Y")
                    days_away = (ed - datetime.now()).days   # how many days until the event

                    # Classify the event risk level based on how soon it is
                    risk = "RED" if days_away <= 5 else ("YELLOW" if days_away <= 10 else "GREEN")
                    events.append({
                        "symbol":     sym,
                        "event_type": item.get("purpose", "UNKNOWN"),   # e.g. "Quarterly Results", "Dividend"
                        "event_date": ed.strftime("%Y-%m-%d"),
                        "days_away":  days_away,
                        "risk_level": risk
                    })
                except Exception:
                    pass   # skip events with unparseable dates
        return events

    def save_events(self, events: list):
        """
        Saves the fetched events to the events_calendar table.
        Deletes stale events first (anything not updated today) to keep the table current.
        action_required is a human-readable instruction derived from the risk level.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        # Delete events that weren't updated today — they're stale (from a previous run)
        self.db.execute("DELETE FROM events_calendar WHERE updated_at < ?", (today,))

        for e in events:
            # Convert risk level to a clear action instruction
            action = "EXIT NOW" if e["risk_level"] == "RED" else (
                "REDUCE SIZE" if e["risk_level"] == "YELLOW" else "MONITOR"
            )
            self.db.execute("""
                INSERT INTO events_calendar
                (symbol, event_type, event_date, days_away, risk_level, action_required)
                VALUES (?,?,?,?,?,?)
            """, (e["symbol"], e["event_type"], e["event_date"],
                  e["days_away"], e["risk_level"], action))
