"""
data_collector.py — All inbound data fetching for the system.

Two external sources feed the system:

1. Dhan broker API (dhanhq SDK)
   - Historical OHLCV (daily) for indicator calculation
   - Intraday 60-min OHLCV for 4H timeframe check
   - Used for both paper and live mode — price data is real in both

2. NSE India public API (no auth, cookie-based via requests.Session)
   - India VIX (fear index) — from /api/allIndices
   - GIFT Nifty (pre-market global signal) — same endpoint
   - FII/DII daily net buying — from /api/fiidiiTradeReact
   - Earnings/results calendar — from /api/event-calendar

NSE requires a browser-like session: we first hit the main page to get a cookie,
then make the real API call with that cookie. Without this, NSE blocks the request
with 403/401. See Config.NSE_HEADERS — they mimic a Chrome browser.

Fallback:
If Dhan is not connected (no credentials, or import failed), _mock_ohlcv() generates
deterministic synthetic price data so the system can run end-to-end in test mode.
The mock uses a fixed seed derived from the symbol name, so the same symbol always
produces the same price history — this makes paper trade runs reproducible.
"""

import requests
from datetime import datetime, timedelta
from typing import Optional

from config import Config, Watchlist, log, LIBS_AVAILABLE
from database import DatabaseManager

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

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
        self.dhan = None
        self._nse_session = requests.Session()
        self._nse_session.headers.update(Config.NSE_HEADERS)
        self._init_dhan()

    def _init_dhan(self):
        """Connect to Dhan using credentials from environment variables.
        If credentials are the defaults ("YOUR_CLIENT_ID"), Dhan will still initialise
        but API calls will return auth errors — the system falls back to mock data."""
        if not LIBS_AVAILABLE or dhanhq is None:
            return
        try:
            self.dhan = dhanhq(Config.DHAN_CLIENT_ID, Config.DHAN_ACCESS_TOKEN)
            log.info("Dhan API connected.")
        except Exception as e:
            log.error(f"Dhan API connection failed: {e}")

    def _nse_get(self, endpoint: str) -> Optional[dict]:
        """
        Two-step NSE request: first hit the homepage to get a session cookie,
        then call the actual JSON endpoint. NSE requires a Referer header and
        a valid session cookie or it returns empty/blocked responses.
        Returns parsed JSON dict, or None on any failure.
        """
        try:
            self._nse_session.get(Config.NSE_BASE, timeout=10)
            resp = self._nse_session.get(Config.NSE_BASE + endpoint, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.error(f"NSE API error {endpoint}: {e}")
            return None

    # ---- OHLCV ----

    def fetch_ohlcv_daily(self, symbol: str, days: int = 250) -> Optional["pd.DataFrame"]:
        """
        Fetches daily candle data from Dhan for the given symbol.
        - 250 days = ~1 trading year — enough for EMA-200 to stabilise
        - 500 days = used for weekly chart (check_weekly_bullish needs 22 weekly bars)
        Returns a sorted DataFrame with lowercase columns: open, high, low, close, volume, date.
        Falls back to _mock_ohlcv() if Dhan is unavailable.
        """
        if not self.dhan or not LIBS_AVAILABLE:
            return self._mock_ohlcv(symbol, days)
        try:
            to_date   = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            data = self.dhan.historical_daily_data(
                symbol=symbol, exchange_segment="NSE_EQ",
                instrument_type="EQUITY",
                from_date=from_date, to_date=to_date, expiry_code=0
            )
            if data and "data" in data:
                df = pd.DataFrame(data["data"])
                df.columns = [c.lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["timestamp"])
                return df.sort_values("date").reset_index(drop=True)
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
        """
        if not self.dhan or not LIBS_AVAILABLE:
            return self._mock_ohlcv(symbol, days * 6)
        try:
            to_date   = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            data = self.dhan.intraday_minute_data(
                symbol=symbol, exchange_segment="NSE_EQ",
                instrument_type="EQUITY", interval=interval,
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
        np.random.seed(abs(hash(symbol)) % 999)
        base   = 1000 + abs(hash(symbol)) % 2000
        dates  = pd.date_range(end=datetime.now(), periods=bars, freq="B")
        closes = base + np.cumsum(np.random.randn(bars) * 15)
        closes = np.maximum(closes, base * 0.5)
        return pd.DataFrame({
            "date":   dates,
            "open":   closes * (1 + np.random.randn(bars) * 0.002),
            "high":   closes * (1 + abs(np.random.randn(bars)) * 0.008),
            "low":    closes * (1 - abs(np.random.randn(bars)) * 0.008),
            "close":  closes,
            "volume": np.random.randint(500000, 5000000, bars).astype(float),
        })

    # ---- FII/DII ----

    def fetch_fii_dii(self) -> dict:
        """
        Fetches today's FII and DII net cash buying/selling from NSE.
        Returns {"fii_net_cash": float, "dii_net_cash": float, "date": str}.
        netVal = FII net (positive = buying, negative = selling).
        This single day's number is stored in fii_history.
        get_fii_consecutive_days() then counts how many consecutive days
        the same direction has persisted to determine FIIFlow.BUYING/SELLING.
        """
        data   = self._nse_get(Config.NSE_FII_DII)
        result = {"fii_net_cash": 0.0, "dii_net_cash": 0.0, "date": ""}
        if data:
            try:
                latest = data[0] if isinstance(data, list) else data
                result["fii_net_cash"] = float(str(latest.get("netVal", 0)).replace(",", ""))
                result["dii_net_cash"] = float(str(latest.get("diiNetVal", 0)).replace(",", ""))
                result["date"] = latest.get("date", "")
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
        """
        rows = self.db.fetchall(
            "SELECT fii_net_cash FROM fii_history ORDER BY date DESC LIMIT 10"
        )
        if not rows:
            return 0, 0
        buy_streak = sell_streak = 0
        for r in rows:
            if r["fii_net_cash"] > 0:
                if sell_streak == 0:
                    buy_streak += 1
                else:
                    break
            else:
                if buy_streak == 0:
                    sell_streak += 1
                else:
                    break
        return buy_streak, sell_streak

    # ---- VIX / GIFT Nifty ----

    def fetch_india_vix(self) -> float:
        """
        Fetches India VIX from the NSE allIndices endpoint.
        VIX >= 28 → CASH mode (no trades). VIX >= 22 → DEFENSIVE mode.
        Defaults to 15.0 (calm market) if the fetch fails — this is intentionally
        conservative: a failed VIX fetch shouldn't accidentally block all trading.
        """
        data = self._nse_get(Config.NSE_VIX)
        if data and "data" in data:
            for item in data["data"]:
                if "INDIA VIX" in str(item.get("index", "")):
                    try:
                        return float(item.get("last", 15.0))
                    except Exception:
                        pass
        log.warning("VIX fetch failed, using 15.0")
        return 15.0

    def fetch_gift_nifty(self) -> float:
        """
        GIFT Nifty is the Singapore-listed Nifty futures contract that trades
        overnight. Its level vs yesterday's Nifty close gives a pre-market
        read on how the Indian market will likely open. Stored in market_snapshots
        for reference in the morning briefing but not used in trading logic.
        """
        data = self._nse_get(Config.NSE_VIX)
        if data and "data" in data:
            for item in data["data"]:
                if "GIFT" in str(item.get("index", "")).upper():
                    try:
                        return float(item.get("last", 0))
                    except Exception:
                        pass
        return 0.0

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
        data   = self._nse_get(f"{Config.NSE_EVENTS}?index=equities&from_date={today}&to_date={future}")
        events = []
        if data:
            items = data if isinstance(data, list) else data.get("data", [])
            watchlist = Watchlist.get_symbols()
            for item in items:
                sym = item.get("symbol", "")
                if sym not in watchlist:
                    continue
                try:
                    ed = datetime.strptime(item.get("date", ""), "%d-%b-%Y")
                    days_away = (ed - datetime.now()).days
                    risk = "RED" if days_away <= 5 else ("YELLOW" if days_away <= 10 else "GREEN")
                    events.append({
                        "symbol":     sym,
                        "event_type": item.get("purpose", "UNKNOWN"),
                        "event_date": ed.strftime("%Y-%m-%d"),
                        "days_away":  days_away,
                        "risk_level": risk
                    })
                except Exception:
                    pass
        return events

    def save_events(self, events: list):
        today = datetime.now().strftime("%Y-%m-%d")
        self.db.execute("DELETE FROM events_calendar WHERE updated_at < ?", (today,))
        for e in events:
            action = "EXIT NOW" if e["risk_level"] == "RED" else (
                "REDUCE SIZE" if e["risk_level"] == "YELLOW" else "MONITOR"
            )
            self.db.execute("""
                INSERT INTO events_calendar
                (symbol, event_type, event_date, days_away, risk_level, action_required)
                VALUES (?,?,?,?,?,?)
            """, (e["symbol"], e["event_type"], e["event_date"],
                  e["days_away"], e["risk_level"], action))
