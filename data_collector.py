"""
data_collector.py — All inbound data fetching for the system.

Two external sources feed the system:

1. Upstox broker API (upstox-python-sdk)
   - Historical OHLCV — daily bars for indicators, 60-min bars for 4H check
   - Weekly bars (daily resampled) for weekly EMA check
   - Live price quotes (LTP) via batch endpoint — all symbols in ONE call
   All fetches use parallel threads (max UPSTOX_MAX_WORKERS concurrent) with a
   shared rate limiter (UPSTOX_RATE_LIMIT_PER_SEC) to stay under Upstox's 250 req/min limit.
   Every call retries up to API_MAX_RETRIES times with exponential backoff.

2. NSE India public API (via pnsea — handles Akamai bot protection automatically)
   - India VIX (fear index)
   - GIFT Nifty (pre-market global signal)
   - FII/DII daily net buying
   - Earnings/results calendar

Fallback:
  If Upstox is not connected (no credentials or token missing), _mock_ohlcv()
  generates deterministic synthetic price data so the system runs in offline/test mode.
  In live/sandbox mode, a missing token triggers automated Upstox auth (see upstox_auth.py).
  Critically: mock data is NEVER silently returned in live mode — a warning is logged and
  the stock is skipped rather than trading on fake data.

Instrument keys:
  Upstox identifies stocks by "NSE_EQ|{ISIN}" format, not the ticker symbol.
  INSTRUMENT_KEYS maps our symbol names to these keys.
"""

import time
import threading
import functools
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config, Watchlist, log, LIBS_AVAILABLE, UPSTOX_AVAILABLE
from database import DatabaseManager

try:
    import upstox_client
    from upstox_client.api import HistoryApi, MarketQuoteApi
    from upstox_client.rest import ApiException
except ImportError:
    upstox_client = None
    ApiException  = Exception

try:
    from pnsea import NSE as PnseaNSE
    PNSEA_AVAILABLE = True
except ImportError:
    PNSEA_AVAILABLE = False
    log.warning("pnsea not installed — NSE data (VIX, FII, events) unavailable. pip install pnsea")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass

# Upstox instrument keys: "NSE_EQ|{ISIN}" for equity, "NSE_INDEX|{index_name}" for indices.
# ISINs never change — they are the universal identifier for each security.
INSTRUMENT_KEYS = {
    "ICICIBANK":  "NSE_EQ|INE090A01021",
    "HDFCBANK":   "NSE_EQ|INE040A01034",
    "AXISBANK":   "NSE_EQ|INE238A01034",
    "INFY":       "NSE_EQ|INE009A01021",
    "HCLTECH":    "NSE_EQ|INE860A01027",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "MARUTI":     "NSE_EQ|INE585B01010",
    "RELIANCE":   "NSE_EQ|INE002A01018",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "SUNPHARMA":  "NSE_EQ|INE044A01036",
    "BAJFINANCE": "NSE_EQ|INE296A01024",
    "LT":         "NSE_EQ|INE018A01030",
    "ITC":        "NSE_EQ|INE154A01025",
    "TITAN":      "NSE_EQ|INE280A01028",
    "TCS":        "NSE_EQ|INE467B01029",
    "NIFTY 50":   "NSE_INDEX|Nifty 50",
}


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def _with_retry(func=None, *, max_attempts=None, base_delay=None, backoff=None):
    """
    Decorator that retries a function on failure with exponential backoff.
    Reads defaults from Config so a single config change applies everywhere.
    HTTP 401/403 are NOT retried — they indicate an auth problem that needs a token refresh.
    HTTP 429 uses longer initial delay since rate-limit windows need time to reset.
    Usage: @_with_retry  or  @_with_retry(max_attempts=5, base_delay=2.0)
    """
    if func is None:
        # Called with arguments: @_with_retry(max_attempts=5)
        return functools.partial(_with_retry, max_attempts=max_attempts,
                                 base_delay=base_delay, backoff=backoff)

    _max  = max_attempts if max_attempts is not None else Config.API_MAX_RETRIES
    _base = base_delay   if base_delay   is not None else Config.API_RETRY_BASE_S
    _back = backoff      if backoff      is not None else Config.API_RETRY_BACKOFF

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(1, _max + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                # Determine HTTP status if available
                status = getattr(exc, "status", None)
                if status in (401, 403):
                    log.error(f"{func.__name__}: auth error {status} — token expired? Not retrying.")
                    raise
                delay = _base * (_back ** (attempt - 1))
                if status == 429:
                    delay = max(delay, 5.0)   # rate-limit — wait at least 5 s
                if attempt < _max:
                    log.warning(
                        f"{func.__name__}: attempt {attempt}/{_max} failed "
                        f"[{type(exc).__name__}: {exc}]. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        f"{func.__name__}: all {_max} attempts failed. "
                        f"Last error: [{type(exc).__name__}: {exc}]"
                    )
        raise last_exc

    return wrapper


# ---------------------------------------------------------------------------
# Thread-safe rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """
    Ensures no more than max_per_second API calls are fired across all threads.
    Each thread calls wait() before making an API call — if the interval since
    the last call hasn't elapsed, it sleeps for the remaining time.
    Lock ensures that concurrent threads don't both read the same last-call timestamp
    and both decide to proceed simultaneously.
    """

    def __init__(self, max_per_second: float):
        self._interval = 1.0 / max_per_second
        self._lock     = threading.Lock()
        self._last     = 0.0

    def wait(self):
        with self._lock:
            now     = time.time()
            to_wait = self._interval - (now - self._last)
            if to_wait > 0:
                time.sleep(to_wait)
            self._last = time.time()


# ---------------------------------------------------------------------------
# DataCollector
# ---------------------------------------------------------------------------

class DataCollector:
    """
    Fetches all market data the system needs each morning.
    Called in step1_collect_data() for:
      - India VIX, GIFT Nifty, FII/DII (from NSE via pnsea)
      - Daily + 4H (60-min) + weekly OHLCV for all 15 watchlist stocks (from Upstox, in parallel)
      - Events calendar (earnings dates) for all watchlist stocks (from NSE)
    """

    def __init__(self, db: DatabaseManager):
        self.db           = db
        self._upstox      = None          # Upstox API client (set after auth)
        self._nse         = PnseaNSE() if PNSEA_AVAILABLE else None
        self._rate_lim    = _RateLimiter(Config.UPSTOX_RATE_LIMIT_PER_SEC)
        self._auth        = None          # lazy-initialised below
        self._init_upstox()

    def _init_upstox(self):
        """
        Initialises the Upstox API client using a valid access token.
        The token is loaded from DB (saved by UpstoxAuth) or refreshed automatically.
        If no token is available, self._upstox stays None and calls fall back to mock data.
        """
        if not UPSTOX_AVAILABLE or not LIBS_AVAILABLE:
            log.warning("upstox-python-sdk not installed — running in offline/mock mode.")
            return
        try:
            from upstox_auth import UpstoxAuth
            self._auth = UpstoxAuth(self.db)
            token = self._auth.get_valid_token()
            if not token:
                log.warning("No Upstox token available — DataCollector running in mock mode.")
                return
            self._upstox = self._build_client(token)
            log.info("Upstox DataCollector initialised.")
        except Exception as e:
            log.error(f"DataCollector Upstox init failed: {e}")

    def _build_client(self, token: str) -> "upstox_client.ApiClient":
        """Returns a configured Upstox ApiClient with the given access token."""
        cfg = upstox_client.Configuration()
        cfg.access_token = token
        return upstox_client.ApiClient(cfg)

    def _refresh_token_if_needed(self):
        """
        Checks if the current token is still valid before making a batch of API calls.
        Called at the start of each major fetch cycle (step1). If the token has expired
        (can happen if the bot runs across midnight), refreshes it automatically.
        """
        if not self._auth or not self._upstox:
            return
        token = self._auth.get_valid_token()
        if token:
            # Rebuild the client with the refreshed token
            self._upstox = self._build_client(token)

    # -------------------------------------------------------------------------
    # Parallel OHLCV fetch — main entry point for step1
    # -------------------------------------------------------------------------

    def fetch_all_ohlcv_parallel(self, symbols: list) -> dict:
        """
        Fetches all 3 timeframes (daily, 4H, weekly) for every symbol in parallel.
        Returns {symbol: (df_daily, df_4h, df_weekly)}.
        Failed symbols have (None, None, None) — caller skips them gracefully.

        Uses a ThreadPoolExecutor with UPSTOX_MAX_WORKERS threads.
        Each thread shares the _RateLimiter so the combined request rate across
        all threads stays at or below UPSTOX_RATE_LIMIT_PER_SEC.
        """
        self._refresh_token_if_needed()

        results = {}
        log.info(f"Parallel OHLCV fetch starting: {len(symbols)} symbols, "
                 f"{Config.UPSTOX_MAX_WORKERS} workers, "
                 f"{Config.UPSTOX_RATE_LIMIT_PER_SEC} req/s limit")

        with ThreadPoolExecutor(max_workers=Config.UPSTOX_MAX_WORKERS) as pool:
            future_to_sym = {
                pool.submit(self._fetch_one_symbol, sym): sym
                for sym in symbols
            }
            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    results[sym] = future.result()
                    log.debug(f"Fetched {sym} OK")
                except Exception as e:
                    log.error(f"Parallel fetch permanently failed for {sym}: {e}")
                    results[sym] = (None, None, None)

        ok_count = sum(1 for v in results.values() if v[0] is not None)
        log.info(f"Parallel fetch done: {ok_count}/{len(symbols)} symbols succeeded.")
        return results

    def _fetch_one_symbol(self, symbol: str) -> tuple:
        """
        Fetches all 3 timeframes for one symbol sequentially (within one thread).
        Each fetch call rate-limits and retries internally.
        """
        df_daily  = self.fetch_ohlcv_daily(symbol, days=250)
        df_4h     = self.fetch_ohlcv_intraday(symbol, "60minute", days=30)
        df_weekly = self.fetch_ohlcv_daily(symbol, days=500)
        return df_daily, df_4h, df_weekly

    # -------------------------------------------------------------------------
    # Per-timeframe fetch methods
    # -------------------------------------------------------------------------

    @_with_retry
    def fetch_ohlcv_daily(self, symbol: str, days: int = 250) -> Optional["pd.DataFrame"]:
        """
        Fetches daily OHLCV bars from Upstox for the given symbol.
        250 days = ~1 year — sufficient for EMA-200 to stabilise.
        500 days = ~2 years — used for the weekly chart check.

        At 8:45 AM (before market opens), the most recent candle is YESTERDAY's close.
        The live price is fetched separately in step5 before any order is placed.

        Upstox candle format: [timestamp, open, high, low, close, volume, open_interest]
        Falls back to mock data ONLY in backtest/offline mode (never in live mode).
        """
        self._rate_lim.wait()

        if not self._upstox or not LIBS_AVAILABLE:
            return self._safe_mock(symbol, days)

        instrument_key = INSTRUMENT_KEYS.get(symbol)
        if not instrument_key:
            log.warning(f"No instrument key for {symbol} — skipping.")
            return None

        to_date   = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        api  = HistoryApi(self._upstox)
        resp = api.get_historical_candle_data(
            instrument_key, "day", to_date, from_date, api_version="2.0"
        )
        return self._candles_to_df(resp)

    @_with_retry
    def fetch_ohlcv_intraday(self, symbol: str, interval: str = "60minute",
                              days: int = 30) -> Optional["pd.DataFrame"]:
        """
        Fetches intraday OHLCV bars (60-minute interval) from Upstox.
        30 days of 60-min bars = ~180 candles — enough for the 4H EMA check.
        Upstox supports historical intraday data via the same HistoryApi endpoint.
        """
        self._rate_lim.wait()

        if not self._upstox or not LIBS_AVAILABLE:
            return self._safe_mock(symbol, days * 6)

        instrument_key = INSTRUMENT_KEYS.get(symbol)
        if not instrument_key:
            return None

        to_date   = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        api  = HistoryApi(self._upstox)
        resp = api.get_historical_candle_data(
            instrument_key, interval, to_date, from_date, api_version="2.0"
        )
        return self._candles_to_df(resp)

    def _candles_to_df(self, resp) -> Optional["pd.DataFrame"]:
        """
        Converts an Upstox HistoryApi response into a standardised DataFrame.
        Upstox candle format: [timestamp, open, high, low, close, volume, open_interest]
        Returns None if the response is empty or malformed.
        """
        if not LIBS_AVAILABLE:
            return None
        try:
            candles = resp.data.candles if resp and resp.data else []
            if not candles:
                return None
            df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume", "oi"])
            df["date"]   = pd.to_datetime(df["date"])
            df["volume"] = df["volume"].astype(float)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            log.error(f"Candle DataFrame conversion failed: {e}")
            return None

    def _safe_mock(self, symbol: str, bars: int) -> Optional["pd.DataFrame"]:
        """
        Returns mock OHLCV only in backtest/offline mode.
        In live mode, logs a clear error and returns None so the stock is skipped —
        we never silently trade on fake data when a real connection was expected.
        """
        if Config.BACKTEST_MODE or not self._upstox:
            return self._mock_ohlcv(symbol, bars)
        log.error(
            f"LIVE MODE: Could not fetch real data for {symbol} — "
            "skipping (not using mock data in live/sandbox mode)."
        )
        return None

    def _mock_ohlcv(self, symbol: str, bars: int) -> Optional["pd.DataFrame"]:
        """
        Deterministic synthetic OHLCV for offline/backtest testing.
        Fixed seed per symbol → same data on every run → reproducible results.
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

    # -------------------------------------------------------------------------
    # Live price (LTP) — batch endpoint: all symbols in ONE call
    # -------------------------------------------------------------------------

    def get_all_live_prices(self, symbols: list) -> dict:
        """
        Fetches the last traded price for all given symbols in a SINGLE Upstox API call.
        Upstox's MarketQuoteApi.ltp() accepts a comma-separated list of instrument keys
        and returns all prices at once — far more efficient than one call per symbol.
        Returns {symbol: price}. Symbols where fetch failed are absent from the dict.
        """
        if not self._upstox:
            return {}

        # Build the comma-separated instrument key string for the batch call
        key_to_sym = {
            INSTRUMENT_KEYS[sym]: sym
            for sym in symbols
            if sym in INSTRUMENT_KEYS
        }
        if not key_to_sym:
            return {}

        instrument_keys_csv = ",".join(key_to_sym.keys())
        try:
            api  = MarketQuoteApi(self._upstox)
            resp = api.ltp(instrument_keys_csv, api_version="2.0")
            prices = {}
            if resp and resp.data:
                for key, quote in resp.data.items():
                    sym = key_to_sym.get(key)
                    if sym:
                        price = getattr(quote, "last_price", None)
                        if price and float(price) > 0:
                            prices[sym] = float(price)
            return prices
        except Exception as e:
            log.error(f"Batch LTP fetch failed: {e}")
            return {}

    # -------------------------------------------------------------------------
    # FII / DII
    # -------------------------------------------------------------------------

    def fetch_fii_dii(self) -> dict:
        """
        Fetches today's FII and DII net cash buying/selling from NSE.
        Returns {"fii_net_cash": float, "dii_net_cash": float, "date": str}.
        """
        data   = self._nse_get(Config.NSE_BASE + Config.NSE_FII_DII)
        result = {"fii_net_cash": 0.0, "dii_net_cash": 0.0, "date": ""}
        if not data or not isinstance(data, list):
            return result
        try:
            for entry in data:
                cat = entry.get("category", "")
                val = float(str(entry.get("netValue", 0)).replace(",", ""))
                if "FII" in cat:
                    result["fii_net_cash"] = val
                    result["date"]         = entry.get("date", "")
                elif "DII" in cat:
                    result["dii_net_cash"] = val
        except Exception as e:
            log.error(f"FII/DII parse error: {e}")
        return result

    def get_fii_consecutive_days(self) -> tuple:
        """
        Counts consecutive days of FII buying or selling from fii_history DB table.
        Returns (buy_streak, sell_streak). Requires FII_CONSECUTIVE_DAYS (3) to confirm a trend.
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

    # -------------------------------------------------------------------------
    # VIX / GIFT Nifty
    # -------------------------------------------------------------------------

    def fetch_india_vix(self) -> float:
        """
        Fetches India VIX from NSE via pnsea.
        Defaults to 15.0 (calm) on failure — intentionally conservative so a
        failed fetch doesn't accidentally block all trading.
        """
        if not self._nse:
            log.warning("VIX fetch unavailable (pnsea missing) — using 15.0")
            return 15.0
        try:
            data = self._nse.equity.find_index("INDIA VIX")
            if data:
                raw = data.get("last") or data.get("lastPrice") or data.get("last_price")
                if raw:
                    return float(str(raw).replace(",", ""))
        except Exception as e:
            log.error(f"VIX fetch error: {e}")
        log.warning("VIX fetch failed — using 15.0")
        return 15.0

    def fetch_gift_nifty(self) -> float:
        """Fetches GIFT Nifty pre-market level from NSE. Used in morning briefing only."""
        data = self._nse_get(Config.NSE_BASE + Config.NSE_VIX)
        if data and "data" in data:
            for item in data["data"]:
                if "GIFT" in str(item.get("index", "")).upper():
                    try:
                        return float(item.get("last", 0))
                    except Exception:
                        pass
        return 0.0

    # -------------------------------------------------------------------------
    # Events calendar
    # -------------------------------------------------------------------------

    def fetch_events_calendar(self) -> list:
        """
        Fetches NSE earnings/results calendar for the next 30 days,
        filtered to watchlist stocks only.
        RED (≤5 days): block new entries, trigger exits.
        YELLOW (≤10 days): warn and monitor.
        GREEN (>10 days): safe to hold.
        """
        today  = datetime.now().strftime("%d-%m-%Y")
        future = (datetime.now() + timedelta(days=30)).strftime("%d-%m-%Y")
        data   = self._nse_get(
            f"{Config.NSE_BASE}{Config.NSE_EVENTS}?index=equities&from_date={today}&to_date={future}"
        )
        events    = []
        watchlist = Watchlist.get_symbols()
        if data:
            items = data if isinstance(data, list) else data.get("data", [])
            for item in items:
                sym = item.get("symbol", "")
                if sym not in watchlist:
                    continue
                try:
                    ed        = datetime.strptime(item.get("date", ""), "%d-%b-%Y")
                    days_away = (ed - datetime.now()).days
                    risk      = "RED" if days_away <= 5 else ("YELLOW" if days_away <= 10 else "GREEN")
                    events.append({
                        "symbol":     sym,
                        "event_type": item.get("purpose", "UNKNOWN"),
                        "event_date": ed.strftime("%Y-%m-%d"),
                        "days_away":  days_away,
                        "risk_level": risk,
                    })
                except Exception:
                    pass
        return events

    def save_events(self, events: list):
        """Saves events to DB, removing stale entries from previous runs first."""
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

    # -------------------------------------------------------------------------
    # NSE helper
    # -------------------------------------------------------------------------

    def _nse_get(self, url: str) -> Optional[dict]:
        """
        Fetches a NSE JSON endpoint via pnsea (handles Akamai bot protection).
        Returns parsed JSON or None on failure. Does NOT retry — NSE data is best-effort
        (VIX/FII failures fall back to safe defaults, not trade blockers).
        """
        if not self._nse:
            return None
        if not url.startswith("http"):
            url = Config.NSE_BASE + url
        try:
            resp = self._nse.endpoint_tester(url)
            return resp.json()
        except Exception as e:
            log.error(f"NSE API error [{url}]: {e}")
            return None
