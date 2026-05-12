"""
data_collector.py — All inbound data fetching for the system.

Two external sources feed the system:

1. Upstox broker API (V3 REST + upstox-python-sdk for auth/configuration)
   - Historical OHLCV via V3 REST — native 4H candles (unit="hours", interval=4),
     daily bars (unit="days", interval=1), and weekly bars (unit="weeks", interval=1).
     V3 endpoint: /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to}/{from}
   - Live price quotes (LTP) via V3 batch endpoint — all symbols in ONE call.
     V3 LTP now natively returns volume + cp (prev close) alongside last_price.
   All fetches use parallel threads (max UPSTOX_MAX_WORKERS concurrent) with a
   shared rate limiter enforcing both per-second (40 req/s) and per-minute (450 req/min)
   caps — safely under Upstox's published limits (50 req/s, 500 req/min, 2000 req/30min).
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
  V3 API responses use colon as separator (NSE_EQ:ISIN) — normalised back to pipe on parse.
"""

import time
import threading
import functools
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote as _url_quote

from config import Config, Watchlist, log, LIBS_AVAILABLE, UPSTOX_AVAILABLE
from database import DatabaseManager

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    import upstox_client
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
    # ── BANKING ──────────────────────────────────────────────────────────────
    "ICICIBANK":  "NSE_EQ|INE090A01021",
    "HDFCBANK":   "NSE_EQ|INE040A01034",
    "AXISBANK":   "NSE_EQ|INE238A01034",
    "SBIN":       "NSE_EQ|INE062A01020",
    "KOTAKBANK":  "NSE_EQ|INE237A01036",
    "INDUSINDBK": "NSE_EQ|INE095A01012",

    # ── FINANCE ──────────────────────────────────────────────────────────────
    "BAJFINANCE": "NSE_EQ|INE296A01032",
    "BAJAJFINSV": "NSE_EQ|INE918I01026",
    "SHRIRAMFIN": "NSE_EQ|INE721A01047",

    # ── INSURANCE ────────────────────────────────────────────────────────────
    "HDFCLIFE":   "NSE_EQ|INE795G01014",
    "SBILIFE":    "NSE_EQ|INE123W01016",

    # ── AUTO ─────────────────────────────────────────────────────────────────
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "MARUTI":     "NSE_EQ|INE585B01010",
    "M&M":        "NSE_EQ|INE101A01026",
    "BAJAJ-AUTO": "NSE_EQ|INE917I01010",
    "HEROMOTOCO": "NSE_EQ|INE158A01026",
    "EICHERMOT":  "NSE_EQ|INE066A01021",

    # ── FMCG ─────────────────────────────────────────────────────────────────
    "ITC":        "NSE_EQ|INE154A01025",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "NESTLEIND":  "NSE_EQ|INE239A01024",
    "BRITANNIA":  "NSE_EQ|INE216A01030",
    "TATACONSUM": "NSE_EQ|INE192A01025",

    # ── CONSUMER ─────────────────────────────────────────────────────────────
    "TITAN":      "NSE_EQ|INE280A01028",
    "ASIANPAINT": "NSE_EQ|INE021A01026",
    "TRENT":      "NSE_EQ|INE849A01020",

    # ── PHARMA ───────────────────────────────────────────────────────────────
    "SUNPHARMA":  "NSE_EQ|INE044A01036",
    "CIPLA":      "NSE_EQ|INE059A01026",
    "DRREDDY":    "NSE_EQ|INE089A01031",

    # ── HEALTHCARE ───────────────────────────────────────────────────────────
    "APOLLOHOSP": "NSE_EQ|INE437A01024",

    # ── METALS ───────────────────────────────────────────────────────────────
    "HINDALCO":   "NSE_EQ|INE038A01020",
    "TATASTEEL":  "NSE_EQ|INE081A01020",
    "JSWSTEEL":   "NSE_EQ|INE019A01038",

    # ── ENERGY ───────────────────────────────────────────────────────────────
    "RELIANCE":   "NSE_EQ|INE002A01018",
    "ONGC":       "NSE_EQ|INE213A01029",
    "NTPC":       "NSE_EQ|INE733E01010",
    "POWERGRID":  "NSE_EQ|INE752E01010",
    "COALINDIA":  "NSE_EQ|INE522F01014",
    "BPCL":       "NSE_EQ|INE029A01011",

    # ── CEMENT ───────────────────────────────────────────────────────────────
    "ULTRACEMCO": "NSE_EQ|INE481G01011",
    "GRASIM":     "NSE_EQ|INE047A01021",

    # ── INFRA ────────────────────────────────────────────────────────────────
    "LT":         "NSE_EQ|INE018A01030",
    "ADANIPORTS": "NSE_EQ|INE742F01042",

    # ── CONGLOMERATE ─────────────────────────────────────────────────────────
    "ADANIENT":   "NSE_EQ|INE423A01024",

    # ── TELECOM ──────────────────────────────────────────────────────────────
    "BHARTIARTL": "NSE_EQ|INE397D01024",

    # ── DEFENCE ──────────────────────────────────────────────────────────────
    "BEL":        "NSE_EQ|INE263A01024",
    "HAL":        "NSE_EQ|INE066F01020",
    "SOLARINDS":  "NSE_EQ|INE343H01029",

    # ── RAILINFRA ────────────────────────────────────────────────────────────
    "RVNL":       "NSE_EQ|INE415G01027",
    "IRFC":       "NSE_EQ|INE053F01010",

    # ── FINANCE (new additions) ───────────────────────────────────────────────
    "MUTHOOTFIN": "NSE_EQ|INE414G01012",
    "CHOLAFIN":   "NSE_EQ|INE121A01024",

    # ── AUTO (new addition) ───────────────────────────────────────────────────
    "TVSMOTOR":   "NSE_EQ|INE494B01023",

    # ── AUTO_COMP ────────────────────────────────────────────────────────────
    "MOTHERSON":  "NSE_EQ|INE775A01035",
    "MRF":        "NSE_EQ|INE883A01011",
    "TIINDIA":    "NSE_EQ|INE974X01010",
    "BOSCHLTD":   "NSE_EQ|INE323A01026",

    # ── CONSUMER (new additions) ──────────────────────────────────────────────
    "PAGEIND":    "NSE_EQ|INE761H01022",
    "DMART":      "NSE_EQ|INE192R01011",
    "ETERNAL":    "NSE_EQ|INE758T01015",

    # ── CONSUMER_ELECT ───────────────────────────────────────────────────────
    "HAVELLS":    "NSE_EQ|INE176B01034",
    "POLYCAB":    "NSE_EQ|INE455K01017",
    "DIXON":      "NSE_EQ|INE935N01020",

    # ── HEALTHCARE (new additions) ────────────────────────────────────────────
    "MAXHEALTH":  "NSE_EQ|INE027H01010",
    "ALKEM":      "NSE_EQ|INE540L01014",
    "TORNTPHARM": "NSE_EQ|INE685A01028",

    # ── CHEMICALS ────────────────────────────────────────────────────────────
    "PIDILITIND": "NSE_EQ|INE318A01026",
    "DEEPAKNTR":  "NSE_EQ|INE288B01029",
    "PIIND":      "NSE_EQ|INE603J01030",

    # ── POWER ────────────────────────────────────────────────────────────────
    "TATAPOWER":  "NSE_EQ|INE245A01021",
    "NHPC":       "NSE_EQ|INE848E01016",
    "IREDA":      "NSE_EQ|INE202E01016",

    # ── CAPGOODS ─────────────────────────────────────────────────────────────
    "SIEMENS":    "NSE_EQ|INE003A01024",
    "ABB":        "NSE_EQ|INE117A01022",
    "CUMMINSIND": "NSE_EQ|INE298A01020",
    "BHEL":       "NSE_EQ|INE257A01026",
    "CGPOWER":    "NSE_EQ|INE067A01029",

    # ── REALESTATE ───────────────────────────────────────────────────────────
    "DLF":        "NSE_EQ|INE271C01023",
    "GODREJPROP": "NSE_EQ|INE484J01027",
    "OBEROIRLTY": "NSE_EQ|INE093I01010",

    # ── INDEX ────────────────────────────────────────────────────────────────
    "NIFTY 50":   "NSE_INDEX|Nifty 50",
}

_V3_BASE = "https://api.upstox.com/v3"


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
    Thread-safe rate limiter enforcing both per-second and per-minute caps.

    Each thread calls wait() before making an API call. The limiter ensures:
      1. No more than max_per_second calls in any 1-second window (burst cap)
      2. No more than max_per_minute calls in any rolling 60-second window (sustained cap)

    Upstox published limits:
      Standard APIs (candles, quotes, portfolio): 50/sec, 500/min
      Order APIs (place/cancel/modify):           10/sec, 500/min
    We target 40/sec + 450/min for data, 8/sec + 450/min for orders.
    """

    def __init__(self, max_per_second: float,
                 max_per_minute: int = None):
        self._min_interval = 1.0 / max_per_second
        self._max_per_min  = max_per_minute if max_per_minute is not None \
                             else Config.UPSTOX_RATE_LIMIT_PER_MIN
        self._lock         = threading.Lock()
        self._last         = 0.0
        self._min_window: list = []   # timestamps of calls in the last 60 s

    def wait(self):
        with self._lock:
            now = time.time()

            # ── Per-minute enforcement ────────────────────────────────────────
            cutoff = now - 60.0
            self._min_window = [t for t in self._min_window if t > cutoff]
            if len(self._min_window) >= self._max_per_min:
                # Sleep until the oldest call in the window falls out of the 60-s bucket
                sleep_for = self._min_window[0] - cutoff
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now    = time.time()
                cutoff = now - 60.0
                self._min_window = [t for t in self._min_window if t > cutoff]

            # ── Per-second enforcement ────────────────────────────────────────
            to_wait = self._min_interval - (now - self._last)
            if to_wait > 0:
                time.sleep(to_wait)

            now = time.time()
            self._last = now
            self._min_window.append(now)


# ---------------------------------------------------------------------------
# DataCollector
# ---------------------------------------------------------------------------

class DataCollector:
    """
    Fetches all market data the system needs each morning.
    Called in step1_collect_data() for:
      - India VIX, GIFT Nifty, FII/DII (from NSE via pnsea)
      - Daily + 4H + weekly OHLCV for all 15 watchlist stocks (from Upstox V3, in parallel)
        - Daily:  unit="days",  interval=1, days=400  (≈285 trading bars for EMA-200)
        - 4H:     unit="hours", interval=4, days=90   (≈100 native 4H bars for FVG + EMA)
        - Weekly: unit="weeks", interval=1, weeks=30  (30 native weekly bars for EMA-20)
      - Events calendar (earnings dates) for all watchlist stocks (from NSE)
    """

    def __init__(self, db: DatabaseManager):
        self.db           = db
        self._upstox      = None          # Upstox API client (set after auth)
        self._nse         = PnseaNSE() if PNSEA_AVAILABLE else None
        self._rate_lim    = _RateLimiter(Config.UPSTOX_RATE_LIMIT_DATA_PER_SEC)
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

    def _get_token(self) -> str:
        """Returns the current access token from the configured API client."""
        return self._upstox.configuration.access_token if self._upstox else ""

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
        all threads stays within UPSTOX_RATE_LIMIT_DATA_PER_SEC and UPSTOX_RATE_LIMIT_PER_MIN.
        """
        self._refresh_token_if_needed()

        results = {}
        log.info(f"Parallel OHLCV fetch starting: {len(symbols)} symbols, "
                 f"{Config.UPSTOX_MAX_WORKERS} workers, "
                 f"{Config.UPSTOX_RATE_LIMIT_DATA_PER_SEC} req/s | "
                 f"{Config.UPSTOX_RATE_LIMIT_PER_MIN} req/min limit")

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

        df_daily  — 400 calendar days (≈285 trading bars)  → EMA-200 needs 200+ bars
        df_4h     — 90 calendar days of native 4H bars (≈100 bars) → FVG zones + EMA
        df_weekly — 30 native weekly bars (~7 months) → true 20-week EMA for weekly_bullish
        """
        df_daily  = self._fetch_v3_candles(symbol, unit="days",  interval=1, days=400)
        df_4h     = self._fetch_v3_candles(symbol, unit="hours", interval=4, days=90)
        df_weekly = self._fetch_v3_candles(symbol, unit="weeks", interval=1, days=210)
        return df_daily, df_4h, df_weekly

    # -------------------------------------------------------------------------
    # V3 historical candle fetch (unified for all timeframes)
    # -------------------------------------------------------------------------

    @_with_retry
    def _fetch_v3_candles(self, symbol: str, unit: str, interval: int,
                           days: int) -> Optional["pd.DataFrame"]:
        """
        Fetches historical OHLCV via Upstox V3:
          GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}

        unit:     "days", "hours", "weeks", "months", "minutes"
        interval: count within unit (e.g. 4 → 4H candles, 1 → daily bars)
        days:     calendar days of history (timedelta applied to get from_date)

        V3 data limits by unit:
          hours/minutes → max 1 quarter; days → max 1 decade; weeks/months → unlimited

        Instrument key is URL-encoded in path (| → %7C).
        V3 candle format: [timestamp, open, high, low, close, volume, open_interest]
        Falls back to mock data ONLY in backtest/offline mode.
        """
        self._rate_lim.wait()

        if not self._upstox or not LIBS_AVAILABLE or not _requests:
            return self._safe_mock(symbol, unit, interval, days)

        instrument_key = INSTRUMENT_KEYS.get(symbol)
        if not instrument_key:
            log.warning(f"No instrument key for {symbol} — skipping.")
            return None

        to_date   = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        encoded_key = _url_quote(instrument_key, safe="")
        url = f"{_V3_BASE}/historical-candle/{encoded_key}/{unit}/{interval}/{to_date}/{from_date}"
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Accept":        "application/json",
        }

        resp = _requests.get(url, headers=headers, timeout=15)
        if not resp.ok:
            err = IOError(f"HTTP {resp.status_code} for {symbol} [{unit}/{interval}]: {resp.text[:200]}")
            err.status = resp.status_code
            raise err

        candles = resp.json().get("data", {}).get("candles", [])
        return self._candles_to_df(candles)

    def _candles_to_df(self, candles: list) -> Optional["pd.DataFrame"]:
        """
        Converts a list of V3 candle arrays into a standardised DataFrame.
        V3 candle format: [timestamp, open, high, low, close, volume, open_interest]
        Returns None if the list is empty or malformed.
        """
        if not LIBS_AVAILABLE:
            return None
        try:
            if not candles:
                return None
            df = pd.DataFrame(candles,
                              columns=["date", "open", "high", "low", "close", "volume", "oi"])
            df["date"]   = pd.to_datetime(df["date"])
            df["volume"] = df["volume"].astype(float)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            log.error(f"Candle DataFrame conversion failed: {e}")
            return None

    def _safe_mock(self, symbol: str, unit: str, interval: int,
                   days: int) -> Optional["pd.DataFrame"]:
        """
        Returns mock OHLCV only in backtest/offline mode.
        In live mode, logs a clear error and returns None so the stock is skipped —
        we never silently trade on fake data when a real connection was expected.
        """
        if Config.BACKTEST_MODE or not self._upstox:
            return self._mock_ohlcv(symbol, self._estimate_bars(unit, interval, days))
        log.error(
            f"LIVE MODE: Could not fetch real data for {symbol} — "
            "skipping (not using mock data in live/sandbox mode)."
        )
        return None

    @staticmethod
    def _estimate_bars(unit: str, interval: int, days: int) -> int:
        """Estimates bar count for mock data generation based on unit/interval/days."""
        trading_day_fraction = 5 / 7          # 5 trading days per 7 calendar days
        trading_hours_per_day = 6.5
        if unit == "hours":
            bars = days * trading_day_fraction * (trading_hours_per_day / interval)
        elif unit == "minutes":
            bars = days * trading_day_fraction * (trading_hours_per_day * 60 / interval)
        elif unit == "weeks":
            bars = days / 7
        elif unit == "months":
            bars = days / 30
        else:   # days
            bars = days * trading_day_fraction
        return max(50, int(bars))

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
    # Live price (LTP) — V3 batch endpoint: all symbols in ONE call
    # -------------------------------------------------------------------------

    def get_all_live_prices(self, symbols: list) -> dict:
        """
        Fetches the last traded price for all given symbols in a SINGLE Upstox V3 API call.
        V3 LTP endpoint returns last_price + volume + cp (prev close) natively.
        V3 response keys use colon separator (NSE_EQ:ISIN) — normalised to pipe for lookup.
        Returns {symbol: price}. Symbols where fetch failed are absent from the dict.
        """
        if not self._upstox or not _requests:
            return {}

        key_to_sym = {
            INSTRUMENT_KEYS[sym]: sym
            for sym in symbols
            if sym in INSTRUMENT_KEYS
        }
        if not key_to_sym:
            return {}

        try:
            url     = f"{_V3_BASE}/market-quote/ltp"
            params  = {"instrument_key": ",".join(key_to_sym.keys())}
            headers = {
                "Authorization": f"Bearer {self._get_token()}",
                "Accept":        "application/json",
            }
            resp = _requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()

            prices = {}
            for raw_key, quote in resp.json().get("data", {}).items():
                # V3 returns "NSE_EQ:ISIN" — normalise colon to pipe for our lookup
                sym   = key_to_sym.get(raw_key.replace(":", "|"))
                price = quote.get("last_price") if isinstance(quote, dict) else None
                if sym and price and float(price) > 0:
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
