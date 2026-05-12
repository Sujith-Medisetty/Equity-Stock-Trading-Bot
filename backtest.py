"""
backtest.py — Historical backtesting engine using real Upstox API data.

HOW IT WORKS:
  1. BacktestDataFetcher fetches daily + 4H + weekly OHLCV from the Upstox V3
     historical-candle API for every watchlist stock + Nifty 50. Results are
     cached to disk (backtest_cache/ folder) so re-runs are instant.

  2. BacktestEngine walks through every trading day between bt_start and bt_end
     using the SAME IndicatorEngine, StockScreener, and StrategyEngine as the live
     system — so the backtest is a true replay, not an approximation.

  3. For each day the engine:
        a. Executes yesterday's pending entries at TODAY's open (no lookahead)
        b. Updates open positions (SL hits, tier upgrades, time exits)
        c. Calculates indicators for all stocks (data sliced to current day)
        d. Derives market mode from Nifty EMAs + synthetic VIX
        e. Screens stocks and evaluates strategies
        f. Queues approved setups for execution at TOMORROW's open

  4. BacktestReport prints the full result table — same format as the live dashboard.

ENTRY PRICE RULE (prevents lookahead bias):
  Signal detected at end of day N → entry at day N+1 OPEN.
  This matches the live system: signals are detected in the 8:45 AM data
  collection, entries are placed at market open (9:15 AM).

SL SIMULATION (conservative — assumes worst-case intraday ordering):
  open < current_sl           → gapped below SL → exit at open
  open >= target (pre-T2)     → gapped above target → T2 actions at open
  low <= SL AND high >= target → SL hit first (worst case — assume down then up)
  low <= SL                   → SL hit at SL price
  high >= target (pre-T2)     → T2 partial exit at target price

3-TIER TRAILING SL (replicated from monitor.py):
  T1 (NoLoss) : high >= entry + risk_per_share → SL = entry
  T1.5 (Adapt): SL = entry + 50% of (close − entry), after T1 is done
  T2 (Partial): high >= target → sell 50%, SL = target − 0.5×ATR
  T3 (Trail)  : after T2, SL = max(current_sl, close − 1×ATR) each day

VIX APPROXIMATION (Upstox does not provide historical India VIX):
  synthetic_vix = rolling_20d_std_of_log_returns × sqrt(252) × 100
  This maps well to real VIX thresholds:
    Normal day σ ≈ 1%  → synth_vix ≈ 16   (maps to "normal" band 13-17)
    Stressed  σ ≈ 1.5% → synth_vix ≈ 24   (maps to "elevated" band 17-25)
    Crisis    σ ≈ 2%   → synth_vix ≈ 32   (maps to "panic" band >29)

FII: Set to NEUTRAL in backtest — historical FII data is not in the Upstox API.

RUN FROM COMMAND LINE:
  python main.py --backtest 2023-01-01 2024-12-31
  python main.py --backtest 2022-01-01 2025-01-01 --refresh    (ignore cache)
  python main.py --backtest 2023-01-01 2024-12-31 --capital 300000
"""

import os
import pickle
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List
from urllib.parse import quote as _url_quote

from config import Config, Watchlist, log, LIBS_AVAILABLE
from data_collector import INSTRUMENT_KEYS, _with_retry, _RateLimiter
from indicators import IndicatorEngine
from market_mode import MarketModeEngine
from risk import ChargesCalculator
from screener import StockScreener
from strategy import StrategyEngine
from models import MarketMode, FIIFlow, StockData, Setup, StrategyType

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = np = None  # type: ignore

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore

_V3_BASE   = "https://api.upstox.com/v3"
_CACHE_DIR = "backtest_cache"
_STCG_RATE = 0.20    # India STCG rate post July 23 2024 budget

# Slippage — applied on every simulated fill to make results realistic.
# Same value as Config.PAPER_SLIPPAGE_PCT (0.2%) used by the live paper mode.
# Buys:  fill at open × (1 + SLIP) — pay slightly more than open
# Sells: fill at price × (1 − SLIP) — receive slightly less than price
# Gap exits: open price is already the real adverse fill, no extra slippage added.
_SLIP = Config.PAPER_SLIPPAGE_PCT   # 0.002 = 0.2%

def _buy_fill(price: float) -> float:
    """Simulated buy fill: slightly worse than the quoted price."""
    return round(price * (1 + _SLIP), 2)

def _sell_fill(price: float) -> float:
    """Simulated sell fill: slightly worse than the quoted price."""
    return round(price * (1 - _SLIP), 2)


# =============================================================================
# MOCK DB — minimal stub so StockScreener can run without a real database.
# Only fetchone() is called by the screener (for RED events check) — returns
# None, which means "no upcoming event" → no event block in backtest.
# =============================================================================

class _MockDB:
    def fetchone(self, *args, **kwargs):
        return None   # no events in backtest → screener never event-blocks

    def fetchall(self, *args, **kwargs):
        return []

    def execute(self, *args, **kwargs):
        pass


# =============================================================================
# BACKTEST DATA FETCHER
# =============================================================================

class BacktestDataFetcher:
    """
    Fetches real historical OHLCV from Upstox V3 and caches it to disk.

    Cache layout (inside backtest_cache/):
      {SYMBOL}_daily_{from}_{to}.pkl
      {SYMBOL}_4h_{from}_{to}.pkl
      {SYMBOL}_weekly_{from}_{to}.pkl

    4H limitation: Upstox allows max 1 quarter (≈90 days) per call,
    and 4H data is only available from January 2022 onwards.
    The fetcher automatically splits requests into 90-day chunks and
    concatenates the results. A 2-year backtest needs ~8 chunks per symbol.

    Use refresh=True to force re-fetch (ignores the cache).
    """

    def __init__(self, token: str, refresh: bool = False):
        self.token   = token
        self.refresh = refresh
        self._rl     = _RateLimiter(Config.UPSTOX_RATE_LIMIT_DATA_PER_SEC)
        os.makedirs(_CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def fetch_all(self, symbols: list, bt_start: str, bt_end: str) -> dict:
        """
        Fetches daily + 4H + weekly for every symbol plus NIFTY 50.

        Returns {symbol: {"daily": df, "4h": df, "weekly": df}}

        Data is fetched from (bt_start − 400 days) to bt_end so there are
        enough bars for EMA-200 warmup before the simulation actually starts.
        """
        warmup_start = (
            datetime.strptime(bt_start, "%Y-%m-%d") - timedelta(days=400)
        ).strftime("%Y-%m-%d")

        all_syms = list(symbols) + ["NIFTY 50"]
        results  = {}
        failed   = []

        for sym in all_syms:
            log.info(f"[BACKTEST] Fetching {sym} ({bt_start} → {bt_end}) …")
            try:
                daily  = self._fetch_daily(sym,  warmup_start, bt_end)
                df_4h  = self._fetch_4h_chunked(sym, warmup_start, bt_end)
                weekly = self._fetch_weekly(sym, warmup_start, bt_end)
            except Exception as e:
                log.error(f"[BACKTEST] FETCH FAILED — {sym}: {type(e).__name__}: {e}")
                log.error(f"[BACKTEST]   → {sym} will be excluded from simulation.")
                failed.append(sym)
                results[sym] = {"daily": None, "4h": None, "weekly": None}
                continue

            results[sym] = {"daily": daily, "4h": df_4h, "weekly": weekly}

            # Report what actually came back so missing data is obvious
            daily_bars  = len(daily)  if daily  is not None else 0
            h4_bars     = len(df_4h)  if df_4h  is not None else 0
            weekly_bars = len(weekly) if weekly is not None else 0
            if daily_bars == 0:
                log.warning(f"  {sym}: NO daily bars returned — symbol may be unlisted or key wrong")
            else:
                log.info(f"  {sym}: {daily_bars} daily | {h4_bars} 4H | {weekly_bars} weekly bars")

        if failed:
            log.warning(f"[BACKTEST] {len(failed)} symbol(s) failed to fetch: {', '.join(failed)}")
        if "NIFTY 50" in failed or results.get("NIFTY 50", {}).get("daily") is None:
            raise RuntimeError("[BACKTEST] NIFTY 50 data fetch failed — cannot run simulation.")

        return results

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------

    def _cache_path(self, symbol: str, tf: str, from_d: str, to_d: str) -> str:
        safe = symbol.replace(" ", "_").replace("|", "_")
        return os.path.join(_CACHE_DIR, f"{safe}_{tf}_{from_d}_{to_d}.pkl")

    def _load(self, path: str):
        if not self.refresh and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                log.warning(f"Cache read failed ({os.path.basename(path)}): {e} — will re-fetch")
        return None

    def _save(self, path: str, data):
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            log.warning(f"Cache write failed ({path}): {e}")

    # -------------------------------------------------------------------------
    # V3 API call
    # -------------------------------------------------------------------------

    @_with_retry
    def _v3_candles(self, symbol: str, unit: str, interval: int,
                    from_date: str, to_date: str) -> list:
        """
        Single GET /v3/historical-candle/{key}/{unit}/{interval}/{to}/{from} call.
        Returns raw candle list: [[timestamp, o, h, l, c, vol, oi], ...]
        """
        ikey = INSTRUMENT_KEYS.get(symbol)
        if not ikey:
            raise ValueError(f"No instrument key for {symbol}")

        self._rl.wait()
        encoded = _url_quote(ikey, safe="")
        url = (
            f"{_V3_BASE}/historical-candle/"
            f"{encoded}/{unit}/{interval}/{to_date}/{from_date}"
        )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept":        "application/json",
        }
        resp = _requests.get(url, headers=headers, timeout=20)
        if not resp.ok:
            err = IOError(f"HTTP {resp.status_code} for {symbol} "
                          f"[{unit}/{interval}]: {resp.text[:200]}")
            err.status = resp.status_code  # type: ignore
            raise err
        return resp.json().get("data", {}).get("candles", [])

    def _to_df(self, candles: list):
        """Converts V3 candle list to a clean sorted DataFrame."""
        if not candles or not LIBS_AVAILABLE:
            return None
        try:
            df = pd.DataFrame(
                candles,
                columns=["date", "open", "high", "low", "close", "volume", "oi"]
            )
            dates = pd.to_datetime(df["date"])
            # Upstox V3 returns tz-aware timestamps — strip timezone so all
            # slice comparisons (sim_ts <= date) stay tz-naive throughout.
            df["date"] = dates.dt.tz_convert(None) if dates.dt.tz is not None else dates
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            log.error(f"Candle→DataFrame failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Timeframe-specific fetchers
    # -------------------------------------------------------------------------

    def _fetch_daily(self, symbol: str, from_date: str, to_date: str):
        path   = self._cache_path(symbol, "daily", from_date, to_date)
        cached = self._load(path)
        if cached is not None:
            return cached
        candles = self._v3_candles(symbol, "days", 1, from_date, to_date)
        df      = self._to_df(candles)
        if df is not None:
            self._save(path, df)
        return df

    def _fetch_weekly(self, symbol: str, from_date: str, to_date: str):
        path   = self._cache_path(symbol, "weekly", from_date, to_date)
        cached = self._load(path)
        if cached is not None:
            return cached
        candles = self._v3_candles(symbol, "weeks", 1, from_date, to_date)
        df      = self._to_df(candles)
        if df is not None:
            self._save(path, df)
        return df

    def _fetch_4h_chunked(self, symbol: str, from_date: str, to_date: str):
        """
        4H data: max 1 quarter per call, available from 2022-01-01.
        Splits the range into 89-day chunks, fetches each, concatenates.
        """
        path   = self._cache_path(symbol, "4h", from_date, to_date)
        cached = self._load(path)
        if cached is not None:
            return cached

        # 4H data only available from Jan 2022
        eff_start = max(from_date, "2022-01-01")
        start_dt  = datetime.strptime(eff_start, "%Y-%m-%d")
        end_dt    = datetime.strptime(to_date,   "%Y-%m-%d")

        all_candles: list = []
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + timedelta(days=89), end_dt)
            try:
                candles = self._v3_candles(
                    symbol, "hours", 4,
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                )
                all_candles.extend(candles)
            except Exception as e:
                log.warning(f"4H chunk failed {symbol} "
                            f"[{chunk_start.date()}→{chunk_end.date()}]: {e}")
            chunk_start = chunk_end + timedelta(days=1)

        if not all_candles:
            return None
        df = self._to_df(all_candles)
        if df is not None:
            df = (df.drop_duplicates(subset=["date"])
                    .sort_values("date")
                    .reset_index(drop=True))
            self._save(path, df)
        return df


# =============================================================================
# POSITION TRACKING
# =============================================================================

@dataclass
class _Position:
    """
    One open simulated trade.

    partial_exits: list of (qty, price) tuples logged when T2 fires.
    These are used at close time to compute the weighted gross PnL.
    """
    symbol:        str
    strategy:      str
    entry_date:    str
    entry_price:   float
    quantity:      int
    initial_sl:    float
    current_sl:    float
    target:        float
    atr:           float
    setup_score:   int
    market_mode:   str

    remaining_qty:  int   = 0
    tier1_done:     bool  = False
    tier2_done:     bool  = False
    peak_close:     float = 0.0   # highest close seen (for T1.5 adaptive trail)

    partial_exits:  list = field(default_factory=list)  # [(qty, price), ...]

    def __post_init__(self):
        if self.remaining_qty == 0:
            self.remaining_qty = self.quantity
        self.peak_close = self.entry_price


# =============================================================================
# CLOSED TRADE RECORD
# =============================================================================

@dataclass
class _ClosedTrade:
    symbol:        str
    strategy:      str
    entry_date:    str
    exit_date:     str
    entry_price:   float
    exit_price:    float       # weighted avg across partial + final exits
    quantity:      int
    gross_pnl:     float
    total_charges: float
    net_pnl:       float
    exit_reason:   str
    holding_days:  int
    setup_score:   int
    market_mode:   str
    tier_reached:  str         # T0 / T1 / T2


# =============================================================================
# MAIN BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Walks through every trading day in [bt_start, bt_end] and simulates the
    full trading pipeline using the same IndicatorEngine → StockScreener →
    StrategyEngine chain as the live system.

    Capital management:
      available_capital is tracked as cash not currently deployed in positions.
      Entry:  available_capital -= entry_price × quantity
      Exit:   available_capital += exit_price × remaining_qty (−charges)
      T2 partial: available_capital += target × sold_qty (charges at final close)
    """

    def __init__(self, initial_capital: float = 200_000):
        self.initial_capital    = initial_capital
        self.available_capital  = initial_capital

        _mdb            = _MockDB()
        self._mode_eng  = MarketModeEngine(_mdb)      # exact same logic as live system
        self._screener  = StockScreener(_mdb)         # events check → always None (no historical events)
        self._strategy  = StrategyEngine()

        self._open:     List[_Position]   = []        # open positions
        self._closed:   List[_ClosedTrade] = []       # completed trades
        self._pending:  Dict[str, Setup]  = {}        # symbol → setup (execute next day)

        # SL hit cooldown: symbol → date string until which re-entry is blocked.
        # After a SL_HIT exit the same stock triggered whipsaw re-entries causing
        # double losses. 5 trading days (~1 week) gives the stock time to stabilise.
        self._sl_cooldown: Dict[str, str] = {}

        # Daily P&L totals for protection rule tracking
        self._daily_pnl:   Dict[str, float] = {}     # date_str → net pnl
        self._weekly_pnl:  Dict[str, float] = {}     # YYYY-Www  → net pnl
        self._monthly_pnl: Dict[str, float] = {}     # YYYY-MM   → net pnl

        # Progress reporting
        self._days_run = 0
        self._signals  = 0

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------

    def run(self, all_data: dict, bt_start: str, bt_end: str) -> "_BacktestReport":
        """
        Runs the full simulation.

        all_data — {symbol: {"daily": df, "4h": df, "weekly": df}}
                   Must include "NIFTY 50".
        bt_start / bt_end — YYYY-MM-DD strings (inclusive).
        """
        if not LIBS_AVAILABLE or pd is None:
            raise RuntimeError("pandas/numpy required for backtesting")

        nifty_full = all_data.get("NIFTY 50", {}).get("daily")
        if nifty_full is None:
            raise RuntimeError("NIFTY 50 daily data is missing — cannot run backtest")

        # Collect all trading days in the simulation window from Nifty daily bars
        start_dt = pd.Timestamp(bt_start)
        end_dt   = pd.Timestamp(bt_end)
        trading_days = sorted(
            d for d in nifty_full["date"].tolist()
            if start_dt <= d <= end_dt
        )

        log.info(f"[BACKTEST] Simulating {len(trading_days)} trading days "
                 f"({bt_start} → {bt_end})  |  capital: ₹{self.initial_capital:,.0f}")

        for sim_ts in trading_days:
            self._simulate_day(sim_ts, all_data)
            self._days_run += 1

            # Progress every 50 days
            if self._days_run % 50 == 0:
                open_cnt   = len(self._open)
                closed_cnt = len(self._closed)
                log.info(f"  … day {self._days_run}/{len(trading_days)}  "
                         f"open={open_cnt}  closed={closed_cnt}  "
                         f"capital=₹{self.available_capital:,.0f}")

        # Force-close any positions still open at bt_end (at last available close)
        if self._open:
            last_date = trading_days[-1].strftime("%Y-%m-%d") if trading_days else bt_end
            self._force_close_all(all_data, last_date)

        log.info(f"[BACKTEST] Done: {len(self._closed)} trades, "
                 f"final capital ₹{self.available_capital:,.0f}")

        return _BacktestReport(
            trades         = self._closed,
            initial_capital= self.initial_capital,
            final_capital  = self.available_capital,
            bt_start       = bt_start,
            bt_end         = bt_end,
            days_run       = self._days_run,
            signals_found  = self._signals,
        )

    # -------------------------------------------------------------------------
    # Per-day simulation
    # -------------------------------------------------------------------------

    def _simulate_day(self, sim_ts: "pd.Timestamp", all_data: dict):
        sim_date = sim_ts.strftime("%Y-%m-%d")

        # Slice each stock's daily df to rows up to (and including) today.
        # This is how we prevent lookahead bias — indicators on day N can only
        # see data through day N.
        daily_slices  = self._slice_daily(all_data, sim_ts)
        nifty_slice   = self._slice_one(all_data.get("NIFTY 50", {}).get("daily"), sim_ts)

        if nifty_slice is None or len(nifty_slice) < 50:
            return   # not enough Nifty history for indicators

        today_bar = nifty_slice.iloc[-1]

        # ── Step A: Execute pending entries at today's OPEN ───────────────────
        # Yesterday's approved setups are entered at today's opening price.
        self._execute_pending(sim_date, daily_slices)

        # ── Step B: Update open positions against today's bar ─────────────────
        self._update_positions(sim_date, daily_slices)

        # ── Step C: Check protection limits ───────────────────────────────────
        if self._protection_blocked(sim_date):
            return   # daily/weekly/monthly loss limit hit — no new setups today

        if len(self._open) >= Config.MAX_SIMULTANEOUS_TRADES:
            return   # slots full

        # ── Step D: Nifty indicators (sliced to today — no lookahead) ─────────
        nifty_data = IndicatorEngine.calculate_all(nifty_slice, "NIFTY50")
        if nifty_data is None:
            return

        # Synthetic VIX — Nifty 20-day rolling std of log-returns × √252 × 100.
        # Upstox does not provide historical India VIX, so this is the best proxy.
        synth_vix = self._synth_vix(nifty_slice)

        # Market mode — calls the EXACT SAME MarketModeEngine.detect() as the live system.
        # FII is set to (net=0, streak=0) which resolves to FIIFlow.NEUTRAL.
        # Historical FII data is not available from Upstox, so the FII modifier is absent.
        market_mode, fii_flow = self._mode_eng.detect(
            vix                  = synth_vix,
            nifty_data           = nifty_data,
            fii_net              = 0.0,   # no historical FII — treated as NEUTRAL
            fii_consecutive_buy  = 0,
            fii_consecutive_sell = 0,
        )

        if market_mode in (MarketMode.DEFENSIVE, MarketMode.CASH):
            return   # no new entries in defensive/panic mode

        # ── Step E: Stock indicators (nifty_slice passed — NOT full nifty df) ──
        # Passing nifty_slice (data up to today) prevents lookahead bias in the
        # RS score, which compares stock 60-day return vs Nifty 60-day return.
        # Passing the full unsliced Nifty df would make day-N RS score use future Nifty prices.
        stocks_data: Dict[str, StockData] = {}

        for sym, df_slice in daily_slices.items():
            if df_slice is None or len(df_slice) < 50:
                continue
            df_4h_slice  = self._slice_4h(all_data.get(sym, {}).get("4h"), sim_ts)
            df_wk_slice  = self._slice_one(all_data.get(sym, {}).get("weekly"), sim_ts)

            # IndicatorEngine.calculate_all is the SAME function used in system.py step1.
            # It computes EMA-20/50/200, RSI, MACD, ATR, volume ratio, OBV, FVG zones,
            # candle patterns, RS score vs Nifty — all from the sliced df (no lookahead).
            s_data = IndicatorEngine.calculate_all(df_slice, sym, nifty_slice)
            if s_data:
                # check_4h_bullish / check_weekly_bullish / detect_4h_fvg_zones /
                # apply_combined_fvg_flags — all the same functions called in system.py step1.
                s_data.h4_bullish      = IndicatorEngine.check_4h_bullish(df_4h_slice)
                s_data.weekly_bullish  = IndicatorEngine.check_weekly_bullish(df_wk_slice)
                s_data.tf_aligned_count = sum([
                    s_data.weekly_bullish, s_data.daily_bullish, s_data.h4_bullish
                ])
                s_data.fvg_zones_4h = IndicatorEngine.detect_4h_fvg_zones(df_4h_slice)
                IndicatorEngine.apply_combined_fvg_flags(s_data, s_data.close)
                stocks_data[sym] = s_data

        if not stocks_data:
            return

        # ── Step F: Screen → Strategy → Risk → Queue ─────────────────────────
        # StockScreener.screen() — same function as live system.
        # Events check always returns None (MockDB) since historical events aren't available.
        open_as_dicts = [{"symbol": p.symbol} for p in self._open]
        screened      = self._screener.screen(stocks_data, open_as_dicts, market_mode)

        cap = self.available_capital
        # Collect ALL valid setups first, then pick the 2 highest-scoring ones.
        # Previously we iterated in watchlist order and took the first 2 —
        # that biased toward alphabetically-first stocks, not the best setups.
        candidate_setups: list = []

        for sym in screened:
            data = stocks_data.get(sym)
            if data is None:
                continue
            # StrategyEngine.evaluate_all() — same function as live system.
            # Tries WEEK52 → BREAKOUT → PULLBACK → SWING in priority order.
            # Returns the highest-priority qualifying strategy or None.
            setup = self._strategy.evaluate_all(sym, data, market_mode, fii_flow)
            if setup is None:
                continue

            # Risk sizing (replicates RiskManager.calculate_setup_risk without DB)
            sized = self._size_setup(setup, data, cap)
            if sized is None or sized.status == "SKIPPED":
                continue

            if sized.score < 65:
                continue
            if sized.rr_ratio < Config.MIN_RR_RATIO:
                continue
            if sym in self._pending:
                continue  # already queued from an earlier day

            # SL cooldown: skip re-entry if stock still near or below EMA_20 after SL.
            # A genuine recovery needs close clearly back above EMA_20 — checked by the
            # close >= ema_20 hard gate in strategy.py. Cooldown here blocks re-entry
            # only if the stock hasn't yet cleared EMA_20 (i.e. gate already handles it).
            # We keep a short 3-day cooldown just to prevent same-week whipsaws.
            if self._sl_cooldown.get(sym, "") >= sim_date:
                continue

            candidate_setups.append((sized.score, sym, sized))

        # Queue only the 2 highest-scoring setups for tomorrow's execution.
        # Sorting by score ensures we always take the best setups, not the first ones.
        candidate_setups.sort(key=lambda x: x[0], reverse=True)
        for _, sym, sized in candidate_setups[:2]:
            if sym in self._pending:
                continue
            self._pending[sym] = sized
            self._signals += 1
            log.debug(f"[BT {sim_date}] Queued {sym} {sized.strategy.value} "
                      f"score={sized.score} RR={sized.rr_ratio:.1f} "
                      f"entry=₹{sized.entry_price:.2f}")

    # -------------------------------------------------------------------------
    # Execute pending entries at today's OPEN
    # -------------------------------------------------------------------------

    def _execute_pending(self, sim_date: str, daily_slices: dict):
        """
        Entries from yesterday's pending queue are filled at today's open price.
        After execution, the setup is removed from pending regardless of outcome.
        """
        executed = []
        for sym, setup in list(self._pending.items()):
            df = daily_slices.get(sym)
            if df is None or len(df) == 0:
                executed.append(sym)
                continue

            open_price = float(df.iloc[-1]["open"])

            # Validate drift on raw open (before slippage) — same as live system
            drift_pct = abs(open_price - setup.entry_price) / setup.entry_price * 100
            drift_limit = (Config.MAX_ENTRY_DRIFT_PCT_WIDE
                           if setup.strategy.value in Config.WIDE_DRIFT_STRATEGIES
                           else Config.MAX_ENTRY_DRIFT_PCT)
            if drift_pct > drift_limit:
                log.debug(f"[BT {sim_date}] {sym} drift {drift_pct:.1f}% > "
                          f"{drift_limit}% — skip entry")
                executed.append(sym)
                continue

            # Apply buy slippage: in live mode a LIMIT BUY at entry×1.001 fills slightly
            # above open. Simulated here as open×(1+SLIP) so PnL reflects real costs.
            entry_fill = _buy_fill(open_price)

            # Capital check uses slippage-adjusted price (true cash spent)
            capital_needed = setup.shares * entry_fill
            if capital_needed > self.available_capital:
                log.debug(f"[BT {sim_date}] {sym} insufficient capital "
                          f"(need ₹{capital_needed:.0f}, have ₹{self.available_capital:.0f})")
                executed.append(sym)
                continue

            # Max simultaneous positions
            if len(self._open) >= Config.effective_max_trades(self.available_capital):
                executed.append(sym)
                continue

            # Record position at the slippage-adjusted entry price.
            # SL and target are kept at their original levels (not adjusted for slippage)
            # because the SL-M order at the exchange fires at exactly sl_price.
            pos = _Position(
                symbol       = sym,
                strategy     = setup.strategy.value,
                entry_date   = sim_date,
                entry_price  = entry_fill,          # slippage-adjusted — true cost basis
                quantity     = setup.shares,
                initial_sl   = setup.sl_price,
                current_sl   = setup.sl_price,
                target       = setup.target_price,
                atr          = setup.atr,
                setup_score  = setup.score,
                market_mode  = setup.market_mode,
                remaining_qty= setup.shares,
            )
            self._open.append(pos)
            self.available_capital -= capital_needed
            executed.append(sym)
            log.info(f"[BT {sim_date}] ENTER {sym} {pos.strategy} "
                     f"×{pos.quantity} @ ₹{entry_fill:.2f} (open ₹{open_price:.2f} +{_SLIP*100:.1f}% slip)  "
                     f"SL=₹{pos.current_sl:.2f}  T=₹{pos.target:.2f}")

        for sym in executed:
            self._pending.pop(sym, None)

    # -------------------------------------------------------------------------
    # Update open positions against today's bar
    # -------------------------------------------------------------------------

    def _update_positions(self, sim_date: str, daily_slices: dict):
        still_open = []
        for pos in self._open:
            df = daily_slices.get(pos.symbol)
            if df is None or len(df) == 0:
                still_open.append(pos)
                continue

            bar      = df.iloc[-1]
            o, h, l, c = (float(bar["open"]), float(bar["high"]),
                          float(bar["low"]),  float(bar["close"]))
            atr      = self._current_atr(df)

            # ── Exit checks (conservative order: SL before target) ────────────
            #
            # Slippage rules:
            #   Gap exits   → open price is already the true adverse fill, no extra slip
            #   Intraday SL → fill = sl_price × (1 − SLIP)  [slipped below SL level]
            #   Target hit  → fill = target   × (1 − SLIP)  [slipped below target]
            #   Time exit   → fill = close    × (1 − SLIP)  [normal market sell]
            #   T3 trail    → fill = sl_level × (1 − SLIP)  [same as intraday SL]

            # Gap DOWN below SL at open.
            # Open already reflects the gap — no additional slippage on top of that.
            if o <= pos.current_sl:
                self._close_position(pos, sim_date, o, "SL_HIT_GAP")
                continue

            # Gap UP above target at open (before T2).
            # Open is already above target — treat it as a favorable fill, no slip.
            if not pos.tier2_done and o >= pos.target:
                pos = self._do_t2(pos, sim_date, o, atr)
                still_open.append(pos)
                continue

            # SL AND target both hit in same day → SL first (worst case) with slippage.
            if l <= pos.current_sl and (not pos.tier2_done) and h >= pos.target:
                self._close_position(pos, sim_date, _sell_fill(pos.current_sl), "SL_HIT")
                continue

            # Intraday SL hit (low touched SL level) — apply sell slippage.
            if l <= pos.current_sl:
                self._close_position(pos, sim_date, _sell_fill(pos.current_sl), "SL_HIT")
                continue

            # Target hit intraday — T2 partial exit with sell slippage.
            if not pos.tier2_done and h >= pos.target:
                pos = self._do_t2(pos, sim_date, _sell_fill(pos.target), atr)
                still_open.append(pos)
                continue

            # Time exit: MAX_HOLD_DAYS without profit — sell at close with slippage.
            entry_dt = datetime.strptime(pos.entry_date, "%Y-%m-%d")
            sim_dt   = datetime.strptime(sim_date,       "%Y-%m-%d")
            hold_days = (sim_dt - entry_dt).days
            if hold_days >= Config.MAX_HOLD_DAYS and c <= pos.entry_price:
                self._close_position(pos, sim_date, _sell_fill(c), "TIME_BASED")
                continue

            # ── Tier advancement ─────────────────────────────────────────────

            risk_per_share = pos.entry_price - pos.initial_sl

            # T1 (MinProfit): SL → entry + 12% of risk when 1:1 RR reached.
            # 12% of risk locks in ~₹180 gross on ₹1,500 risk = ~₹85 net after charges.
            # Breakeven (entry) exits net -₹96 after charges, making "wins" worse than losses.
            if not pos.tier1_done and h >= pos.entry_price + risk_per_share:
                pos.current_sl = pos.entry_price + risk_per_share * 0.12
                pos.tier1_done = True
                log.debug(f"[BT {sim_date}] {pos.symbol} T1 MinProfit: SL → ₹{pos.current_sl:.2f}")

            # T1.5 (Adaptive): SL = entry + 30% of gain (only after T1)
            if pos.tier1_done and not pos.tier2_done:
                gain          = max(0.0, c - pos.entry_price)
                adaptive_sl   = pos.entry_price + 0.3 * gain
                pos.current_sl = max(pos.current_sl, round(adaptive_sl, 2))

            # T3 (Trail): after T2, SL = max(current_sl, close − 0.5×ATR)
            if pos.tier2_done and atr > 0:
                trail_sl       = c - atr * 0.5
                pos.current_sl = max(pos.current_sl, round(trail_sl, 2))
                # T3 SL breached — exit with slippage
                if l <= pos.current_sl:
                    self._close_position(pos, sim_date, _sell_fill(pos.current_sl), "TRAILING_SL")
                    continue

            pos.peak_close = max(pos.peak_close, c)
            still_open.append(pos)

        self._open = still_open

    # -------------------------------------------------------------------------
    # T2 partial exit
    # -------------------------------------------------------------------------

    def _do_t2(self, pos: _Position, sim_date: str,
               fill_price: float, atr: float) -> _Position:
        """
        Sells floor(remaining_qty / 2) shares at fill_price.
        Sets new SL = target − 0.5×ATR. Marks tier2_done.
        Partial proceeds go back to available_capital immediately.
        """
        sold_qty = max(1, pos.remaining_qty // 2)
        pos.partial_exits.append((sold_qty, fill_price))
        pos.remaining_qty -= sold_qty
        pos.tier2_done     = True
        new_sl = fill_price - 0.5 * atr
        pos.current_sl = max(pos.current_sl, round(new_sl, 2))

        # Return partial proceeds to capital (charges computed at full close)
        proceeds = sold_qty * fill_price
        self.available_capital += proceeds

        log.info(f"[BT {sim_date}] {pos.symbol} T2 PARTIAL: sold {sold_qty}× "
                 f"@ ₹{fill_price:.2f}  remaining={pos.remaining_qty}  "
                 f"new SL=₹{pos.current_sl:.2f}")
        return pos

    # -------------------------------------------------------------------------
    # Close a position fully
    # -------------------------------------------------------------------------

    def _close_position(self, pos: _Position, sim_date: str,
                        exit_price: float, reason: str):
        """
        Finalises the position: computes gross PnL, charges, and net PnL.
        Updates available_capital with exit proceeds (minus charges).
        Records a _ClosedTrade and updates the daily/weekly/monthly P&L buckets.
        """
        entry_dt  = datetime.strptime(pos.entry_date, "%Y-%m-%d")
        exit_dt   = datetime.strptime(sim_date,        "%Y-%m-%d")
        hold_days = (exit_dt - entry_dt).days

        # Total buy amount (all shares at entry)
        buy_val  = pos.entry_price * pos.quantity

        # Total sell amount (partial exits + final exit)
        sell_val  = sum(qty * price for qty, price in pos.partial_exits)
        sell_val += pos.remaining_qty * exit_price

        # Gross PnL = sell total - buy total
        gross_pnl = sell_val - buy_val

        # Charges — one buy leg, one ChargesCalculator call per sell leg (each gets own DP charge)
        charges  = ChargesCalculator._manual_buy_charges(buy_val)
        charges += sum(
            ChargesCalculator._manual_sell_charges(qty * price)
            for qty, price in pos.partial_exits
        )
        charges += ChargesCalculator._manual_sell_charges(pos.remaining_qty * exit_price)
        charges  = round(charges, 2)

        net_pnl = gross_pnl - charges

        # Tier reached
        tier = ("T2" if pos.tier2_done else ("T1" if pos.tier1_done else "T0"))

        # Weighted average exit price
        total_proceeds = sum(qty * price for qty, price in pos.partial_exits) + pos.remaining_qty * exit_price
        avg_exit       = total_proceeds / pos.quantity

        trade = _ClosedTrade(
            symbol        = pos.symbol,
            strategy      = pos.strategy,
            entry_date    = pos.entry_date,
            exit_date     = sim_date,
            entry_price   = pos.entry_price,
            exit_price    = round(avg_exit, 2),
            quantity      = pos.quantity,
            gross_pnl     = round(gross_pnl, 2),
            total_charges = round(charges, 2),
            net_pnl       = round(net_pnl, 2),
            exit_reason   = reason,
            holding_days  = hold_days,
            setup_score   = pos.setup_score,
            market_mode   = pos.market_mode,
            tier_reached  = tier,
        )
        self._closed.append(trade)

        # Capital: return the remaining shares' proceeds (partials already returned in _do_t2)
        self.available_capital += (pos.remaining_qty * exit_price) - charges

        # SL cooldown: block re-entry for 3 calendar days after a SL hit.
        # Re-entering the same stock immediately after SL creates whipsaw double-losses
        # when the stock is trending down. The cooldown lets the stock stabilise.
        if reason in ("SL_HIT", "SL_HIT_GAP"):
            cooldown_until = (
                datetime.strptime(sim_date, "%Y-%m-%d") + timedelta(days=3)
            ).strftime("%Y-%m-%d")
            self._sl_cooldown[pos.symbol] = cooldown_until

        # Update P&L buckets
        self._record_pnl(sim_date, net_pnl)

        sign = "+" if net_pnl >= 0 else ""
        log.info(f"[BT {sim_date}] CLOSE {pos.symbol} [{reason}] "
                 f"{hold_days}d  avg_exit=₹{avg_exit:.2f}  "
                 f"net={sign}₹{net_pnl:,.0f}  tier={tier}")

    # -------------------------------------------------------------------------
    # Force-close all open positions at backtest end
    # -------------------------------------------------------------------------

    def _force_close_all(self, all_data: dict, last_date: str):
        last_ts = pd.Timestamp(last_date)
        for pos in list(self._open):
            df = self._slice_one(all_data.get(pos.symbol, {}).get("daily"), last_ts)
            if df is not None and len(df) > 0:
                close_price = float(df.iloc[-1]["close"])
            else:
                close_price = pos.entry_price
            # Apply sell slippage — same as a normal close via place_sell_order()
            self._close_position(pos, last_date, _sell_fill(close_price), "BT_END_FORCE_CLOSE")
        self._open = []

    # -------------------------------------------------------------------------
    # Risk sizing (replicates RiskManager.calculate_setup_risk without DB)
    # -------------------------------------------------------------------------

    def _size_setup(self, setup: Setup, data: StockData,
                    available_capital: float) -> Optional[Setup]:
        """
        Replicates RiskManager.calculate_setup_risk() without any DB dependency.
        Returns the setup with sl_price / target_price / shares filled in,
        or None if the trade can't be sized correctly.
        """
        max_risk    = Config.risk_per_trade(available_capital)
        max_capital = available_capital * Config.CAPITAL_PER_TRADE_PCT
        atr         = data.atr

        if max_risk == 0 or atr <= 0:
            return None

        entry = setup.entry_price
        mult  = Config.ATR_MULT.get(setup.strategy.value, 1.5)
        sl    = entry - (atr * mult)

        if setup.strategy == StrategyType.BREAKOUT:
            d20_low = data.close * (1 - data.consolidation_range_pct / 100)
            sl = d20_low - (atr * 0.5)
        elif setup.strategy == StrategyType.WEEK52:
            sl = data.week_52_high - (atr * 1.5)
        elif setup.strategy == StrategyType.PULLBACK:
            sl = data.ema_20 - (atr * 1.5)

        risk_per_share = entry - sl
        if risk_per_share <= 0:
            return None

        target = entry + risk_per_share * Config.MIN_RR_RATIO

        # FVG target override for SWING / PULLBACK
        if (setup.strategy in (StrategyType.PULLBACK, StrategyType.SWING)
                and data.fvg_target > entry):
            fvg_rr = (data.fvg_target - entry) / risk_per_share
            if fvg_rr >= Config.MIN_RR_RATIO and data.fvg_target > target:
                target = data.fvg_target

        shares = int(max_risk / risk_per_share)
        if shares <= 0:
            return None

        capital_needed = shares * entry
        if capital_needed > max_capital:
            shares         = int(max_capital / entry)
            capital_needed = shares * entry

        if capital_needed < Config.MIN_POSITION_VALUE:
            return None
        if shares < Config.MIN_QUANTITY:
            return None

        rr = (target - entry) / risk_per_share
        if rr < Config.MIN_RR_RATIO:
            return None

        setup.sl_price         = round(sl, 2)
        setup.target_price     = round(target, 2)
        setup.atr              = round(atr, 2)
        setup.risk_per_share   = round(risk_per_share, 2)
        setup.shares           = shares
        setup.capital_required = round(capital_needed, 2)
        setup.actual_risk      = round(shares * risk_per_share, 2)
        setup.rr_ratio         = round(rr, 2)
        return setup

    @staticmethod
    def _synth_vix(nifty_df: "pd.DataFrame") -> float:
        """
        Computes synthetic VIX from Nifty 50 daily returns.
        Formula: rolling_20d_std(log_returns) × sqrt(252) × 100
        Defaults to 15.0 if insufficient data.
        """
        if nifty_df is None or len(nifty_df) < 21:
            return 15.0
        try:
            closes  = nifty_df["close"].values[-21:]
            returns = np.log(closes[1:] / closes[:-1])
            return float(np.std(returns) * math.sqrt(252) * 100)
        except Exception:
            return 15.0

    # -------------------------------------------------------------------------
    # Protection gate (daily / weekly / monthly loss limits)
    # -------------------------------------------------------------------------

    def _protection_blocked(self, sim_date: str) -> bool:
        daily   = self._daily_pnl.get(sim_date, 0.0)
        week    = _week_key(sim_date)
        monthly = sim_date[:7]

        weekly_pnl  = self._weekly_pnl.get(week,    0.0)
        monthly_pnl = self._monthly_pnl.get(monthly, 0.0)

        if daily   < -Config.DAILY_LOSS_LIMIT:
            return True
        if weekly_pnl  < -Config.WEEKLY_LOSS_LIMIT:
            return True
        if monthly_pnl < -Config.MONTHLY_LOSS_LIMIT:
            return True
        return False

    def _record_pnl(self, sim_date: str, net_pnl: float):
        week    = _week_key(sim_date)
        monthly = sim_date[:7]

        self._daily_pnl[sim_date]  = self._daily_pnl.get(sim_date, 0.0)  + net_pnl
        self._weekly_pnl[week]     = self._weekly_pnl.get(week,    0.0)   + net_pnl
        self._monthly_pnl[monthly] = self._monthly_pnl.get(monthly, 0.0)  + net_pnl

    # -------------------------------------------------------------------------
    # DataFrame slicing helpers (no lookahead)
    # -------------------------------------------------------------------------

    @staticmethod
    def _slice_daily(all_data: dict, sim_ts: "pd.Timestamp") -> dict:
        """Returns {symbol: df_up_to_sim_ts} for all symbols."""
        slices = {}
        for sym, data_dict in all_data.items():
            if sym == "NIFTY 50":
                continue
            slices[sym] = BacktestEngine._slice_one(data_dict.get("daily"), sim_ts)
        return slices

    @staticmethod
    def _slice_one(df, sim_ts: "pd.Timestamp"):
        """Returns df rows where date <= sim_ts."""
        if df is None:
            return None
        mask = df["date"] <= sim_ts
        sub  = df[mask]
        return sub.reset_index(drop=True) if len(sub) > 0 else None

    @staticmethod
    def _slice_4h(df_4h, sim_ts: "pd.Timestamp"):
        """Returns 4H rows where date.date() <= sim_ts.date()."""
        if df_4h is None:
            return None
        sim_date = sim_ts.date()
        mask     = df_4h["date"].dt.date <= sim_date
        sub      = df_4h[mask]
        return sub.reset_index(drop=True) if len(sub) > 0 else None

    @staticmethod
    def _current_atr(df) -> float:
        """Returns the last ATR value from the dataframe if column exists, else 0."""
        if df is None or len(df) == 0:
            return 0.0
        if "atr" in df.columns:
            return float(df["atr"].iloc[-1])
        # ATR not pre-computed — estimate from last bar's range
        last = df.iloc[-1]
        return float(last["high"] - last["low"])


def _week_key(date_str: str) -> str:
    """Returns YYYY-Www ISO week key for grouping weekly P&L."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.strftime("%G-W%V")
    except Exception:
        return date_str[:7]


# =============================================================================
# BACKTEST REPORT
# =============================================================================

class _BacktestReport:
    """
    Analyses the list of _ClosedTrade records and prints a full dashboard.

    Sections:
      1. Overview       — capital curve, total return, Sharpe-like ratio
      2. Trade stats    — win rate, avg win/loss, real R:R
      3. Strategy       — breakdown per strategy (SWING/BREAKOUT/PULLBACK/WEEK52)
      4. Exit reasons   — how trades closed
      5. Monthly P&L    — month-by-month net PnL table
      6. Tax estimate   — India STCG at 20% on total net profit
      7. Trade log      — every trade (symbol, dates, entry, exit, PnL, reason)
    """

    def __init__(self, trades: List[_ClosedTrade], initial_capital: float,
                 final_capital: float, bt_start: str, bt_end: str,
                 days_run: int, signals_found: int):
        self.trades          = trades
        self.initial_capital = initial_capital
        self.final_capital   = final_capital
        self.bt_start        = bt_start
        self.bt_end          = bt_end
        self.days_run        = days_run
        self.signals_found   = signals_found

    def print_report(self, verbose: bool = False):
        trades = self.trades
        W      = 74

        print("\n" + "=" * W)
        print("  BACKTEST RESULTS  (Real Upstox Data — No Lookahead)")
        print("=" * W)
        print(f"  Period  : {self.bt_start}  →  {self.bt_end}")
        print(f"  Capital : ₹{self.initial_capital:>12,.0f}  (starting)")
        print(f"  Final   : ₹{self.final_capital:>12,.0f}")
        total_return = self.final_capital - self.initial_capital
        pct_return   = total_return / self.initial_capital * 100
        print(f"  Return  : ₹{total_return:>+12,.0f}  ({pct_return:+.1f}%)")
        print(f"  Days    : {self.days_run}  |  Signals found: {self.signals_found}"
              f"  |  Executed: {len(trades)}")

        if not trades:
            print("\n  No trades executed in this period.")
            print("=" * W + "\n")
            return

        wins   = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]

        total_gross   = sum(t.gross_pnl     for t in trades)
        total_charges = sum(t.total_charges for t in trades)
        total_net     = sum(t.net_pnl       for t in trades)

        avg_win  = sum(t.net_pnl for t in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0
        rr_real  = abs(avg_win / avg_loss) if avg_loss else 0
        avg_hold = sum(t.holding_days for t in trades) / len(trades)

        # ── Trade statistics ─────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print("  TRADE STATISTICS")
        print(f"{'─' * W}")
        print(f"  Total trades : {len(trades)}  "
              f"({len(wins)} wins / {len(losses)} losses)")
        print(f"  Win rate     : {len(wins)/len(trades)*100:.1f}%  "
              f"(need >50% to be consistently profitable)")
        print(f"  Avg win      : ₹{avg_win:>+10,.2f}")
        print(f"  Avg loss     : ₹{avg_loss:>+10,.2f}")
        rr_note = "Good" if rr_real >= 2.0 else ("Acceptable" if rr_real >= 1.5 else "Low")
        print(f"  Real R:R     : {rr_real:.2f}:1  [{rr_note}]  (target ≥ 2.0:1)")
        print(f"  Avg hold     : {avg_hold:.1f} days")
        print(f"  Gross P&L    : ₹{total_gross:>+12,.2f}")
        print(f"  Charges paid : ₹{total_charges:>12,.2f}")
        print(f"  Net P&L      : ₹{total_net:>+12,.2f}  ← what you actually made/lost")

        # ── Strategy breakdown ───────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print("  STRATEGY BREAKDOWN")
        print(f"{'─' * W}")
        by_strat: dict = {}
        for t in trades:
            s = t.strategy
            if s not in by_strat:
                by_strat[s] = {"n": 0, "wins": 0, "pnl": 0.0}
            by_strat[s]["n"]    += 1
            by_strat[s]["wins"] += 1 if t.net_pnl > 0 else 0
            by_strat[s]["pnl"]  += t.net_pnl

        print(f"  {'Strategy':<12} {'Trades':>6}  {'Win%':>6}  {'Net P&L':>12}  Verdict")
        print(f"  {'─'*12} {'─'*6}  {'─'*6}  {'─'*12}  {'─'*18}")
        for strat, s in sorted(by_strat.items(), key=lambda x: x[1]["pnl"], reverse=True):
            wr = s["wins"] / s["n"] * 100 if s["n"] else 0
            v  = ("Working" if s["pnl"] > 0 and wr >= 50 else
                  "Profitable low WR" if s["pnl"] > 0 else "Losing — review")
            print(f"  {strat:<12} {s['n']:>6}  {wr:>5.1f}%  ₹{s['pnl']:>11,.0f}  {v}")

        # ── Exit reasons ─────────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print("  EXIT REASONS")
        print(f"{'─' * W}")
        by_exit: dict = {}
        for t in trades:
            by_exit[t.exit_reason] = by_exit.get(t.exit_reason, 0) + 1

        labels = {
            "SL_HIT":            "Stop loss hit (intraday)",
            "SL_HIT_GAP":        "Stop loss hit (gap down at open)",
            "TRAILING_SL":       "Tier 3 trail SL hit",
            "TIME_BASED":        "Stale trade (15 days, no profit)",
            "BT_END_FORCE_CLOSE":"Force closed at backtest end",
        }
        for reason, cnt in sorted(by_exit.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cnt:>3}×  {labels.get(reason, reason)}")

        # ── Tier breakdown ───────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print("  TIER BREAKDOWN  (how far each trade progressed)")
        print(f"{'─' * W}")
        by_tier: dict = {}
        for t in trades:
            by_tier[t.tier_reached] = by_tier.get(t.tier_reached, 0) + 1
        tier_labels = {
            "T0": "T0 — exited before reaching breakeven (SL or time)",
            "T1": "T1 — SL moved to entry (breakeven protected)",
            "T2": "T2 — 50% profit locked, trailed remainder",
        }
        for tier in ["T0", "T1", "T2"]:
            cnt = by_tier.get(tier, 0)
            print(f"  {cnt:>3}×  {tier_labels[tier]}")

        # ── Monthly P&L ──────────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print("  MONTHLY P&L")
        print(f"{'─' * W}")
        monthly: dict = {}
        for t in trades:
            m = t.exit_date[:7]
            monthly[m] = monthly.get(m, {"net": 0.0, "n": 0, "wins": 0})
            monthly[m]["net"]  += t.net_pnl
            monthly[m]["n"]    += 1
            monthly[m]["wins"] += 1 if t.net_pnl > 0 else 0

        cum   = 0.0
        print(f"  {'Month':<10}  {'Trades':>6}  {'Win%':>6}  {'Net P&L':>11}  {'Cumulative':>12}")
        print(f"  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*11}  {'─'*12}")
        for m in sorted(monthly):
            ms  = monthly[m]
            wr  = ms["wins"] / ms["n"] * 100 if ms["n"] else 0
            cum += ms["net"]
            arrow = "▲" if ms["net"] >= 0 else "▼"
            print(f"  {m:<10}  {ms['n']:>6}  {wr:>5.1f}%  "
                  f"{arrow}₹{abs(ms['net']):>9,.0f}  ₹{cum:>+11,.0f}")

        # ── Tax estimate ─────────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print("  TAX ESTIMATE (India STCG — all trades < 1 year = Short-Term)")
        print(f"{'─' * W}")
        taxable    = max(0.0, total_net)
        tax_est    = taxable * _STCG_RATE
        print(f"  Total net P&L   : ₹{total_net:>+12,.2f}")
        print(f"  Taxable STCG    : ₹{taxable:>12,.2f}  (0 if net is negative)")
        print(f"  STCG rate       : 20%  (post July 23, 2024 budget)")
        print(f"  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  ESTIMATED TAX : ₹{tax_est:>10,.2f}                    │")
        print(f"  └─────────────────────────────────────────────────────┘")
        print(f"  Charges (₹{total_charges:,.0f}) already deducted from net P&L above.")
        print(f"  This is a planning estimate — consult a CA for actual filing.")

        # ── Per-trade log ────────────────────────────────────────────────────
        if verbose:
            print(f"\n{'─' * W}")
            print("  FULL TRADE LOG")
            print(f"{'─' * W}")
            hdr = (f"  {'#':>3}  {'Symbol':<12} {'Strategy':<9} "
                   f"{'Entry date':<12} {'Exit date':<12} "
                   f"{'Entry':>8} {'Exit':>8} {'Days':>4} "
                   f"{'Net P&L':>10}  {'Reason'}")
            print(hdr)
            print(f"  {'─'*3}  {'─'*12} {'─'*9} {'─'*12} {'─'*12} "
                  f"{'─'*8} {'─'*8} {'─'*4} {'─'*10}  {'─'*16}")
            for i, t in enumerate(sorted(trades, key=lambda x: x.entry_date), 1):
                sign = "+" if t.net_pnl >= 0 else ""
                print(f"  {i:>3}  {t.symbol:<12} {t.strategy:<9} "
                      f"{t.entry_date:<12} {t.exit_date:<12} "
                      f"₹{t.entry_price:>7.2f} ₹{t.exit_price:>7.2f} "
                      f"{t.holding_days:>4} "
                      f"{sign}₹{t.net_pnl:>8,.0f}  {t.exit_reason}")

        print("=" * W + "\n")


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_backtest(bt_start: str, bt_end: str,
                 initial_capital: float = 200_000,
                 refresh: bool          = False,
                 verbose: bool          = False):
    """
    Full backtest pipeline. Called from main.py --backtest.

    bt_start / bt_end — date strings YYYY-MM-DD (inclusive).
    initial_capital   — starting cash (default ₹2,00,000).
    refresh           — if True, re-fetches all data ignoring the cache.
    verbose           — if True, prints the full per-trade log at the end.

    Requires a valid Upstox access token saved in the DB (run --auth first).
    The token is used only for historical data fetching — no orders are placed.
    """
    if not LIBS_AVAILABLE or pd is None:
        print("pandas and numpy are required: pip install pandas numpy")
        return

    if _requests is None:
        print("requests is required: pip install requests")
        return

    # Load auth token from DB (same path as the live system)
    from database import DatabaseManager
    from upstox_auth import UpstoxAuth

    db    = DatabaseManager()
    auth  = UpstoxAuth(db)
    token = auth.get_valid_token()

    if not token:
        print("No Upstox token found. Run:  python main.py --auth")
        return

    symbols = Watchlist.get_symbols()

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    print(f"\n[BACKTEST] Fetching data for {len(symbols)} stocks "
          f"({bt_start} → {bt_end}){' [REFRESH]' if refresh else ''} …")
    print(f"  Data cached in {_CACHE_DIR}/ — subsequent runs are instant.\n")

    fetcher  = BacktestDataFetcher(token=token, refresh=refresh)
    all_data = fetcher.fetch_all(symbols, bt_start, bt_end)

    # ── 2. Run simulation ─────────────────────────────────────────────────────
    engine = BacktestEngine(initial_capital=initial_capital)
    report = engine.run(all_data, bt_start, bt_end)

    # ── 3. Print report ───────────────────────────────────────────────────────
    report.print_report(verbose=verbose)
