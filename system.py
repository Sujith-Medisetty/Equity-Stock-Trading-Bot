"""
system.py — The master coordinator that wires all modules together.

TradingSystem is the only class that knows about all other classes.
It owns the daily workflow: collect data → detect market mode → screen stocks
→ find setups → execute trades → monitor → end of day.

Daily timeline (all times IST):
  08:45  step1_collect_data()          — fetch VIX, FII, OHLCV for all 15 stocks
  09:00  step2_detect_market_mode()    — decide today's market regime
  09:10  step3-5 (combined)            — screen, evaluate strategies, place entries
  09:15  step8_morning_briefing()      — print today's action plan
  09:30-15:30  step6_monitor_trades()  — every 15 mins: check exits, update trailing SL
  15:35  step7_end_of_day()            — dashboard, consecutive loss check, tax alert

State held in memory (reset each run):
  self.market_mode    — current MarketMode enum (AGGRESSIVE/NORMAL/SELECTIVE/CAUTIOUS/DEFENSIVE/CASH)
  self.fii_flow       — current FIIFlow enum (BUYING/SELLING/NEUTRAL)
  self.vix            — today's India VIX reading
  self.fii_net        — today's FII net cash (₹Cr)
  self.stocks_data    — dict of symbol → StockData with all indicators calculated
  self.nifty_data     — StockData for Nifty 50 index (used for RS calculation and mode detection)
  self.todays_setups  — list of Setup objects identified today (used in morning briefing)

Scheduler:
  start_scheduler() runs the entire workflow automatically on weekdays.
  Uses the `schedule` library — one job per event per day.
  Monitoring runs every 15 mins; sync_with_broker() runs inside each monitor cycle.

Modes:
  python main.py              → run_once_test()     single cycle, print dashboard
  python main.py --scheduler  → paper trade, full auto scheduler
  python main.py --live       → requires explicit Enter confirmation, sets PAPER_TRADE=False
  python main.py --dashboard  → just print the analytics dashboard, no trading
"""

import time
from datetime import datetime

from config import Config, Watchlist, log, SCHEDULE_AVAILABLE
from models import MarketMode, FIIFlow
from database import DatabaseManager
from data_collector import DataCollector
from indicators import IndicatorEngine
from market_mode import MarketModeEngine
from screener import StockScreener
from strategy import StrategyEngine
from risk import RiskManager
from protection import ProtectionEngine
from orders import OrderManager
from monitor import TradeMonitor
from analytics import PerformanceAnalytics, MorningBriefing

if SCHEDULE_AVAILABLE:
    import schedule


class TradingSystem:
    """
    Owns and orchestrates every module. One instance per run.
    On startup, immediately calls sync_with_broker() to catch any SL orders
    the broker executed since the last run (overnight, weekend, or after a crash).
    """

    def __init__(self):
        log.info("Initialising Nifty 50 Swing Trading System...")

        # Each module is instantiated once and held for the full session.
        # They share a single DatabaseManager (self.db) so all reads/writes
        # go to the same SQLite connection and transaction context.
        self.db            = DatabaseManager()
        self.collector     = DataCollector(self.db)           # fetches OHLCV, VIX, FII from Upstox + NSE
        self.market_mode_eng = MarketModeEngine(self.db)      # converts VIX + Nifty + FII → MarketMode enum
        self.screener      = StockScreener(self.db)           # hard filters — eliminates structurally bad stocks
        self.strategy_eng  = StrategyEngine()                 # evaluates 5 strategies and scores setups
        self.risk_mgr      = RiskManager(self.db)             # sizes positions and runs pre-trade checklist
        self.protection    = ProtectionEngine(self.db)        # circuit breakers: loss limits, cooldowns, timing
        self.order_mgr     = OrderManager(self.db)            # places/cancels orders at Upstox broker
        self.monitor       = TradeMonitor(self.db, self.order_mgr, self.protection)  # 15-min exit + trailing SL loop
        self.analytics     = PerformanceAnalytics(self.db)    # PnL dashboard + tax computation
        self.briefing      = MorningBriefing(self.db)         # 9:15 AM human-readable summary

        # In-memory state — reset every run.
        # These are updated by step1 and step2 and consumed by steps 3–8.
        self.market_mode       = MarketMode.CAUTIOUS      # safe default until step2 runs
        self.fii_flow          = FIIFlow.NEUTRAL           # safe default until step2 runs
        self.vix               = 15.0                     # safe calm default until step1 fetches real VIX
        self.fii_net           = 0.0                      # net FII buying in ₹Cr — positive=buying, negative=selling
        self.stocks_data       = {}                       # symbol → StockData dict, populated in step1
        self.nifty_data        = None                     # StockData for NIFTY 50, used for RS calculation
        self.todays_setups     = []                       # all Setup objects from step4, used in morning briefing
        self.available_capital = float(Config.TOTAL_CAPITAL)  # refreshed from Upstox fund limits in step1

        # Pending watchlist for PULLBACK and BREAKOUT confirmation.
        # A setup is added here on the first cycle it qualifies.
        # It is only entered on the NEXT cycle if still valid (15-min confirmation).
        # This avoids entering on a momentary spike — requires the setup to hold for one full cycle.
        # Format: {symbol: {"strategy": str, "since": datetime}}
        # Cleared at end of day (step7) so it starts fresh each morning.
        self.pending_watchlist: dict = {}

        # On startup: immediately reconcile DB against actual Upstox holdings.
        # Catches any SL orders the broker executed since last run (overnight, weekend).
        # Without this, the DB would still show those trades as OPEN — causing screener
        # to block their sector and cap count to be wrong for the whole day.
        self.monitor.sync_with_broker()
        log.info("System ready.")

    # =========================================================================
    # STEP 1: Pre-market data collection (8:45 AM)
    # Runs before market open. All 15 stocks + Nifty are fetched and indicators
    # are calculated. The 4H (60-min) and weekly charts are fetched for each stock
    # to compute tf_aligned_count (how many timeframes agree on bullish direction).
    # Results stored in self.stocks_data and also persisted to stock_snapshots DB table.
    # =========================================================================

    def step1_collect_data(self):
        log.info("STEP 1: Collecting market data...")

        # Once-a-day stale-trade exit (>15 days held without profit).
        # Uses date arithmetic only — running this every 15 min during the day
        # would re-evaluate the same value 26 times. Pre-market exits queue at
        # the broker and fill in the opening auction.
        self.monitor.check_stale_trades()

        # --- Market-wide signals ---
        self.vix     = self.collector.fetch_india_vix()    # India VIX — fear index. >=29 → CASH mode, >=25 → DEFENSIVE / tighten SLs
        fii_data     = self.collector.fetch_fii_dii()      # today's FII + DII net buying from NSE API
        self.fii_net = fii_data.get("fii_net_cash", 0.0)  # positive = FII buying, negative = selling

        # Count how many consecutive days FII has been buying or selling.
        # MarketModeEngine needs 3+ consecutive days above ₹2000 Cr to confirm a trend.
        buy_streak, sell_streak = self.collector.get_fii_consecutive_days()

        # Persist today's FII reading to fii_history so get_fii_consecutive_days()
        # can look back across previous days and count the streak.
        self.db.execute("""
            INSERT OR REPLACE INTO fii_history
            (date, fii_net_cash, dii_net_cash, consecutive_buying_days, consecutive_selling_days)
            VALUES (?,?,?,?,?)
        """, (datetime.now().strftime("%Y-%m-%d"),
              self.fii_net, fii_data.get("dii_net_cash", 0.0),
              buy_streak, sell_streak))

        # --- Nifty benchmark data ---
        # 250 days needed so EMA-200 is reliable (needs ~200+ bars to stabilise).
        # nifty_df is passed to IndicatorEngine.calculate_all() for each stock
        # so it can compute the relative strength (RS) score = stock return − Nifty return.
        nifty_ohlcv = self.collector.fetch_all_ohlcv_parallel(["NIFTY 50"])
        nifty_df = nifty_ohlcv.get("NIFTY 50", (None, None, None))[0]
        if nifty_df is not None:
            # Calculate indicators for Nifty itself — used in step2 to detect market mode
            # (is Nifty above EMA20? EMA50? EMA200? What is Nifty's RSI?)
            self.nifty_data = IndicatorEngine.calculate_all(nifty_df, "NIFTY50")

        # --- Per-stock data collection (parallel fetch across all 15+ symbols) ---
        # fetch_all_ohlcv_parallel() fires UPSTOX_MAX_WORKERS threads simultaneously,
        # with a shared rate limiter so combined request rate stays under Upstox's limit.
        # Each call retries up to API_MAX_RETRIES times with exponential backoff.
        # Returns {symbol: (df_daily, df_4h, df_weekly)} — None tuples for failed symbols.
        all_ohlcv = self.collector.fetch_all_ohlcv_parallel(Watchlist.get_symbols())

        for sym, (df_daily, df_4h, df_weekly) in all_ohlcv.items():
            if df_daily is None:
                log.warning(f"Skipping {sym} — daily OHLCV fetch failed (no mock in live mode).")
                continue

            # Build the StockData object with all daily indicators.
            # nifty_df is passed here so RS score can be computed inside calculate_all()
            s_data = IndicatorEngine.calculate_all(df_daily, sym, nifty_df)

            if s_data:
                s_data.h4_bullish     = IndicatorEngine.check_4h_bullish(df_4h)
                s_data.weekly_bullish = IndicatorEngine.check_weekly_bullish(df_weekly)
                s_data.tf_aligned_count = sum([
                    s_data.weekly_bullish, s_data.daily_bullish, s_data.h4_bullish
                ])
                # Detect 4H FVG zones and merge with daily zones into the three FVG flags.
                # Daily FVGs are rare (big structural gaps); 4H FVGs occur more frequently
                # and give finer-grained confluence signals for intraday entries.
                # apply_combined_fvg_flags() updates in_fvg_zone, fvg_pullback, fvg_target
                # using OR logic across both timeframes.
                s_data.fvg_zones_4h = IndicatorEngine.detect_4h_fvg_zones(df_4h)
                IndicatorEngine.apply_combined_fvg_flags(s_data, s_data.close)
                self.stocks_data[sym] = s_data
                self.db.save_stock_snapshot(s_data)

        # Fetch and save upcoming earnings/results dates for all 15 stocks.
        # Events within 5 days get risk_level='RED' — screener blocks entries on those stocks.
        events = self.collector.fetch_events_calendar()
        self.collector.save_events(events)

        # Refresh available capital from Upstox's fund limits API.
        # This is the real cash available AFTER existing positions are deployed.
        # Used in step5 to cap position sizes correctly.
        self.available_capital = self.order_mgr.get_available_capital()

        log.info(f"Data collected: {len(self.stocks_data)} stocks | VIX: {self.vix:.1f} | "
                 f"FII: ₹{self.fii_net:.0f}Cr | Available capital: ₹{self.available_capital:,.0f}")

    # =========================================================================
    # STEP 2: Market mode detection (9:00 AM)
    # Reads VIX, Nifty technicals, and FII streak from step 1 data.
    # Sets self.market_mode (the master switch for the whole day) and self.fii_flow.
    # Also saves the full market picture to market_snapshots for historical analysis.
    # =========================================================================

    def step2_detect_market_mode(self):
        log.info("STEP 2: Detecting market mode...")

        # Re-read streaks from DB (they were written in step1) — ensures consistency
        # even if step2 is called independently after a restart.
        buy_s, sell_s = self.collector.get_fii_consecutive_days()

        # Returns a tuple: (MarketMode enum, FIIFlow enum)
        # MarketMode controls which strategies are allowed and how many positions we take.
        # FIIFlow is stored separately — used in step4 to check FII_FLOW strategy eligibility.
        self.market_mode, self.fii_flow = self.market_mode_eng.detect(
            vix=self.vix, nifty_data=self.nifty_data,
            fii_net=self.fii_net,
            fii_consecutive_buy=buy_s, fii_consecutive_sell=sell_s
        )

        # Save the full market snapshot to DB.
        # Used for two purposes:
        # 1. Historical analysis: "what was the market mode when this trade was taken?"
        # 2. Morning briefing: GIFT Nifty is fetched here and shown at 9:15 AM
        nifty = self.nifty_data
        self.db.save_market_snapshot({
            "date":                datetime.now().strftime("%Y-%m-%d"),
            "nifty_close":         nifty.close   if nifty else 0,
            "nifty_ema20":         nifty.ema_20  if nifty else 0,
            "nifty_ema50":         nifty.ema_50  if nifty else 0,
            "nifty_ema200":        nifty.ema_200 if nifty else 0,
            "nifty_rsi":           nifty.rsi     if nifty else 50,
            "india_vix":           self.vix,
            "gift_nifty":          self.collector.fetch_gift_nifty(),  # pre-market global signal
            "fii_net_cash":        self.fii_net,
            "dii_net_cash":        0.0,
            "fii_consecutive_days": buy_s,
            "market_mode":         self.market_mode.value,
            "fii_flow_label":      self.fii_flow.value
        })

    # =========================================================================
    # STEP 3: Screen stocks (9:10 AM)
    # Hard-filter pass over all 15 stocks. Eliminates: high volatility, already held,
    # same sector open, RED events within 5 days, low volume, defensive/cash mode.
    # Returns the subset of symbols that are eligible for strategy evaluation.
    # =========================================================================

    def step3_screen_stocks(self) -> list:
        log.info("STEP 3: Screening stocks...")
        open_trades = list(self.db.get_open_trades())  # currently held positions — screener uses these to check sector overlap

        # screen() returns a list of symbol strings that passed all 6 hard filters.
        # Only these symbols will be evaluated by the strategy engine in step4.
        passed = self.screener.screen(self.stocks_data, open_trades, self.market_mode)
        log.info(f"Screening result: {passed}")
        return passed

    # =========================================================================
    # STEP 4: Strategy matching (9:20 AM)
    # For each screened stock: evaluate all 5 strategies, pick the best setup,
    # then size the position (RiskManager: SL, target, shares, R:R check).
    # Setups passing risk validation are saved to DB and collected in a list.
    # Final list is sorted by score descending — best setups execute first.
    # =========================================================================

    def step4_find_setups(self, screened_symbols: list) -> list:
        log.info("STEP 4: Finding setups...")
        setups        = []
        fii_sectors   = self._get_fii_buying_sectors()  # which sectors have FII inflows today (for FII_FLOW strategy)

        for symbol in screened_symbols:
            data = self.stocks_data.get(symbol)
            if not data:
                continue  # shouldn't happen since screened_symbols came from stocks_data, but guard anyway

            # Check if this stock's sector is actively receiving FII inflows.
            # Passed to evaluate_all() so the FII_FLOW strategy can gate on it.
            fii_sector_buying = Watchlist.get_sector(symbol) in fii_sectors

            # evaluate_all() tries all 5 strategies and returns the highest-priority
            # one that passes its checks. Returns None if no strategy qualifies.
            setup = self.strategy_eng.evaluate_all(
                symbol=symbol, data=data,
                market_mode=self.market_mode,
                fii_flow=self.fii_flow,
                fii_sector_buying=fii_sector_buying
            )

            if setup:
                # calculate_setup_risk() fills in sl_price, target_price, shares, rr_ratio.
                # Uses available_capital to derive the max risk budget and position cap.
                # Sets setup.status = "SKIPPED" if the math doesn't work (R:R too low, position too small, etc.)
                setup = self.risk_mgr.calculate_setup_risk(setup, data, self.available_capital)

                if setup.status != "SKIPPED":
                    # Persist the setup to DB (status=PENDING) so we can audit what was found today.
                    # Saved even if we don't trade it (e.g. portfolio is full) — helps review missed setups.
                    self.db.save_setup(setup)
                    setups.append(setup)
                    log.info(f"SETUP: {symbol} | {setup.strategy.value} | Score:{setup.score} | "
                             f"Entry:₹{setup.entry_price:.2f} SL:₹{setup.sl_price:.2f} T:₹{setup.target_price:.2f}")

        # Sort highest score first — step5 will trade them in this order,
        # stopping when the position limit is reached. Best setups get priority.
        setups.sort(key=lambda x: x.score, reverse=True)
        log.info(f"Total setups: {len(setups)}")
        return setups

    def _get_fii_buying_sectors(self) -> set:
        """
        Returns the set of sectors currently receiving FII inflows.
        Used in step4 to identify which stocks qualify for the FII_FLOW strategy.

        Logic: if FII is in BUYING state (3+ days, ₹2000Cr+ net), look at each
        stock's RS score. Stocks with rs_score > 3 are outperforming Nifty — this
        is used as a proxy for "FII is flowing into this sector". The sector of each
        such stock is added to the result set.
        """
        if self.fii_flow != FIIFlow.BUYING:
            return set()  # FII not in buying mode → no sectors qualify → FII_FLOW strategy inactive

        # Find sectors where outperforming stocks (rs_score > 3) exist.
        # rs_score = stock's 60-day return minus Nifty's 60-day return.
        # rs_score > 3 means the stock outperformed Nifty by more than 3 percentage points — strong signal.
        return {
            Watchlist.get_sector(sym)
            for sym, data in self.stocks_data.items()
            if data and data.rs_score > 3
        }

    # =========================================================================
    # STEP 5: Execute trades (9:30 AM)
    # For each approved setup (score >= 60, checklist passed):
    # place the entry order + SL order at the broker.
    # Stops adding new trades once effective_max_trades(available_capital) is reached.
    # The full pre-trade checklist (10 conditions) runs here as the final gate.
    # =========================================================================

    def step5_execute_trades(self, setups: list):
        log.info("STEP 5: Evaluating trades...")

        # Check all protection circuit breakers before placing any order.
        # Returns (True, "OK") only if all pass. On first failure → (False, reason).
        allowed, reason = self.protection.is_trading_allowed()
        if not allowed:
            log.warning(f"Trading blocked: {reason}")
            return

        # Account-level floor: if available capital is below ₹20k, position sizing
        # would produce positions too small to be profitable after charges.
        if self.available_capital < Config.MIN_TRADE_CAPITAL:
            log.warning(
                f"INSUFFICIENT CAPITAL: ₹{self.available_capital:,.0f} available — "
                f"minimum required is ₹{Config.MIN_TRADE_CAPITAL:,}. No new trades placed."
            )
            return

        # --- Pending watchlist: expire stale candidates ---
        # A candidate that hasn't re-qualified within 45 minutes is dropped.
        # This prevents acting on a setup that was valid once but drifted away.
        stale = [k for k, v in self.pending_watchlist.items()
                 if (datetime.now() - v["since"]).total_seconds() > 2700]
        for k in stale:
            log.info(f"WATCHLIST EXPIRE: {k} ({self.pending_watchlist[k]['strategy']}) — "
                     f"setup not confirmed within 45 min, dropping")
            del self.pending_watchlist[k]

        open_trades = list(self.db.get_open_trades())  # current holdings — used to check position count and portfolio risk

        # Strategies that require a 15-min confirmation before entry.
        # On the first cycle a setup qualifies: add to watchlist, don't enter.
        # On the next cycle (15 min later), if still valid: enter.
        # In BACKTEST_MODE we skip this — backtests run one cycle per "day" so
        # a two-cycle confirmation would block all PULLBACK/BREAKOUT entries.
        CONFIRM_STRATEGIES = {"PULLBACK", "BREAKOUT"}

        for setup in setups:
            # Stop if we've hit the maximum allowed concurrent positions.
            # effective_max_trades() scales with capital: at ₹50k → max 3, at ₹200k → max 4.
            if len(open_trades) >= Config.effective_max_trades(self.available_capital):
                break

            # Minimum score gate: anything below 60 is too low-confidence to enter.
            if setup.score < 60:
                continue

            # --- 15-min confirmation gate for PULLBACK and BREAKOUT ---
            strategy_name = setup.strategy.value if setup.strategy else ""
            if strategy_name in CONFIRM_STRATEGIES and not Config.BACKTEST_MODE:
                if setup.symbol not in self.pending_watchlist:
                    # First time this setup qualifies — add to watchlist, wait for next cycle.
                    # A real PULLBACK or BREAKOUT holds for at least 15 minutes.
                    # A momentary price spike that immediately reverses won't survive confirmation.
                    self.pending_watchlist[setup.symbol] = {
                        "strategy": strategy_name,
                        "since": datetime.now(),
                    }
                    log.info(f"WATCHLIST ADD: {setup.symbol} {strategy_name} score:{setup.score} "
                             f"— waiting 15-min confirmation before entry")
                    continue  # do NOT enter this cycle
                else:
                    # Already in watchlist from a previous cycle — setup held for ≥15 min, confirmed.
                    age_mins = (datetime.now() - self.pending_watchlist[setup.symbol]["since"]).total_seconds() / 60
                    log.info(f"WATCHLIST CONFIRMED: {setup.symbol} {strategy_name} "
                             f"held for {age_mins:.0f} min — proceeding to entry")
                    del self.pending_watchlist[setup.symbol]  # remove; order placement follows below

            # --- Live price validation ---
            # setup.entry_price was set to yesterday's close during step4 (8:45 AM data).
            # By now the market is open and price has moved. We must check the live price
            # before placing an order — we never buy at yesterday's closing price.
            live = self._get_live_price(setup.symbol)
            if live > 0:
                # drift = how far the live price has moved from the setup price (yesterday's close).
                # Expressed as a percentage.
                drift = abs(live - setup.entry_price) / setup.entry_price * 100

                # FII_FLOW, BREAKOUT, WEEK52 strategies often have gap-ups on the entry signal
                # (e.g. a stock breaks out overnight). Allow 3% drift for these.
                # Other strategies (SWING, PULLBACK) expect price near yesterday's close — use 1.5%.
                strategy_name = setup.strategy.value if setup.strategy else ""
                max_drift = (Config.MAX_ENTRY_DRIFT_PCT_WIDE
                             if strategy_name in Config.WIDE_DRIFT_STRATEGIES
                             else Config.MAX_ENTRY_DRIFT_PCT)

                if drift > max_drift:
                    # Price has moved too much from the setup — the setup's technical picture
                    # (RSI zone, EMA distance, etc.) may no longer be valid at this new price.
                    log.info(
                        f"SKIP {setup.symbol}: live ₹{live:.2f} drifted "
                        f"{drift:.1f}% from setup ₹{setup.entry_price:.2f} (limit {max_drift}%)"
                    )
                    setup.status = "SKIPPED"
                    setup.skip_reason = f"Price drifted {drift:.1f}% from setup entry (limit {max_drift}%)"
                    continue

                # Price is within acceptable range — update entry to the actual live price.
                # Orders will be placed at this live price, not at yesterday's close.
                setup.entry_price = round(live, 2)

                data = self.stocks_data.get(setup.symbol)
                if data:
                    import copy

                    # Build a live-price-patched copy of StockData.
                    # This mirrors exactly what run_midday_scan() does.
                    # We must re-evaluate the strategy with the live close because several
                    # strategy checks directly depend on the price relative to EMAs:
                    #   PULLBACK hard gate : abs(close - ema_20) / ema_20 <= 0.5%
                    #                        → if price gapped away, pullback setup is gone
                    #   WEEK52 gate        : close >= week_52_high (within 1%)
                    #                        → if price gapped back below 52W high, breakout is invalid
                    #   SWING/BREAKOUT/FII : close > ema_20, close > ema_50
                    #                        → can flip on a 1%+ gap
                    # Without this re-check, we'd enter a trade whose setup conditions
                    # were only valid at yesterday's close, not at the actual live price.
                    live_data = copy.copy(data)          # shallow copy — keeps all indicator values
                    live_data.close = live               # replace yesterday's close with live price
                    live_data.high  = max(data.high, live)   # intraday high could be above yesterday's high
                    live_data.low   = min(data.low, live)    # intraday low could be below yesterday's low
                    # Recompute daily_bullish with live price vs morning EMA values.
                    # EMAs themselves don't change (daily indicators on 250 bars) but
                    # whether price is above/below them changes if there's a gap.
                    live_data.daily_bullish = (live > data.ema_20 and data.ema_20 > data.ema_50)
                    # Re-evaluate FVG flags (in_fvg_zone, fvg_pullback, fvg_target) against live price.
                    # Uses both daily and 4H zone lists already stored in live_data (copied from data).
                    IndicatorEngine.apply_combined_fvg_flags(live_data, live)

                    # Re-run all 5 strategy checks against the live-price-patched data.
                    # Returns the best qualifying setup, or None if none qualifies at the live price.
                    fii_sector_buying = Watchlist.get_sector(setup.symbol) in self._get_fii_buying_sectors()
                    fresh_setup = self.strategy_eng.evaluate_all(
                        symbol=setup.symbol, data=live_data,
                        market_mode=self.market_mode,
                        fii_flow=self.fii_flow,
                        fii_sector_buying=fii_sector_buying
                    )

                    if not fresh_setup:
                        # No strategy qualifies at the live price — the gap invalidated all checks.
                        # e.g. PULLBACK: price drifted 0.8% from EMA20 (within drift limit but outside 0.5% gate)
                        log.info(
                            f"SKIP {setup.symbol}: {setup.strategy.value} no longer valid at "
                            f"live ₹{live:.2f} (was valid at close ₹{data.close:.2f})"
                        )
                        setup.status = "SKIPPED"
                        setup.skip_reason = (
                            f"{setup.strategy.value} conditions failed at live price ₹{live:.2f} "
                            f"(setup was scored at yesterday's close ₹{data.close:.2f})"
                        )
                        continue

                    if fresh_setup.strategy != setup.strategy:
                        # The live price triggered a different strategy than what was scored at 9:10 AM.
                        # e.g. was FII_FLOW at ₹1200 close but only qualifies as SWING at ₹1208 live.
                        # We skip rather than switch strategies mid-flight — the SL, risk params,
                        # and score were all computed for the original strategy type.
                        log.info(
                            f"SKIP {setup.symbol}: strategy shifted from {setup.strategy.value} "
                            f"→ {fresh_setup.strategy.value} at live ₹{live:.2f}. "
                            f"Re-scoring against original strategy not safe — skipping."
                        )
                        setup.status = "SKIPPED"
                        setup.skip_reason = (
                            f"Strategy changed {setup.strategy.value}→{fresh_setup.strategy.value} "
                            f"at live price"
                        )
                        continue

                    # Same strategy still qualifies at the live price — update the score.
                    # Score may differ slightly because daily_bullish and close vs EMA comparisons
                    # are now evaluated at the live price.
                    if fresh_setup.score != setup.score:
                        log.info(
                            f"{setup.symbol}: score updated {setup.score}→{fresh_setup.score} "
                            f"at live ₹{live:.2f}"
                        )
                    setup.score = fresh_setup.score

                    # Recalculate SL, target, shares using live_data so all SL formulas
                    # (including BREAKOUT's consolidation-box low and PULLBACK's EMA20 level)
                    # are consistent with the live price context.
                    setup = self.risk_mgr.calculate_setup_risk(
                        setup, live_data, self.available_capital
                    )
                    if setup.status == "SKIPPED":
                        # R:R fell below 2.0 at live price, or position too small, etc.
                        log.info(f"SKIP {setup.symbol} after live-price recalc: {setup.skip_reason}")
                        continue

            # --- Final gate: 10-point pre-trade checklist ---
            # All 10 conditions must pass — any single failure rejects the trade.
            # Checks: market mode, VIX, no imminent event, score >= 60 AND >= 80,
            #         valid SL, R:R >= 2.0, capital per trade, total portfolio risk.
            approved, failed = self.risk_mgr.run_pre_trade_checklist(
                setup, self.market_mode, self.vix, open_trades, self.available_capital
            )
            if not approved:
                setup.status = "SKIPPED"
                setup.skip_reason = f"Failed: {failed}"  # list of failed check names for debugging
                continue

            # --- Place the order ---
            # place_entry_order() places a BUY + SL SELL order at the broker.
            # Returns trade_id (a string like "ICICIBANK_20240509093015") on success, None on failure.
            trade_id = self.order_mgr.place_entry_order(setup)
            if trade_id:
                setup.status = "TAKEN"
                open_trades = list(self.db.get_open_trades())  # refresh the list so next loop iteration has updated count
            else:
                log.error(f"Order failed: {setup.symbol}")

        # Save all setups (including SKIPPED ones) to self.todays_setups
        # so the morning briefing can show what was found and what was entered today.
        self.todays_setups = setups

    # =========================================================================
    # MIDDAY SCAN — re-runs setup search on every 15-min cycle (via _conditional_monitor)
    # Uses daily indicators already calculated at 8:45 AM (stored in stocks_data)
    # but replaces each stock's close, volume_ratio, open, and FVG flags with live values.
    # This catches setups that didn't exist at open — e.g. a stock that pulls
    # back to EMA20 at 11 AM, forming a valid PULLBACK entry missed at 9:10 AM.
    # =========================================================================

    def run_midday_scan(self):
        if not self.stocks_data:
            log.warning("Midday scan: no stock data — morning scan must run first")
            return

        allowed, reason = self.protection.is_trading_allowed()
        if not allowed:
            log.info(f"Midday scan skipped: {reason}")
            return

        log.info("MIDDAY SCAN: re-running with live prices + intraday volume...")

        # Fetch live quotes (price + today's traded volume) for all 15 stocks in one call.
        # get_all_live_quotes() uses the full-quote endpoint instead of ltp() so we get
        # today's intraday volume to refresh volume_ratio alongside the live price.
        live_quotes = self._get_all_live_quotes()
        if not live_quotes:
            log.warning("Midday scan: could not fetch live quotes — skipping")
            return

        import copy

        refreshed = {}
        for symbol, data in self.stocks_data.items():
            quote = live_quotes.get(symbol)
            if not quote:
                refreshed[symbol] = data
                continue

            live         = quote["price"]
            today_volume = quote["volume"]

            if live <= 0:
                refreshed[symbol] = data
                continue

            updated               = copy.copy(data)
            updated.close         = live
            updated.high          = max(data.high, live)
            updated.low           = min(data.low, live)
            updated.daily_bullish = (live > data.ema_20 and data.ema_20 > data.ema_50)

            # Refresh today's session open.
            # data.open is yesterday's daily open — meaningless for the PULLBACK check
            # "close > open" (green candle = buyers stepped in today).
            # The full quote returns today's intraday OHLC, so we can use the real open.
            # If open is 0 (fallback ltp path), leave as yesterday's — stale but safe.
            today_open = quote.get("open", 0)
            if today_open > 0:
                updated.open = today_open

            # Refresh volume_ratio with today's real intraday volume.
            # vol_avg (20-day average) is back-calculated from morning data:
            #   morning volume_ratio = yesterday_volume / vol_avg
            #   → vol_avg = yesterday_volume / morning_volume_ratio
            # Then: intraday_volume_ratio = today_volume / vol_avg
            # This makes BREAKOUT's 2x gate and PULLBACK's low-volume check honest
            # for the actual session rather than relying on yesterday's volumes.
            if today_volume > 0 and data.volume > 0 and data.volume_ratio > 0:
                vol_avg              = data.volume / data.volume_ratio  # 20-day avg from morning
                updated.volume_ratio = round(today_volume / vol_avg, 2)
                log.debug(f"{symbol}: volume_ratio refreshed "
                          f"{data.volume_ratio:.2f}→{updated.volume_ratio:.2f} "
                          f"(today vol {today_volume:,.0f}, avg {vol_avg:,.0f})")
            # If volume=0 (fallback ltp path), volume_ratio stays from morning — stale but safe.

            # Re-evaluate all FVG flags (daily + 4H) against the live price.
            # Zone boundaries are fixed from the morning — only whether live price
            # falls inside them changes intraday. OR logic across both timeframes.
            IndicatorEngine.apply_combined_fvg_flags(updated, live)

            refreshed[symbol] = updated

        # Temporarily swap stocks_data with the refreshed version,
        # run the full screen → strategy → execute pipeline, then restore original.
        original         = self.stocks_data
        self.stocks_data = refreshed
        screened         = self.step3_screen_stocks()
        setups           = self.step4_find_setups(screened)
        self.step5_execute_trades(setups)
        self.stocks_data = original
        log.info("MIDDAY SCAN complete")

    def _get_live_price(self, symbol: str) -> float:
        """Returns the current live price for one symbol via Upstox batch LTP.
        Returns 0.0 if Upstox is unavailable or the fetch fails."""
        prices = self.order_mgr.get_all_live_prices([symbol])
        return prices.get(symbol, 0.0)

    def _get_all_live_prices(self) -> dict:
        """
        Returns live prices for all watchlist symbols in ONE Upstox batch LTP call.
        Dict of {symbol: price}. Used by monitoring — only needs price, fast.
        """
        return self.order_mgr.get_all_live_prices(Watchlist.get_symbols())

    def _get_all_live_quotes(self) -> dict:
        """
        Returns live quotes (price + today's intraday volume) for all watchlist symbols.
        Dict of {symbol: {"price": float, "volume": float}}.
        Used by run_midday_scan() to refresh volume_ratio alongside the live price.
        """
        return self.order_mgr.get_all_live_quotes(Watchlist.get_symbols())

    # =========================================================================
    # STEP 6: Intraday monitoring (every 15 mins from 9:30 AM to 3:30 PM)
    # Fetches current prices from Upstox (real prices in both paper and live mode),
    # then passes them to TradeMonitor for exit checks and trailing SL updates.
    # =========================================================================

    def step6_monitor_trades(self):
        # Get current live prices for all open positions.
        # Falls back to last known price if Upstox is unavailable.
        current_prices = self._get_current_prices()

        # TradeMonitor does: broker sync → exit checks → trailing SL updates.
        # vix is passed so VIX-spike exits can fire even if we didn't run step2 today.
        self.monitor.monitor_all_trades(current_prices, self.vix)

    def _get_current_prices(self) -> dict:
        """
        Fetches live prices for all open positions via a single Upstox batch LTP call.
        Falls back to the last known price from DB if the API call fails — so the
        monitoring cycle still runs even if Upstox is temporarily unreachable.
        """
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return {}

        open_symbols = [t["symbol"] for t in open_trades]

        # Single batch call — all open symbols in one API request
        live = self.order_mgr.get_all_live_prices(open_symbols)

        prices = {}
        for t in open_trades:
            sym = t["symbol"]
            if sym in live:
                prices[sym] = live[sym]
            else:
                # Upstox couldn't return this price — use last known from DB
                fallback = t["current_price"] or t["entry_price"]
                prices[sym] = fallback
                log.warning(f"Live price unavailable for {sym} — using last known ₹{fallback:.2f}")

        return prices

    # =========================================================================
    # STEP 7: End of day (15:35 — after market close at 15:30)
    # Prints the full analytics dashboard, checks for 3 consecutive losses
    # (soft alert to review the system), and warns if advance tax is due.
    # =========================================================================

    def step7_end_of_day(self):
        log.info("STEP 7: End of day tasks...")

        # Clear the pending watchlist — any unconfirmed setups from today are stale.
        # Tomorrow morning's scan will re-identify candidates fresh from new data.
        if self.pending_watchlist:
            log.info(f"Clearing {len(self.pending_watchlist)} unconfirmed watchlist entries at EOD")
            self.pending_watchlist.clear()

        # Print the full dashboard: today's closed trades, open positions, monthly stats,
        # strategy breakdown, and tax estimate for the current financial year.
        self.analytics.print_dashboard(self.available_capital)

        # Soft warning — if last 3 closed trades were all losses, log a review reminder.
        # This is NOT a hard block; the system continues trading the next day.
        if self.protection.check_consecutive_losses():
            log.warning("3 CONSECUTIVE LOSSES — Review your system.")

    # =========================================================================
    # STEP 8: Morning briefing (9:15 AM — after steps 3-5 complete)
    # Prints a human-readable summary of market conditions and today's setups.
    # Includes any RED-risk events (earnings within 2 days) as action items.
    # =========================================================================

    def step8_morning_briefing(self):
        # Fetch RED events within 2 days — these are shown as urgent action items.
        # The screener already blocked entries on these stocks, but the operator
        # should also know if any OPEN positions need to be manually reviewed.
        events_today = self.db.fetchall(
            "SELECT * FROM events_calendar WHERE days_away <= 2 AND risk_level='RED'"
        )
        print(self.briefing.generate(
            market_mode=self.market_mode, fii_flow=self.fii_flow,
            vix=self.vix, fii_net=self.fii_net,
            setups=self.todays_setups, events_today=list(events_today)
        ))

    # =========================================================================
    # MASTER DAILY RUN
    # =========================================================================

    def run_daily(self):
        log.info("=" * 60)
        log.info("DAILY RUN STARTING")
        log.info("=" * 60)
        self.step1_collect_data()         # 8:45 AM: fetch all data, calculate indicators
        self.step2_detect_market_mode()   # 9:00 AM: decide regime
        screened = self.step3_screen_stocks()    # 9:10 AM: hard filters
        setups   = self.step4_find_setups(screened)  # 9:10 AM: strategy evaluation
        self.step5_execute_trades(setups)        # 9:30 AM: place orders
        self.step8_morning_briefing()            # 9:15 AM: print summary
        log.info("Daily setup complete. Monitoring starts at 9:30 AM.")

    # =========================================================================
    # SCHEDULER
    # =========================================================================

    def start_scheduler(self):
        if not SCHEDULE_AVAILABLE:
            log.error("schedule package not installed. pip install schedule")
            return

        log.info(f"Starting scheduler | Paper Trade: {Config.PAPER_TRADE}")

        # Register all daily jobs for Monday through Friday.
        # Each job fires at the specified time on each weekday.
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            getattr(schedule.every(), day).at("08:45").do(self.step1_collect_data)
            getattr(schedule.every(), day).at("09:00").do(self.step2_detect_market_mode)
            getattr(schedule.every(), day).at("09:10").do(
                # Chain steps 3→4→5 as a single job so they run in sequence at 9:10.
                # Lambda defers execution until the scheduled time.
                lambda: self.step5_execute_trades(
                    self.step4_find_setups(self.step3_screen_stocks())
                )
            )
            getattr(schedule.every(), day).at("09:15").do(self.step8_morning_briefing)
            getattr(schedule.every(), day).at("15:35").do(self.step7_end_of_day)  # 5 min after market close
            # run_midday_scan removed from fixed slots — now fires every 15 mins via _conditional_monitor

        # Monitoring fires every 15 minutes all day, but _conditional_monitor()
        # checks if it's a weekday and within market hours before actually running.
        schedule.every(15).minutes.do(self._conditional_monitor)

        # Weekly review runs Sunday evening — recap of the week + DB cleanup.
        schedule.every().sunday.at("20:00").do(self._weekly_review)

        print("\nScheduler running. Press Ctrl+C to stop.\n")
        try:
            while True:
                schedule.run_pending()  # fires any jobs whose scheduled time has passed
                time.sleep(30)          # check every 30s — frequent enough, not excessive
        except KeyboardInterrupt:
            log.info("System stopped.")

    def _conditional_monitor(self):
        """
        Fires every 15 mins. Does two things if it's a weekday:
        1. step6_monitor_trades — checks exits + trailing SL for all open trades (9:15 AM – 3:30 PM)
        2. run_midday_scan     — scans for NEW entries with live prices (9:30 AM – 1:30 PM)
           The 9:30 AM start avoids opening auction noise.
           The 1:30 PM cutoff stops new entries with insufficient time left in the day.
        """
        now = datetime.now()
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return

        market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        entry_start  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        entry_cutoff = now.replace(hour=13, minute=30, second=0, microsecond=0)

        if market_open <= now <= market_close:
            self.step6_monitor_trades()

        if entry_start <= now <= entry_cutoff:
            self.run_midday_scan()

    def _weekly_review(self):
        """
        Runs Sunday evening. Prints a brief weekly recap and
        cleans up DB rows older than 90 days to keep the file size manageable.
        """
        m  = self.analytics.monthly_summary()   # reuse monthly stats for the weekly print
        print("\n" + "=" * 60)
        print("  WEEKLY REVIEW")
        print("=" * 60)
        if "total_trades" in m:
            print(f"  Trades   : {m['total_trades']}  WR: {m['win_rate']}%  Net: ₹{m['net_pnl']:,.2f}")
        print("=" * 60 + "\n")

        # Prune stock_snapshots, setups, and trailing_sl_log older than 90 days.
        # Trades, daily_pnl, and fii_history are never pruned (needed for tax + streaks).
        self.db.cleanup_old_data(days=90)

    def run_once_test(self):
        """Single test cycle: run the full daily workflow once and print the dashboard.
        Used in: python main.py (no flags). Good for verifying the system is working."""
        print("\n" + "=" * 60)
        print("RUNNING SINGLE TEST CYCLE")
        print("=" * 60 + "\n")
        self.run_daily()
        self.analytics.print_dashboard(self.available_capital)
