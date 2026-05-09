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
from risk import RiskManager, ChargesCalculator
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
        self.collector     = DataCollector(self.db)           # fetches OHLCV, VIX, FII from Dhan + NSE
        self.market_mode_eng = MarketModeEngine(self.db)      # converts VIX + Nifty + FII → MarketMode enum
        self.screener      = StockScreener(self.db)           # hard filters — eliminates structurally bad stocks
        self.strategy_eng  = StrategyEngine()                 # evaluates 5 strategies and scores setups
        self.risk_mgr      = RiskManager(self.db)             # sizes positions and runs pre-trade checklist
        self.protection    = ProtectionEngine(self.db)        # circuit breakers: loss limits, cooldowns, timing
        self.order_mgr     = OrderManager(self.db)            # places/cancels orders at Dhan broker
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
        self.available_capital = float(Config.TOTAL_CAPITAL)  # refreshed from Dhan fund limits in step1

        # On startup: immediately reconcile DB against actual Dhan holdings.
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

        # --- Market-wide signals ---
        self.vix     = self.collector.fetch_india_vix()    # India VIX — fear index. >=28 → CASH mode, >=22 → DEFENSIVE
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
        nifty_df = self.collector.fetch_ohlcv_daily("NIFTY 50", days=250)
        if nifty_df is not None:
            # Calculate indicators for Nifty itself — used in step2 to detect market mode
            # (is Nifty above EMA20? EMA50? EMA200? What is Nifty's RSI?)
            self.nifty_data = IndicatorEngine.calculate_all(nifty_df, "NIFTY50")

        # --- Per-stock data collection ---
        for sym in Watchlist.get_symbols():
            # Daily bars for indicator calculation (EMAs, RSI, MACD, ATR, OBV, etc.)
            df_daily  = self.collector.fetch_ohlcv_daily(sym, days=250)

            # 60-min bars for 4H timeframe check: is price above 20-EMA on the hourly chart?
            # 30 days of 60-min bars = ~180 candles — enough for a 20-period EMA on hourly
            df_4h     = self.collector.fetch_ohlcv_intraday(sym, "60", 30)

            # 500 days for the weekly check: price above 20-week EMA?
            # We use daily bars resampled to weekly — 500 daily bars ≈ 100 weekly bars
            df_weekly = self.collector.fetch_ohlcv_daily(sym, days=500)

            # Build the StockData object with all daily indicators.
            # nifty_df is passed here so RS score can be computed inside calculate_all()
            s_data    = IndicatorEngine.calculate_all(df_daily, sym, nifty_df)

            if s_data:
                # Attach multi-timeframe checks to StockData.
                # These are calculated from intraday and weekly data — separate from daily.
                s_data.h4_bullish     = IndicatorEngine.check_4h_bullish(df_4h)
                s_data.weekly_bullish = IndicatorEngine.check_weekly_bullish(df_weekly)

                # tf_aligned_count = how many of the 3 timeframes (weekly/daily/4H) are bullish.
                # Most strategies require at least 2 out of 3 for confirmation.
                s_data.tf_aligned_count = sum([s_data.weekly_bullish, s_data.daily_bullish, s_data.h4_bullish])

                # Store in memory for immediate use by steps 3-5 today
                self.stocks_data[sym] = s_data

                # Also persist to DB so we have a historical record of indicators
                # (useful for debugging: "what was ICICIBANK's RSI on 2024-05-09?")
                self.db.save_stock_snapshot(s_data)

        # Fetch and save upcoming earnings/results dates for all 15 stocks.
        # Events within 5 days get risk_level='RED' — screener blocks entries on those stocks.
        events = self.collector.fetch_events_calendar()
        self.collector.save_events(events)

        # Refresh available capital from Dhan's fund limits API.
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

        open_trades = list(self.db.get_open_trades())  # current holdings — used to check position count and portfolio risk

        for setup in setups:
            # Stop if we've hit the maximum allowed concurrent positions.
            # effective_max_trades() scales with capital: at ₹50k → max 3, at ₹200k → max 4.
            if len(open_trades) >= Config.effective_max_trades(self.available_capital):
                break

            # Minimum score gate: anything below 60 is too low-confidence to enter.
            if setup.score < 60:
                continue

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
    # MIDDAY SCAN — re-runs setup search at 11:30 AM and 1:30 PM
    # Uses daily indicators already calculated at 8:45 AM (stored in stocks_data)
    # but replaces each stock's 'close' with the current live price.
    # This catches setups that didn't exist at open — e.g. a stock that pulls
    # back to EMA20 at 11 AM, forming a valid PULLBACK entry missed at 9:10 AM.
    # =========================================================================

    def run_midday_scan(self):
        if not self.stocks_data:
            log.warning("Midday scan: no stock data — morning scan must run first")
            return

        # Check all circuit breakers — don't enter trades if protection is active
        allowed, reason = self.protection.is_trading_allowed()
        if not allowed:
            log.info(f"Midday scan skipped: {reason}")
            return

        log.info("MIDDAY SCAN: re-running with live prices...")

        # Fetch current live prices for all 15 stocks in one pass
        live_prices = self._get_all_live_prices()
        if not live_prices:
            log.warning("Midday scan: could not fetch live prices — skipping")
            return

        import copy

        # Build a refreshed snapshot dict: same indicators as this morning,
        # but close/high/low replaced with live values so setup conditions
        # (e.g. "price within 0.5% of EMA20") are evaluated against current price.
        refreshed = {}
        for symbol, data in self.stocks_data.items():
            live = live_prices.get(symbol, 0)
            if live > 0:
                updated              = copy.copy(data)    # shallow copy — preserves all indicator fields
                updated.close        = live               # the main thing strategies use for price comparisons
                # Update high/low to reflect intraday movement so indicators
                # (candle patterns, consolidation range) use coherent OHLC.
                updated.high         = max(data.high, live)    # if price moved above morning's high, record it
                updated.low          = min(data.low, live)     # if price dipped below morning's low, record it
                # Recompute daily_bullish using the live price vs the morning's EMA values.
                # EMA values themselves don't change intraday (they're daily indicators on 250 bars).
                updated.daily_bullish = (live > data.ema_20 and data.ema_20 > data.ema_50)
                refreshed[symbol]    = updated
            else:
                refreshed[symbol] = data  # couldn't fetch live price — use morning data unchanged

        # Temporarily swap stocks_data with the live-price-refreshed version,
        # run the full screen → strategy → execute pipeline, then restore original.
        original          = self.stocks_data
        self.stocks_data  = refreshed
        screened          = self.step3_screen_stocks()
        setups            = self.step4_find_setups(screened)
        self.step5_execute_trades(setups)
        self.stocks_data  = original   # restore so the monitoring cycle still has the morning snapshot
        log.info("MIDDAY SCAN complete")

    def _get_live_price(self, symbol: str) -> float:
        """Fetches current live price for one symbol from Dhan market feed.
        Returns 0.0 if Dhan is not connected or the API call fails.
        Callers check > 0 before using the result."""
        if not self.order_mgr.dhan:
            return 0.0
        try:
            quote = self.order_mgr.dhan.get_market_feed_quote(
                security_id=self.order_mgr._get_security_id(symbol),
                exchange_segment="NSE_EQ"
            )
            if quote and "data" in quote:
                return float(quote["data"].get("ltp", 0))  # ltp = Last Traded Price
        except Exception as e:
            log.warning(f"Live price fetch failed for {symbol}: {e}")
        return 0.0

    def _get_all_live_prices(self) -> dict:
        """Fetches current live prices for all 15 watchlist stocks.
        Returns a dict of {symbol: price}. Symbols where fetch failed are absent from the dict."""
        prices = {}
        for symbol in Watchlist.get_symbols():
            p = self._get_live_price(symbol)
            if p > 0:
                prices[symbol] = p
        return prices

    # =========================================================================
    # STEP 6: Intraday monitoring (every 15 mins from 9:30 AM to 3:30 PM)
    # Fetches current prices from Dhan (real prices in both paper and live mode),
    # then passes them to TradeMonitor for exit checks and trailing SL updates.
    # =========================================================================

    def step6_monitor_trades(self):
        # Get current live prices for all open positions.
        # Falls back to last known price if Dhan is unavailable.
        current_prices = self._get_current_prices()

        # TradeMonitor does: broker sync → exit checks → trailing SL updates.
        # vix is passed so VIX-spike exits can fire even if we didn't run step2 today.
        self.monitor.monitor_all_trades(current_prices, self.vix)

    def _get_current_prices(self) -> dict:
        """
        Always fetch real live prices from Dhan — both paper and live mode.
        Paper mode only skips order placement, not data fetching.
        Falls back to last known price if Dhan is unavailable.
        """
        prices      = {}
        open_trades = self.db.get_open_trades()
        if not open_trades:
            return prices  # nothing open → nothing to fetch

        if self.order_mgr.dhan:
            try:
                for t in open_trades:
                    sym   = t["symbol"]
                    quote = self.order_mgr.dhan.get_market_feed_quote(
                        security_id=self.order_mgr._get_security_id(sym),
                        exchange_segment="NSE_EQ"
                    )
                    if quote and "data" in quote:
                        # Use live LTP if available, fall back to last stored current_price
                        prices[sym] = float(quote["data"].get("ltp", t["current_price"]))
                    else:
                        # Quote response malformed — use last known price from DB
                        prices[sym] = t["current_price"] or t["entry_price"]
            except Exception as e:
                log.error(f"Price fetch failed: {e}")
                # On exception, use last known prices so monitoring cycle still runs
                for t in open_trades:
                    prices[t["symbol"]] = t["current_price"] or t["entry_price"]
        else:
            # Dhan not connected — use last known price (no random simulation)
            for t in open_trades:
                prices[t["symbol"]] = t["current_price"] or t["entry_price"]
            log.warning("Dhan not connected — using last known prices, monitoring may be stale")

        return prices

    # =========================================================================
    # STEP 7: End of day (15:35 — after market close at 15:30)
    # Prints the full analytics dashboard, checks for 3 consecutive losses
    # (soft alert to review the system), and warns if advance tax is due.
    # =========================================================================

    def step7_end_of_day(self):
        log.info("STEP 7: End of day tasks...")

        # Print the full dashboard: today's closed trades, open positions, monthly stats,
        # strategy breakdown, and tax estimate for the current financial year.
        self.analytics.print_dashboard(self.available_capital)

        # Soft warning — if last 3 closed trades were all losses, log a review reminder.
        # This is NOT a hard block; the system continues trading the next day.
        if self.protection.check_consecutive_losses():
            log.warning("3 CONSECUTIVE LOSSES — Review your system.")

        # Compute STCG tax for the current financial year (April 1 → March 31).
        # Alerts if annual tax liability exceeds ₹10,000 (advance tax payment required).
        tax = self.analytics.tax_summary()
        if tax["advance_tax_required"]:
            log.warning(f"ADVANCE TAX DUE: ₹{tax['total_tax']:,.2f}")

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
            getattr(schedule.every(), day).at("11:30").do(self.run_midday_scan)   # midday check 1
            getattr(schedule.every(), day).at("13:30").do(self.run_midday_scan)   # midday check 2
            getattr(schedule.every(), day).at("15:35").do(self.step7_end_of_day)  # 5 min after market close

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
        Guards the monitoring cycle: only runs step6 if it's a weekday
        and within market hours (9:15 + 15 min wait → 9:30 AM to 3:30 PM).
        The 15-min wait after open avoids the chaotic opening auction period.
        """
        now = datetime.now()
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday — skip weekends
            return
        if now.replace(hour=9, minute=15) <= now <= now.replace(hour=15, minute=30):
            self.step6_monitor_trades()

    def _weekly_review(self):
        """
        Runs Sunday evening. Prints a brief weekly recap and
        cleans up DB rows older than 90 days to keep the file size manageable.
        """
        m  = self.analytics.monthly_summary()   # reuse monthly stats for the weekly print
        tx = self.analytics.tax_summary()
        print("\n" + "=" * 60)
        print("  WEEKLY REVIEW")
        print("=" * 60)
        if "total_trades" in m:
            print(f"  Trades   : {m['total_trades']}  WR: {m['win_rate']}%  Net: ₹{m['net_pnl']:,.2f}")
        print(f"  STCG Tax : ₹{tx['total_tax']:,.2f}")
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
        tx = self.analytics.tax_summary()
        print(f"Tax this FY — STCG: ₹{tx['annual_stcg']:,.2f} | Tax: ₹{tx['total_tax']:,.2f}")
