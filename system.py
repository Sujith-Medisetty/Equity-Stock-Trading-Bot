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
        self.db            = DatabaseManager()
        self.collector     = DataCollector(self.db)
        self.market_mode_eng = MarketModeEngine(self.db)
        self.screener      = StockScreener(self.db)
        self.strategy_eng  = StrategyEngine()
        self.risk_mgr      = RiskManager(self.db)
        self.protection    = ProtectionEngine(self.db)
        self.order_mgr     = OrderManager(self.db)
        self.monitor       = TradeMonitor(self.db, self.order_mgr, self.protection)
        self.analytics     = PerformanceAnalytics(self.db)
        self.briefing      = MorningBriefing(self.db)

        self.market_mode   = MarketMode.CAUTIOUS
        self.fii_flow      = FIIFlow.NEUTRAL
        self.vix           = 15.0
        self.fii_net       = 0.0
        self.stocks_data   = {}
        self.nifty_data    = None
        self.todays_setups = []

        # On startup: immediately reconcile DB against actual Dhan holdings.
        # Catches any SL orders the broker executed since last run (overnight, weekend).
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

        self.vix     = self.collector.fetch_india_vix()
        fii_data     = self.collector.fetch_fii_dii()
        self.fii_net = fii_data.get("fii_net_cash", 0.0)

        buy_streak, sell_streak = self.collector.get_fii_consecutive_days()
        self.db.execute("""
            INSERT OR REPLACE INTO fii_history
            (date, fii_net_cash, dii_net_cash, consecutive_buying_days, consecutive_selling_days)
            VALUES (?,?,?,?,?)
        """, (datetime.now().strftime("%Y-%m-%d"),
              self.fii_net, fii_data.get("dii_net_cash", 0.0),
              buy_streak, sell_streak))

        nifty_df = self.collector.fetch_ohlcv_daily("NIFTY 50", days=250)
        if nifty_df is not None:
            self.nifty_data = IndicatorEngine.calculate_all(nifty_df, "NIFTY50")

        for sym in Watchlist.get_symbols():
            df_daily  = self.collector.fetch_ohlcv_daily(sym, days=250)
            df_4h     = self.collector.fetch_ohlcv_intraday(sym, "60", 30)
            df_weekly = self.collector.fetch_ohlcv_daily(sym, days=500)
            s_data    = IndicatorEngine.calculate_all(df_daily, sym, nifty_df)
            if s_data:
                s_data.h4_bullish     = IndicatorEngine.check_4h_bullish(df_4h)
                s_data.weekly_bullish = IndicatorEngine.check_weekly_bullish(df_weekly)
                s_data.tf_aligned_count = sum([s_data.weekly_bullish, s_data.daily_bullish, s_data.h4_bullish])
                self.stocks_data[sym] = s_data
                self.db.save_stock_snapshot(s_data)

        events = self.collector.fetch_events_calendar()
        self.collector.save_events(events)
        log.info(f"Data collected: {len(self.stocks_data)} stocks | VIX: {self.vix:.1f} | FII: ₹{self.fii_net:.0f}Cr")

    # =========================================================================
    # STEP 2: Market mode detection (9:00 AM)
    # Reads VIX, Nifty technicals, and FII streak from step 1 data.
    # Sets self.market_mode (the master switch for the whole day) and self.fii_flow.
    # Also saves the full market picture to market_snapshots for historical analysis.
    # =========================================================================

    def step2_detect_market_mode(self):
        log.info("STEP 2: Detecting market mode...")
        buy_s, sell_s = self.collector.get_fii_consecutive_days()
        self.market_mode, self.fii_flow = self.market_mode_eng.detect(
            vix=self.vix, nifty_data=self.nifty_data,
            fii_net=self.fii_net,
            fii_consecutive_buy=buy_s, fii_consecutive_sell=sell_s
        )
        nifty = self.nifty_data
        self.db.save_market_snapshot({
            "date":                datetime.now().strftime("%Y-%m-%d"),
            "nifty_close":         nifty.close   if nifty else 0,
            "nifty_ema20":         nifty.ema_20  if nifty else 0,
            "nifty_ema50":         nifty.ema_50  if nifty else 0,
            "nifty_ema200":        nifty.ema_200 if nifty else 0,
            "nifty_rsi":           nifty.rsi     if nifty else 50,
            "india_vix":           self.vix,
            "gift_nifty":          self.collector.fetch_gift_nifty(),
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
        open_trades = list(self.db.get_open_trades())
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
        fii_sectors   = self._get_fii_buying_sectors()

        for symbol in screened_symbols:
            data = self.stocks_data.get(symbol)
            if not data:
                continue
            fii_sector_buying = Watchlist.get_sector(symbol) in fii_sectors
            setup = self.strategy_eng.evaluate_all(
                symbol=symbol, data=data,
                market_mode=self.market_mode,
                fii_flow=self.fii_flow,
                fii_sector_buying=fii_sector_buying
            )
            if setup:
                setup = self.risk_mgr.calculate_setup_risk(setup, data)
                if setup.status != "SKIPPED":
                    self.db.save_setup(setup)
                    setups.append(setup)
                    log.info(f"SETUP: {symbol} | {setup.strategy.value} | Score:{setup.score} | "
                             f"Entry:₹{setup.entry_price:.2f} SL:₹{setup.sl_price:.2f} T:₹{setup.target_price:.2f}")

        setups.sort(key=lambda x: x.score, reverse=True)
        log.info(f"Total setups: {len(setups)}")
        return setups

    def _get_fii_buying_sectors(self) -> set:
        if self.fii_flow != FIIFlow.BUYING:
            return set()
        return {
            Watchlist.get_sector(sym)
            for sym, data in self.stocks_data.items()
            if data and data.rs_score > 3
        }

    # =========================================================================
    # STEP 5: Execute trades (9:30 AM)
    # For each approved setup (score >= 60, checklist passed):
    # place the entry order + SL order at the broker.
    # Stops adding new trades once MAX_SIMULTANEOUS_TRADES (4) is reached.
    # The full pre-trade checklist (10 conditions) runs here as the final gate.
    # =========================================================================

    def step5_execute_trades(self, setups: list):
        log.info("STEP 5: Evaluating trades...")
        allowed, reason = self.protection.is_trading_allowed()
        if not allowed:
            log.warning(f"Trading blocked: {reason}")
            return

        open_trades = list(self.db.get_open_trades())
        for setup in setups:
            if len(open_trades) >= Config.MAX_SIMULTANEOUS_TRADES:
                break
            if setup.score < 60:
                continue

            approved, failed = self.risk_mgr.run_pre_trade_checklist(
                setup, self.market_mode, self.vix, open_trades
            )
            if not approved:
                setup.status = "SKIPPED"
                setup.skip_reason = f"Failed: {failed}"
                continue

            trade_id = self.order_mgr.place_entry_order(setup)
            if trade_id:
                setup.status = "TAKEN"
                open_trades = list(self.db.get_open_trades())
            else:
                log.error(f"Order failed: {setup.symbol}")

        self.todays_setups = setups

    # =========================================================================
    # STEP 6: Intraday monitoring (every 15 mins from 9:30 AM to 3:30 PM)
    # Fetches current prices from Dhan (real prices in both paper and live mode),
    # then passes them to TradeMonitor for exit checks and trailing SL updates.
    # =========================================================================

    def step6_monitor_trades(self):
        current_prices = self._get_current_prices()
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
            return prices

        if self.order_mgr.dhan:
            try:
                for t in open_trades:
                    sym   = t["symbol"]
                    quote = self.order_mgr.dhan.get_market_feed_quote(
                        security_id=self.order_mgr._get_security_id(sym),
                        exchange_segment="NSE_EQ"
                    )
                    if quote and "data" in quote:
                        prices[sym] = float(quote["data"].get("ltp", t["current_price"]))
                    else:
                        prices[sym] = t["current_price"] or t["entry_price"]
            except Exception as e:
                log.error(f"Price fetch failed: {e}")
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
        self.analytics.print_dashboard()
        if self.protection.check_consecutive_losses():
            log.warning("3 CONSECUTIVE LOSSES — Review your system.")
        tax = self.analytics.tax_summary()
        if tax["advance_tax_required"]:
            log.warning(f"ADVANCE TAX DUE: ₹{tax['total_tax']:,.2f}")

    # =========================================================================
    # STEP 8: Morning briefing (9:15 AM — after steps 3-5 complete)
    # Prints a human-readable summary of market conditions and today's setups.
    # Includes any RED-risk events (earnings within 2 days) as action items.
    # =========================================================================

    def step8_morning_briefing(self):
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
        self.step1_collect_data()
        self.step2_detect_market_mode()
        screened = self.step3_screen_stocks()
        setups   = self.step4_find_setups(screened)
        self.step5_execute_trades(setups)
        self.step8_morning_briefing()
        log.info("Daily setup complete. Monitoring starts at 9:30 AM.")

    # =========================================================================
    # SCHEDULER
    # =========================================================================

    def start_scheduler(self):
        if not SCHEDULE_AVAILABLE:
            log.error("schedule package not installed. pip install schedule")
            return

        log.info(f"Starting scheduler | Paper Trade: {Config.PAPER_TRADE}")

        for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            getattr(schedule.every(), day).at("08:45").do(self.step1_collect_data)
            getattr(schedule.every(), day).at("09:00").do(self.step2_detect_market_mode)
            getattr(schedule.every(), day).at("09:10").do(
                lambda: self.step5_execute_trades(
                    self.step4_find_setups(self.step3_screen_stocks())
                )
            )
            getattr(schedule.every(), day).at("09:15").do(self.step8_morning_briefing)
            getattr(schedule.every(), day).at("15:35").do(self.step7_end_of_day)

        schedule.every(15).minutes.do(self._conditional_monitor)
        schedule.every().sunday.at("20:00").do(self._weekly_review)

        print("\nScheduler running. Press Ctrl+C to stop.\n")
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            log.info("System stopped.")

    def _conditional_monitor(self):
        now = datetime.now()
        if now.weekday() >= 5:
            return
        if now.replace(hour=9, minute=15) <= now <= now.replace(hour=15, minute=30):
            self.step6_monitor_trades()

    def _weekly_review(self):
        m  = self.analytics.monthly_summary()
        tx = self.analytics.tax_summary()
        print("\n" + "=" * 60)
        print("  WEEKLY REVIEW")
        print("=" * 60)
        if "total_trades" in m:
            print(f"  Trades   : {m['total_trades']}  WR: {m['win_rate']}%  Net: ₹{m['net_pnl']:,.2f}")
        print(f"  STCG Tax : ₹{tx['total_tax']:,.2f}")
        print("=" * 60 + "\n")

    def run_once_test(self):
        print("\n" + "=" * 60)
        print("RUNNING SINGLE TEST CYCLE")
        print("=" * 60 + "\n")
        self.run_daily()
        self.analytics.print_dashboard()
        tx = self.analytics.tax_summary()
        print(f"Tax this FY — STCG: ₹{tx['annual_stcg']:,.2f} | Tax: ₹{tx['total_tax']:,.2f}")
