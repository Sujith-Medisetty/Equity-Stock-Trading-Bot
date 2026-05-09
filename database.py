"""
database.py — SQLite persistence layer for the trading system.

This is NOT the source of truth for what positions you hold. That's Dhan.
The database stores:
  - Historical records so we can analyse performance, compute tax, and audit decisions
  - Live trade metadata (entry price, SL level, tier progress, broker SL order ID)
    needed because Dhan's API doesn't store our custom fields like trailing SL tiers
  - Protection state (cooldown_until, trading_halted) that must survive restarts
  - Market/FII snapshots for backtesting and trend analysis

Every morning on startup, TradeMonitor.sync_with_broker() reconciles the trades
table against live Dhan holdings — any trade that the broker already exited gets
marked CLOSED in the DB so the two are in sync before the day begins.

Tables:
  market_snapshots    → daily: VIX, Nifty levels, FII flow, market mode
  stock_snapshots     → daily per-stock: all technical indicators calculated by IndicatorEngine
  setups              → every setup evaluated each day (PENDING / TAKEN / SKIPPED)
  trades              → live and historical trades with tier tracking and broker SL order ID
  trailing_sl_log     → audit trail of every SL move (for post-trade analysis)
  daily_pnl           → aggregated daily PNL summary
  fii_history         → FII/DII net buying per day + streak counters
  events_calendar     → upcoming earnings/results for watchlist stocks
  protection_state    → key-value store for runtime flags (cooldown, halt)
"""

import sqlite3
from datetime import datetime, timedelta

from config import Config, log
from models import Trade, Setup, StockData


class DatabaseManager:
    """
    Thin wrapper around SQLite providing typed read/write methods for every table.

    Why SQLite?
    - Zero setup: no server, no credentials, single file (trading.db)
    - Sufficient for 15 stocks × 250 trading days — not a scalability concern
    - Easy to inspect with any DB browser tool for debugging

    Thread safety: check_same_thread=False because the scheduler runs the monitor
    in a separate thread from the main setup cycle. SQLite handles concurrent reads
    fine; writes are brief and infrequent so contention is not a problem.
    """

    def __init__(self, db_path: str = Config.DB_PATH):
        """
        Opens (or creates) the SQLite file and runs CREATE TABLE IF NOT EXISTS
        for every table. Safe to call on every startup — existing data is never touched.
        row_factory = sqlite3.Row lets callers access columns by name (row["symbol"])
        instead of by position, which is much safer when columns are added later.
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        log.info(f"Database ready: {db_path}")

    def _create_tables(self):
        """
        Creates all 9 tables on first run. Subsequent runs hit the IF NOT EXISTS guard.
        Also runs a safe migration for sl_order_id (ALTER TABLE with try/except so the
        column is added once and ignored on every run after that).
        """
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                date TEXT PRIMARY KEY,
                nifty_close REAL, nifty_ema20 REAL, nifty_ema50 REAL,
                nifty_ema200 REAL, nifty_rsi REAL,
                india_vix REAL, gift_nifty REAL,
                fii_net_cash REAL, dii_net_cash REAL,
                fii_consecutive_days INTEGER,
                market_mode TEXT, fii_flow_label TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS stock_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, symbol TEXT,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                ema_20 REAL, ema_50 REAL, ema_200 REAL,
                rsi REAL, macd REAL, macd_signal REAL, macd_hist REAL,
                atr REAL, bb_upper REAL, bb_lower REAL,
                volume_ratio REAL, week_52_high REAL, week_52_low REAL,
                rs_score REAL, candle_pattern TEXT,
                weekly_bullish INTEGER, daily_bullish INTEGER, h4_bullish INTEGER,
                tf_aligned_count INTEGER,
                consolidation_range_pct REAL, obv_rising INTEGER,
                atr_ratio REAL,
                UNIQUE(date, symbol)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS setups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, symbol TEXT, strategy TEXT,
                score INTEGER, entry_price REAL, sl_price REAL,
                target_price REAL, atr REAL,
                risk_per_share REAL, shares INTEGER,
                capital_required REAL, actual_risk REAL,
                rr_ratio REAL, market_mode TEXT, fii_flow TEXT,
                status TEXT DEFAULT 'PENDING',
                skip_reason TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT, strategy TEXT,
                entry_date TEXT, entry_price REAL, quantity INTEGER,
                initial_sl REAL, initial_target REAL, current_sl REAL,
                current_price REAL,
                tier1_done INTEGER DEFAULT 0, tier1_price REAL, tier1_qty INTEGER,
                tier2_done INTEGER DEFAULT 0, tier2_price REAL, tier2_qty INTEGER,
                remaining_qty INTEGER,
                stt REAL DEFAULT 0, dp_charge REAL DEFAULT 0,
                exchange_charge REAL DEFAULT 0, stamp_duty REAL DEFAULT 0,
                gst REAL DEFAULT 0, sebi REAL DEFAULT 0,
                total_charges REAL DEFAULT 0,
                gross_pnl REAL DEFAULT 0, net_pnl REAL DEFAULT 0,
                setup_score INTEGER DEFAULT 0,
                market_mode_at_entry TEXT,
                status TEXT DEFAULT 'OPEN',
                exit_reason TEXT DEFAULT '',
                exit_date TEXT DEFAULT '',
                holding_days INTEGER DEFAULT 0,
                sl_order_id TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            c.execute("ALTER TABLE trades ADD COLUMN sl_order_id TEXT DEFAULT ''")
        except Exception:
            pass

        c.execute("""
            CREATE TABLE IF NOT EXISTS trailing_sl_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT, timestamp TEXT,
                old_sl REAL, new_sl REAL,
                current_price REAL, reason TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT PRIMARY KEY,
                realised_pnl REAL DEFAULT 0,
                unrealised_pnl REAL DEFAULT 0,
                total_charges REAL DEFAULT 0,
                trades_opened INTEGER DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                running_stcg REAL DEFAULT 0,
                stcg_tax_estimate REAL DEFAULT 0
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS fii_history (
                date TEXT PRIMARY KEY,
                fii_net_cash REAL, dii_net_cash REAL,
                consecutive_buying_days INTEGER DEFAULT 0,
                consecutive_selling_days INTEGER DEFAULT 0,
                sector_flow TEXT DEFAULT '{}'
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS events_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, event_type TEXT,
                event_date TEXT, days_away INTEGER,
                risk_level TEXT, action_required TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS protection_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()

    # ---- Generic helpers ----
    # These three methods cover every DB operation in the system.
    # Callers write plain SQL — no ORM abstraction, deliberately.
    # Reading raw SQL is faster to debug than figuring out what an ORM generated.

    def execute(self, sql, params=()):
        """Run an INSERT / UPDATE / DELETE and commit immediately."""
        c = self.conn.cursor()
        c.execute(sql, params)
        self.conn.commit()
        return c

    def fetchone(self, sql, params=()):
        """Run a SELECT and return the first row (or None). Used for lookups."""
        c = self.conn.cursor()
        c.execute(sql, params)
        return c.fetchone()

    def fetchall(self, sql, params=()):
        """Run a SELECT and return all matching rows as a list."""
        c = self.conn.cursor()
        c.execute(sql, params)
        return c.fetchall()

    # ---- Market snapshot ----
    # Called once per day (step2_detect_market_mode) to record the overall market
    # picture: Nifty levels, VIX, FII, and the resulting market mode.
    # INSERT OR REPLACE means re-running after a crash won't create duplicate rows.

    def save_market_snapshot(self, data: dict):
        self.execute("""
            INSERT OR REPLACE INTO market_snapshots
            (date, nifty_close, nifty_ema20, nifty_ema50, nifty_ema200,
             nifty_rsi, india_vix, gift_nifty, fii_net_cash, dii_net_cash,
             fii_consecutive_days, market_mode, fii_flow_label)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["date"], data["nifty_close"],
            data["nifty_ema20"], data["nifty_ema50"], data["nifty_ema200"],
            data["nifty_rsi"], data["india_vix"], data["gift_nifty"],
            data["fii_net_cash"], data["dii_net_cash"],
            data["fii_consecutive_days"],
            data["market_mode"], data["fii_flow_label"]
        ))

    # ---- Stock snapshot ----
    # Called once per stock per day (step1_collect_data) after indicators are calculated.
    # Stores the full technical picture so we can review why we entered or skipped a trade.
    # The UNIQUE(date, symbol) constraint + INSERT OR REPLACE prevents duplicate rows on reruns.

    def save_stock_snapshot(self, s: StockData):
        self.execute("""
            INSERT OR REPLACE INTO stock_snapshots
            (date, symbol, open, high, low, close, volume,
             ema_20, ema_50, ema_200, rsi, macd, macd_signal, macd_hist,
             atr, bb_upper, bb_lower, volume_ratio, week_52_high, week_52_low,
             rs_score, candle_pattern, weekly_bullish, daily_bullish, h4_bullish,
             tf_aligned_count, consolidation_range_pct, obv_rising, atr_ratio)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            s.date, s.symbol, s.open, s.high, s.low, s.close, s.volume,
            s.ema_20, s.ema_50, s.ema_200, s.rsi,
            s.macd, s.macd_signal, s.macd_hist,
            s.atr, s.bb_upper, s.bb_lower,
            s.volume_ratio, s.week_52_high, s.week_52_low,
            s.rs_score, s.candle_pattern,
            int(s.weekly_bullish), int(s.daily_bullish), int(s.h4_bullish),
            s.tf_aligned_count, s.consolidation_range_pct,
            int(s.obv_rising), s.atr_ratio
        ))

    # ---- Setups ----
    # A Setup is a candidate trade identified by StrategyEngine and sized by RiskManager.
    # Saved regardless of whether we actually enter the trade (status = PENDING/TAKEN/SKIPPED).
    # This lets us audit: "how many good setups did we miss because the portfolio was full?"

    def save_setup(self, s: Setup):
        self.execute("""
            INSERT INTO setups
            (date, symbol, strategy, score, entry_price, sl_price,
             target_price, atr, risk_per_share, shares, capital_required,
             actual_risk, rr_ratio, market_mode, fii_flow, status, skip_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            s.date, s.symbol, s.strategy.value, s.score,
            s.entry_price, s.sl_price, s.target_price,
            s.atr, s.risk_per_share, s.shares, s.capital_required,
            s.actual_risk, s.rr_ratio, s.market_mode, s.fii_flow,
            s.status, s.skip_reason
        ))

    def get_pending_setups(self, date: str):
        return self.fetchall(
            "SELECT * FROM setups WHERE date=? AND status='PENDING' ORDER BY score DESC",
            (date,)
        )

    # ---- Trades ----
    # Core trade record. Created at entry, updated throughout the trade lifecycle:
    #   - current_price updated every 15 mins by TradeMonitor
    #   - current_sl updated when trailing SL moves up (tiers 1/2/3)
    #   - tier1_done / tier2_done / remaining_qty updated as partial exits happen
    #   - sl_order_id updated when the broker SL order is cancelled+replaced
    #   - status changes from OPEN → CLOSED when the trade ends
    # sl_order_id is critical: it's needed to cancel the old SL order at the broker
    # when the trailing SL level changes. Without it we can't do cancel+replace.

    def save_trade(self, t: Trade):
        self.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, symbol, strategy, entry_date, entry_price, quantity,
             initial_sl, initial_target, current_sl, current_price,
             tier1_done, tier1_price, tier1_qty,
             tier2_done, tier2_price, tier2_qty, remaining_qty,
             stt, dp_charge, exchange_charge, stamp_duty, gst, sebi,
             total_charges, gross_pnl, net_pnl,
             setup_score, market_mode_at_entry,
             status, exit_reason, exit_date, holding_days, sl_order_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            t.trade_id, t.symbol, t.strategy,
            t.entry_date, t.entry_price, t.quantity,
            t.initial_sl, t.initial_target, t.current_sl, t.current_price,
            int(t.tier1_done), t.tier1_price, t.tier1_qty,
            int(t.tier2_done), t.tier2_price, t.tier2_qty, t.remaining_qty,
            t.stt, t.dp_charge, t.exchange_charge, t.stamp_duty,
            t.gst, t.sebi, t.total_charges,
            t.gross_pnl, t.net_pnl,
            t.setup_score, t.market_mode_at_entry,
            t.status, t.exit_reason, t.exit_date, t.holding_days, t.sl_order_id
        ))

    def get_open_trades(self):
        """All trades with status=OPEN. Note: sync_with_broker() should be called first
        to ensure closed broker positions are already marked CLOSED in the DB."""
        return self.fetchall("SELECT * FROM trades WHERE status='OPEN'")

    def get_trade(self, trade_id: str):
        return self.fetchone("SELECT * FROM trades WHERE trade_id=?", (trade_id,))

    def update_trade_sl(self, trade_id: str, new_sl: float):
        """Update only the SL level in the DB. Call replace_sl_order() separately
        to also update the actual SL order sitting at the broker exchange."""
        self.execute("UPDATE trades SET current_sl=? WHERE trade_id=?", (new_sl, trade_id))

    def update_sl_order_id(self, trade_id: str, sl_order_id: str):
        """Store the new Dhan order ID after a cancel+replace. Without this stored,
        the next SL update won't know which order to cancel."""
        self.execute("UPDATE trades SET sl_order_id=? WHERE trade_id=?", (sl_order_id, trade_id))

    def close_trade(self, trade_id: str, exit_price: float,
                    exit_reason: str, net_pnl: float,
                    gross_pnl: float, total_charges: float):
        today = datetime.now().strftime("%Y-%m-%d")
        entry = self.get_trade(trade_id)
        holding = 0
        if entry:
            try:
                holding = (datetime.now() - datetime.strptime(entry["entry_date"], "%Y-%m-%d")).days
            except Exception:
                pass
        self.execute("""
            UPDATE trades SET status='CLOSED', exit_reason=?,
            exit_date=?, current_price=?, gross_pnl=?,
            net_pnl=?, total_charges=?, holding_days=?
            WHERE trade_id=?
        """, (exit_reason, today, exit_price, gross_pnl,
              net_pnl, total_charges, holding, trade_id))

    def log_trailing_sl(self, trade_id: str, old_sl: float,
                        new_sl: float, price: float, reason: str):
        """Append-only audit trail: every time the SL moves, record the old level,
        new level, current price, and which tier triggered it.
        Useful post-trade to see exactly how the trailing SL progressed."""
        self.execute("""
            INSERT INTO trailing_sl_log (trade_id, timestamp, old_sl, new_sl, current_price, reason)
            VALUES (?,?,?,?,?,?)
        """, (trade_id, datetime.now().isoformat(), old_sl, new_sl, price, reason))

    # ---- Protection state ----
    # Key-value store for runtime flags that must survive process restarts.
    # Currently used for:
    #   "cooldown_until"  → ISO timestamp; no trades until after this datetime
    #   "trading_halted"  → "1" means manual halt in effect (set by operator)
    # If we stored these only in memory, a crash or restart would clear them
    # and the protection would silently stop working.

    def set_state(self, key: str, value: str):
        self.execute(
            "INSERT OR REPLACE INTO protection_state (key, value, updated_at) VALUES (?,?,?)",
            (key, value, datetime.now().isoformat())
        )

    def get_state(self, key: str, default: str = "") -> str:
        row = self.fetchone("SELECT value FROM protection_state WHERE key=?", (key,))
        return row["value"] if row else default

    # ---- PNL queries ----
    # These four methods are called by ProtectionEngine to check loss limits,
    # and by PerformanceAnalytics for the dashboard and tax summary.
    # All look at net_pnl (after charges) on CLOSED trades only.
    # The FY start date (April 1) matches India's financial year for STCG computation.

    def get_today_realised_pnl(self) -> float:
        today = datetime.now().strftime("%Y-%m-%d")
        row = self.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE exit_date=? AND status='CLOSED'", (today,)
        )
        return row["total"] or 0.0

    def get_week_realised_pnl(self) -> float:
        week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")
        row = self.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE exit_date>=? AND status='CLOSED'", (week_start,)
        )
        return row["total"] or 0.0

    def get_month_realised_pnl(self) -> float:
        row = self.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE exit_date>=? AND status='CLOSED'",
            (datetime.now().strftime("%Y-%m-01"),)
        )
        return row["total"] or 0.0

    def get_annual_stcg(self) -> float:
        year = datetime.now().year if datetime.now().month >= 4 else datetime.now().year - 1
        fy_start = f"{year}-04-01"
        row = self.fetchone(
            "SELECT SUM(net_pnl) as total FROM trades WHERE exit_date>=? AND status='CLOSED' AND net_pnl>0",
            (fy_start,)
        )
        return row["total"] or 0.0

    # ---- Maintenance ----

    def cleanup_old_data(self, days: int = 90):
        """
        Prunes old rows from the 3 high-volume tables.
        Called weekly from system._weekly_review().

        What gets pruned vs kept:
          stock_snapshots  → keep last `days` days (indicators are recalculated fresh daily anyway)
          setups           → keep last `days` days (older skipped setups have no audit value)
          trailing_sl_log  → keep last `days` days (old closed-trade SL history not needed)

        What is NEVER pruned:
          trades           → financial records, needed for full tax history
          daily_pnl        → needed for drawdown calculation across the account lifetime
          fii_history      → needed for consecutive-day streak calculation
          market_snapshots → useful for long-term regime analysis
          protection_state → runtime state, tiny and always current

        After deletion, VACUUM reclaims the freed pages so the file actually shrinks.
        """
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        before = self.fetchone("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")

        self.execute("DELETE FROM stock_snapshots WHERE date < ?", (cutoff,))
        self.execute("DELETE FROM setups WHERE date < ?", (cutoff,))
        self.execute("DELETE FROM trailing_sl_log WHERE timestamp < ?", (cutoff + "T00:00:00",))

        self.conn.execute("VACUUM")
        self.conn.commit()

        after = self.fetchone("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        freed = ((before["size"] or 0) - (after["size"] or 0)) / 1024
        log.info(f"DB cleanup done (cutoff: {cutoff}) | freed: {freed:.1f} KB | tables pruned: stock_snapshots, setups, trailing_sl_log")
