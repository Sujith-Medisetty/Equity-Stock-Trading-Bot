"""
protection.py — All trading restrictions and circuit breakers.

This is the safety layer that prevents emotional or runaway trading.
Every protection check runs against the database, so protections survive restarts.

Protections implemented:

1. Loss circuit breakers (cascading limits):
   - Daily loss ≥ ₹3,000   → stop for today
   - Weekly loss ≥ ₹6,000  → stop for the week
   - Monthly loss ≥ ₹10,000 → stop for the month
   - Monthly loss ≥ ₹20,000 → full stop (Max drawdown — needs review before resuming)

   Why cascading? A daily limit reset at midnight. A weekly limit resets Monday.
   But if we lost ₹6k this week, continuing next week isn't obviously safe.
   The monthly and drawdown limits catch what the shorter windows miss.

2. Loss cooldown (2 hours after any SL hit):
   Prevents revenge trading — the urge to immediately recover a loss with
   another trade. The 2-hour wait forces a mental reset.
   Stored as "cooldown_until" ISO timestamp in protection_state so it survives
   if the process restarts during the cooldown period.

3. Manual halt:
   Operator can set protection_state "trading_halted" = "1" to stop trading
   without touching the code. Useful for news-driven market events.

4. Timing rules:
   - Market open wait (9:15 + 15 min): first 15 minutes after open are chaotic —
     opening gaps, order book imbalances, large institutions filling overnight orders.
     We wait for price discovery to settle.
   - Monday wait (30 min): weekend news creates larger opening gaps on Monday.
   - Friday 2 PM cutoff: no new entries after 2 PM Friday — we don't want to
     hold through the weekend when we can't monitor or react.

5. Event guard:
   Checks the events_calendar for upcoming earnings/results.
   Returns a warning at 10 days, forces exit at 5 days.
   Called by TradeMonitor to exit open positions before risk events.

6. Consecutive loss warning:
   If the last 3 closed trades are all losses, logs a warning.
   Not a hard block — it's a flag to review the system logic and market conditions.
"""

from datetime import datetime, timedelta

from config import Config, log
from database import DatabaseManager


class ProtectionEngine:
    """
    All trading restrictions in one place. Called by:
    - TradingSystem.step5_execute_trades() → is_trading_allowed() gate
    - TradeMonitor._check_exit_conditions() → check_event_guard() for open positions
    - TradeMonitor._execute_exit() → start_loss_cooldown() after SL hits
    - TradingSystem.step7_end_of_day() → check_consecutive_losses() for review flag
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def is_trading_allowed(self) -> tuple:
        """
        Master gate — called before placing any new trade.
        Checks all protections in priority order: manual halt → loss limits
        → cooldown → timing rules.
        Returns (True, "OK") only when every check passes.
        Returns (False, reason_string) on the first failure found.
        """

        # --- Check 1: Manual halt ---
        # An operator can set trading_halted="1" in the DB (e.g. via a script or directly)
        # to stop all trading immediately — e.g. during a breaking news event or system issue.
        if self.db.get_state("trading_halted") == "1":
            return False, "Trading manually halted"

        # --- Check 2: Daily loss limit ---
        # Reads sum of net_pnl for all trades closed today from the DB.
        # If we've lost ₹3,000+ today, stop trading for the rest of the day.
        # Resets at midnight (new day = new daily_pnl calculation).
        daily = self.db.get_today_realised_pnl()
        if daily <= -Config.DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit hit: ₹{abs(daily):.0f}"

        # --- Check 3: Weekly loss limit ---
        # Reads sum of net_pnl for all trades closed since Monday of this week.
        # If we've lost ₹6,000+ this week, stop for the rest of the week.
        weekly = self.db.get_week_realised_pnl()
        if weekly <= -Config.WEEKLY_LOSS_LIMIT:
            return False, f"Weekly loss limit hit: ₹{abs(weekly):.0f}"

        # --- Check 4: Monthly loss limit ---
        # If we've lost ₹10,000+ this calendar month, stop for the month.
        monthly = self.db.get_month_realised_pnl()
        if monthly <= -Config.MONTHLY_LOSS_LIMIT:
            return False, f"Monthly loss limit hit: ₹{abs(monthly):.0f}"

        # --- Check 5: Maximum drawdown ---
        # If monthly loss hits ₹20,000, this is a hard stop — the system needs
        # a manual review before trading can resume (operator must clear trading_halted flag).
        # Note: using monthly PnL as proxy for drawdown — close enough for ₹2L account.
        if monthly <= -Config.MAX_DRAWDOWN:
            return False, f"Max drawdown hit: ₹{abs(monthly):.0f}"

        # --- Check 6: Loss cooldown ---
        # After any SL hit, a 2-hour cooldown is started (stored in DB as ISO timestamp).
        # During this window, no new trades are allowed — prevents emotional revenge trading.
        cooldown_until = self.db.get_state("cooldown_until")
        if cooldown_until:
            try:
                cd = datetime.fromisoformat(cooldown_until)   # parse the stored ISO timestamp
                if datetime.now() < cd:
                    # Cooldown still active — calculate remaining minutes for the log message
                    mins = int((cd - datetime.now()).total_seconds() / 60)
                    return False, f"Cooldown active: {mins} mins remaining"
            except Exception:
                pass  # malformed timestamp in DB — ignore and continue

        now = datetime.now()

        # --- Check 7: Monday 30-minute wait ---
        # Monday mornings have larger gaps due to weekend news.
        # Wait 30 minutes after 9:15 AM open for gap resolution.
        if now.weekday() == 0:   # 0 = Monday
            market_open = now.replace(hour=9, minute=15, second=0)
            if now < market_open + timedelta(minutes=Config.MONDAY_NO_ENTRY_MINS):
                return False, "Monday 30-min wait period"

        # --- Check 8: Friday 2 PM cutoff ---
        # No new entries after 2 PM on Friday.
        # Reason: new positions would be held through the weekend (Sat+Sun) when
        # we can't monitor or exit. Weekend news could gap the stock against us.
        if now.weekday() == 4 and now.hour >= Config.FRIDAY_NO_ENTRY_HOUR:  # 4 = Friday
            return False, "No new entries after 2 PM Friday"

        # --- Check 9: Daily entry cutoff (1:30 PM) ---
        # No new entries at or after 1:30 PM on any day.
        # Reason: any position taken this late must be held overnight (market closes at 3:30 PM
        # and we need time to place and confirm orders). Holding overnight without monitoring
        # opportunity is too risky for swing-sized positions.
        cutoff = now.replace(hour=Config.MAX_ENTRY_HOUR,
                             minute=Config.MAX_ENTRY_MINUTE, second=0, microsecond=0)
        if now >= cutoff:
            return False, f"No new entries after {Config.MAX_ENTRY_HOUR}:{Config.MAX_ENTRY_MINUTE:02d} PM"

        # --- Check 10: Market open wait (15 minutes after 9:15 AM) ---
        # The first 15 minutes after open are chaotic: opening auction fills, gap resolution,
        # institutions filling large overnight orders. Prices are erratic and setups unreliable.
        # We wait for the order book to stabilise before entering.
        market_open = now.replace(hour=9, minute=15 + Config.MARKET_OPEN_WAIT_MINS, second=0)
        if now < market_open:
            return False, "Waiting for market to settle"

        return True, "OK"  # all checks passed — trading is allowed

    def start_loss_cooldown(self):
        """
        Starts a 2-hour trading pause after a loss. Persisted to DB so a restart
        during the cooldown period still enforces the remaining wait time.
        Also called by sync_with_broker() when broker-executed SL exits are detected.
        """
        # Store the "don't trade until this time" timestamp in the DB.
        # is_trading_allowed() reads this every time before placing a trade.
        until = (datetime.now() + timedelta(hours=Config.COOLDOWN_AFTER_LOSS_HR)).isoformat()
        self.db.set_state("cooldown_until", until)
        log.warning(f"Loss cooldown started — no trades for {Config.COOLDOWN_AFTER_LOSS_HR}h")

    def check_consecutive_losses(self) -> bool:
        """
        Returns True if the last 3 closed trades are all losses (net_pnl < 0).
        This is a soft warning, not a hard block. When True, the EOD step logs
        a review reminder — it means either the system parameters need adjustment
        or market conditions have changed.
        """
        # Fetch the 5 most recently closed trades (only 3 needed, but 5 gives context)
        rows = self.db.fetchall(
            "SELECT net_pnl FROM trades WHERE status='CLOSED' ORDER BY exit_date DESC LIMIT 5"
        )
        if len(rows) < 3:
            return False  # not enough history to determine a streak

        # Check only the 3 most recent — all must be negative for the warning to trigger
        last_three = [r["net_pnl"] for r in rows[:3]]
        return all(p < 0 for p in last_three)

    def check_event_guard(self, symbol: str) -> tuple:
        """
        Checks for upcoming earnings/results for this symbol.
        Returns (action: str, message: str) where action is one of:
          "SAFE"    → no event or event far away — hold normally
          "WARN"    → event within 10 days — log, no action required yet
          "TIGHTEN" → event within 5 days — tighten SL to breakeven (or exit if already at a loss)
          "EXIT"    → event within 2 days — force exit unconditionally, too close to hold through

        Why staged instead of immediate exit at 5 days?
        At 5 days we still have time — dumping the position immediately at whatever the
        current price is could lock in a worse loss than waiting for price to recover
        or at least tightening the SL to limit downside. Only at 2 days is the gap risk
        (earnings can move ±10% overnight) too large to justify holding any longer.
        """
        # Find the nearest upcoming event for this symbol (by days_away ascending)
        event = self.db.fetchone(
            "SELECT * FROM events_calendar WHERE symbol=? ORDER BY days_away ASC LIMIT 1",
            (symbol,)
        )
        if not event:
            return "SAFE", "No upcoming events"

        days = event["days_away"]

        if days <= Config.EVENT_FORCE_EXIT_DAYS:
            # 2 days or less — gap risk is now too large. Force exit immediately regardless of P&L.
            # Earnings can open ±10% against us with no time to react.
            return "EXIT", f"Force exit — {event['event_type']} in {days} day(s)"

        if days <= Config.EVENT_EXIT_DAYS:
            # 3–5 days away — tighten SL to breakeven (entry price).
            # If trade is already at a loss, monitor.py will exit it.
            # If trade is in profit, worst case becomes zero loss from here.
            return "TIGHTEN", f"{event['event_type']} in {days} days — tightening SL"

        if days <= Config.EVENT_WARN_DAYS:
            # 6–10 days away — log a warning but no mechanical action yet.
            # Monitor for weakness; the tighten/exit tiers will fire as we get closer.
            return "WARN", f"Warning: {event['event_type']} in {days} days"

        # Event is far enough away to not be a concern for a swing trade
        return "SAFE", "Event far away, safe to hold"
