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
        if self.db.get_state("trading_halted") == "1":
            return False, "Trading manually halted"

        daily = self.db.get_today_realised_pnl()
        if daily <= -Config.DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit hit: ₹{abs(daily):.0f}"

        weekly = self.db.get_week_realised_pnl()
        if weekly <= -Config.WEEKLY_LOSS_LIMIT:
            return False, f"Weekly loss limit hit: ₹{abs(weekly):.0f}"

        monthly = self.db.get_month_realised_pnl()
        if monthly <= -Config.MONTHLY_LOSS_LIMIT:
            return False, f"Monthly loss limit hit: ₹{abs(monthly):.0f}"

        if monthly <= -Config.MAX_DRAWDOWN:
            return False, f"Max drawdown hit: ₹{abs(monthly):.0f}"

        cooldown_until = self.db.get_state("cooldown_until")
        if cooldown_until:
            try:
                cd = datetime.fromisoformat(cooldown_until)
                if datetime.now() < cd:
                    mins = int((cd - datetime.now()).total_seconds() / 60)
                    return False, f"Cooldown active: {mins} mins remaining"
            except Exception:
                pass

        now = datetime.now()
        if now.weekday() == 0:
            market_open = now.replace(hour=9, minute=15, second=0)
            if now < market_open + timedelta(minutes=Config.MONDAY_NO_ENTRY_MINS):
                return False, "Monday 30-min wait period"

        if now.weekday() == 4 and now.hour >= Config.FRIDAY_NO_ENTRY_HOUR:
            return False, "No new entries after 2 PM Friday"

        cutoff = now.replace(hour=Config.MAX_ENTRY_HOUR,
                             minute=Config.MAX_ENTRY_MINUTE, second=0, microsecond=0)
        if now >= cutoff:
            return False, f"No new entries after {Config.MAX_ENTRY_HOUR}:{Config.MAX_ENTRY_MINUTE:02d} PM"

        market_open = now.replace(hour=9, minute=15 + Config.MARKET_OPEN_WAIT_MINS, second=0)
        if now < market_open:
            return False, "Waiting for market to settle"

        return True, "OK"

    def start_loss_cooldown(self):
        """
        Starts a 2-hour trading pause after a loss. Persisted to DB so a restart
        during the cooldown period still enforces the remaining wait time.
        Also called by sync_with_broker() when broker-executed SL exits are detected.
        """
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
        rows = self.db.fetchall(
            "SELECT net_pnl FROM trades WHERE status='CLOSED' ORDER BY exit_date DESC LIMIT 5"
        )
        if len(rows) < 3:
            return False
        last_three = [r["net_pnl"] for r in rows[:3]]
        return all(p < 0 for p in last_three)

    def check_event_guard(self, symbol: str) -> tuple:
        """
        Checks for upcoming earnings/results for this symbol.
        Returns (safe: bool, message: str).
        - days <= EVENT_EXIT_DAYS (5): (False, "Exit! ...") — trade should be closed
        - days <= 10: (True, "Warning: ...") — hold but monitor closely
        - far away: (True, "Event far away, safe to hold")
        Called every 15 mins by TradeMonitor for each open position.
        """
        event = self.db.fetchone(
            "SELECT * FROM events_calendar WHERE symbol=? ORDER BY days_away ASC LIMIT 1",
            (symbol,)
        )
        if not event:
            return True, "No upcoming events"
        days = event["days_away"]
        if days <= Config.EVENT_EXIT_DAYS:
            return False, f"Exit! {event['event_type']} in {days} days"
        if days <= 10:
            return True, f"Warning: {event['event_type']} in {days} days"
        return True, "Event far away, safe to hold"
