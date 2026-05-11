"""
analytics.py — Performance reporting, tax calculation, and morning briefing.

Two classes:

1. PerformanceAnalytics
   Reads closed trade records from the DB and computes statistics.
   Called at end of day (step7_end_of_day) and on demand via --dashboard.

   Key metrics tracked:
   - Win rate: % of trades that closed with positive net PNL
   - Avg win / Avg loss: absolute rupee amounts
   - Realised R:R: the actual avg_win / avg_loss ratio (target: > 2.0)
   - Strategy breakdown: which of the 5 strategies is performing best
   - Tax: STCG estimate for the current financial year (April 1 onwards)

   Why track realised R:R separately from the minimum 2.0 we set at entry?
   Because partial fills, slippage, and early exits can reduce the actual R:R.
   Monitoring realised R:R tells us if the system is executing as designed.

2. MorningBriefing
   Prints a human-readable summary before the trading day begins (9:15 AM).
   Shows: market mode, VIX, FII flow, any RED events today, and the top setups
   that the system identified and is about to (or already did) enter.
   This is the operator's daily check-in — confirms the system is operating
   with the expected market context before real positions are taken.
"""

from datetime import datetime

from config import Config, log
from models import MarketMode, FIIFlow
from database import DatabaseManager


class PerformanceAnalytics:
    """
    Reads the trades table and computes daily, monthly, and annual statistics.
    All PNL figures are net (after broker charges). Tax is on profitable trades only.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def daily_summary(self) -> dict:
        """
        Today's closed trade stats + current open position unrealised PNL.
        Unrealised = sum of (current_price - entry_price) × remaining_qty for all OPEN trades.
        current_price is updated every 15 mins by TradeMonitor so this figure is near-live.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        rows  = self.db.fetchall(
            "SELECT * FROM trades WHERE exit_date=? AND status='CLOSED'", (today,)
        )
        wins   = [r for r in rows if r["net_pnl"] > 0]
        losses = [r for r in rows if r["net_pnl"] <= 0]
        open_trades = self.db.get_open_trades()
        unrealised  = sum(
            (t["current_price"] - t["entry_price"]) * t["remaining_qty"]
            for t in open_trades
        )
        return {
            "date":           today,
            "trades_closed":  len(rows),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       round(len(wins) / len(rows) * 100, 1) if rows else 0,
            "realised_pnl":   round(sum(r["net_pnl"] for r in rows), 2),
            "unrealised_pnl": round(unrealised, 2),
            "total_charges":  round(sum(r["total_charges"] for r in rows), 2),
            "open_positions": len(open_trades),
        }

    def monthly_summary(self) -> dict:
        """
        Month-to-date stats from the 1st of the current month.
        strategy_breakdown shows which strategies are contributing most to PNL —
        useful for deciding if any strategy should be temporarily disabled.
        rr_realised is the actual R:R delivered, not the theoretical minimum.
        If this drops below 1.5, SL management or exit timing needs review.
        """
        rows = self.db.fetchall(
            "SELECT * FROM trades WHERE exit_date>=? AND status='CLOSED'",
            (datetime.now().strftime("%Y-%m-01"),)
        )
        if not rows:
            return {"message": "No closed trades this month"}

        wins   = [r for r in rows if r["net_pnl"] > 0]
        losses = [r for r in rows if r["net_pnl"] <= 0]
        net_pnl  = sum(r["net_pnl"] for r in rows)
        avg_win  = sum(r["net_pnl"] for r in wins)  / len(wins)  if wins  else 0
        avg_loss = sum(r["net_pnl"] for r in losses) / len(losses) if losses else 0

        by_strategy: dict = {}
        for r in rows:
            s = r["strategy"]
            if s not in by_strategy:
                by_strategy[s] = {"trades": 0, "wins": 0, "pnl": 0}
            by_strategy[s]["trades"] += 1
            by_strategy[s]["wins"]   += 1 if r["net_pnl"] > 0 else 0
            by_strategy[s]["pnl"]    += r["net_pnl"]

        return {
            "month":              datetime.now().strftime("%B %Y"),
            "total_trades":       len(rows),
            "wins":               len(wins),
            "losses":             len(losses),
            "win_rate":           round(len(wins) / len(rows) * 100, 1),
            "net_pnl":            round(net_pnl, 2),
            "total_charges":      round(sum(r["total_charges"] for r in rows), 2),
            "avg_win":            round(avg_win, 2),
            "avg_loss":           round(avg_loss, 2),
            "rr_realised":        round(abs(avg_win / avg_loss), 2) if avg_loss else 0,
            "strategy_breakdown": by_strategy,
        }

    def print_dashboard(self, available_capital: float = None):
        d  = self.daily_summary()
        m  = self.monthly_summary()

        max_trades = (
            Config.effective_max_trades(available_capital)
            if available_capital is not None
            else Config.MAX_SIMULTANEOUS_TRADES
        )

        print("\n" + "=" * 65)
        print("  NIFTY 50 SWING TRADING — DASHBOARD")
        print("=" * 65)
        print(f"  Date          : {d['date']}")
        print(f"  Open Positions: {d['open_positions']} / {max_trades}")
        print(f"\n  TODAY")
        print(f"     Closed : {d['trades_closed']}  ({d['wins']}W / {d['losses']}L)  WR: {d['win_rate']}%")
        print(f"     Realised   : ₹{d['realised_pnl']:,.2f}")
        print(f"     Unrealised : ₹{d['unrealised_pnl']:,.2f}")
        print(f"     Charges    : ₹{d['total_charges']:,.2f}")

        if isinstance(m, dict) and "total_trades" in m:
            print(f"\n  {m.get('month', 'MONTH')}")
            print(f"     Trades  : {m['total_trades']}  WR: {m['win_rate']}%")
            print(f"     Net P&L : ₹{m['net_pnl']:,.2f}  (Avg W: ₹{m['avg_win']:,.2f} / L: ₹{m['avg_loss']:,.2f})")
            print(f"     Real RR : {m['rr_realised']}:1")
            if m.get("strategy_breakdown"):
                print(f"\n  STRATEGY BREAKDOWN")
                for strat, stats in m["strategy_breakdown"].items():
                    wr = round(stats["wins"] / stats["trades"] * 100, 1)
                    print(f"     {strat:12s}: {stats['trades']} trades | {wr}% WR | ₹{stats['pnl']:,.0f}")

        open_trades = self.db.get_open_trades()
        if open_trades:
            print(f"\n  OPEN POSITIONS")
            for t in open_trades:
                unr = (t["current_price"] - t["entry_price"]) * t["remaining_qty"]
                print(f"     {t['symbol']:12s} | {t['strategy']:9s} | "
                      f"Entry:₹{t['entry_price']:.2f} Now:₹{t['current_price']:.2f} "
                      f"SL:₹{t['current_sl']:.2f} | Unr:₹{unr:,.2f}")
        print("=" * 65 + "\n")


class MorningBriefing:
    """
    Generates a formatted text summary of market conditions and today's setups.
    Called in step8_morning_briefing() after setup discovery is complete.
    Printed to stdout so the operator can review before market open at 9:30 AM.
    The output is human-readable, not machine-parseable — it's a daily sanity check.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def generate(self, market_mode: MarketMode, fii_flow: FIIFlow,
                  vix: float, fii_net: float,
                  setups: list, events_today: list) -> str:
        """
        Builds and returns the morning briefing as a multi-line string.
        setups is the sorted list from step4_find_setups() — shows top 4 max.
        events_today are RED-risk events within 2 days — these are action items.
        In DEFENSIVE or CASH mode, replaces the setup list with a clear warning.
        """
        today = datetime.now().strftime("%A, %d %B %Y")
        mode_emoji = {
            "AGGRESSIVE": "🚀", "NORMAL": "✅", "SELECTIVE": "⚠️",
            "CAUTIOUS": "🟡", "DEFENSIVE": "🛑", "CASH": "💰"
        }.get(market_mode.value, "❓")

        lines = [
            "=" * 60,
            f"  MORNING BRIEFING — {today}",
            "=" * 60,
            f"  Market Mode : {mode_emoji} {market_mode.value}",
            f"  India VIX   : {vix:.1f}",
            f"  FII Flow    : {fii_flow.value} (₹{fii_net:,.0f} Cr)",
            "",
        ]

        if events_today:
            lines.append("  EVENTS TODAY / TOMORROW:")
            for e in events_today:
                lines.append(f"     {e['symbol']} — {e['event_type']} in {e['days_away']} days")
            lines.append("")

        if market_mode in [MarketMode.DEFENSIVE, MarketMode.CASH]:
            lines.append("  NO TRADING TODAY — Market in protection mode")
        elif not setups:
            lines.append("  No valid setups found today.")
        else:
            lines.append(f"  {len(setups)} SETUP(S):")
            for i, s in enumerate(setups[:4], 1):
                lines.append(
                    f"  {i}. {s.symbol:12s} | {s.strategy.value:10s} | Score:{s.score} | "
                    f"Entry:₹{s.entry_price:.2f} SL:₹{s.sl_price:.2f} T:₹{s.target_price:.2f}"
                )
            lines.append(f"\n  RECOMMENDED: {setups[0].symbol} via {setups[0].strategy.value}")

        daily_pnl = self.db.get_today_realised_pnl()
        lines.append(f"\n  Today P&L: ₹{daily_pnl:,.2f}")
        lines.append("=" * 60)
        return "\n".join(lines)
