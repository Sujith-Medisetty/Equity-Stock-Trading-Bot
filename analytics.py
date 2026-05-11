"""
analytics.py — Complete performance reporting, tax calculation, and morning briefing.

DATA SOURCES — what comes from where and why:

  LOCAL DB  (trading.db) — the only place that stores:
    • Strategy used per trade (PULLBACK / BREAKOUT / SWING / WEEK52)
      → Upstox has no concept of our strategies; it only records order IDs
    • Tier status (tier1_done, tier2_done, trailing SL levels)
      → Upstox doesn't know our exit tiers; it just sees individual sell orders
    • Target price per trade
      → Upstox doesn't store targets; we set them, we track them
    • Net P&L after charges per trade (calculated at exit using Upstox brokerage API)
      → Upstox's /trade/profit-loss API gives gross (sell - buy), not net after charges
    • Setup score, market mode at entry
      → System metadata — broker has no concept of these
    • SL order linkage (sl_order_id → entry order)
      → No Upstox API field links an SL order back to its parent position
    • FII history, market mode snapshots, events calendar
      → System metadata, not stored by any broker
    • Protection state (cooldown timer, halt flag)
      → Must survive process restarts — stored as key-value in DB

  UPSTOX LIVE API — what we fetch from the broker:
    • Available cash: /v3/user/get-funds-and-margin (already used in get_available_capital)
    • Current prices for open positions: already polled every 15 mins by TradeMonitor
      and written to trades.current_price in DB → analytics reads from DB (fresh enough)
    • Brokerage charges per trade: /v2/charges/brokerage (already used at exit time,
      stored as total_charges in DB)

  TAX (computed here from DB, not fetched from Upstox):
    Why not Upstox /v2/trade/profit-loss/charges?
    → That API gives FY-aggregate charges, not net P&L per trade.
      Our DB already has net_pnl (after charges) calculated precisely at exit.
      Using our DB gives per-trade accuracy that the Upstox API can't match.

    India STCG (Short-Term Capital Gains) rules:
    → All bot trades hold ≤ 15 days → ALL are STCG (< 1 year)
    → STCG tax rate (post July 23, 2024 budget): 20%
    → Taxable base = total net P&L across all trades in the FY (losses offset gains)
    → Only pay tax if the NET result for the year is positive
    → India FY: April 1 → March 31
    → Advance tax required if total annual tax > ₹10,000:
        Jun 15  → 15% of annual estimate
        Sep 15  → 45% cumulative
        Dec 15  → 75% cumulative
        Mar 15  → 100%

Two classes:
  PerformanceAnalytics — end-of-day and on-demand dashboard + tax
  MorningBriefing      — 9:15 AM pre-market summary
"""

from datetime import datetime, date, timedelta

from config import Config, log
from models import MarketMode, FIIFlow
from database import DatabaseManager


# India STCG rate post July 23 2024 budget: 20%
_STCG_RATE = 0.20


def _fy_start() -> str:
    """
    Returns April 1 of the CURRENT Indian financial year as YYYY-MM-DD.
    India FY: April 1 (year N) → March 31 (year N+1).
    Example: called in May 2026 → returns 2026-04-01
             called in Feb 2027 → returns 2026-04-01 (still same FY)
    """
    today = date.today()
    fy_year = today.year if today.month >= 4 else today.year - 1
    return f"{fy_year}-04-01"


def _fy_year() -> int:
    """Returns the starting year of the current Indian FY (e.g. 2026 for FY 2026-27)."""
    today = date.today()
    return today.year if today.month >= 4 else today.year - 1


def _progress_bar(used_pct: float, width: int = 10) -> str:
    """ASCII progress bar. 0% = all empty, 100% = all filled."""
    filled = min(width, int(used_pct / 100 * width))
    return f"[{'█' * filled}{'░' * (width - filled)}] {used_pct:5.1f}%"


def _status_label(used_pct: float) -> str:
    if used_pct >= 100: return "STOPPED"
    if used_pct >= 75:  return "DANGER"
    if used_pct >= 50:  return "WARN"
    return "OK"


class PerformanceAnalytics:
    """
    Reads trade records from local DB and produces a complete dashboard.

    Call flow:
      print_dashboard(available_capital)
        → daily_summary()      — today's closed trades + open position unrealised PnL
        → monthly_summary()    — month-to-date stats + strategy breakdown
        → fy_tax_summary()     — financial year P&L + STCG tax estimate + advance tax
        → protection_status()  — circuit breaker usage (daily/weekly/monthly limits)
        → DB.get_open_trades() — detailed open position table with tier status

    The dashboard is designed so you never need to open the code or the database
    to understand how the system is performing.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    # =========================================================================
    # DAILY SUMMARY
    # =========================================================================

    def daily_summary(self) -> dict:
        """
        Today's closed trades + unrealised PnL on open positions.

        unrealised = sum of (current_price - entry_price) × remaining_qty
        current_price is written to DB every 15 mins by TradeMonitor, so this
        figure is always within 15 minutes of the actual live price.
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
            if t["current_price"] and t["entry_price"]
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

    # =========================================================================
    # MONTHLY SUMMARY
    # =========================================================================

    def monthly_summary(self) -> dict:
        """
        Month-to-date stats from the 1st of the current calendar month.

        strategy_breakdown: which strategies are actually making money.
          If SWING has 30% win rate over 3 months, consider raising its score threshold.
          If PULLBACK is consistently 60%+, it's the core edge of the system.

        rr_realised: the actual reward:risk delivered (not the theoretical 2.0 minimum).
          If this drops below 1.5, it means exits are too early or SL is too tight.
          Target: stay above 2.0 — that's what the system was designed to deliver.

        exit_breakdown: how trades are closing.
          SL_HIT, BROKER_EXECUTED → losses (SL fired)
          TIER2_PARTIAL_EXIT      → partial profit locked
          MARKET_CRASH, VIX_PANIC → emergency VIX exits
          EVENT_EXIT              → earnings-related exits
          TIME_BASED              → stale trade exits (held 15 days without profit)
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

        # Strategy breakdown
        by_strategy: dict = {}
        for r in rows:
            s = r["strategy"] or "UNKNOWN"
            if s not in by_strategy:
                by_strategy[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_strategy[s]["trades"] += 1
            by_strategy[s]["wins"]   += 1 if r["net_pnl"] > 0 else 0
            by_strategy[s]["pnl"]    += r["net_pnl"]

        # Exit reason breakdown — tells you WHY trades are closing
        by_exit: dict = {}
        for r in rows:
            reason = r["exit_reason"] or "UNKNOWN"
            by_exit[reason] = by_exit.get(reason, 0) + 1

        # Average holding days
        holding_days = [r["holding_days"] for r in rows if r["holding_days"]]
        avg_hold = round(sum(holding_days) / len(holding_days), 1) if holding_days else 0

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
            "avg_hold_days":      avg_hold,
            "strategy_breakdown": by_strategy,
            "exit_breakdown":     by_exit,
        }

    # =========================================================================
    # FINANCIAL YEAR TAX SUMMARY
    # =========================================================================

    def fy_tax_summary(self) -> dict:
        """
        Full financial year tax picture from April 1 to today.

        Source: local DB only.
          net_pnl in DB = sell_amount - buy_amount - all_charges
          (calculated precisely at exit using Upstox /v2/charges/brokerage API)

        Why not use Upstox /v2/trade/profit-loss/data?
          → That API gives gross P&L (sell - buy) without charges deducted.
          → Our DB already has net_pnl (charges already subtracted), which is
             more accurate than trying to match Upstox's aggregate charges API.

        STCG tax logic (India):
          All bot trades hold ≤ 15 days → ALL are Short-Term Capital Gains.
          Taxable amount = total net P&L for the FY (losses offset wins).
          If total net is negative → tax = ₹0 (losses carried forward, but
          that's a CA discussion — the bot just shows estimate on current trades).
          Tax rate: 20% (post July 23, 2024 budget).

        Advance tax (required if annual tax > ₹10,000):
          Jun 15 → pay 15% of full-year estimate
          Sep 15 → pay 45% cumulative
          Dec 15 → pay 75% cumulative
          Mar 15 → pay 100%

        Charges (STT, DP, GST, SEBI, brokerage) are DEDUCTIBLE EXPENSES
        and are already subtracted in our net_pnl calculation.
        Consult a CA for actual filing — this is a planning estimate only.
        """
        fy    = _fy_start()
        fy_yr = _fy_year()
        rows  = self.db.fetchall(
            "SELECT * FROM trades WHERE exit_date>=? AND status='CLOSED'", (fy,)
        )

        wins   = [r for r in rows if r["net_pnl"] > 0]
        losses = [r for r in rows if r["net_pnl"] <= 0]

        total_gross   = sum(r["gross_pnl"]      for r in rows)
        total_charges = sum(r["total_charges"]   for r in rows)
        total_net     = sum(r["net_pnl"]         for r in rows)

        # Tax base = net profit for the year (losses offset gains)
        taxable       = max(0.0, total_net)
        tax_estimate  = round(taxable * _STCG_RATE, 2)

        # Advance tax quarters for this FY
        quarters = [
            ("Jun 15", date(fy_yr, 6, 15),      0.15),
            ("Sep 15", date(fy_yr, 9, 15),      0.45),
            ("Dec 15", date(fy_yr, 12, 15),     0.75),
            ("Mar 15", date(fy_yr + 1, 3, 15),  1.00),
        ]
        today_d   = date.today()
        next_due  = None
        last_paid = None
        for label, due, pct in quarters:
            if due < today_d:
                last_paid = {"date": label, "amount": round(tax_estimate * pct, 2), "pct": int(pct * 100)}
            elif next_due is None:
                next_due  = {"date": label, "due": str(due), "amount": round(tax_estimate * pct, 2), "pct": int(pct * 100)}

        return {
            "fy_start":           fy,
            "fy_label":           f"FY {fy_yr}-{str(fy_yr + 1)[2:]}",
            "total_trades":       len(rows),
            "winning_trades":     len(wins),
            "losing_trades":      len(losses),
            "total_gross_pnl":    round(total_gross, 2),
            "total_charges_paid": round(total_charges, 2),
            "total_net_pnl":      round(total_net, 2),
            "taxable_stcg":       round(taxable, 2),
            "stcg_rate":          f"{int(_STCG_RATE * 100)}%",
            "stcg_tax_estimate":  tax_estimate,
            "next_advance_tax":   next_due,
            "last_advance_tax":   last_paid,
        }

    # =========================================================================
    # PROTECTION STATUS
    # =========================================================================

    def protection_status(self) -> dict:
        """
        How close are we to each loss circuit breaker?

        Daily  limit: ₹3,000  → stops trading for the rest of today
        Weekly limit: ₹6,000  → stops trading for the rest of the week
        Monthly limit:₹10,000 → stops trading for the rest of the month
        Drawdown:     ₹20,000 → full system halt, needs manual review

        Cooldown: 2-hour pause after any SL hit (prevents revenge trading).
          Stored as ISO timestamp in DB → survives restarts.

        All values read from DB closed trades (net_pnl after charges).
        """
        daily   = self.db.get_today_realised_pnl()
        weekly  = self.db.get_week_realised_pnl()
        monthly = self.db.get_month_realised_pnl()

        # Cooldown: how many minutes remain on the post-loss pause
        cooldown_mins = 0
        cooldown_until = self.db.get_state("cooldown_until")
        if cooldown_until:
            try:
                cd   = datetime.fromisoformat(cooldown_until)
                secs = (cd - datetime.now()).total_seconds()
                cooldown_mins = max(0, int(secs / 60))
            except Exception:
                pass

        # Consecutive loss check: last 3 closed trades all negative?
        rows = self.db.fetchall(
            "SELECT net_pnl FROM trades WHERE status='CLOSED' ORDER BY exit_date DESC LIMIT 3"
        )
        consec_losses = len(rows) >= 3 and all(r["net_pnl"] < 0 for r in rows)

        def _pct(loss_val: float, limit: float) -> float:
            return round(abs(loss_val) / limit * 100, 1) if loss_val < 0 else 0.0

        return {
            "daily_pnl":         round(daily, 2),
            "daily_limit":       Config.DAILY_LOSS_LIMIT,
            "daily_used_pct":    _pct(daily,   Config.DAILY_LOSS_LIMIT),
            "weekly_pnl":        round(weekly, 2),
            "weekly_limit":      Config.WEEKLY_LOSS_LIMIT,
            "weekly_used_pct":   _pct(weekly,  Config.WEEKLY_LOSS_LIMIT),
            "monthly_pnl":       round(monthly, 2),
            "monthly_limit":     Config.MONTHLY_LOSS_LIMIT,
            "monthly_used_pct":  _pct(monthly, Config.MONTHLY_LOSS_LIMIT),
            "drawdown_pct":      _pct(monthly, Config.MAX_DRAWDOWN),
            "cooldown_active":   cooldown_mins > 0,
            "cooldown_mins":     cooldown_mins,
            "halted":            self.db.get_state("trading_halted") == "1",
            "consec_losses":     consec_losses,
        }

    # =========================================================================
    # MAIN DASHBOARD PRINTER
    # =========================================================================

    def print_dashboard(self, available_capital: float = None):
        """
        Prints the complete system dashboard. Call at end of day (step7) or
        via: python main.py --dashboard

        Sections printed:
          1. Account snapshot    — cash available, open slot count
          2. TODAY               — closed trades, realised + unrealised PnL, charges
          3. OPEN POSITIONS      — per-trade detail: entry, current, target, SL, tier, days
          4. THIS MONTH          — win rate, net P&L, avg win/loss, real RR, charges
          5. STRATEGY BREAKDOWN  — which strategy is working and which needs review
          6. EXIT REASONS        — why trades are closing (SL/target/VIX/event/time)
          7. PROTECTION STATUS   — circuit breaker bars (daily/weekly/monthly)
          8. TAX SUMMARY         — FY P&L, STCG estimate, advance tax due date
        """
        d    = self.daily_summary()
        m    = self.monthly_summary()
        tax  = self.fy_tax_summary()
        prot = self.protection_status()

        open_trades = self.db.get_open_trades()
        max_trades  = (
            Config.effective_max_trades(available_capital)
            if available_capital is not None
            else Config.MAX_SIMULTANEOUS_TRADES
        )

        W = 72   # dashboard width

        print("\n" + "=" * W)
        print("  NIFTY 50 SWING TRADING — FULL DASHBOARD")
        print("=" * W)
        mode_str = "LIVE MONEY" if not Config.PAPER_TRADE else "PAPER TRADE (no real money)"
        print(f"  Date    : {d['date']}    Mode: {mode_str}")
        if available_capital is not None:
            print(f"  Cash    : ₹{available_capital:,.0f}  available for new trades")
        print(f"  Slots   : {d['open_positions']} open / {max_trades} max allowed  "
              f"({'FULL — no new trades today' if d['open_positions'] >= max_trades else 'slots available'})")

        # ── TODAY ────────────────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print("  TODAY")
        print(f"{'─' * W}")
        if d["trades_closed"] == 0:
            print("  No trades closed today.")
        else:
            print(f"  Closed  : {d['trades_closed']} trades  ({d['wins']} wins / {d['losses']} losses)  "
                  f"Win Rate: {d['win_rate']}%")
        print(f"  Realised   P&L : ₹{d['realised_pnl']:>10,.2f}  ← actual money settled today")
        print(f"  Unrealised P&L : ₹{d['unrealised_pnl']:>10,.2f}  ← open positions (can still go up/down)")
        print(f"  Charges Paid   : ₹{d['total_charges']:>10,.2f}  ← STT + DP charge + GST + SEBI today")

        # ── OPEN POSITIONS ───────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        if open_trades:
            print(f"  OPEN POSITIONS  ({len(open_trades)} active)")
            print(f"{'─' * W}")
            hdr = f"  {'Stock':<12} {'Strategy':<9} {'Entry':>8} {'Now':>8} {'Target':>8} {'SL':>8}  {'Tier':<14} {'Days':>4}  {'Unr P&L':>10}"
            print(hdr)
            print(f"  {'-'*12} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*14} {'-'*4}  {'-'*10}")
            for t in open_trades:
                cp  = t["current_price"] or t["entry_price"]
                unr = (cp - t["entry_price"]) * t["remaining_qty"]
                pct = (cp - t["entry_price"]) / t["entry_price"] * 100 if t["entry_price"] else 0
                try:
                    hdays = (datetime.now() - datetime.strptime(t["entry_date"], "%Y-%m-%d")).days
                except Exception:
                    hdays = t.get("holding_days", 0)

                # Tier label — what stage is this trade at?
                # T0-Entry    : SL at original level. Full ₹1,500 risk still live.
                # T1-NoLoss   : SL moved to entry price. Worst case = ₹0 loss from here.
                # T2-Trailing : 50% sold at target. Remaining shares trailing with SL.
                if t["tier2_done"]:
                    tier = "T2-Trailing"
                elif t["tier1_done"]:
                    tier = "T1-NoLoss  "
                else:
                    tier = "T0-Entry   "

                print(f"  {t['symbol']:<12} {t['strategy']:<9} "
                      f"₹{t['entry_price']:>7.2f} ₹{cp:>7.2f} "
                      f"₹{t['initial_target']:>7.2f} ₹{t['current_sl']:>7.2f}  "
                      f"{tier:<14} {hdays:>4}d  "
                      f"₹{unr:>9,.2f} ({pct:+.1f}%)")

            print()
            print("  Tier Guide:")
            print("    T0-Entry    SL at original stop loss.  Risk = ₹1,500 if SL fires.")
            print("    T1-NoLoss   SL moved to your entry price. Worst case = ₹0 loss.")
            print("    T2-Trailing Half shares sold at target (profit locked). Rest trailing.")
            print(f"    Max hold = {Config.MAX_HOLD_DAYS} days — auto-exits flat/losing trades after that.")
        else:
            print("  OPEN POSITIONS")
            print(f"{'─' * W}")
            print("  No open positions.")

        # ── THIS MONTH ───────────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        if isinstance(m, dict) and "total_trades" in m:
            print(f"  {m['month'].upper()}")
            print(f"{'─' * W}")
            print(f"  Trades    : {m['total_trades']}  ({m['wins']} wins / {m['losses']} losses)  Win Rate: {m['win_rate']}%")
            print(f"  Net P&L   : ₹{m['net_pnl']:>10,.2f}  (after all charges)")
            print(f"  Avg Win   : ₹{m['avg_win']:>10,.2f}")
            print(f"  Avg Loss  : ₹{m['avg_loss']:>10,.2f}")
            rr_note = "Good" if m["rr_realised"] >= 2.0 else ("Review exits — too early?" if m["rr_realised"] >= 1.5 else "LOW — exit timing problem")
            print(f"  Real RR   : {m['rr_realised']}:1  ← {rr_note}  (target ≥ 2.0:1)")
            print(f"  Avg Hold  : {m['avg_hold_days']} days per trade")
            print(f"  Charges   : ₹{m['total_charges']:>10,.2f}  (total brokerage + taxes this month)")

            # Strategy breakdown
            if m.get("strategy_breakdown"):
                print(f"\n  STRATEGY BREAKDOWN")
                print(f"  {'Strategy':<12} {'Trades':>6}  {'Win%':>6}  {'Net P&L':>10}  {'Verdict'}")
                print(f"  {'-'*12} {'-'*6}  {'-'*6}  {'-'*10}  {'-'*20}")
                for strat, s in sorted(m["strategy_breakdown"].items(),
                                        key=lambda x: x[1]["pnl"], reverse=True):
                    wr      = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
                    verdict = ("Working well" if s["pnl"] > 0 and wr >= 50 else
                               "Profitable, low WR" if s["pnl"] > 0 else
                               "Review — losing")
                    print(f"  {strat:<12} {s['trades']:>6}  {wr:>5.1f}%  ₹{s['pnl']:>9,.0f}  {verdict}")

            # Exit reasons breakdown
            if m.get("exit_breakdown"):
                print(f"\n  EXIT REASONS  (why trades closed)")
                exit_labels = {
                    "SL_HIT":             "SL hit (backtest)",
                    "BROKER_EXECUTED":    "SL hit by broker (live/paper)",
                    "TIER2_PARTIAL_EXIT": "Partial exit at target (tier 2)",
                    "TIER3_TRAIL":        "Trailed out past target (tier 3)",
                    "MARKET_CRASH":       "VIX panic exit",
                    "EVENT_EXIT":         "Earnings event exit",
                    "TIME_BASED":         "Stale trade exit (15 days no profit)",
                }
                for reason, count in sorted(m["exit_breakdown"].items(),
                                             key=lambda x: x[1], reverse=True):
                    label = exit_labels.get(reason, reason)
                    print(f"  {count:>3}×  {label}")
        else:
            print("  THIS MONTH")
            print(f"{'─' * W}")
            print("  No closed trades this month yet.")

        # ── PROTECTION STATUS ─────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print("  PROTECTION STATUS  (trading circuit breakers)")
        print(f"{'─' * W}")
        print("  These limits auto-stop trading to prevent blowing up the account.")
        print()

        for label, pnl, limit, pct in [
            ("Daily  ", prot["daily_pnl"],   prot["daily_limit"],   prot["daily_used_pct"]),
            ("Weekly ", prot["weekly_pnl"],  prot["weekly_limit"],  prot["weekly_used_pct"]),
            ("Monthly", prot["monthly_pnl"], prot["monthly_limit"], prot["monthly_used_pct"]),
        ]:
            bar    = _progress_bar(pct)
            status = _status_label(pct)
            loss_str = f"₹{abs(pnl):>7,.0f} lost" if pnl < 0 else f"₹{pnl:>7,.0f} made"
            print(f"  {label}: {loss_str} / ₹{limit:,} limit  {bar}  [{status}]")

        print()
        if prot["cooldown_active"]:
            print(f"  *** COOLDOWN: No new trades for {prot['cooldown_mins']} more minutes.")
            print(f"      (2-hour pause triggered after a stop loss hit — prevents revenge trading)")
        if prot["halted"]:
            print(f"  *** MANUAL HALT: All trading stopped.")
            print(f"      To resume: set trading_halted='' in protection_state table in trading.db")
        if prot["consec_losses"]:
            print(f"  *** WARNING: Last 3 closed trades were all losses.")
            print(f"      Review market conditions and strategy parameters before continuing.")

        all_ok = (not prot["cooldown_active"] and not prot["halted"]
                  and prot["daily_used_pct"] < 100
                  and prot["weekly_used_pct"] < 100
                  and prot["monthly_used_pct"] < 100)
        print()
        print(f"  Trading today: {'ALLOWED' if all_ok else 'BLOCKED'}")

        # ── TAX SUMMARY ──────────────────────────────────────────────────────
        print(f"\n{'─' * W}")
        print(f"  TAX SUMMARY  ({tax['fy_label']}  from {tax['fy_start']})")
        print(f"{'─' * W}")
        print(f"  Total trades this FY : {tax['total_trades']}  "
              f"({tax['winning_trades']} profitable / {tax['losing_trades']} at loss)")
        print()
        print(f"  Gross P&L  (sell − buy, before charges)  : ₹{tax['total_gross_pnl']:>10,.2f}")
        print(f"  Charges paid (STT + DP + GST + SEBI + brokerage) : ₹{tax['total_charges_paid']:>10,.2f}")
        print(f"  Net P&L    (after all charges deducted)  : ₹{tax['total_net_pnl']:>10,.2f}")
        print()
        print(f"  Taxable STCG base    : ₹{tax['taxable_stcg']:>10,.2f}")
        print(f"  (= net P&L if positive; ₹0 if net is negative — losses offset gains)")
        print(f"  STCG tax rate        : {tax['stcg_rate']}  (all bot trades < 1 year = Short-Term)")
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  STCG TAX ESTIMATE : ₹{tax['stcg_tax_estimate']:>10,.2f}  — SET THIS ASIDE   │")
        print(f"  └─────────────────────────────────────────────────────────┘")

        if tax["next_advance_tax"]:
            nd = tax["next_advance_tax"]
            print(f"\n  Next advance tax due : {nd['date']} ({nd['due']})")
            print(f"  Pay {nd['pct']}% of annual estimate  = ₹{nd['amount']:,.2f}")
            print(f"  (Advance tax required only if total annual tax > ₹10,000)")

        print()
        print("  Notes:")
        print("    • Charges (STT, DP, brokerage, GST, SEBI) are DEDUCTIBLE — already")
        print("      subtracted from net_pnl, so they reduce your taxable base.")
        print("    • Losses in this FY can offset gains within the SAME FY only.")
        print("    • This is an ESTIMATE for planning. Consult a CA for actual filing.")
        print("=" * W + "\n")


# =============================================================================
# MORNING BRIEFING
# =============================================================================

class MorningBriefing:
    """
    Generates the 9:15 AM pre-market briefing printed before trading starts.

    Shows the operator:
      1. Market conditions — mode, VIX, FII flow, Gift Nifty direction
      2. Risk events — any earnings/results within 2 days for watchlist stocks
      3. Today's setups — what the system found, scores, entry/SL/target
      4. Protection status — any circuit breakers already active this morning
      5. Today's P&L so far — in case early SL hits happened overnight

    In DEFENSIVE or CASH mode: replaces the setup list with a clear
    "NO TRADING TODAY" warning so the operator doesn't need to check code.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def generate(self, market_mode: MarketMode, fii_flow: FIIFlow,
                 vix: float, fii_net: float,
                 setups: list, events_today: list) -> str:

        today = datetime.now().strftime("%A, %d %B %Y")

        # Market mode labels with what they mean for today's trading
        mode_info = {
            "AGGRESSIVE": ("All 4 strategies active. Max 4 positions.",   "FULL TRADING"),
            "NORMAL":     ("Most strategies active. Normal conditions.",   "NORMAL TRADING"),
            "SELECTIVE":  ("Only score ≥ 80 setups. Be very selective.",  "SELECTIVE TRADING"),
            "CAUTIOUS":   ("1–2 positions max. Mixed market.",            "CAUTIOUS"),
            "DEFENSIVE":  ("No new entries. Protect existing positions.", "NO NEW TRADES"),
            "CASH":       ("VIX panic. Exit all positions.",              "EXIT ALL — PANIC MODE"),
        }
        minfo   = mode_info.get(market_mode.value, ("", ""))
        m_desc  = minfo[0]
        m_label = minfo[1]

        # VIX reading with plain-English meaning
        if vix >= 29:
            vix_note = "PANIC — exit all positions"
        elif vix >= 25:
            vix_note = "NERVOUS — no new entries, tighten SLs"
        elif vix >= 17:
            vix_note = "Elevated — trade carefully"
        elif vix >= 13:
            vix_note = "Normal — good conditions"
        else:
            vix_note = "Very calm — excellent conditions"

        # FII flow with what it means
        fii_note = ""
        if fii_flow == FIIFlow.BUYING:
            fii_note = "(3+ days buying ₹2000Cr+ → institutional tailwind)"
        elif fii_flow == FIIFlow.SELLING:
            fii_note = "(3+ days selling ₹2000Cr+ → institutions exiting)"
        else:
            fii_note = "(no clear sustained direction)"

        lines = [
            "=" * 62,
            f"  MORNING BRIEFING — {today}",
            "=" * 62,
            f"  Market Mode : {market_mode.value} — {m_label}",
            f"              : {m_desc}",
            f"  India VIX   : {vix:.1f}  ({vix_note})",
            f"  FII Flow    : {fii_flow.value}  ₹{fii_net:,.0f} Cr today  {fii_note}",
            "",
        ]

        # RED events — these are URGENT action items
        if events_today:
            lines.append("  URGENT — EVENTS TODAY / TOMORROW:")
            for e in events_today:
                lines.append(f"    {e['symbol']:12} — {e['event_type']} in {e['days_away']} day(s)")
                lines.append(f"    Action: bot will FORCE EXIT {e['symbol']} before results.")
            lines.append("")

        # Setup list or no-trading message
        if market_mode in [MarketMode.DEFENSIVE, MarketMode.CASH]:
            lines.append("  ─" * 31)
            lines.append(f"  NO TRADING TODAY — {market_mode.value} mode")
            lines.append("  ─" * 31)
            if market_mode == MarketMode.CASH:
                lines.append("  VIX is in panic territory. Bot will exit all open positions.")
            else:
                lines.append("  Nifty below EMA200 — trend is down. Bot sits out.")
        elif not setups:
            lines.append("  No valid setups found today (all scored below 80 or filtered).")
            lines.append("  Bot will re-scan every 15 mins during market hours for new setups.")
        else:
            taken  = [s for s in setups if s.status == "TAKEN"]
            found  = [s for s in setups if s.status != "TAKEN"]
            lines.append(f"  {len(setups)} SETUP(S) FOUND   ({len(taken)} entered / {len(found)} pending/skipped)")
            lines.append(f"  {'#':>2}  {'Stock':<12} {'Strategy':<10} {'Score':>5}  {'Entry':>8} {'SL':>8} {'Target':>8}  {'Status'}")
            lines.append(f"  {'─'*2}  {'─'*12} {'─'*10} {'─'*5}  {'─'*8} {'─'*8} {'─'*8}  {'─'*10}")
            for i, s in enumerate(setups[:6], 1):
                status = getattr(s, "status", "PENDING")
                lines.append(
                    f"  {i:>2}  {s.symbol:<12} {s.strategy.value:<10} {s.score:>5}  "
                    f"₹{s.entry_price:>7.2f} ₹{s.sl_price:>7.2f} ₹{s.target_price:>7.2f}  {status}"
                )
            if setups:
                best = setups[0]
                lines.append("")
                lines.append(f"  Top pick: {best.symbol} via {best.strategy.value}  "
                             f"(score {best.score},  R:R {getattr(best, 'rr_ratio', 0):.1f}:1)")

        # Protection status snapshot
        lines.append("")
        lines.append("  ─" * 31)

        daily_pnl  = self.db.get_today_realised_pnl()
        cooldown   = self.db.get_state("cooldown_until")
        halted     = self.db.get_state("trading_halted") == "1"

        cooldown_mins = 0
        if cooldown:
            try:
                cd = datetime.fromisoformat(cooldown)
                cooldown_mins = max(0, int((cd - datetime.now()).total_seconds() / 60))
            except Exception:
                pass

        if halted:
            lines.append("  TRADING HALTED manually — check protection_state in DB to resume")
        elif cooldown_mins > 0:
            lines.append(f"  COOLDOWN: {cooldown_mins} mins remaining (post-loss pause active)")
        else:
            lines.append("  Protection: OK — no circuit breakers active")

        lines.append(f"  Today P&L so far: ₹{daily_pnl:,.2f}")
        lines.append("=" * 62)
        return "\n".join(lines)
