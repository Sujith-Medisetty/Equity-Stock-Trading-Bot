"""
monitor.py — Intraday trade monitoring, exit conditions, and trailing SL.

TradeMonitor runs every 15 minutes (step6_monitor_trades) from 9:30 AM to 3:30 PM.
It does four things on each cycle:

1. Broker sync (sync_with_broker)
   Compares the DB's open trades against actual Dhan holdings.
   Any trade that is in the DB as OPEN but no longer in Dhan holdings =
   the broker already fired the SL order. We mark it CLOSED in the DB with
   reason "BROKER_EXECUTED" and start the loss cooldown.
   This is critical for live mode: without this, the DB would show trades as
   still open when they've already been exited at the exchange level.
   Skipped in paper mode (no real broker holdings to compare against).

2. Exit condition checks (_check_exit_conditions)
   Evaluates four exit triggers each cycle:
   a) SL hit — paper mode only. In live mode, the broker's SL order handles this.
      We never manually sell in live mode because the broker already sold us out.
   b) VIX spike — if VIX crosses the nervous threshold (22), exit immediately
      regardless of position profit. Market chaos makes all setups invalid.
   c) Time-based exit — if a trade is open for 15+ days without profit, exit.
      Dead money: capital tied up in a trade going nowhere could be redeployed.
   d) Event guard — if an earnings/results event is within 5 days, exit.
      We don't hold through results — gap risk is too high for swing size positions.

3. Trailing SL updates (_update_trailing_sl)
   3-tier system that locks in progressively more profit as the trade works:
   Tier 1: price reaches 1:1 (profit = initial risk) → SL moves to breakeven.
            No-loss guarantee from this point. We stay in for more upside.
   Tier 2: price reaches 2:1 (initial target) → sell 50% of remaining shares.
            Lock in realised profit. SL tightens to (target − 0.5×ATR).
   Tier 3: beyond 2:1 → trail remaining shares with SL at (price − 1×ATR).
            SL rises every time price makes a new high. Ride the trend, let winners run.

   Each tier update also updates the actual SL order at the broker exchange
   (cancel old order + place new one) via replace_sl_order(). This ensures
   the broker-level protection matches our trailing SL.

4. Current price tracking
   Updates current_price in the trades table every cycle.
   This is used for unrealised PNL calculation in the dashboard.
"""

from datetime import datetime

from config import Config, log
from models import ExitReason, Trade
from database import DatabaseManager
from orders import OrderManager
from protection import ProtectionEngine
from risk import ChargesCalculator


class TradeMonitor:
    """
    Monitors all open trades every 15 minutes during market hours.
    Handles exits (paper SL, VIX spike, time, events) and trailing SL management.
    Reconciles DB state with broker state on every cycle.
    """

    def __init__(self, db: DatabaseManager, order_mgr: OrderManager,
                 protection: ProtectionEngine):
        self.db         = db
        self.order_mgr  = order_mgr
        self.protection = protection

    def sync_with_broker(self):
        """
        Two-way reconciliation between Dhan holdings and the local DB.

        Direction 1 — Dhan → DB (close):
          DB trade is OPEN but symbol no longer in Dhan holdings.
          Broker already fired the SL order. Mark it CLOSED with "BROKER_EXECUTED".

        Direction 2 — Dhan → DB (import):
          Dhan has a holding that has no matching OPEN trade in DB.
          Could be a manual buy in the Dhan app, or a crash before DB write.
          Import it as an OPEN trade with conservative defaults so the
          monitor and screener know about it.

        Direction 3 — DB → Dhan (re-place SL):
          DB trade is OPEN and symbol is in Dhan, but sl_order_id is empty.
          The SL order was never placed or was lost. Re-place it now so the
          position is protected at the exchange level.

        No-op in BACKTEST_MODE (no real broker to talk to).
        """
        # Skip entirely in backtest mode — there's no broker to sync with
        if Config.BACKTEST_MODE or not self.order_mgr.dhan:
            return

        try:
            resp = self.order_mgr.dhan.get_holdings()
            # get_holdings() returns {"data": [...]} or just a list depending on SDK version
            holdings = resp.get("data", []) if isinstance(resp, dict) else []

            # Build a map: {symbol: {qty, avg_cost}} for everything Dhan says we hold
            held_map = {}
            for h in holdings:
                sym = h.get("tradingSymbol") or h.get("symbol", "")
                qty = int(h.get("availableQty", 0) or h.get("totalQty", 0) or 0)
                if sym and qty > 0:
                    # Dhan uses different field names across SDK versions — try all known variants
                    avg_cost = float(
                        h.get("avgCostPrice") or h.get("costPrice") or
                        h.get("buyAvg") or h.get("avgBuyPrice") or 0
                    )
                    held_map[sym] = {"qty": qty, "avg_cost": avg_cost}

        except Exception as e:
            log.error(f"Broker sync failed: {e}")
            return  # if we can't get holdings, don't make any DB changes — safer to do nothing

        # Build the current DB open trades as a map for easy lookup
        db_open = {t["symbol"]: t for t in self.db.get_open_trades()}

        # --- Direction 1: DB OPEN but not in Dhan → broker already closed it ---
        for symbol, trade in db_open.items():
            if symbol in held_map:
                continue  # still in Dhan → still open → no action needed

            # Symbol is in our DB as OPEN but Dhan no longer has it in holdings.
            # Conclusion: the broker's exchange-level SL order fired and we were sold out.
            # Default exit price = the current_sl stored in DB (the SL level that triggered)
            exit_price = trade["current_sl"]

            # Try to find the actual fill price from the order history (may be slightly different from SL)
            try:
                orders_resp = self.order_mgr.dhan.get_order_list()
                # Walk through orders in reverse (newest first) to find the most recent SELL fill for this symbol
                for o in reversed(orders_resp.get("data", [])):
                    if (o.get("tradingSymbol") == symbol and
                            o.get("transactionType") == "SELL" and
                            o.get("orderStatus") == "TRADED"):
                        exit_price = float(o.get("tradedPrice") or o.get("price") or exit_price)
                        break  # found the fill — stop looking
            except Exception as e:
                log.warning(f"Could not fetch order history for {symbol}: {e}")
                # Continue with the SL price as the exit price

            # Calculate net PnL after charges for the trade that just closed
            pnl = ChargesCalculator.calculate_trade_pnl(
                trade["entry_price"], exit_price, trade["remaining_qty"]
            )

            # Mark the trade CLOSED in our DB with the actual exit details
            self.db.close_trade(
                trade["trade_id"], exit_price, "BROKER_EXECUTED",
                pnl["net_pnl"], pnl["gross_pnl"], pnl["total_charges"]
            )
            log.info(f"SYNC ← {symbol} closed by broker @ ₹{exit_price:.2f} | Net: ₹{pnl['net_pnl']:.2f}")

            # Start the 2-hour loss cooldown — prevents revenge trading after an SL hit
            self.protection.start_loss_cooldown()

        # --- Direction 2: Dhan has holding not in DB → import it ---
        for symbol, holding in held_map.items():
            if symbol in db_open:
                continue  # already tracked in our DB → no action needed

            # This symbol is in Dhan but not in our DB.
            # Could be: manual buy in Dhan app, or a crash happened after the order was placed but before DB write.
            if symbol not in self.order_mgr.SECURITY_IDS:
                # Not one of our 15 watchlist stocks — could be a mutual fund, ETF, or other stock.
                # Ignore it so we don't accidentally manage positions we didn't intend to trade.
                continue

            avg_cost = holding["avg_cost"]
            qty      = holding["qty"]
            if avg_cost <= 0:
                log.warning(f"SYNC → {symbol} found in Dhan but avg_cost unknown — skipping import")
                continue

            # Look up the most recent ATR for this stock from our snapshots table.
            # ATR is needed to set a meaningful SL for the imported position.
            atr_row = self.db.fetchone(
                "SELECT atr FROM stock_snapshots WHERE symbol=? AND atr > 0 ORDER BY date DESC LIMIT 1",
                (symbol,)
            )
            if atr_row and atr_row["atr"]:
                atr    = atr_row["atr"]
                # Use SWING multiplier (1.5) as the default — it's the most conservative general SL
                sl     = round(avg_cost - atr * Config.ATR_MULT["SWING"], 2)
                target = round(avg_cost + (avg_cost - sl) * Config.MIN_RR_RATIO, 2)
                sl_method = f"ATR {atr:.2f} × {Config.ATR_MULT['SWING']}"
            else:
                # No ATR data available (e.g. first ever run, or snapshot table empty).
                # Fall back to a 5% SL and 10% target as safe defaults.
                sl     = round(avg_cost * 0.95, 2)
                target = round(avg_cost * 1.10, 2)
                sl_method = "5% fallback (no ATR data yet)"

            # Safety bounds: SL must be between 90% of cost and just below cost.
            # Prevents absurd SL values if ATR calculation produced something extreme.
            sl = max(sl, round(avg_cost * 0.90, 2))   # never more than 10% below cost
            sl = min(sl, avg_cost - 0.01)              # never above or equal to entry

            # Create a trade ID that signals this was imported, not placed by the system
            trade_id = f"IMPORTED_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            trade = Trade(
                trade_id=trade_id,
                symbol=symbol,
                strategy="IMPORTED",
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                entry_price=avg_cost,
                quantity=qty,
                initial_sl=sl,
                initial_target=target,
                current_sl=sl,
                current_price=avg_cost,
                remaining_qty=qty,
                market_mode_at_entry="IMPORTED",
            )
            self.db.save_trade(trade)
            log.info(f"SYNC → {symbol} imported | {qty}×₹{avg_cost:.2f} | SL ₹{sl:.2f} ({sl_method}) | T ₹{target:.2f}")

        # --- Direction 3: DB OPEN + in Dhan but SL order missing → re-place SL ---
        # Re-fetch after imports so newly imported trades are also checked for missing SL orders.
        db_open = {t["symbol"]: t for t in self.db.get_open_trades()}
        for symbol, trade in db_open.items():
            if symbol not in held_map:
                continue  # not in Dhan — already handled in direction 1

            if trade["sl_order_id"]:
                continue  # SL order already exists — nothing to do

            # SL order ID is missing — means either:
            # 1. The initial order placement succeeded but saving sl_order_id to DB failed (crash)
            # 2. The SL order was placed but its ID was lost somehow
            # Either way: re-place the SL at the current SL level stored in DB
            sl = trade["current_sl"]
            if not sl or sl <= 0:
                continue  # no SL level to use — skip rather than place a ₹0 SL

            new_id = self.order_mgr.replace_sl_order(symbol, trade["remaining_qty"], sl, "")
            if new_id:
                self.db.update_sl_order_id(trade["trade_id"], new_id)
                log.info(f"SYNC → {symbol} SL order re-placed @ ₹{sl:.2f} | id: {new_id}")

    def monitor_all_trades(self, current_prices: dict, vix: float):
        """
        Main monitoring loop called every 15 mins from step6_monitor_trades().
        For each open trade:
        1. Updates current_price in the DB
        2. Checks exit conditions (SL hit / VIX / time / event)
        3. Checks if trailing SL should move up (tier 1/2/3 progression)
        Always syncs with broker first to ensure we're not monitoring already-closed trades.
        """
        # Always sync first — if broker closed a trade since the last cycle, we need to know
        # before we try to check exits or update trailing SL for that position.
        self.sync_with_broker()

        open_trades = self.db.get_open_trades()
        if not open_trades:
            return  # nothing to monitor

        log.info(f"Monitoring {len(open_trades)} open trades...")
        for trade in open_trades:
            price = current_prices.get(trade["symbol"], 0)
            if price <= 0:
                continue  # couldn't get price for this stock — skip this cycle, try next time

            # Update current_price in DB every cycle.
            # This is used for unrealised PNL in the dashboard: (current_price - entry) × qty.
            self.db.execute(
                "UPDATE trades SET current_price=? WHERE trade_id=?",
                (price, trade["trade_id"])
            )

            # Check if any exit condition has been triggered (SL hit, VIX spike, time, event)
            self._check_exit_conditions(trade, price, vix)

            # Check if the trailing SL should advance to the next tier
            self._update_trailing_sl(trade, price)

    def _check_exit_conditions(self, trade: dict, price: float, vix: float):
        trade_id = trade["trade_id"]
        symbol   = trade["symbol"]
        sl       = trade["current_sl"]      # current stop-loss level (starts at initial_sl, moves up with trailing)
        entry    = trade["entry_price"]
        qty      = trade["remaining_qty"]   # shares still held (decreases after tier 2 partial sell)

        # --- Exit 1: SL hit (backtest mode only) ---
        # In live/sandbox mode, the broker's exchange-level SL order handles this automatically.
        # The broker fires a SELL order when price touches sl_price. We detect it via sync_with_broker().
        # In backtest mode there's no broker, so we manually check if price crossed the SL.
        if Config.BACKTEST_MODE and price <= sl:
            log.warning(f"SL HIT: {symbol} @ ₹{price:.2f} (SL: ₹{sl:.2f})")
            self._execute_exit(trade, price, qty, ExitReason.SL_HIT.value)
            self.protection.start_loss_cooldown()  # 2-hour pause after a loss
            return  # return immediately — no point checking other conditions on a closed trade

        # --- Exit 2: VIX spike ---
        # If VIX crosses 22 (VIX_NERVOUS threshold), market conditions have deteriorated sharply.
        # All open positions are exited immediately regardless of profit — chaos makes SLs unreliable.
        if vix > Config.VIX_NERVOUS:
            log.warning(f"VIX SPIKE EXIT: {symbol} | VIX: {vix:.1f}")
            self._execute_exit(trade, price, qty, ExitReason.MARKET_CRASH.value)
            return

        # --- Exit 3: Time-based exit ---
        # If a trade has been open for MAX_HOLD_DAYS (15 days) and is NOT showing a profit,
        # it's "dead money" — capital tied up in a trade going nowhere.
        # We exit to free up capital for better opportunities.
        # Note: we DON'T exit if the trade is in profit (even after 15 days) — let winners run.
        entry_date   = datetime.strptime(trade["entry_date"], "%Y-%m-%d")
        holding_days = (datetime.now() - entry_date).days
        if holding_days >= Config.MAX_HOLD_DAYS and price <= entry:
            log.info(f"TIME EXIT: {symbol} held {holding_days} days without profit")
            self._execute_exit(trade, price, qty, ExitReason.TIME_BASED.value)
            return

        # --- Exit 4: Upcoming earnings/results event ---
        # check_event_guard() looks at events_calendar for this symbol.
        # If an event is within 5 days (EVENT_EXIT_DAYS), it returns (False, "Exit! ...")
        # We exit now to avoid holding through a gap-risk event (earnings can move ±10%).
        safe, msg = self.protection.check_event_guard(symbol)
        if not safe:
            log.info(f"EVENT EXIT: {symbol} — {msg}")
            self._execute_exit(trade, price, qty, ExitReason.EVENT_EXIT.value)

    def _update_trailing_sl(self, trade: dict, price: float):
        """
        Checks all 3 tier conditions and advances to the next tier if triggered.
        Only one tier can advance per monitoring cycle (elif chain).
        Each tier update:
        1. Updates current_sl in the DB (update_trade_sl)
        2. Logs the SL move to trailing_sl_log for post-trade review
        3. Cancels the old broker SL order and places a new one (replace_sl_order)
        4. Saves the new order ID in the DB (update_sl_order_id)

        atr is fetched from the most recent stock_snapshots row —
        not stored in the trade record because ATR changes daily as volatility changes.
        """
        trade_id = trade["trade_id"]
        entry    = trade["entry_price"]
        sl       = trade["current_sl"]       # current trailing SL level
        target   = trade["initial_target"]   # initial 2:1 target price (not updated as price moves)
        qty      = trade["remaining_qty"]    # shares still held
        atr      = self._get_atr(trade["symbol"])  # today's ATR for this stock (used in tier 3 trail)
        risk     = entry - trade["initial_sl"]     # initial risk per share (entry - original SL)

        if risk <= 0:
            return  # invalid trade data — skip (shouldn't happen but guard anyway)

        # ============================================================
        # TIER 1: Move SL to breakeven when price reaches 1:1 profit
        # Condition: price >= entry + risk (i.e. profit equals our risk amount)
        # Action: SL moves up to entry price (breakeven)
        # Effect: from this point on, worst case = 0 loss (not counting charges)
        # ============================================================
        if not trade["tier1_done"] and price >= entry + risk and sl < entry:
            log.info(f"TIER 1 → Breakeven: {trade['symbol']}")
            self.db.update_trade_sl(trade_id, entry)   # move SL to entry in DB
            self.db.log_trailing_sl(trade_id, sl, entry, price, "TIER1_BREAKEVEN")  # audit log
            # Mark tier1 as done and record the price at which it triggered
            self.db.execute("UPDATE trades SET tier1_done=1, tier1_price=? WHERE trade_id=?",
                            (price, trade_id))
            # Cancel the old SL order at the broker and place a new one at entry price
            new_id = self.order_mgr.replace_sl_order(trade["symbol"], qty, entry, trade["sl_order_id"])
            if new_id:
                self.db.update_sl_order_id(trade_id, new_id)  # store new order ID for next cancel+replace

        # ============================================================
        # TIER 1.5: Adaptive trail between breakeven and target
        # Condition: tier1 done, tier2 not yet done, price between entry and target
        # Action: SL = entry + 50% of current gain above entry
        # Effect: as price rises, SL locks in 50% of each new gain — rises continuously
        # Example: entry=₹100, price=₹110, SL = 100 + (110-100)×0.5 = ₹105
        # ============================================================
        elif trade["tier1_done"] and not trade["tier2_done"] and price < target:
            adaptive_sl = entry + (price - entry) * 0.5  # locks 50% of gain above entry
            if adaptive_sl > sl:
                # Only move SL up, never down — trailing SL can only tighten, never loosen
                log.info(f"ADAPTIVE TRAIL: {trade['symbol']} SL {sl:.2f} → {adaptive_sl:.2f} "
                         f"(price ₹{price:.2f}, protecting 50% of ₹{price - entry:.2f} gain)")
                self.db.update_trade_sl(trade_id, adaptive_sl)
                self.db.log_trailing_sl(trade_id, sl, adaptive_sl, price, "ADAPTIVE_TRAIL")
                new_id = self.order_mgr.replace_sl_order(trade["symbol"], qty, adaptive_sl, trade["sl_order_id"])
                if new_id:
                    self.db.update_sl_order_id(trade_id, new_id)

        # ============================================================
        # TIER 2: Partial sell (50%) when price hits the initial 2:1 target
        # Condition: tier1 done, tier2 not done, price >= target
        # Action: sell half the remaining shares, tighten SL near the target level
        # Effect: locked in profit on 50%, remaining 50% still running with tighter protection
        # ============================================================
        elif not trade["tier2_done"] and trade["tier1_done"] and price >= target:
            exit_qty  = max(1, qty // 2)    # sell half, minimum 1 share
            remaining = qty - exit_qty       # shares that stay in the position

            # Before selling, check if the partial exit is profitable after charges.
            # The DP charge (₹15.34 flat) on a tiny partial sell can make it net-negative.
            partial_pnl = ChargesCalculator.calculate_trade_pnl(entry, price, exit_qty)
            if partial_pnl["net_pnl"] <= 0:
                # Partial sell would be a net loss after charges — skip the sell,
                # but still mark tier2 done and start trailing with 1×ATR.
                log.info(f"TIER 2 SKIP: {trade['symbol']} partial sell of {exit_qty} shares "
                         f"not profitable after charges (net ₹{partial_pnl['net_pnl']:.2f}). "
                         f"Trailing full qty instead.")
                new_trail_sl = price - atr  # trail 1×ATR below current price with full qty
                if new_trail_sl > sl:
                    self.db.update_trade_sl(trade_id, new_trail_sl)
                    self.db.log_trailing_sl(trade_id, sl, new_trail_sl, price, "TIER2_SKIP_TRAIL")
                    new_id = self.order_mgr.replace_sl_order(trade["symbol"], qty, new_trail_sl, trade["sl_order_id"])
                    if new_id:
                        self.db.update_sl_order_id(trade_id, new_id)
                # Mark tier2 done with qty=0 (no partial sell happened) and keep full qty
                self.db.execute("UPDATE trades SET tier2_done=1, tier2_price=?, tier2_qty=0, remaining_qty=? WHERE trade_id=?",
                                (price, qty, trade_id))
                return

            # Execute the partial sell order
            log.info(f"TIER 2 PARTIAL EXIT: {trade['symbol']} selling {exit_qty}")
            if self.order_mgr.place_sell_order(trade["symbol"], exit_qty, price, "TIER2_PARTIAL_EXIT"):
                # Tighten SL to just below the target level: target - 0.5×ATR.
                # Ensures if price pulls back from the target, we exit near it (not at breakeven).
                new_sl = max(sl, target - (atr * 0.5))
                self.db.execute(
                    "UPDATE trades SET tier2_done=1, tier2_price=?, tier2_qty=?, remaining_qty=?, current_sl=? WHERE trade_id=?",
                    (price, exit_qty, remaining, new_sl, trade_id)
                )
                self.db.log_trailing_sl(trade_id, sl, new_sl, price, "TIER2_PARTIAL_EXIT")
                # Update the broker SL order to protect only the remaining shares
                new_id = self.order_mgr.replace_sl_order(trade["symbol"], remaining, new_sl, trade["sl_order_id"])
                if new_id:
                    self.db.update_sl_order_id(trade_id, new_id)
                pnl = ChargesCalculator.calculate_trade_pnl(entry, price, exit_qty)
                log.info(f"TIER 2 Net PNL so far: ₹{pnl['net_pnl']:.2f}")

        # ============================================================
        # TIER 3: Dynamic trail with 1×ATR after tier 2
        # Condition: tier2 done, still have shares remaining
        # Action: SL = current_price - 1×ATR (recalculated each cycle)
        # Effect: SL rises as price rises, giving the trade room to breathe
        #         but locking in more profit as new highs are made
        # Note: ATR used here is TODAY's ATR (not at-entry ATR) so the trail
        #       widens/tightens with current volatility.
        # ============================================================
        elif trade["tier2_done"] and qty > 0 and atr > 0:
            new_trail_sl = price - atr  # 1 ATR below current price
            if new_trail_sl > sl:
                # Only move SL up — never let it go down
                log.info(f"TRAIL SL: {trade['symbol']} {sl:.2f} → {new_trail_sl:.2f}")
                self.db.update_trade_sl(trade_id, new_trail_sl)
                self.db.log_trailing_sl(trade_id, sl, new_trail_sl, price, "TIER3_TRAIL")
                # Update the broker SL order to the new higher level
                new_id = self.order_mgr.replace_sl_order(trade["symbol"], qty, new_trail_sl, trade["sl_order_id"])
                if new_id:
                    self.db.update_sl_order_id(trade_id, new_id)

    def _execute_exit(self, trade: dict, price: float, qty: int, reason: str):
        """
        Places a sell order and closes the trade in DB.
        In paper mode: sell always "succeeds" (logged only).
        In live mode: if the sell API call fails, we abort — better to leave the
        position open and retry than to close it in DB but still hold shares at the broker.
        PNL calculation uses ChargesCalculator to get net_pnl after all brokerage charges.
        """
        symbol   = trade["symbol"]
        entry    = trade["entry_price"]
        trade_id = trade["trade_id"]

        # Place the sell order at the broker
        sold = self.order_mgr.place_sell_order(symbol, qty, price, reason)

        if not sold and not Config.BACKTEST_MODE:
            # Sell order failed in live mode — do NOT close the trade in DB.
            # The position is still open at the broker. Leaving DB as OPEN means
            # the next monitoring cycle will retry the exit.
            log.error(f"EXIT ORDER FAILED: {symbol}")
            return

        # Calculate net PnL: gross PnL minus all NSE delivery charges (STT, DP, etc.)
        pnl = ChargesCalculator.calculate_trade_pnl(entry, price, qty)

        # Mark trade as CLOSED in the DB with all financial details
        self.db.close_trade(trade_id, price, reason,
                            pnl["net_pnl"], pnl["gross_pnl"], pnl["total_charges"])
        log.info(f"TRADE CLOSED: {symbol} | {reason} | Net: ₹{pnl['net_pnl']:.2f}")

    def _get_atr(self, symbol: str) -> float:
        """
        Fetches the most recent ATR for a symbol from the stock_snapshots table.
        ATR changes every day as the 14-day average true range recalculates.
        Using today's ATR (not the ATR at entry) for trailing SL keeps the trail
        proportional to the stock's current volatility — wider when it's volatile,
        tighter when it's calm.
        Defaults to 20.0 if no snapshot exists (conservative fallback).
        """
        row = self.db.fetchone(
            "SELECT atr FROM stock_snapshots WHERE symbol=? ORDER BY date DESC LIMIT 1",
            (symbol,)
        )
        return row["atr"] if row and row["atr"] else 20.0  # 20.0 = ₹20 default — roughly 1.5% of a ₹1300 stock
