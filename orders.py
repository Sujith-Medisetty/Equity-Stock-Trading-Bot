"""
orders.py — All broker order placement via the Dhan API.

Three types of orders this system places:

1. Entry order (place_entry_order)
   A LIMIT BUY order placed at entry_price × 1.001 (0.1% above close) so it
   fills reliably rather than missing by a tick. CNC product type = delivery
   (we hold overnight, not intraday). Simultaneously places a STOP-LOSS SELL
   order at sl_price so the SL is sitting at the exchange from the moment we enter.
   The broker will fire the SL automatically if price falls to that level —
   we don't need to monitor and manually sell.

2. SL order cancel+replace (replace_sl_order)
   Called every time the trailing SL moves up (tier 1 breakeven, tier 2 tighten,
   tier 3 trail). Process: cancel the old SL order (identified by sl_order_id
   stored in the DB) → place a new SL order at the new level. Returns the new
   order ID which is stored back in the DB via update_sl_order_id().
   If the cancel fails (e.g. order already executed), we log a warning and
   still try to place the new SL.

3. Exit sell order (place_sell_order)
   A LIMIT SELL placed at price × 0.999 (slightly below market) for fast fills.
   Used for: tier 2 partial exit, VIX spike exit, time-based exit, event exit.
   NOT used for SL exits in live mode — those are handled by the exchange-level
   SL order placed at entry time.

Paper trade behaviour:
   All three methods log the intended order but skip the actual API call.
   Paper mode uses real market prices (fetched via get_market_feed_quote)
   so the paper trade results are realistic — only the order placement is fake.

SECURITY_IDS:
   Each NSE stock has a numeric security ID that Dhan requires for API calls.
   These are fixed values — they never change. Stored here as a simple lookup dict
   rather than making an API call to resolve them each time.
"""

from datetime import datetime
from typing import Optional

from config import Config, log, LIBS_AVAILABLE
from models import Trade, Setup
from database import DatabaseManager

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None


class OrderManager:
    """
    Handles all broker interactions for order placement.
    Maintains its own Dhan connection (separate from DataCollector's connection).
    Called by TradingSystem.step5_execute_trades() and TradeMonitor.
    """

    SECURITY_IDS = {
        "ICICIBANK": "4963",  "HDFCBANK": "1333",
        "AXISBANK":  "5900",  "INFY":     "1594",
        "HCLTECH":   "7229",  "TATAMOTORS": "3456",
        "MARUTI":    "10999", "RELIANCE": "2885",
        "BHARTIARTL": "10604","SUNPHARMA": "3351",
        "BAJFINANCE": "317",  "LT":       "11483",
        "ITC":       "1660",  "TITAN":    "3506",
        "TCS":       "11536",
    }

    def __init__(self, db: DatabaseManager):
        self.db   = db
        self.dhan = None
        self._init_dhan()

    def _init_dhan(self):
        if not LIBS_AVAILABLE or dhanhq is None or Config.BACKTEST_MODE:
            return
        try:
            if Config.PAPER_TRADE:
                self.dhan = dhanhq(Config.DHAN_SANDBOX_CLIENT_ID, Config.DHAN_SANDBOX_ACCESS_TOKEN)
                log.info("Dhan initialised in SANDBOX mode")
            else:
                self.dhan = dhanhq(Config.DHAN_CLIENT_ID, Config.DHAN_ACCESS_TOKEN)
                log.info("Dhan initialised in LIVE mode")
        except Exception as e:
            log.error(f"OrderManager Dhan init failed: {e}")

    def _get_security_id(self, symbol: str) -> str:
        return self.SECURITY_IDS.get(symbol, "0")

    def get_available_capital(self) -> float:
        """
        Returns the cash balance available for new trades.
        In live/sandbox mode this comes from Dhan's fund-limits API so it
        reflects your real account after existing positions are deployed.
        Falls back to Config.TOTAL_CAPITAL in backtest mode or if the API fails.
        """
        if Config.BACKTEST_MODE or not self.dhan:
            return max(0.0, float(Config.TOTAL_CAPITAL) - Config.CAPITAL_RESERVE)
        try:
            resp = self.dhan.get_fund_limits()
            data = resp.get("data", {}) if isinstance(resp, dict) else {}
            # Dhan returns the field with a typo in some SDK versions — try both
            balance = float(
                data.get("availableBalance") or
                data.get("availabelBalance") or
                data.get("net") or 0
            )
            if balance <= 0:
                log.warning("Fund limits returned zero/negative — falling back to Config.TOTAL_CAPITAL")
                return max(0.0, float(Config.TOTAL_CAPITAL) - Config.CAPITAL_RESERVE)
            return max(0.0, balance - Config.CAPITAL_RESERVE)
        except Exception as e:
            log.warning(f"Could not fetch fund limits: {e} — falling back to Config.TOTAL_CAPITAL")
            return max(0.0, float(Config.TOTAL_CAPITAL) - Config.CAPITAL_RESERVE)

    def place_entry_order(self, setup: Setup) -> Optional[str]:
        """
        Places a BUY order and immediately places an SL SELL order.
        In paper mode: skips API calls, records trade directly with simulated slippage.
        In live mode: places both orders via Dhan, stores sl_order_id in DB.
        Returns the trade_id string on success, None on failure.
        """
        trade_id = f"{setup.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        mode_tag = "[SANDBOX]" if Config.PAPER_TRADE else "[LIVE]"

        # Apply slippage to paper entries — simulate worse fills than LTP.
        # Buy fills are higher (adverse), making paper PNL more conservative.
        if Config.PAPER_TRADE and not Config.BACKTEST_MODE:
            setup.entry_price = round(setup.entry_price * (1 + Config.PAPER_SLIPPAGE_PCT), 2)

        if Config.BACKTEST_MODE:
            log.info(f"[BACKTEST] BUY {setup.shares}×{setup.symbol} @ ₹{setup.entry_price:.2f} "
                     f"SL:₹{setup.sl_price:.2f} T:₹{setup.target_price:.2f}")
            self._record_trade(trade_id, setup)
            return trade_id

        if not self.dhan:
            log.error("Dhan API not initialized")
            return None

        try:
            self.dhan.place_order(
                security_id=self._get_security_id(setup.symbol),
                exchange_segment=self.dhan.NSE,
                transaction_type=self.dhan.BUY,
                quantity=setup.shares,
                order_type=self.dhan.LIMIT,
                product_type=self.dhan.CNC,
                price=round(setup.entry_price * 1.001, 2)
            )
            sl_order = self.dhan.place_order(
                security_id=self._get_security_id(setup.symbol),
                exchange_segment=self.dhan.NSE,
                transaction_type=self.dhan.SELL,
                quantity=setup.shares,
                order_type=self.dhan.SL,
                product_type=self.dhan.CNC,
                price=setup.sl_price,
                trigger_price=setup.sl_price
            )
            sl_order_id = str(
                (sl_order.get("data") or {}).get("orderId") or
                sl_order.get("orderId", "")
            )
            log.info(f"{mode_tag} orders placed: {setup.symbol} | SL order id: {sl_order_id}")
            self._record_trade(trade_id, setup, sl_order_id)
            return trade_id

        except Exception as e:
            log.error(f"Order placement failed for {setup.symbol}: {e}")
            return None

    def _record_trade(self, trade_id: str, setup: Setup, sl_order_id: str = ""):
        """Creates a Trade object and saves it to DB. Called after successful order placement.
        remaining_qty starts equal to quantity — it decreases as partial exits happen at tier 2."""
        trade = Trade(
            trade_id=trade_id,
            symbol=setup.symbol,
            strategy=setup.strategy.value,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            entry_price=setup.entry_price,
            quantity=setup.shares,
            initial_sl=setup.sl_price,
            initial_target=setup.target_price,
            current_sl=setup.sl_price,
            current_price=setup.entry_price,
            remaining_qty=setup.shares,
            setup_score=setup.score,
            market_mode_at_entry=setup.market_mode,
            sl_order_id=sl_order_id
        )
        self.db.save_trade(trade)
        log.info(f"Trade recorded: {trade_id}")

    def replace_sl_order(self, symbol: str, qty: int,
                          new_sl: float, old_order_id: str) -> str:
        """
        Cancels the old exchange-level SL order and places a new one at new_sl.
        Called whenever the trailing SL moves up: tier 1 (breakeven), tier 2 (tighten),
        or tier 3 (1×ATR trail after each new high).

        Retries up to SL_REPLACE_MAX_RETRIES times on failure. If all retries fail,
        places an emergency market sell to prevent an unprotected position.

        Returns the new Dhan order ID so it can be saved back to the DB.
        Returns "" in paper mode or if the API call fails.
        """
        if Config.BACKTEST_MODE:
            log.info(f"[BACKTEST] SL replaced: {symbol} → ₹{new_sl:.2f} qty {qty}")
            return ""
        if not self.dhan:
            return ""

        mode_tag = "[SANDBOX]" if Config.PAPER_TRADE else "[LIVE]"

        if old_order_id:
            try:
                self.dhan.cancel_order(order_id=old_order_id)
                log.info(f"Cancelled SL order {old_order_id} for {symbol}")
            except Exception as e:
                log.warning(f"Could not cancel SL order {old_order_id}: {e}")

        import time
        for attempt in range(1, Config.SL_REPLACE_MAX_RETRIES + 1):
            try:
                sl_order = self.dhan.place_order(
                    security_id=self._get_security_id(symbol),
                    exchange_segment=self.dhan.NSE,
                    transaction_type=self.dhan.SELL,
                    quantity=qty,
                    order_type=self.dhan.SL,
                    product_type=self.dhan.CNC,
                    price=new_sl,
                    trigger_price=new_sl
                )
                new_id = str(
                    (sl_order.get("data") or {}).get("orderId") or
                    sl_order.get("orderId", "")
                )
                log.info(f"{mode_tag} New SL order placed: {symbol} @ ₹{new_sl:.2f} | id: {new_id}")
                return new_id
            except Exception as e:
                log.error(f"SL placement attempt {attempt}/{Config.SL_REPLACE_MAX_RETRIES} "
                          f"failed for {symbol}: {e}")
                if attempt < Config.SL_REPLACE_MAX_RETRIES:
                    time.sleep(Config.SL_REPLACE_RETRY_DELAY)

        log.critical(
            f"EMERGENCY: All {Config.SL_REPLACE_MAX_RETRIES} SL placement attempts failed for "
            f"{symbol}. Placing emergency MARKET SELL to protect capital."
        )
        try:
            self.dhan.place_order(
                security_id=self._get_security_id(symbol),
                exchange_segment=self.dhan.NSE,
                transaction_type=self.dhan.SELL,
                quantity=qty,
                order_type=self.dhan.MARKET,
                product_type=self.dhan.CNC,
                price=0
            )
            log.critical(f"EMERGENCY SELL placed for {symbol} × {qty} shares")
        except Exception as e2:
            log.critical(f"EMERGENCY SELL ALSO FAILED for {symbol}: {e2} — POSITION UNPROTECTED")
        return ""

    def place_sell_order(self, symbol: str, qty: int,
                          price: float, reason: str) -> bool:
        """
        Places a LIMIT SELL slightly below market (price × 0.999) for fast fills.
        In paper mode: applies slippage simulation so paper exits are conservative.
        Used for: tier 2 partial exit, VIX spike exit, time-based exit, event exit.
        NOT for SL exits — in live mode, the exchange-level SL order handles those.
        Returns True on success (or in paper mode), False on API failure.
        """
        if Config.BACKTEST_MODE:
            log.info(f"[BACKTEST] SELL {qty}×{symbol} @ ₹{price:.2f} | {reason}")
            return True
        if not self.dhan:
            return False

        # Sell fills are lower (adverse) in paper mode — simulate real slippage.
        if Config.PAPER_TRADE:
            price = round(price * (1 - Config.PAPER_SLIPPAGE_PCT), 2)

        mode_tag = "[SANDBOX]" if Config.PAPER_TRADE else "[LIVE]"
        log.info(f"{mode_tag} SELL {qty}×{symbol} @ ₹{price:.2f} | {reason}")
        try:
            self.dhan.place_order(
                security_id=self._get_security_id(symbol),
                exchange_segment=self.dhan.NSE,
                transaction_type=self.dhan.SELL,
                quantity=qty,
                order_type=self.dhan.LIMIT,
                product_type=self.dhan.CNC,
                price=round(price * 0.999, 2)
            )
            return True
        except Exception as e:
            log.error(f"Sell order failed {symbol}: {e}")
            return False
