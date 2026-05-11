"""
orders.py — All broker order placement via the Upstox API.

Three types of orders this system places:

1. Entry order (place_entry_order)
   A LIMIT BUY at entry_price × 1.001 (0.1% above close) for reliable fills.
   Simultaneously places a STOP-LOSS SELL at sl_price — exchange-level protection
   that fires automatically even if the bot goes offline.

2. SL order cancel + replace (replace_sl_order)
   Called whenever the trailing SL moves up (tier 1/2/3).
   Cancels the old SL order and places a new one at the new level.
   Retries up to SL_REPLACE_MAX_RETRIES times; places an emergency MARKET SELL
   if all retries fail so the position is never left unprotected.

3. Exit sell order (place_sell_order)
   LIMIT SELL at price × 0.999 (0.1% below market) for fast fills.
   Used for tier 2 partial exit, VIX exit, time exit, event exit.
   NOT used for SL exits — the exchange-level SL order handles those in live mode.

API versions used:
   Order placement (place/emergency) → Upstox V3 via direct HTTP to api-hft.upstox.com.
     V3 response returns order_ids[] (array) instead of order_id (string).
   Cancel order, holdings, order book, fund/margin → V2 SDK (no V3 variant documented).
   Market quotes (LTP, OHLC) → V3 via direct HTTP to api.upstox.com.

Paper trade behaviour:
   All three methods log the intended order but skip the actual API call.
   Live prices are always fetched from Upstox (real prices in both paper and live mode).

Upstox instrument keys:
   Upstox identifies stocks by "NSE_EQ|{ISIN}" format.
   INSTRUMENT_KEYS maps our symbols to these keys (shared with data_collector.py).

Holdings / order book:
   get_holdings() and get_order_list() return standardised Python dicts so
   TradeMonitor.sync_with_broker() doesn't need to know Upstox's internal format.
"""

import time
from datetime import datetime
from typing import Optional, List

from config import Config, log, LIBS_AVAILABLE, UPSTOX_AVAILABLE
from data_collector import INSTRUMENT_KEYS, _V3_BASE, _with_retry
from models import Trade, Setup
from database import DatabaseManager

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    import upstox_client
    from upstox_client.api import OrderApi, PortfolioApi, UserApi
    from upstox_client.rest import ApiException
except ImportError:
    upstox_client = None
    OrderApi      = None
    PortfolioApi  = None
    UserApi       = None
    ApiException  = Exception

# V3 order placement uses a dedicated HFT subdomain — different from the standard API host.
_V3_HFT_BASE = "https://api-hft.upstox.com/v3"


class OrderManager:
    """
    Handles all broker interactions for order placement and account queries.
    Maintains its own Upstox API client (separate from DataCollector's client).
    Called by TradingSystem.step5_execute_trades() and TradeMonitor.
    """

    def __init__(self, db: DatabaseManager):
        self.db      = db
        self._client = None   # Upstox ApiClient — initialised after auth
        self._auth   = None
        self._init_upstox()

    def _init_upstox(self):
        """
        Connects to Upstox using the access token from UpstoxAuth.
        In BACKTEST_MODE: no broker connection needed — all operations are simulated.
        In PAPER_TRADE mode: uses real Upstox API for prices + order flow, but
        orders go to a paper/sandbox account (same API key, different account type).
        """
        if not UPSTOX_AVAILABLE or not LIBS_AVAILABLE or Config.BACKTEST_MODE:
            return
        try:
            from upstox_auth import UpstoxAuth
            self._auth   = UpstoxAuth(self.db)
            token = self._auth.get_valid_token()
            if not token:
                log.warning("OrderManager: no Upstox token — order placement will be skipped.")
                return
            self._client = self._build_client(token)
            mode = "PAPER" if Config.PAPER_TRADE else "LIVE"
            log.info(f"Upstox OrderManager initialised [{mode} mode].")
        except Exception as e:
            log.error(f"OrderManager Upstox init failed: {e}")

    def _build_client(self, token: str) -> "upstox_client.ApiClient":
        cfg = upstox_client.Configuration()
        cfg.access_token = token
        return upstox_client.ApiClient(cfg)

    def _refresh_client_if_needed(self):
        """Refreshes the API client token before critical operations (order placement)."""
        if not self._auth or not self._client:
            return
        token = self._auth.get_valid_token()
        if token:
            self._client = self._build_client(token)

    def _get_token(self) -> str:
        """Returns the current access token from the configured API client."""
        return self._client.configuration.access_token if self._client else ""

    def _ikey(self, symbol: str) -> str:
        """Returns the Upstox instrument key for a symbol. Raises ValueError if not mapped."""
        key = INSTRUMENT_KEYS.get(symbol)
        if not key:
            raise ValueError(f"No instrument key mapped for symbol '{symbol}'")
        return key

    def _post_v3_order(self, payload: dict) -> str:
        """
        Places an order via the Upstox V3 HFT endpoint and returns the first order_id.
        V3 uses api-hft.upstox.com (not api.upstox.com) — direct HTTP required.
        V3 response: {"data": {"order_ids": ["id1", ...]}} — takes the first element.
        Raises IOError with .status attribute on non-2xx responses (for @_with_retry).
        """
        url     = f"{_V3_HFT_BASE}/order/place"
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        resp = _requests.post(url, json=payload, headers=headers, timeout=10)
        if not resp.ok:
            err = IOError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            err.status = resp.status_code
            raise err
        ids = resp.json().get("data", {}).get("order_ids", [])
        return str(ids[0]) if ids else ""

    # -------------------------------------------------------------------------
    # Capital
    # -------------------------------------------------------------------------

    def get_available_capital(self) -> float:
        """
        Returns available cash for new trades from Upstox's fund/margin API.
        Falls back to Config.TOTAL_CAPITAL in backtest mode or on API failure.
        """
        if Config.BACKTEST_MODE or not self._client:
            return max(0.0, float(Config.TOTAL_CAPITAL))
        try:
            api  = UserApi(self._client)
            resp = api.get_fund_and_margin(api_version="2.0", segment="SEC")
            equity = resp.data.equity if resp and resp.data else None
            if equity:
                balance = float(
                    getattr(equity, "available_margin", None) or
                    getattr(equity, "net",              None) or
                    getattr(equity, "payin_amount",     None) or 0
                )
                if balance > 0:
                    return balance
            log.warning("Fund/margin API returned zero — falling back to Config.TOTAL_CAPITAL")
        except Exception as e:
            log.warning(f"get_available_capital failed: {e} — falling back to Config.TOTAL_CAPITAL")
        return max(0.0, float(Config.TOTAL_CAPITAL))

    def get_brokerage(self, symbol: str, qty: int, price: float, transaction_type: str) -> Optional[dict]:
        """
        Fetches exact brokerage and statutory charges from Upstox V2 API.
        Returns a dict with all charge components, or None on failure (caller falls back to manual).

        transaction_type: "BUY" or "SELL"
        product: always "D" (delivery) for this system.

        Response fields used:
          charges.total          → total round-trip charges
          charges.brokerage      → broker commission
          charges.taxes.stt      → Securities Transaction Tax
          charges.taxes.gst      → GST
          charges.taxes.stamp_duty
          charges.other_charges.transaction  → NSE transaction charge
          charges.other_charges.sebi_turnover
          charges.other_charges.clearing
          charges.other_charges.ipft
          charges.dp_plan.min_expense        → DP charge (sell side only)
        """
        if Config.BACKTEST_MODE or not self._client or not _requests:
            return None
        instrument_token = INSTRUMENT_KEYS.get(symbol)
        if not instrument_token:
            return None
        try:
            self._refresh_client_if_needed()
            url = "https://api.upstox.com/v2/charges/brokerage"
            params = {
                "instrument_token": instrument_token,
                "quantity":         qty,
                "product":          "D",
                "transaction_type": transaction_type.upper(),
                "price":            price,
            }
            headers = {
                "Authorization": f"Bearer {self._get_token()}",
                "Accept":        "application/json",
            }
            resp = _requests.get(url, params=params, headers=headers, timeout=10)
            if not resp.ok:
                log.warning(f"get_brokerage HTTP {resp.status_code} for {symbol}")
                return None
            ch = resp.json().get("data", {}).get("charges", {})
            taxes       = ch.get("taxes", {})
            other       = ch.get("other_charges", {})
            dp_plan     = ch.get("dp_plan", {})
            return {
                "total":        float(ch.get("total", 0)),
                "brokerage":    float(ch.get("brokerage", 0)),
                "stt":          float(taxes.get("stt", 0)),
                "gst":          float(taxes.get("gst", 0)),
                "stamp_duty":   float(taxes.get("stamp_duty", 0)),
                "transaction":  float(other.get("transaction", 0)),
                "sebi":         float(other.get("sebi_turnover", 0)),
                "clearing":     float(other.get("clearing", 0)),
                "ipft":         float(other.get("ipft", 0)),
                "dp_charge":    float(dp_plan.get("min_expense", 0)),
            }
        except Exception as e:
            log.warning(f"get_brokerage failed for {symbol}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Live prices — V3 batch endpoints (used by system.py and monitor.py)
    # -------------------------------------------------------------------------

    def get_all_live_prices(self, symbols: list) -> dict:
        """
        Batch LTP fetch — returns {symbol: price} for all given symbols in ONE V3 API call.
        V3 LTP also returns volume and cp (prev close) natively alongside last_price.
        V3 response keys use colon separator (NSE_EQ:ISIN) — normalised to pipe for lookup.
        Symbols where fetch failed are absent from the returned dict.
        Used by monitoring (only needs price, fast).
        """
        if not self._client or not _requests:
            return {}
        key_to_sym = {INSTRUMENT_KEYS[s]: s for s in symbols if s in INSTRUMENT_KEYS}
        if not key_to_sym:
            return {}
        try:
            url     = f"{_V3_BASE}/market-quote/ltp"
            params  = {"instrument_key": ",".join(key_to_sym.keys())}
            headers = {
                "Authorization": f"Bearer {self._get_token()}",
                "Accept":        "application/json",
            }
            resp = _requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            prices = {}
            for raw_key, quote in resp.json().get("data", {}).items():
                sym   = key_to_sym.get(raw_key.replace(":", "|"))
                price = quote.get("last_price") if isinstance(quote, dict) else None
                if sym and price and float(price) > 0:
                    prices[sym] = float(price)
            return prices
        except Exception as e:
            log.error(f"Batch LTP fetch failed: {e}")
            return {}

    def get_all_live_quotes(self, symbols: list) -> dict:
        """
        Batch full-quote fetch — returns {symbol: {"price": float, "volume": float, "open": float}}.
        Uses the V3 OHLC endpoint (interval=1d) which returns live_ohlc with today's volume
        and session open — eliminating the need for a separate full market quote call.
        V3 response keys use colon separator (NSE_EQ:ISIN) — normalised to pipe for lookup.
        Falls back to LTP-only data (volume=0) if the richer call fails, so the midday
        scan still runs on price alone rather than being skipped entirely.
        """
        if not self._client or not _requests:
            return {}
        key_to_sym = {INSTRUMENT_KEYS[s]: s for s in symbols if s in INSTRUMENT_KEYS}
        if not key_to_sym:
            return {}
        try:
            url     = f"{_V3_BASE}/market-quote/ohlc"
            params  = {"instrument_key": ",".join(key_to_sym.keys()), "interval": "1d"}
            headers = {
                "Authorization": f"Bearer {self._get_token()}",
                "Accept":        "application/json",
            }
            resp = _requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            quotes = {}
            for raw_key, quote in resp.json().get("data", {}).items():
                sym   = key_to_sym.get(raw_key.replace(":", "|"))
                price = quote.get("last_price", 0) if isinstance(quote, dict) else 0
                if not sym or not price or float(price) <= 0:
                    continue
                # V3 OHLC nests today's session data under live_ohlc.
                # live_ohlc.volume = total traded volume today (not bid-side depth).
                # live_ohlc.open   = today's session open (for green candle check in midday scan).
                live_ohlc  = quote.get("live_ohlc") or {}
                volume     = float(live_ohlc.get("volume", 0) or 0)
                today_open = float(live_ohlc.get("open",   0) or 0)
                quotes[sym] = {
                    "price":  float(price),
                    "volume": volume,
                    "open":   today_open,
                }
            return quotes
        except Exception as e:
            log.warning(f"get_all_live_quotes failed: {e} — falling back to LTP only")
            prices = self.get_all_live_prices(symbols)
            return {sym: {"price": p, "volume": 0.0, "open": 0.0} for sym, p in prices.items()}

    # -------------------------------------------------------------------------
    # Holdings & order book (used by TradeMonitor.sync_with_broker)
    # -------------------------------------------------------------------------

    def get_holdings(self) -> List[dict]:
        """
        Returns current demat holdings from Upstox as a list of standardised dicts:
          [{symbol, qty, avg_cost}, ...]
        Returns [] if the API fails — caller (sync_with_broker) handles gracefully.
        """
        if Config.BACKTEST_MODE or not self._client:
            return []
        try:
            api  = PortfolioApi(self._client)
            resp = api.get_holdings(api_version="2.0")
            holdings = []
            for h in (resp.data or []):
                sym = getattr(h, "tradingsymbol", None)
                qty = int(getattr(h, "quantity", 0) or 0)
                avg = float(getattr(h, "average_price", 0) or 0)
                if sym and qty > 0:
                    holdings.append({"symbol": sym, "qty": qty, "avg_cost": avg})
            return holdings
        except Exception as e:
            log.error(f"get_holdings failed: {e}")
            return []

    def get_order_list(self) -> List[dict]:
        """
        Returns today's order book from Upstox as standardised dicts:
          [{symbol, transaction_type, status, average_price}, ...]
        Returns [] on failure. Used by sync_with_broker to find filled SL exits.
        """
        if Config.BACKTEST_MODE or not self._client:
            return []
        try:
            api  = OrderApi(self._client)
            resp = api.get_order_book(api_version="2.0")
            orders = []
            for o in (resp.data or []):
                orders.append({
                    "symbol":           getattr(o, "tradingsymbol",    ""),
                    "transaction_type": getattr(o, "transaction_type", ""),
                    "status":           getattr(o, "status",           ""),
                    "average_price":    float(getattr(o, "average_price", 0) or 0),
                })
            return orders
        except Exception as e:
            log.error(f"get_order_list failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # Entry order
    # -------------------------------------------------------------------------

    def place_entry_order(self, setup: Setup) -> Optional[str]:
        """
        Places a BUY LIMIT order + an SL SELL order via Upstox V3 HFT endpoint.
        Paper mode: skips API, records the trade directly (with slippage).
        Backtest mode: logs intent, records trade immediately.
        Returns trade_id string on success, None on failure.
        """
        trade_id = f"{setup.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        mode_tag = "[PAPER]" if Config.PAPER_TRADE else "[LIVE]"

        # Apply paper slippage to entry — makes paper PnL realistically conservative
        if Config.PAPER_TRADE and not Config.BACKTEST_MODE:
            setup.entry_price = round(setup.entry_price * (1 + Config.PAPER_SLIPPAGE_PCT), 2)

        if Config.BACKTEST_MODE:
            log.info(f"[BACKTEST] BUY {setup.shares}×{setup.symbol} @ ₹{setup.entry_price:.2f} "
                     f"SL ₹{setup.sl_price:.2f} T ₹{setup.target_price:.2f}")
            self._record_trade(trade_id, setup)
            return trade_id

        if not self._client or not _requests:
            log.error("Upstox client not initialised — cannot place order.")
            return None

        self._refresh_client_if_needed()

        try:
            instrument_key = self._ikey(setup.symbol)

            # BUY LIMIT at 0.1% above LTP for reliable fill
            buy_price    = round(setup.entry_price * 1.001, 2)
            buy_order_id = self._post_v3_order({
                "quantity":           setup.shares,
                "product":            "D",           # D = Delivery (CNC equivalent)
                "validity":           "DAY",
                "price":              buy_price,
                "instrument_token":   instrument_key,
                "order_type":         "LIMIT",
                "transaction_type":   "BUY",
                "disclosed_quantity": 0,
                "trigger_price":      0.0,
                "is_amo":             False,
                "tag":                "bot_entry",
            })
            log.info(f"{mode_tag} BUY placed: {setup.shares}×{setup.symbol} @ ₹{buy_price:.2f} "
                     f"| order_id: {buy_order_id}")

            # SL SELL — exchange-level stop, fires even if bot goes offline
            sl_order_id = self._post_v3_order({
                "quantity":           setup.shares,
                "product":            "D",
                "validity":           "DAY",
                "price":              setup.sl_price,    # limit price = trigger for immediate fill on breach
                "instrument_token":   instrument_key,
                "order_type":         "SL",              # stop-loss order type
                "transaction_type":   "SELL",
                "disclosed_quantity": 0,
                "trigger_price":      setup.sl_price,    # activates when price falls to this level
                "is_amo":             False,
                "tag":                "bot_sl",
            })
            log.info(f"{mode_tag} SL placed: {setup.symbol} @ ₹{setup.sl_price:.2f} "
                     f"| sl_order_id: {sl_order_id}")

            self._record_trade(trade_id, setup, sl_order_id)
            return trade_id

        except Exception as e:
            log.error(f"Entry order failed for {setup.symbol}: {e}")
            return None

    def _record_trade(self, trade_id: str, setup: Setup, sl_order_id: str = ""):
        """Creates a Trade object and persists it to DB after successful order placement."""
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
            sl_order_id=sl_order_id,
        )
        self.db.save_trade(trade)
        log.info(f"Trade recorded: {trade_id}")

    # -------------------------------------------------------------------------
    # SL replace (trailing SL updates)
    # -------------------------------------------------------------------------

    @_with_retry(max_attempts=Config.SL_REPLACE_MAX_RETRIES, base_delay=Config.SL_REPLACE_RETRY_DELAY)
    def replace_sl_order(self, symbol: str, qty: int,
                          new_sl: float, old_order_id: str) -> str:
        """
        Cancels the old exchange-level SL order and places a new one at new_sl.
        Called on every trailing SL advance (tier 1 breakeven / tier 2 tighten / tier 3 trail).
        Returns the new Upstox order ID so it can be stored in DB for the next cancel+replace.
        Returns "" in paper/backtest mode or if placement fails after all retries.

        Decorated with @_with_retry(max_attempts=SL_REPLACE_MAX_RETRIES) — on total failure
        (all retries exhausted), the decorator raises the last exception. The calling code in
        monitor.py catches this and triggers the emergency MARKET SELL below.
        """
        if Config.BACKTEST_MODE:
            log.info(f"[BACKTEST] SL replaced: {symbol} → ₹{new_sl:.2f}")
            return ""

        if not self._client or not _requests:
            return ""

        mode_tag = "[PAPER]" if Config.PAPER_TRADE else "[LIVE]"

        # Cancel old SL via V2 SDK — failure acceptable (order may have already executed)
        if old_order_id:
            try:
                OrderApi(self._client).cancel_order(old_order_id, api_version="2.0")
                log.info(f"Cancelled SL order {old_order_id} for {symbol}")
            except Exception as e:
                log.warning(f"Could not cancel SL {old_order_id} for {symbol}: {e} (may already be filled)")

        new_id = self._post_v3_order({
            "quantity":           qty,
            "product":            "D",
            "validity":           "DAY",
            "price":              new_sl,
            "instrument_token":   self._ikey(symbol),
            "order_type":         "SL",
            "transaction_type":   "SELL",
            "disclosed_quantity": 0,
            "trigger_price":      new_sl,
            "is_amo":             False,
            "tag":                "bot_sl",
        })
        log.info(f"{mode_tag} New SL placed: {symbol} @ ₹{new_sl:.2f} | id: {new_id}")
        return new_id

    def replace_sl_order_safe(self, symbol: str, qty: int,
                               new_sl: float, old_order_id: str) -> str:
        """
        Wrapper around replace_sl_order that catches all-retry-exhausted failures
        and fires an emergency MARKET SELL to avoid leaving the position unprotected.
        This is the method TradeMonitor should call — never call replace_sl_order directly.
        """
        try:
            return self.replace_sl_order(symbol, qty, new_sl, old_order_id)
        except Exception:
            # All SL_REPLACE_MAX_RETRIES attempts failed — position is unprotected.
            log.critical(
                f"EMERGENCY: SL replacement exhausted all retries for {symbol} "
                f"({Config.SL_REPLACE_MAX_RETRIES} attempts). Placing emergency MARKET SELL."
            )
            self._emergency_sell(symbol, qty)
            return ""

    def _emergency_sell(self, symbol: str, qty: int):
        """Last-resort MARKET SELL when SL order placement has failed repeatedly."""
        if not self._client or not _requests:
            return
        try:
            order_id = self._post_v3_order({
                "quantity":           qty,
                "product":            "D",
                "validity":           "DAY",
                "price":              0.0,           # price=0 for market orders in Upstox
                "instrument_token":   self._ikey(symbol),
                "order_type":         "MARKET",
                "transaction_type":   "SELL",
                "disclosed_quantity": 0,
                "trigger_price":      0.0,
                "is_amo":             False,
                "tag":                "bot_emergency",
            })
            log.critical(f"EMERGENCY MARKET SELL placed: {qty}×{symbol} | order_id: {order_id}")
        except Exception as e2:
            log.critical(f"EMERGENCY SELL ALSO FAILED for {symbol}: {e2} — POSITION UNPROTECTED")

    # -------------------------------------------------------------------------
    # Exit sell (tier 2 partial / VIX / time / event exits)
    # -------------------------------------------------------------------------

    def place_sell_order(self, symbol: str, qty: int,
                          price: float, reason: str) -> bool:
        """
        Places a LIMIT SELL at price × 0.999 (0.1% below market) for fast fills.
        Paper mode: applies slippage but still records as success.
        Returns True on success, False on API failure.
        NOT used for SL exits — those are handled by the exchange-level SL order.
        """
        if Config.BACKTEST_MODE:
            log.info(f"[BACKTEST] SELL {qty}×{symbol} @ ₹{price:.2f} | {reason}")
            return True

        if not self._client or not _requests:
            return False

        if Config.PAPER_TRADE:
            price = round(price * (1 - Config.PAPER_SLIPPAGE_PCT), 2)

        mode_tag   = "[PAPER]" if Config.PAPER_TRADE else "[LIVE]"
        sell_price = round(price * 0.999, 2)   # 0.1% below market for reliable fill
        log.info(f"{mode_tag} SELL {qty}×{symbol} @ ₹{sell_price:.2f} | {reason}")

        try:
            self._post_v3_order({
                "quantity":           qty,
                "product":            "D",
                "validity":           "DAY",
                "price":              sell_price,
                "instrument_token":   self._ikey(symbol),
                "order_type":         "LIMIT",
                "transaction_type":   "SELL",
                "disclosed_quantity": 0,
                "trigger_price":      0.0,
                "is_amo":             False,
                "tag":                f"bot_{reason[:20].lower().replace(' ', '_')}",
            })
            return True
        except Exception as e:
            log.error(f"Sell order failed for {symbol}: {e}")
            return False
