"""
risk.py — Position sizing, stop loss placement, and pre-trade validation.

Two classes live here:

1. RiskManager
   Sizes every trade so the maximum possible loss is exactly ₹1,500.
   The formula: shares = (available_capital × RISK_PER_TRADE_PCT) / risk_per_share
   where risk_per_share = entry_price - stop_loss_price.

   Stop loss placement is strategy-specific:
   - SWING:    entry − (ATR × 1.5)         standard swing SL
   - BREAKOUT: low of the consolidation box − (ATR × 0.5)  tight SL just below the box
   - PULLBACK: EMA20 − (ATR × 1.5)         SL below the level being tested
   - WEEK52:   52W high − (ATR × 1.5)      SL just below the breakout level

   After sizing, RiskManager checks that the R:R ratio is at least 2.0 (1 risk : 2 reward).
   Setups that don't make 2:1 are skipped — no point entering a trade that requires
   price to move ₹1 to make ₹1 when it might move ₹1 against us and cost ₹1.

   run_pre_trade_checklist() is the final gate before order placement.
   It checks 10 conditions and returns (approved=True, failed=[]) or (False, [failed list]).
   All 10 must pass — there are no partial passes.

2. ChargesCalculator
   Calculates exact NSE delivery charges for buy and sell legs.
   Used to compute net PNL (after charges) for every trade.
   Also computes the STCG tax estimate for the tax summary dashboard.

   Why bother tracking charges?
   On a ₹50k trade: STT ≈ ₹50, DP charge = ₹15.34, total round-trip ≈ ₹120-150.
   Across 50 trades a year that's ₹6,000-7,500 in charges — meaningful on a ₹2L account.
   The tax summary uses net_pnl (after charges) so tax is calculated correctly.
"""

from config import Config, log
from models import StrategyType, Setup, StockData, MarketMode
from database import DatabaseManager


class RiskManager:
    """
    Calculates stop loss, target, share count, and validates R:R for each setup.
    Called in step4_find_setups() after StrategyEngine identifies a candidate.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def calculate_setup_risk(self, setup: Setup, stock_data: StockData,
                              available_capital: float = None) -> Setup:
        """
        Fills in sl_price, target_price, shares, capital_required, actual_risk, rr_ratio.
        Sets setup.status = "SKIPPED" with a reason if the math doesn't work out.

        Limits are derived from available_capital (trading capital after reserve):
          max capital per trade = available_capital × CAPITAL_PER_TRADE_PCT (25%)
          max risk per trade    = Config.risk_per_trade(available_capital) — fixed tiers:
                                    > ₹80k  → ₹1,500
                                    > ₹20k  → ₹1,000
                                    ≤ ₹20k  → skip (below trading floor)

        If shares × entry exceeds the capital cap, shares are reduced to fit.
        Risk may then be less than the tier target but the trade is still valid.
        """
        # Fall back to Config total if no live capital provided (e.g. backtest mode)
        if available_capital is None:
            available_capital = float(Config.TOTAL_CAPITAL)

        # Max capital that can go into this single trade: 25% of available capital.
        # This prevents over-concentration in one position.
        max_capital = available_capital * Config.CAPITAL_PER_TRADE_PCT

        # Fixed-rupee risk budget per trade, tiered by account size.
        # This is the max loss we'll accept if SL is hit.
        max_risk    = Config.risk_per_trade(available_capital)

        if max_risk == 0:
            # Below the ₹20k trading floor — position sizing math would produce worthless results
            setup.skip_reason = f"Trading capital ₹{available_capital:.0f} ≤ floor ₹{Config.MIN_TRADE_CAPITAL:,}"
            setup.status = "SKIPPED"
            return setup

        atr = stock_data.atr
        if atr <= 0:
            # ATR = 0 means indicator didn't calculate — can't place a meaningful SL
            setup.skip_reason = "ATR is zero"
            setup.status = "SKIPPED"
            return setup

        entry = setup.entry_price

        # Default SL: entry − (ATR × strategy multiplier).
        # The multiplier is strategy-specific — wider for volatile strategies (FII_FLOW),
        # tighter for precise setups (BREAKOUT, PULLBACK).
        mult  = Config.ATR_MULT.get(setup.strategy.value, 1.5)  # default 1.5x if strategy not in map
        sl    = entry - (atr * mult)   # preliminary SL — may be overridden below for specific strategies

        # Strategy-specific SL overrides that place the SL at a more logical level
        # than a simple ATR distance from entry:

        if setup.strategy == StrategyType.BREAKOUT:
            # For BREAKOUT: SL goes below the bottom of the consolidation box, not below entry.
            # The logic: if the breakout was real, price should NOT re-enter the consolidation box.
            # consolidation_range_pct is the 20-day range as % of price.
            # d20_low ≈ the bottom of the box that just broke out.
            d20_low = stock_data.close * (1 - stock_data.consolidation_range_pct / 100)
            sl = d20_low - (atr * 0.5)   # small ATR buffer below the box bottom

        elif setup.strategy == StrategyType.WEEK52:
            # For WEEK52: SL just below the old 52W high (now support level).
            # The logic: the 52W high was resistance for a year. Once broken, it becomes support.
            # If price falls back BELOW the 52W high, the breakout has failed.
            sl = stock_data.week_52_high - (atr * 1.5)

        elif setup.strategy == StrategyType.PULLBACK:
            # For PULLBACK: SL 1.5×ATR below EMA20.
            # Entry is within 0.3×ATR of EMA20, so the gap to SL is at least 1.2×ATR —
            # enough buffer to survive normal daily noise without premature SL hits.
            # The logic: we're buying at EMA20 support. If price breaks BELOW EMA20, the setup is invalid.
            sl = stock_data.ema_20 - (atr * 1.5)

        # How much can we lose per share if SL is hit?
        risk_per_share = entry - sl
        if risk_per_share <= 0:
            # SL is above entry — invalid (would mean we profit if SL hits, which is wrong)
            setup.skip_reason = "Invalid SL"
            setup.status = "SKIPPED"
            return setup

        # Target = entry + (risk_per_share × minimum R:R).
        # At min R:R of 2.0: if we risk ₹10/share, we target ₹20/share profit.
        target = entry + (risk_per_share * Config.MIN_RR_RATIO)

        # FVG target override for SWING and PULLBACK:
        # If there's an unfilled bullish FVG above price (within 8%), its bottom is a
        # natural magnet — institutions left orders there. Use it as target if it gives
        # better R:R than the standard 2×ATR target AND still meets the 2:1 minimum.
        if (setup.strategy in (StrategyType.PULLBACK, StrategyType.SWING) and
                stock_data.fvg_target > entry):
            fvg_rr = (stock_data.fvg_target - entry) / risk_per_share
            if fvg_rr >= Config.MIN_RR_RATIO and stock_data.fvg_target > target:
                log.info(
                    f"{setup.symbol}: FVG target ₹{stock_data.fvg_target:.2f} used "
                    f"(R:R {fvg_rr:.1f}:1 vs standard {(target - entry) / risk_per_share:.1f}:1)"
                )
                target = stock_data.fvg_target

        # Share count: how many shares can we buy so that if SL hits, we lose exactly max_risk?
        # Formula: shares = max_risk / risk_per_share
        # e.g. max_risk=₹1500, risk_per_share=₹15 → 100 shares
        shares = int(max_risk / risk_per_share)
        if shares <= 0:
            setup.skip_reason = "Too few shares after risk calc"
            setup.status = "SKIPPED"
            return setup

        # Apply the 25% capital cap: if 100 shares × ₹1200 = ₹120,000 but max is ₹50,000,
        # reduce shares to int(50000/1200) = 41 shares. This means actual risk < max_risk,
        # which is acceptable — we're just not deploying the full risk budget.
        capital_needed = shares * entry
        if capital_needed > max_capital:
            shares = int(max_capital / entry)    # reduce to fit within capital cap
            capital_needed = shares * entry      # recalculate after reduction

        # Floor check: if position value is below ₹15,000, the DP charge (₹15.34 flat)
        # and STT on a small profit would make the trade unprofitable even if we're right.
        if capital_needed < Config.MIN_POSITION_VALUE:
            setup.skip_reason = (
                f"Position too small: ₹{capital_needed:.0f} < MIN_POSITION_VALUE ₹{Config.MIN_POSITION_VALUE:,} "
                f"— fixed charges would eat into profit"
            )
            setup.status = "SKIPPED"
            return setup

        # Minimum quantity for the 3-tier exit system.
        # Tier 2 sells floor(qty/2). With qty < 3:
        #   qty=1 → tier 2 sells the only share → position fully closes at tier 2 (tier 3 never runs)
        #   qty=2 → tier 2 sells 1, tier 3 trails 1 share (too small to matter)
        #   qty=3 → tier 2 sells 1, tier 3 trails 2 properly ✓
        if shares < Config.MIN_QUANTITY:
            setup.skip_reason = (
                f"Too few shares: {shares} < MIN_QUANTITY {Config.MIN_QUANTITY} "
                f"— tier exit system needs at least {Config.MIN_QUANTITY} shares to function"
            )
            setup.status = "SKIPPED"
            return setup

        # Actual risk = what we'd lose if SL hits at exactly sl_price (shares might be reduced from max_risk)
        actual_risk = shares * risk_per_share

        # R:R ratio for this specific setup: how much reward vs how much risk?
        # At min 2.0: we make ₹2 if right, lose ₹1 if wrong.
        rr = (target - entry) / risk_per_share

        if rr < Config.MIN_RR_RATIO:
            # R:R below 2.0 — not worth taking. Even a 50% win rate would break even,
            # but charges would make it a net loser.
            setup.skip_reason = f"RR too low: {rr:.2f}"
            setup.status = "SKIPPED"
            return setup

        # All checks passed — populate the setup fields
        setup.sl_price         = round(sl, 2)
        setup.target_price     = round(target, 2)
        setup.atr              = round(atr, 2)
        setup.risk_per_share   = round(risk_per_share, 2)   # how much we lose per share if SL hits
        setup.shares           = shares                      # number of shares to buy
        setup.capital_required = round(capital_needed, 2)   # total rupees deployed
        setup.actual_risk      = round(actual_risk, 2)       # total max loss in rupees
        setup.rr_ratio         = round(rr, 2)                # reward-to-risk ratio
        return setup

    def run_pre_trade_checklist(self, setup: Setup, market_mode: MarketMode,
                                 vix: float, open_trades: list,
                                 available_capital: float = None) -> tuple:
        """
        10-point checklist before placing any order. ALL must pass.
        Returns (approved: bool, failed: list of failed check names).

        Checks 9 and 10 use limits derived from available_capital so they
        auto-scale as the account grows or shrinks with realised PnL.

        Check 10 is the most important guard against over-leverage:
        portfolio risk = sum of (entry - current_sl) × remaining_qty for all open trades.
        Adding this trade must not push the total above PORTFOLIO_RISK_PCT of capital.
        """
        if available_capital is None:
            available_capital = float(Config.TOTAL_CAPITAL)

        max_capital    = available_capital * Config.CAPITAL_PER_TRADE_PCT   # 25% per trade cap
        portfolio_risk = available_capital * Config.PORTFOLIO_RISK_PCT      # 3% total portfolio risk cap

        checks = {
            # Check 1: Market must not be in protection mode (no entries in DEFENSIVE or CASH)
            "1_market_mode":   market_mode not in [MarketMode.DEFENSIVE, MarketMode.CASH],

            # Check 2: VIX must be below the "nervous" threshold — above 22 means too much volatility
            "2_vix":           vix < Config.VIX_NERVOUS,

            # Check 3: No earnings/results event within 1 day for this specific stock.
            #          (Screener blocked entries within 5 days, this is a tighter final check for 1 day)
            "3_no_event":      self.db.fetchone(
                "SELECT * FROM events_calendar WHERE symbol=? AND days_away<=1 AND risk_level='RED'",
                (setup.symbol,)
            ) is None,  # None = no event found = safe to trade

            # Check 4: Setup must have a minimum quality score of 60
            "4_setup_score":   setup.score >= 60,

            # Check 5: Setup must not have been rejected by calculate_setup_risk() already
            "5_strategy_ok":   setup.status != "SKIPPED",

            # Check 6: High confidence bar — score must be 80+ for actual order placement.
            #          This catches setups that scored exactly 60-79: they were found and evaluated,
            #          but are considered insufficient for real capital deployment.
            "6_candle":        setup.score >= 80,

            # Check 7: SL must be a valid positive number below entry.
            #          SL >= entry means the SL would fire immediately — clearly wrong.
            "7_sl_valid":      setup.sl_price > 0 and setup.sl_price < setup.entry_price,

            # Check 8: Risk:reward ratio must be at least 2.0 (risk ₹1, target ₹2)
            "8_rr":            setup.rr_ratio >= Config.MIN_RR_RATIO,

            # Check 9: Capital needed for this trade must fit within the 25% per-trade cap.
            #          This should already be enforced by calculate_setup_risk(), but double-check here.
            "9_capital":       setup.capital_required <= max_capital,

            # Check 10: The most important portfolio-level risk guard.
            #   Part A: open trade count must be below the dynamic max (scales with capital).
            #   Part B: total open risk (sum of (entry - current_sl) × remaining_qty for all open trades)
            #           PLUS this new trade's risk must not exceed 3% of available capital.
            #   This prevents a scenario where 4 trades all at max risk = 4 × ₹1500 = ₹6000 simultaneous loss.
            "10_portfolio":    (
                len(open_trades) < Config.effective_max_trades(available_capital) and
                sum(
                    (t["entry_price"] - t["current_sl"]) * t["remaining_qty"]
                    for t in open_trades if t["status"] == "OPEN"
                ) + setup.actual_risk <= portfolio_risk
            ),
        }

        # Collect all the check names that evaluated to False
        failed   = [k for k, v in checks.items() if not v]
        approved = len(failed) == 0  # approved only if EVERY check passed

        if approved:
            log.info(f"Checklist PASS: {setup.symbol} | {setup.strategy.value} | Score: {setup.score}")
        else:
            log.info(f"Checklist FAIL: {setup.symbol} | {failed}")  # log which checks failed — useful for debugging

        return approved, failed


class ChargesCalculator:
    """
    Fetches exact brokerage and statutory charges from the Upstox /v2/charges/brokerage API.
    Falls back to manual rates (Config constants) if the API is unavailable.

    calculate_trade_pnl() calls the API twice — once for BUY leg, once for SELL leg —
    and sums them for a precise round-trip cost. The symbol→instrument_token mapping
    (NSE_EQ|ISIN) is resolved inside OrderManager.get_brokerage().
    """

    @staticmethod
    def _manual_buy_charges(buy_value: float) -> float:
        """
        NSE equity delivery buy-leg charges.
        Components mirror what the Upstox /v2/charges/brokerage API returns:
          stt          → taxes.stt           (0.1% on buy)
          stamp        → taxes.stamp_duty    (0.015% on buy, state-mandated)
          exchange     → other_charges.transaction  (NSE transaction 0.00297%)
          gst          → taxes.gst           (18% on exchange charge)
          sebi         → other_charges.sebi_turnover (₹10 per crore)
          clearing     → other_charges.clearing      (NSE clearing 0.000325%)
          ipft         → other_charges.ipft           (₹1 per crore)
        Brokerage = ₹0 (Upstox delivery).
        """
        exchange = buy_value * Config.EXCHANGE_CHARGE
        return (
            buy_value * Config.STT_DELIVERY          # STT on buy
            + buy_value * Config.STAMP_DUTY          # stamp duty on buy
            + exchange                               # NSE transaction charge
            + exchange * Config.GST_RATE             # GST on exchange
            + buy_value * Config.SEBI_CHARGE         # SEBI turnover fee
            + buy_value * Config.CLEARING_CHARGE     # NSE clearing charge
            + buy_value * Config.IPFT_CHARGE         # IPFT
        )

    @staticmethod
    def _manual_sell_charges(sell_value: float) -> float:
        """
        NSE equity delivery sell-leg charges.
        DP charge (₹15.34) is flat per sell transaction — included here once per call.
        Call this separately for each sell leg (T2 partial + final exit) so each gets its own DP.
        """
        exchange = sell_value * Config.EXCHANGE_CHARGE
        return (
            sell_value * Config.STT_DELIVERY         # STT on sell
            + Config.DP_CHARGE                       # DP charge: ₹15.34 flat per sell transaction
            + exchange                               # NSE transaction charge
            + exchange * Config.GST_RATE             # GST on exchange
            + sell_value * Config.SEBI_CHARGE        # SEBI turnover fee
            + sell_value * Config.CLEARING_CHARGE    # NSE clearing charge
            + sell_value * Config.IPFT_CHARGE        # IPFT
        )

    @staticmethod
    def calculate_trade_pnl(symbol: str, entry_price: float, exit_price: float,
                            qty: int, order_mgr=None) -> dict:
        """
        Returns gross_pnl, total_charges, and net_pnl for a round-trip trade.
        Charges are fetched from Upstox API (accurate to the rupee).
        Falls back to manual Config rates if the API is unavailable.
        """
        buy_value  = entry_price * qty
        sell_value = exit_price  * qty
        gross_pnl  = sell_value - buy_value

        buy_total = sell_total = 0.0
        if order_mgr is not None:
            buy_ch  = order_mgr.get_brokerage(symbol, qty, entry_price, "BUY")
            sell_ch = order_mgr.get_brokerage(symbol, qty, exit_price,  "SELL")
            if buy_ch is not None and sell_ch is not None:
                buy_total  = buy_ch["total"]
                sell_total = sell_ch["total"]
            else:
                log.warning(f"Brokerage API unavailable for {symbol} — using manual rates")
                buy_total  = ChargesCalculator._manual_buy_charges(buy_value)
                sell_total = ChargesCalculator._manual_sell_charges(sell_value)
        else:
            buy_total  = ChargesCalculator._manual_buy_charges(buy_value)
            sell_total = ChargesCalculator._manual_sell_charges(sell_value)

        total_ch = buy_total + sell_total
        net_pnl  = gross_pnl - total_ch
        return {
            "gross_pnl":     round(gross_pnl, 2),
            "total_charges": round(total_ch, 2),
            "net_pnl":       round(net_pnl, 2),
        }
