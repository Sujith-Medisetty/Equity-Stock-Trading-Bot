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
   - PULLBACK: EMA20 − (ATR × 1.0)         SL below the level being tested
   - FII_FLOW: entry − (ATR × 2.0)         wider SL for volatile institutional moves
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
            # For PULLBACK: SL just below the EMA20 being tested.
            # The logic: we're buying at EMA20 support. If price breaks BELOW EMA20, the setup is invalid.
            sl = stock_data.ema_20 - (atr * 1.0)  # small ATR buffer below EMA20

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
    Calculates exact NSE equity delivery charges for any trade.
    All charge rates are defined in Config and reflect 2026 NSE fee schedule.

    Buy charges: STT (0.1%) + stamp duty (0.015%) + exchange charge + GST on exchange + SEBI
    Sell charges: STT (0.1%) + DP charge (₹15.34 flat) + exchange charge + GST on exchange + SEBI

    Note: stamp duty is only on the buy side (as per SEBI rules).
    DP (depository participant) charge is only on the sell side — it's the cost
    of moving shares out of your demat account when you sell.

    annual_tax_summary() computes STCG tax on all profitable trades in the current
    financial year (April 1 to March 31). STCG rate = 20% + 4% cess = 20.8% effective.
    Also breaks down advance tax quarterly instalments — required if annual tax > ₹10,000.
    """

    @staticmethod
    def calculate_buy_charges(buy_value: float) -> dict:
        """Calculates all charges on the BUY leg of a delivery trade.
        All rates are from Config — change them there if NSE updates the schedule."""
        stt      = buy_value * Config.STT_DELIVERY     # 0.1% STT on buy side for delivery
        stamp    = buy_value * Config.STAMP_DUTY        # 0.015% stamp duty (only on buy, not sell)
        exchange = buy_value * Config.EXCHANGE_CHARGE   # NSE transaction charge (~0.003%)
        gst      = exchange  * Config.GST_RATE          # 18% GST applied only on the exchange charge
        sebi     = buy_value * Config.SEBI_CHARGE       # SEBI regulatory fee (tiny, ~0.0001%)
        total    = stt + stamp + exchange + gst + sebi  # sum of all buy-side charges
        return {
            "stt": round(stt, 2), "stamp_duty": round(stamp, 2),
            "exchange": round(exchange, 2), "gst": round(gst, 2),
            "sebi": round(sebi, 2), "total": round(total, 2)
        }

    @staticmethod
    def calculate_sell_charges(sell_value: float) -> dict:
        """Calculates all charges on the SELL leg of a delivery trade.
        Key difference from buy: stamp duty is absent, DP charge is present."""
        stt      = sell_value * Config.STT_DELIVERY     # 0.1% STT on sell side for delivery
        dp       = Config.DP_CHARGE                     # ₹15.34 flat per sell transaction (demat debit charge)
        exchange = sell_value * Config.EXCHANGE_CHARGE   # NSE transaction charge
        gst      = exchange   * Config.GST_RATE          # 18% GST on exchange charge
        sebi     = sell_value * Config.SEBI_CHARGE       # SEBI regulatory fee
        total    = stt + dp + exchange + gst + sebi      # sum of all sell-side charges
        return {
            "stt": round(stt, 2), "dp_charge": round(dp, 2),
            "exchange": round(exchange, 2), "gst": round(gst, 2),
            "sebi": round(sebi, 2), "total": round(total, 2)
        }

    @staticmethod
    def calculate_trade_pnl(entry_price: float, exit_price: float, qty: int) -> dict:
        """
        Calculates gross PnL, total charges, and net PnL for a complete trade round-trip.
        Also computes the STCG tax estimate (20.8%) on the net profit.
        Called by TradeMonitor when closing a trade so the correct net_pnl is stored in DB.
        """
        buy_value  = entry_price * qty    # total capital deployed on entry
        sell_value = exit_price  * qty    # total proceeds from exit
        gross_pnl  = sell_value - buy_value   # raw profit/loss before charges
        buy_ch     = ChargesCalculator.calculate_buy_charges(buy_value)
        sell_ch    = ChargesCalculator.calculate_sell_charges(sell_value)
        total_ch   = buy_ch["total"] + sell_ch["total"]  # combined round-trip charges
        net_pnl    = gross_pnl - total_ch                 # actual profit/loss in hand
        # STCG tax: only on profitable trades (max(0,net_pnl) to avoid tax on losses)
        stcg_tax   = max(0, net_pnl) * Config.EFFECTIVE_TAX   # 20.8% = 20% + 4% cess
        return {
            "gross_pnl":        round(gross_pnl, 2),
            "total_charges":    round(total_ch, 2),
            "net_pnl":          round(net_pnl, 2),
            "stcg_tax_estimate": round(stcg_tax, 2),
        }

    @staticmethod
    def annual_tax_summary(annual_stcg: float) -> dict:
        """
        Computes STCG tax for the full financial year.
        annual_stcg = sum of all profitable trade net_pnl since April 1 (from DB).
        Breaks down advance tax quarters — required if total tax > ₹10,000.
        """
        stcg_tax  = max(0, annual_stcg) * Config.STCG_RATE    # 20% on profits
        cess      = stcg_tax * Config.CESS_RATE                # 4% health & education cess on the tax
        total_tax = stcg_tax + cess                             # effective 20.8%
        return {
            "annual_stcg":          round(annual_stcg, 2),
            "stcg_tax_20pct":       round(stcg_tax, 2),
            "cess_4pct":            round(cess, 2),
            "total_tax":            round(total_tax, 2),
            "take_home":            round(max(0, annual_stcg) - total_tax, 2),  # profit after tax
            "advance_tax_required": total_tax > Config.ADVANCE_TAX_THRESHOLD,   # True if tax > ₹10k
            # Advance tax installment schedule (% of annual liability due each quarter)
            "advance_tax_quarters": {
                "jun_15_15pct":  round(total_tax * 0.15, 2),   # 15% by June 15
                "sep_15_45pct":  round(total_tax * 0.30, 2),   # another 30% by Sep 15 (cumulative 45%)
                "dec_15_75pct":  round(total_tax * 0.30, 2),   # another 30% by Dec 15 (cumulative 75%)
                "mar_15_100pct": round(total_tax * 0.25, 2),   # remaining 25% by Mar 15 (100%)
            }
        }
