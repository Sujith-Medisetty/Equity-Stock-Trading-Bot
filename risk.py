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
        if available_capital is None:
            available_capital = float(Config.TOTAL_CAPITAL)
        max_capital = available_capital * Config.CAPITAL_PER_TRADE_PCT
        max_risk    = Config.risk_per_trade(available_capital)
        if max_risk == 0:
            setup.skip_reason = f"Trading capital ₹{available_capital:.0f} ≤ floor ₹{Config.MIN_TRADE_CAPITAL:,}"
            setup.status = "SKIPPED"
            return setup

        atr = stock_data.atr
        if atr <= 0:
            setup.skip_reason = "ATR is zero"
            setup.status = "SKIPPED"
            return setup

        entry = setup.entry_price
        mult  = Config.ATR_MULT.get(setup.strategy.value, 1.5)
        sl    = entry - (atr * mult)

        if setup.strategy == StrategyType.BREAKOUT:
            d20_low = stock_data.close * (1 - stock_data.consolidation_range_pct / 100)
            sl = d20_low - (atr * 0.5)
        elif setup.strategy == StrategyType.WEEK52:
            sl = stock_data.week_52_high - (atr * 1.5)
        elif setup.strategy == StrategyType.PULLBACK:
            sl = stock_data.ema_20 - (atr * 1.0)

        risk_per_share = entry - sl
        if risk_per_share <= 0:
            setup.skip_reason = "Invalid SL"
            setup.status = "SKIPPED"
            return setup

        target = entry + (risk_per_share * Config.MIN_RR_RATIO)
        shares = int(max_risk / risk_per_share)
        if shares <= 0:
            setup.skip_reason = "Too few shares after risk calc"
            setup.status = "SKIPPED"
            return setup

        capital_needed = shares * entry
        if capital_needed > max_capital:
            shares = int(max_capital / entry)
            capital_needed = shares * entry

        if capital_needed < Config.MIN_POSITION_VALUE:
            setup.skip_reason = (
                f"Position too small: ₹{capital_needed:.0f} < MIN_POSITION_VALUE ₹{Config.MIN_POSITION_VALUE:,} "
                f"— Dhan fixed charges would eat into profit"
            )
            setup.status = "SKIPPED"
            return setup

        if shares < Config.MIN_QUANTITY:
            setup.skip_reason = (
                f"Too few shares: {shares} < MIN_QUANTITY {Config.MIN_QUANTITY} "
                f"— tier exit system needs at least {Config.MIN_QUANTITY} shares to function"
            )
            setup.status = "SKIPPED"
            return setup

        actual_risk = shares * risk_per_share
        rr = (target - entry) / risk_per_share

        if rr < Config.MIN_RR_RATIO:
            setup.skip_reason = f"RR too low: {rr:.2f}"
            setup.status = "SKIPPED"
            return setup

        setup.sl_price         = round(sl, 2)
        setup.target_price     = round(target, 2)
        setup.atr              = round(atr, 2)
        setup.risk_per_share   = round(risk_per_share, 2)
        setup.shares           = shares
        setup.capital_required = round(capital_needed, 2)
        setup.actual_risk      = round(actual_risk, 2)
        setup.rr_ratio         = round(rr, 2)
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
        max_capital    = available_capital * Config.CAPITAL_PER_TRADE_PCT
        portfolio_risk = available_capital * Config.PORTFOLIO_RISK_PCT

        checks = {
            "1_market_mode":   market_mode not in [MarketMode.DEFENSIVE, MarketMode.CASH],
            "2_vix":           vix < Config.VIX_NERVOUS,
            "3_no_event":      self.db.fetchone(
                "SELECT * FROM events_calendar WHERE symbol=? AND days_away<=1 AND risk_level='RED'",
                (setup.symbol,)
            ) is None,
            "4_setup_score":   setup.score >= 60,
            "5_strategy_ok":   setup.status != "SKIPPED",
            "6_candle":        setup.score >= 80,
            "7_sl_valid":      setup.sl_price > 0 and setup.sl_price < setup.entry_price,
            "8_rr":            setup.rr_ratio >= Config.MIN_RR_RATIO,
            "9_capital":       setup.capital_required <= max_capital,
            "10_portfolio":    (
                len(open_trades) < Config.effective_max_trades(available_capital) and
                sum(
                    (t["entry_price"] - t["current_sl"]) * t["remaining_qty"]
                    for t in open_trades if t["status"] == "OPEN"
                ) + setup.actual_risk <= portfolio_risk
            ),
        }

        failed   = [k for k, v in checks.items() if not v]
        approved = len(failed) == 0

        if approved:
            log.info(f"Checklist PASS: {setup.symbol} | {setup.strategy.value} | Score: {setup.score}")
        else:
            log.info(f"Checklist FAIL: {setup.symbol} | {failed}")
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
        stt      = buy_value * Config.STT_DELIVERY
        stamp    = buy_value * Config.STAMP_DUTY
        exchange = buy_value * Config.EXCHANGE_CHARGE
        gst      = exchange  * Config.GST_RATE
        sebi     = buy_value * Config.SEBI_CHARGE
        total    = stt + stamp + exchange + gst + sebi
        return {
            "stt": round(stt, 2), "stamp_duty": round(stamp, 2),
            "exchange": round(exchange, 2), "gst": round(gst, 2),
            "sebi": round(sebi, 2), "total": round(total, 2)
        }

    @staticmethod
    def calculate_sell_charges(sell_value: float) -> dict:
        stt      = sell_value * Config.STT_DELIVERY
        dp       = Config.DP_CHARGE
        exchange = sell_value * Config.EXCHANGE_CHARGE
        gst      = exchange   * Config.GST_RATE
        sebi     = sell_value * Config.SEBI_CHARGE
        total    = stt + dp + exchange + gst + sebi
        return {
            "stt": round(stt, 2), "dp_charge": round(dp, 2),
            "exchange": round(exchange, 2), "gst": round(gst, 2),
            "sebi": round(sebi, 2), "total": round(total, 2)
        }

    @staticmethod
    def calculate_trade_pnl(entry_price: float, exit_price: float, qty: int) -> dict:
        buy_value  = entry_price * qty
        sell_value = exit_price  * qty
        gross_pnl  = sell_value - buy_value
        buy_ch     = ChargesCalculator.calculate_buy_charges(buy_value)
        sell_ch    = ChargesCalculator.calculate_sell_charges(sell_value)
        total_ch   = buy_ch["total"] + sell_ch["total"]
        net_pnl    = gross_pnl - total_ch
        stcg_tax   = max(0, net_pnl) * Config.EFFECTIVE_TAX
        return {
            "gross_pnl":        round(gross_pnl, 2),
            "total_charges":    round(total_ch, 2),
            "net_pnl":          round(net_pnl, 2),
            "stcg_tax_estimate": round(stcg_tax, 2),
        }

    @staticmethod
    def annual_tax_summary(annual_stcg: float) -> dict:
        stcg_tax  = max(0, annual_stcg) * Config.STCG_RATE
        cess      = stcg_tax * Config.CESS_RATE
        total_tax = stcg_tax + cess
        return {
            "annual_stcg":          round(annual_stcg, 2),
            "stcg_tax_20pct":       round(stcg_tax, 2),
            "cess_4pct":            round(cess, 2),
            "total_tax":            round(total_tax, 2),
            "take_home":            round(max(0, annual_stcg) - total_tax, 2),
            "advance_tax_required": total_tax > Config.ADVANCE_TAX_THRESHOLD,
            "advance_tax_quarters": {
                "jun_15_15pct":  round(total_tax * 0.15, 2),
                "sep_15_45pct":  round(total_tax * 0.30, 2),
                "dec_15_75pct":  round(total_tax * 0.30, 2),
                "mar_15_100pct": round(total_tax * 0.25, 2),
            }
        }
