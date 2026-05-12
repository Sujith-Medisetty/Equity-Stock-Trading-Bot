"""
analyze_params.py — Parameter optimization using cached backtest data.

IMPORTANT: This simulation matches the ACTUAL live system exactly:
  - NO partial sells at target. T2 just tightens the SL near the target.
  - ALL exits are SL-based (the broker fires the SL-M order).
  - Trailing SL progression: T1 (breakeven) → T1.5 (adaptive) → T2 (tighten) → T3 (ATR trail)
  - Time exit: sell at close after 15 days if still at a loss.

WHAT IT FINDS:
  1. Best initial SL placement (ATR multiplier)
  2. Best RSI range for PULLBACK entry
  3. Best volume threshold
  4. Best RR ratio (target distance, which controls when T2 tightening kicks in)
  5. Best T3 trail multiplier (how tightly to trail after T2)
  6. Best T1.5 adaptive % (how much of the gain to lock in between T1 and T2)
  7. Best stocks and sectors

RUN:  python3 analyze_params.py
"""

import os, pickle, sys
from collections import defaultdict
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = "backtest_cache"
DATE_TAG  = "_2022-11-27_2024-12-31.pkl"
DATE_TAG2 = "_2021-11-27_2023-12-31.pkl"
MAX_HOLD  = 15    # days — force exit if no profit after this many days

STOCKS = [
    "ICICIBANK","HDFCBANK","AXISBANK","SBIN","KOTAKBANK","INDUSINDBK",
    "BAJFINANCE","BAJAJFINSV","SHRIRAMFIN","HDFCLIFE","SBILIFE",
    "TATAMOTORS","MARUTI","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT",
    "ITC","HINDUNILVR","NESTLEIND","BRITANNIA","TATACONSUM",
    "TITAN","ASIANPAINT","TRENT",
    "SUNPHARMA","CIPLA","DRREDDY","APOLLOHOSP",
    "HINDALCO","TATASTEEL","JSWSTEEL",
    "RELIANCE","ONGC","NTPC","POWERGRID","COALINDIA","BPCL",
    "ULTRACEMCO","GRASIM","LT","ADANIPORTS","ADANIENT",
    "BHARTIARTL","BEL",
]
SECTOR = {
    "ICICIBANK":"BANKING","HDFCBANK":"BANKING","AXISBANK":"BANKING",
    "SBIN":"BANKING","KOTAKBANK":"BANKING","INDUSINDBK":"BANKING",
    "BAJFINANCE":"FINANCE","BAJAJFINSV":"FINANCE","SHRIRAMFIN":"FINANCE",
    "HDFCLIFE":"INSURANCE","SBILIFE":"INSURANCE",
    "TATAMOTORS":"AUTO","MARUTI":"AUTO","M&M":"AUTO",
    "BAJAJ-AUTO":"AUTO","HEROMOTOCO":"AUTO","EICHERMOT":"AUTO",
    "ITC":"FMCG","HINDUNILVR":"FMCG","NESTLEIND":"FMCG",
    "BRITANNIA":"FMCG","TATACONSUM":"FMCG",
    "TITAN":"CONSUMER","ASIANPAINT":"CONSUMER","TRENT":"CONSUMER",
    "SUNPHARMA":"PHARMA","CIPLA":"PHARMA","DRREDDY":"PHARMA",
    "APOLLOHOSP":"HEALTHCARE",
    "HINDALCO":"METALS","TATASTEEL":"METALS","JSWSTEEL":"METALS",
    "RELIANCE":"ENERGY","ONGC":"ENERGY","NTPC":"ENERGY",
    "POWERGRID":"ENERGY","COALINDIA":"ENERGY","BPCL":"ENERGY",
    "ULTRACEMCO":"CEMENT","GRASIM":"CEMENT",
    "LT":"INFRA","ADANIPORTS":"INFRA","ADANIENT":"CONGLOMERATE",
    "BHARTIARTL":"TELECOM","BEL":"DEFENCE",
}

# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi(c, p=14):
    d  = c.diff()
    ag = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(com=p-1, adjust=False).mean()
    return 100 - 100/(1 + ag/al.replace(0,1e-9))

def _atr(df, p=14):
    h,l,c = df["high"],df["low"],df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()

def _macd(c, f=12, s=26, sig=9):
    ml = _ema(c,f) - _ema(c,s)
    sl = _ema(ml,sig)
    return ml, sl, ml-sl

def _obv(df):
    return (np.sign(df["close"].diff()).fillna(0)*df["volume"]).cumsum()

def _vrat(v, p=20):
    return v / v.rolling(p).mean().replace(0,1.0)

def add_indicators(df):
    df = df.copy().reset_index(drop=True)
    c  = df["close"]
    df["e20"]  = _ema(c,20)
    df["e50"]  = _ema(c,50)
    df["e200"] = _ema(c,200)
    df["rsi"]  = _rsi(c)
    df["atr"]  = _atr(df)
    df["ml"],df["ms"],df["mh"] = _macd(c)
    df["vr"]   = _vrat(df["volume"])
    df["obv"]  = _obv(df)
    df["obv_r"]= df["obv"].diff(3) > 0
    rh = df["high"].rolling(20).max()
    rl = df["low"].rolling(20).min()
    df["cpct"] = (rh - rl)/c*100
    return df.dropna(subset=["e200"])

def load_daily(sym):
    safe = sym.replace(" ","_").replace("|","_")
    for tag in [DATE_TAG, DATE_TAG2]:
        p = os.path.join(CACHE_DIR, f"{safe}_daily{tag}")
        if os.path.exists(p):
            try:
                with open(p,"rb") as f:
                    df = pickle.load(f)
                if df is not None and len(df)>150:
                    return df
            except: pass
    return None


# ── Core simulation — matches live system EXACTLY ─────────────────────────────
# No partial sells. All exits via SL only.
# SL progression: T1(breakeven) → T1.5(adaptive) → T2(tighten near target) → T3(ATR trail)

def simulate_sl_only(df, entry_idx, initial_sl, target,
                     adaptive_pct=0.50,   # T1.5: lock this % of gain above entry
                     t2_tighten_atr=0.50, # T2: SL = target - this×ATR
                     t3_trail_atr=1.00,   # T3: SL = close - this×ATR
                     max_hold=MAX_HOLD):
    """
    Simulates a trade from entry_idx+1's open.
    Returns (pnl_pct, outcome, holding_days).

    SL trail matches monitor.py _update_trailing_sl() exactly:
      T1  : price >= entry+risk  →  SL = entry
      T1.5: T1 done, price < target  →  SL = entry + adaptive_pct*(price-entry)
      T2  : T1 done, price >= target →  SL = max(SL, target - t2_tighten_atr*ATR)
      T3  : T2 done →  SL = max(SL, close - t3_trail_atr*ATR)
    """
    n = len(df)
    if entry_idx + 1 >= n:
        return None, "SKIP", 0

    # Entry at next-day open + 0.2% slippage (same as live system LIMIT BUY at entry×1.001)
    entry = float(df.iloc[entry_idx+1]["open"]) * 1.002
    risk  = entry - initial_sl
    if risk <= 0:
        return None, "SKIP", 0

    sl       = initial_sl
    t1_done  = False
    t2_done  = False

    for j in range(entry_idx+1, min(entry_idx+max_hold+1, n)):
        bar = df.iloc[j]
        o   = float(bar["open"])
        h   = float(bar["high"])
        l   = float(bar["low"])
        c   = float(bar["close"])
        a   = float(bar["atr"]) if float(bar["atr"]) > 0 else risk

        days = j - (entry_idx+1)

        # ── Gap down: open below SL ───────────────────────────────────────
        if o <= sl:
            pnl = (o - entry) / entry * 100
            return pnl, "SL_GAP", days

        # ── Intraday SL hit ───────────────────────────────────────────────
        if l <= sl:
            pnl = (sl - entry) / entry * 100
            return pnl, "SL_HIT", days

        # ── Trailing SL update (same order as monitor.py elif chain) ──────

        if not t1_done and c >= entry + risk:
            # T1: move SL to breakeven
            sl      = max(sl, entry)
            t1_done = True

        elif t1_done and not t2_done and c < target:
            # T1.5: adaptive trail — lock adaptive_pct of gain above entry
            adaptive = entry + (c - entry) * adaptive_pct
            sl = max(sl, adaptive)

        elif t1_done and not t2_done and c >= target:
            # T2: price reached target → tighten SL near target (NO sell)
            new_sl  = target - t2_tighten_atr * a
            sl      = max(sl, new_sl)
            t2_done = True

        elif t2_done:
            # T3: trail at t3_trail_atr×ATR below close
            trail = c - t3_trail_atr * a
            sl    = max(sl, trail)

        # ── Time exit: 15 days held, still at a loss ──────────────────────
        if days >= max_hold and c <= entry:
            pnl = (c * 0.998 - entry) / entry * 100   # sell at close -0.2% slip
            return pnl, "TIME", days

    # Reached max_hold in profit (or at end of data)
    last_c = float(df.iloc[min(entry_idx+max_hold, n-1)]["close"]) * 0.998
    pnl    = (last_c - entry) / entry * 100
    return pnl, "TIME_PROFIT" if last_c > entry else "TIME", max_hold


# ── Load data ─────────────────────────────────────────────────────────────────

print("="*72)
print("  PARAMETER OPTIMIZATION  (SL-only exits, matches live system)")
print("="*72)
print("\nLoading …", end=" ", flush=True)

stock_dfs = {}
for sym in STOCKS:
    df = load_daily(sym)
    if df is None: continue
    stock_dfs[sym] = add_indicators(df)

nifty_raw = load_daily("NIFTY 50")
if nifty_raw is None:
    sys.exit("NIFTY 50 data not found in cache.")
nifty_ind = add_indicators(nifty_raw)
nifty_lkp = nifty_ind.set_index("date")[["close","e200"]].to_dict("index")
nifty_dates = sorted(nifty_lkp.keys())

def nifty_up(dt):
    sub = [d for d in nifty_dates if d <= dt]
    if not sub: return False
    r = nifty_lkp[sub[-1]]
    return r["close"] > r["e200"]

print(f"done. {len(stock_dfs)} stocks.\n")


# ── PRE-COMPUTE candidate rows ────────────────────────────────────────────────
# For each (stock, day) that satisfies the broad PULLBACK base condition,
# pre-compute trade outcomes for every (atr_mult, rr, adaptive_pct, t3_mult) combo.
# This means the grid search is just a fast filter+lookup, not re-simulating.

print("Pre-computing trade outcomes for all parameter combinations …")
print("(This takes a few minutes — runs once, results used by all sections)\n")

ATR_MULTS     = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
RR_VALS       = [1.5, 2.0, 2.5, 3.0]
ADAPTIVE_PCTS = [0.30, 0.50, 0.70]        # T1.5: what % of gain to lock in
T3_MULTS      = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # T3 trail ATR multiplier

candidate_rows = []

for sym, df in stock_dfs.items():
    n      = len(df)
    arr_c  = df["close"].values
    arr_o  = df["open"].values
    arr_h  = df["high"].values
    arr_l  = df["low"].values
    arr_a  = df["atr"].values
    arr_r  = df["rsi"].values
    arr_vr = df["vr"].values
    arr_e20  = df["e20"].values
    arr_e50  = df["e50"].values
    arr_e200 = df["e200"].values
    arr_ml   = df["ml"].values
    arr_ms   = df["ms"].values
    arr_mh   = df["mh"].values
    arr_obvr = df["obv_r"].values.astype(bool)
    arr_cpct = df["cpct"].values

    for i in range(250, n - MAX_HOLD - 2):
        c   = arr_c[i]; a = arr_a[i]
        e20 = arr_e20[i]; e50 = arr_e50[i]; e200 = arr_e200[i]
        rsi = arr_r[i];   vr  = arr_vr[i]

        if a<=0 or e20<=0 or e50<=0 or e200<=0: continue
        # Broad PULLBACK base: uptrend + price near EMA20
        if not (e20 > e50 and c > e200): continue
        dist_atr = (c - e20) / a
        # Allow -0.5 to +0.4 ATR from EMA20 (slightly wider scan than current 0-0.3)
        if not (-0.5 <= dist_atr <= 0.4): continue

        # Forward returns for correlation analysis
        fwd10 = (arr_c[min(i+10, n-1)] - c) / c * 100

        # Pre-compute outcomes for all combos
        outcomes = {}
        for atr_m in ATR_MULTS:
            sl_p = e20 - atr_m * a
            r_p  = c - sl_p
            if r_p <= 0: continue
            for rr in RR_VALS:
                tgt = c + r_p * rr
                for adp in ADAPTIVE_PCTS:
                    for t3 in T3_MULTS:
                        pnl, rsn, days = simulate_sl_only(
                            df, i, sl_p, tgt,
                            adaptive_pct=adp,
                            t2_tighten_atr=0.50,
                            t3_trail_atr=t3
                        )
                        outcomes[(atr_m, rr, adp, t3)] = (pnl, rsn, days)

        candidate_rows.append({
            "sym":     sym,
            "sec":     SECTOR.get(sym,"?"),
            "i":       i,
            "c":       c, "a": a, "e20": e20, "e50": e50, "e200": e200,
            "rsi":     rsi, "vr": vr,
            "macd_l":  arr_ml[i], "macd_s": arr_ms[i], "macd_h": arr_mh[i],
            "obv_r":   bool(arr_obvr[i]),
            "cpct":    arr_cpct[i],
            "dist_atr":dist_atr,
            "nifty_up":nifty_up(df.iloc[i]["date"]),
            "fwd10":   fwd10,
            "outcomes":outcomes,
        })

print(f"  {len(candidate_rows):,} candidate rows ready.\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FORWARD RETURN CORRELATION
# What indicator values at entry predict positive 10-day returns?
# ═════════════════════════════════════════════════════════════════════════════

cdf = pd.DataFrame([{k:v for k,v in r.items() if k!="outcomes"} for r in candidate_rows])

print("─"*72)
print("  SECTION 1: ENTRY CONDITION CORRELATION  (10-day forward return)")
print("─"*72)

# RSI
rsi_bins  = [0,40,45,50,55,60,65,70,100]
rsi_lbls  = ["<40","40-45","45-50","50-55","55-60","60-65","65-70",">70"]
cdf["rb"] = pd.cut(cdf["rsi"], bins=rsi_bins, labels=rsi_lbls)
rs = cdf.groupby("rb", observed=False)["fwd10"].agg(
    n="count",
    win=lambda x:(x>0).mean()*100,
    mean="mean",
    wm=lambda x:x[x>0].mean() if (x>0).any() else 0,
    lm=lambda x:x[x<=0].mean() if (x<=0).any() else 0,
)
print("\n  RSI at entry → 10d forward return:")
print(f"  {'RSI':^10}  {'N':>5}  {'Win%':>6}  {'Avg 10d':>8}  {'Avg Win':>8}  {'Avg Loss':>9}")
print(f"  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*9}")
best_rsi_win = rs.loc[rs["n"]>=30,"win"].max()
for lbl, row in rs.iterrows():
    if row["n"]<30: continue
    flag = "  ◀ BEST WIN%" if abs(row["win"]-best_rsi_win)<0.01 else ""
    print(f"  {str(lbl):^10}  {int(row['n']):>5}  {row['win']:>5.1f}%  "
          f"{row['mean']:>+7.2f}%  {row['wm']:>+7.2f}%  {row['lm']:>+8.2f}%{flag}")

# Volume ratio
vbins  = [0,0.8,1.0,1.2,1.5,2.0,2.5,100]
vlbls  = ["<0.8","0.8-1","1-1.2","1.2-1.5","1.5-2","2-2.5",">2.5"]
cdf["vb"] = pd.cut(cdf["vr"], bins=vbins, labels=vlbls)
vs = cdf.groupby("vb", observed=False)["fwd10"].agg(
    n="count", win=lambda x:(x>0).mean()*100, mean="mean"
)
print("\n  Volume ratio at entry → 10d forward return:")
print(f"  {'Vol Ratio':^10}  {'N':>5}  {'Win%':>6}  {'Avg 10d':>8}")
print(f"  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*8}")
bv = vs.loc[vs["n"]>=30,"win"].max()
for lbl,row in vs.iterrows():
    if row["n"]<30: continue
    flag = "  ◀ BEST" if abs(row["win"]-bv)<0.01 else ""
    print(f"  {str(lbl):^10}  {int(row['n']):>5}  {row['win']:>5.1f}%  {row['mean']:>+7.2f}%{flag}")

# EMA20 distance
db   = [-10,-0.5,-0.2,0,0.1,0.2,0.3,0.5,1.0,10]
dlbl = ["<-0.5","-0.5~-0.2","-0.2~0","0~0.1","0.1~0.2","0.2~0.3","0.3~0.5","0.5~1",">1"]
cdf["db"] = pd.cut(cdf["dist_atr"], bins=db, labels=dlbl)
ds = cdf.groupby("db", observed=False)["fwd10"].agg(
    n="count", win=lambda x:(x>0).mean()*100, mean="mean"
)
print("\n  Distance from EMA20 (ATR units) → 10d forward return:")
print(f"  {'EMA20 Dist':^12}  {'N':>5}  {'Win%':>6}  {'Avg 10d':>8}")
print(f"  {'─'*12}  {'─'*5}  {'─'*6}  {'─'*8}")
for lbl,row in ds.iterrows():
    if row["n"]<30: continue
    note = ""
    if "0~0.1" in str(lbl) or "0.1~0.2" in str(lbl): note = "  ◀ CLOSEST TO EMA20"
    print(f"  {str(lbl):^12}  {int(row['n']):>5}  {row['win']:>5.1f}%  {row['mean']:>+7.2f}%{note}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INITIAL SL ATR MULTIPLIER SWEEP
# Fix RR=2.0, adaptive=50%, T3=1.0×  →  sweep ATR mult only
# Shows: what happens to SL hit rate, win rate, expectancy as we widen SL
# ═════════════════════════════════════════════════════════════════════════════

print()
print("─"*72)
print("  SECTION 2: INITIAL SL PLACEMENT SWEEP  (fixed RR=2.0, T3=1×ATR)")
print("─"*72)
print("  Filter: close within ±0.3ATR of EMA20, RSI 38-58, vol<1.5, OBV rising")
print()

# Base PULLBACK filter
pb_mask = [
    r for r in candidate_rows
    if (-0.3 <= r["dist_atr"] <= 0.3) and (38 <= r["rsi"] <= 58)
    and r["vr"] < 1.5 and r["obv_r"]
]
print(f"  PULLBACK candidate setups: {len(pb_mask)}")
print()

RR_FIXED  = 2.0
ADP_FIXED = 0.50
T3_FIXED  = 1.00

print(f"  {'ATR Mult':<9}  {'N':>5}  {'SL%':>6}  {'Win%':>6}  "
      f"{'AvgWin':>7}  {'AvgLoss':>8}  {'Expectancy':>11}  Insight")
print(f"  {'─'*9}  {'─'*5}  {'─'*6}  {'─'*6}  "
      f"{'─'*7}  {'─'*8}  {'─'*11}  {'─'*20}")

sl_results = {}
for am in ATR_MULTS:
    pnls, rsns = [], []
    for r in pb_mask:
        tpl = r["outcomes"].get((am, RR_FIXED, ADP_FIXED, T3_FIXED))
        if tpl and tpl[0] is not None:
            pnls.append(tpl[0]); rsns.append(tpl[1])
    if len(pnls) < 10: continue
    wins   = [p for p in pnls if p>0]
    losses = [p for p in pnls if p<=0]
    sl_pct = sum(1 for r in rsns if "SL" in r)/len(rsns)*100
    win_pct= len(wins)/len(pnls)*100
    exp    = float(np.mean(pnls))
    sl_results[am] = exp
    curr_flag = "  ← CURRENT" if am==1.5 else ""
    print(f"  {am:<9.1f}  {len(pnls):>5}  {sl_pct:>5.1f}%  {win_pct:>5.1f}%  "
          f"{(np.mean(wins) if wins else 0):>+6.2f}%  "
          f"{(np.mean(losses) if losses else 0):>+7.2f}%  "
          f"{exp:>+10.3f}%{curr_flag}")

best_atr = max(sl_results, key=sl_results.get) if sl_results else 1.5
print(f"\n  Best ATR multiplier: {best_atr}×  (expectancy {sl_results.get(best_atr,0):+.3f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — T3 TRAIL MULTIPLIER SWEEP
# Fix entry conditions + initial SL = best_atr, sweep T3 trail ATR mult
# ═════════════════════════════════════════════════════════════════════════════

print()
print("─"*72)
print(f"  SECTION 3: T3 TRAILING SL MULTIPLIER SWEEP  (initial SL={best_atr}×ATR)")
print("─"*72)
print()
print(f"  {'T3 Mult':<8}  {'N':>5}  {'Win%':>6}  {'AvgWin':>7}  {'AvgLoss':>8}  {'Expectancy':>11}")
print(f"  {'─'*8}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*11}")

t3_results = {}
for t3 in T3_MULTS:
    pnls = []
    for r in pb_mask:
        tpl = r["outcomes"].get((best_atr, RR_FIXED, ADP_FIXED, t3))
        if tpl and tpl[0] is not None:
            pnls.append(tpl[0])
    if len(pnls)<10: continue
    wins  = [p for p in pnls if p>0]
    losses= [p for p in pnls if p<=0]
    exp   = float(np.mean(pnls))
    t3_results[t3] = exp
    curr_flag = "  ← CURRENT" if t3==1.0 else ""
    print(f"  {t3:<8.2f}  {len(pnls):>5}  {len(wins)/len(pnls)*100:>5.1f}%  "
          f"{(np.mean(wins) if wins else 0):>+6.2f}%  "
          f"{(np.mean(losses) if losses else 0):>+7.2f}%  "
          f"{exp:>+10.3f}%{curr_flag}")

best_t3 = max(t3_results, key=t3_results.get) if t3_results else 1.0
print(f"\n  Best T3 trail multiplier: {best_t3}×ATR  (expectancy {t3_results.get(best_t3,0):+.3f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — T1.5 ADAPTIVE TRAIL PCT SWEEP
# Fix SL=best_atr, T3=best_t3, sweep adaptive_pct
# ═════════════════════════════════════════════════════════════════════════════

print()
print("─"*72)
print(f"  SECTION 4: T1.5 ADAPTIVE TRAIL SWEEP  (SL={best_atr}×, T3={best_t3}×)")
print("  (How much of the gain above entry to lock in between T1 and T2)")
print("─"*72)
print()
print(f"  {'Adaptive%':<10}  {'N':>5}  {'Win%':>6}  {'AvgWin':>7}  {'AvgLoss':>8}  {'Expectancy':>11}")
print(f"  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*11}")

adp_results = {}
for adp in ADAPTIVE_PCTS:
    pnls = []
    for r in pb_mask:
        tpl = r["outcomes"].get((best_atr, RR_FIXED, adp, best_t3))
        if tpl and tpl[0] is not None:
            pnls.append(tpl[0])
    if len(pnls)<10: continue
    wins  = [p for p in pnls if p>0]
    losses= [p for p in pnls if p<=0]
    exp   = float(np.mean(pnls))
    adp_results[adp] = exp
    curr_flag = "  ← CURRENT" if adp==0.50 else ""
    print(f"  {adp*100:.0f}%{'':>7}  {len(pnls):>5}  {len(wins)/len(pnls)*100:>5.1f}%  "
          f"{(np.mean(wins) if wins else 0):>+6.2f}%  "
          f"{(np.mean(losses) if losses else 0):>+7.2f}%  "
          f"{exp:>+10.3f}%{curr_flag}")

best_adp = max(adp_results, key=adp_results.get) if adp_results else 0.50
print(f"\n  Best adaptive trail: {best_adp*100:.0f}%  (expectancy {adp_results.get(best_adp,0):+.3f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RR RATIO SWEEP
# Fix SL=best_atr, T3=best_t3, adp=best_adp
# ═════════════════════════════════════════════════════════════════════════════

print()
print("─"*72)
print(f"  SECTION 5: RR RATIO SWEEP  (SL={best_atr}×, T3={best_t3}×, adp={best_adp*100:.0f}%)")
print("  (RR controls how far the target is → when T2 SL tightening kicks in)")
print("─"*72)
print()
print(f"  {'RR Ratio':<9}  {'N':>5}  {'Win%':>6}  {'AvgWin':>7}  {'AvgLoss':>8}  "
      f"{'Expectancy':>11}  AvgHold")
print(f"  {'─'*9}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*11}  {'─'*7}")

rr_results = {}
for rr in RR_VALS:
    pnls, days_all = [], []
    for r in pb_mask:
        tpl = r["outcomes"].get((best_atr, rr, best_adp, best_t3))
        if tpl and tpl[0] is not None:
            pnls.append(tpl[0]); days_all.append(tpl[2])
    if len(pnls)<10: continue
    wins  = [p for p in pnls if p>0]
    losses= [p for p in pnls if p<=0]
    exp   = float(np.mean(pnls))
    rr_results[rr] = exp
    curr_flag = "  ← CURRENT" if rr==2.0 else ""
    print(f"  {rr:<9.1f}  {len(pnls):>5}  {len(wins)/len(pnls)*100:>5.1f}%  "
          f"{(np.mean(wins) if wins else 0):>+6.2f}%  "
          f"{(np.mean(losses) if losses else 0):>+7.2f}%  "
          f"{exp:>+10.3f}%  {np.mean(days_all):.1f}d{curr_flag}")

best_rr = max(rr_results, key=rr_results.get) if rr_results else 2.0
print(f"\n  Best RR ratio: {best_rr}  (expectancy {rr_results.get(best_rr,0):+.3f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RSI RANGE GRID (entry filter)
# Fix SL=best_atr, T3=best_t3, adp=best_adp, RR=best_rr
# Sweep RSI lo/hi and volume cap
# ═════════════════════════════════════════════════════════════════════════════

print()
print("─"*72)
print(f"  SECTION 6: RSI RANGE + VOLUME THRESHOLD GRID")
print(f"  (SL={best_atr}×, T3={best_t3}×, adp={best_adp*100:.0f}%, RR={best_rr})")
print("─"*72)
print()

RSI_RANGES = [(35,50),(38,52),(40,52),(40,55),(42,55),(45,55),(45,58),(45,60),(48,58),(50,60)]
VOL_CAPS   = [1.0, 1.2, 1.5, 2.0]

entry_grid = []
for rlo,rhi in RSI_RANGES:
    for vcap in VOL_CAPS:
        pnls = []
        for r in candidate_rows:
            if not (r["e20"]>r["e50"] and r["c"]>r["e200"]): continue
            if not (-0.3 <= r["dist_atr"] <= 0.3): continue
            if not (rlo <= r["rsi"] <= rhi): continue
            if not (r["vr"] < vcap and r["obv_r"]): continue
            tpl = r["outcomes"].get((best_atr, best_rr, best_adp, best_t3))
            if tpl and tpl[0] is not None:
                pnls.append(tpl[0])
        if len(pnls)<15: continue
        wins   = [p for p in pnls if p>0]
        losses = [p for p in pnls if p<=0]
        entry_grid.append({
            "rlo":rlo,"rhi":rhi,"vcap":vcap,"n":len(pnls),
            "win":len(wins)/len(pnls)*100,
            "aw":float(np.mean(wins)) if wins else 0,
            "al":float(np.mean(losses)) if losses else 0,
            "exp":float(np.mean(pnls)),
        })

eg = pd.DataFrame(entry_grid).sort_values("exp",ascending=False)

print(f"  {'RSI Range':<10}  {'Vol<':<5}  {'N':>5}  {'Win%':>6}  "
      f"{'AvgWin':>7}  {'AvgLoss':>8}  {'Expectancy':>11}")
print(f"  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*11}")
for _,row in eg.head(20).iterrows():
    curr = (row["rlo"]==40 and row["rhi"]==52 and row["vcap"]==1.5)
    flag = "  ← CURRENT" if curr else ""
    print(f"  {row['rlo']:.0f}–{row['rhi']:.0f}{'':>6}  {row['vcap']:<5.1f}  {int(row['n']):>5}  "
          f"{row['win']:>5.1f}%  {row['aw']:>+6.2f}%  {row['al']:>+7.2f}%  "
          f"{row['exp']:>+10.3f}%{flag}")

if not eg.empty:
    curr_row = eg[(eg["rlo"]==40)&(eg["rhi"]==52)&(eg["vcap"]==1.5)]
    if not curr_row.empty:
        rank = list(eg.index).index(curr_row.index[0])+1
        print(f"\n  Current entry params rank: #{rank} of {len(eg)} tested")

best_entry = eg.iloc[0] if not eg.empty else None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BEST STOCKS  (using optimal params found above)
# ═════════════════════════════════════════════════════════════════════════════

print()
print("─"*72)
print("  SECTION 7: BEST AND WORST STOCKS  (optimal params)")
print("─"*72)

rlo = int(best_entry["rlo"]) if best_entry is not None else 40
rhi = int(best_entry["rhi"]) if best_entry is not None else 52
vcap= float(best_entry["vcap"]) if best_entry is not None else 1.5

by_sym = defaultdict(list)
for r in candidate_rows:
    if not (r["e20"]>r["e50"] and r["c"]>r["e200"]): continue
    if not (-0.3 <= r["dist_atr"] <= 0.3): continue
    if not (rlo <= r["rsi"] <= rhi): continue
    if not (r["vr"]<vcap and r["obv_r"]): continue
    tpl = r["outcomes"].get((best_atr, best_rr, best_adp, best_t3))
    if tpl and tpl[0] is not None:
        by_sym[r["sym"]].append(tpl[0])

sym_rows = []
for sym,pnls in by_sym.items():
    if len(pnls)<6: continue
    wins = [p for p in pnls if p>0]
    sym_rows.append({
        "sym":sym,"sec":SECTOR.get(sym,"?"),"n":len(pnls),
        "win":len(wins)/len(pnls)*100,
        "exp":float(np.mean(pnls)),
    })
sym_rows.sort(key=lambda x:x["exp"],reverse=True)

print()
print(f"  {'Symbol':<14}  {'Sector':<14}  {'N':>5}  {'Win%':>6}  {'Expectancy':>11}  Verdict")
print(f"  {'─'*14}  {'─'*14}  {'─'*5}  {'─'*6}  {'─'*11}  {'─'*12}")
for r in sym_rows:
    v = "✓ Trade" if r["exp"]>0 else "✗ Avoid"
    print(f"  {r['sym']:<14}  {r['sec']:<14}  {r['n']:>5}  {r['win']:>5.1f}%  "
          f"{r['exp']:>+10.3f}%  {v}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SECTOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

print()
print("─"*72)
print("  SECTION 8: SECTOR ANALYSIS")
print("─"*72)
by_sec = defaultdict(list)
for r in sym_rows:
    by_sec[r["sec"]].append(r)
sec_rows = []
for sec, syms in by_sec.items():
    all_pnls = []
    for r in candidate_rows:
        if SECTOR.get(r["sym"])!=sec: continue
        if not (-0.3<=r["dist_atr"]<=0.3 and rlo<=r["rsi"]<=rhi and r["vr"]<vcap and r["obv_r"]): continue
        tpl = r["outcomes"].get((best_atr,best_rr,best_adp,best_t3))
        if tpl and tpl[0] is not None:
            all_pnls.append(tpl[0])
    if len(all_pnls)<10:
        continue
    wins = [p for p in all_pnls if p>0]
    sec_rows.append({
        "sec":sec,"n":len(all_pnls),
        "win":len(wins)/len(all_pnls)*100,
        "exp":float(np.mean(all_pnls)),
    })
sec_rows.sort(key=lambda x:x["exp"],reverse=True)
print()
print(f"  {'Sector':<16}  {'N':>5}  {'Win%':>6}  {'Expectancy':>11}  Verdict")
print(f"  {'─'*16}  {'─'*5}  {'─'*6}  {'─'*11}  {'─'*12}")
for r in sec_rows:
    v = "✓ Trade" if r["exp"]>0 and r["win"]>=48 else ("Marginal" if r["exp"]>0 else "✗ Avoid")
    print(f"  {r['sec']:<16}  {r['n']:>5}  {r['win']:>5.1f}%  {r['exp']:>+10.3f}%  {v}")


# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print()
print("="*72)
print("  OPTIMAL PARAMETER SUMMARY  (data-driven, 2022–2024, SL-only exits)")
print("="*72)

best_rlo  = int(best_entry["rlo"])  if best_entry is not None else 40
best_rhi  = int(best_entry["rhi"])  if best_entry is not None else 52
best_vcap = float(best_entry["vcap"]) if best_entry is not None else 1.5

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  ENTRY CONDITIONS (strategy.py → _check_pullback)              │
  ├─────────────────────────────────────────────────────────────────┤
  │  RSI range:      current 40–52   →  optimal {best_rlo}–{best_rhi}           │
  │  Volume cap:     current <1.5×   →  optimal <{best_vcap:.1f}×           │
  │  EMA20 distance: current 0–0.3×ATR  (no change needed)         │
  ├─────────────────────────────────────────────────────────────────┤
  │  RISK PARAMETERS (config.py)                                    │
  ├─────────────────────────────────────────────────────────────────┤
  │  ATR_MULT['PULLBACK']: current 1.5×  →  optimal {best_atr}×          │
  │  MIN_RR_RATIO:         current 2.0   →  optimal {best_rr}           │
  ├─────────────────────────────────────────────────────────────────┤
  │  TRAILING SL (monitor.py → _update_trailing_sl)                │
  ├─────────────────────────────────────────────────────────────────┤
  │  T1.5 adaptive %:  current 50%  →  optimal {best_adp*100:.0f}%          │
  │  T3 trail ATR×:    current 1.0× →  optimal {best_t3}×          │
  └─────────────────────────────────────────────────────────────────┘

  HOW EACH CHANGE HELPS:
  ─────────────────────
  ATR_MULT {best_atr}×  : SL placed further below EMA20 → fewer whipsaw exits
               → allows trade to breathe through intraday noise
  RSI {best_rlo}–{best_rhi}  : captures stocks that have genuinely cooled off at EMA20
               → avoids entries on exhausted moves
  Vol <{best_vcap:.1f}× : lower volume on pullback = clean consolidation dip
               → not panic selling / distribution
  RR  {best_rr}    : target is {best_rr}×risk away → T2 tightening only kicks in
               after a real move, not near-entry noise
  T3  {best_t3}×    : trails at {best_t3}×ATR below close → gives winners room to run
               without giving back too much

  TOP STOCKS TO FOCUS ON (PULLBACK, optimal params):""")

for r in sym_rows[:8]:
    print(f"  {'✓' if r['exp']>0 else '✗'} {r['sym']:<14}  win={r['win']:.0f}%  "
          f"exp={r['exp']:+.3f}%  ({r['sec']})")

print(f"""
  SECTORS TO TRADE vs AVOID:""")
for r in sec_rows:
    icon = "✓" if r["exp"]>0 and r["win"]>=48 else ("~" if r["exp"]>0 else "✗")
    print(f"  {icon} {r['sec']:<16}  win={r['win']:.0f}%  exp={r['exp']:+.3f}%")

print(f"""
  APPLY IN THIS ORDER:
  1. config.py   →  ATR_MULT['PULLBACK'] = {best_atr}
  2. config.py   →  MIN_RR_RATIO = {best_rr}
  3. strategy.py →  _check_pullback(): RSI {best_rlo}–{best_rhi}, vol <{best_vcap:.1f}×
  4. monitor.py  →  _update_trailing_sl(): adaptive_pct = {best_adp:.2f}, T3 = {best_t3}×ATR
  5. Re-run backtest to verify improvement
  6. Paper trade 2 weeks before going live
""")
print("="*72)
