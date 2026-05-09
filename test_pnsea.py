"""
test_pnsea.py — Quick test to verify pnsea can reach NSE endpoints from a server.

Run: python3 test_pnsea.py

Checks:
  1. India VIX       — via nse.equity.find_index()
  2. FII/DII data    — via nse.endpoint_tester() with correct endpoint
  3. Events calendar — via nse.endpoint_tester() with correct endpoint

If all 3 print real data, we can swap pnsea into data_collector.py.
"""

from datetime import datetime, timedelta

try:
    from pnsea import NSE
except ImportError:
    print("FAIL: pnsea not installed. Run: pip install pnsea")
    exit(1)

nse = NSE()

print("\n" + "=" * 50)
print("  PNSEA NSE ENDPOINT TEST")
print("=" * 50)

# ── 1. India VIX ─────────────────────────────────────
print("\n[1] India VIX")
try:
    vix_data = nse.equity.find_index("INDIA VIX")
    if vix_data:
        last = vix_data.get("last") or vix_data.get("last_price") or vix_data.get("lastPrice")
        print(f"    OK  — VIX = {last}")
        print(f"    Raw keys: {list(vix_data.keys())}")
    else:
        print("    FAIL — returned empty/None")
except Exception as e:
    print(f"    FAIL — {e}")

# ── 2. FII / DII ─────────────────────────────────────
print("\n[2] FII / DII")
try:
    resp = nse.endpoint_tester("https://www.nseindia.com/api/fiidiiTradeReact")
    data = resp.json()
    if data and isinstance(data, list) and len(data) > 0:
        fii = next((x for x in data if "FII" in x.get("category", "")), data[0])
        print(f"    OK  — {fii.get('category')} | date: {fii.get('date')} | net: {fii.get('netValue')}")
        print(f"    Full response ({len(data)} entries): {data}")
    else:
        print(f"    FAIL — unexpected response: {data}")
except Exception as e:
    print(f"    FAIL — {e}")

# ── 3. Events Calendar ───────────────────────────────
print("\n[3] Events Calendar (next 30 days)")
try:
    today  = datetime.now().strftime("%d-%m-%Y")
    future = (datetime.now() + timedelta(days=30)).strftime("%d-%m-%Y")
    url    = f"https://www.nseindia.com/api/event-calendar?index=equities&from_date={today}&to_date={future}"
    resp   = nse.endpoint_tester(url)
    data   = resp.json()
    items  = data if isinstance(data, list) else data.get("data", [])
    watchlist = [
        "ICICIBANK", "HDFCBANK", "AXISBANK", "INFY", "HCLTECH",
        "TATAMOTORS", "MARUTI", "RELIANCE", "BHARTIARTL", "SUNPHARMA",
        "BAJFINANCE", "LT", "ITC", "TITAN", "TCS"
    ]
    hits = [x for x in items if x.get("symbol") in watchlist]
    if items:
        print(f"    OK  — {len(items)} total events, {len(hits)} in our watchlist")
        for h in hits[:3]:
            print(f"          {h.get('symbol')} | {h.get('purpose')} | {h.get('date')}")
    else:
        print(f"    FAIL — empty response. Raw: {data}")
except Exception as e:
    print(f"    FAIL — {e}")

print("\n" + "=" * 50)
print("  Done. If all 3 show OK, pnsea works from this server.")
print("=" * 50 + "\n")
