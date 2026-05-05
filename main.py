"""
main.py — Entry point. Run this file to start the trading system.

Usage:
  python main.py                  Default: run one full cycle and print dashboard.
                                  Safe for testing — PAPER_TRADE stays True.

  python main.py --scheduler      Run the full auto scheduler in paper trade mode.
                                  Schedules all 8 steps on weekdays automatically.
                                  Monitoring runs every 15 mins during market hours.

  python main.py --live           Run the auto scheduler in LIVE mode.
                                  CAUTION: real orders will be placed at Dhan.
                                  Requires explicit Enter confirmation before starting.
                                  Requires DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN set
                                  as environment variables.

  python main.py --dashboard      Just print today's analytics dashboard and exit.
                                  No trading, no data collection. Quick status check.

Environment variables required for live/paper trading with real data:
  export DHAN_CLIENT_ID="your_client_id"
  export DHAN_ACCESS_TOKEN="your_access_token"
  Without these, the system uses mock OHLCV data and skips all Dhan API calls.
"""

import sys
from config import Config
from system import TradingSystem


def main():
    system = TradingSystem()

    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "--live":
        print("LIVE MODE — Real orders will be placed!")
        print("Press Enter to confirm or Ctrl+C to cancel...")
        input()
        Config.PAPER_TRADE = False
        system.start_scheduler()

    elif mode == "--scheduler":
        print("Starting scheduler in PAPER TRADE mode...")
        system.start_scheduler()

    elif mode == "--dashboard":
        system.analytics.print_dashboard()

    else:
        system.run_once_test()


if __name__ == "__main__":
    main()
