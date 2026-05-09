"""
main.py — Entry point. Run this file to start the trading system.

Usage:
  python main.py                  Default: run one full cycle and print dashboard.
                                  Safe for testing — PAPER_TRADE stays True.

  python main.py --auth           One-time Upstox auth setup (or force token refresh).
                                  Run this before first use, or whenever the stored
                                  token is lost. Saves token to DB for daily auto-refresh.

  python main.py --scheduler      Run the full auto scheduler in paper trade mode.
                                  Schedules all 8 steps on weekdays automatically.
                                  Monitoring runs every 15 mins during market hours.

  python main.py --live           Run the auto scheduler in LIVE mode.
                                  CAUTION: real orders will be placed at Upstox.
                                  Requires explicit Enter confirmation before starting.

  python main.py --dashboard      Just print today's analytics dashboard and exit.
                                  No trading, no data collection. Quick status check.

Required environment variables:
  UPSTOX_API_KEY       — from https://developer.upstox.com
  UPSTOX_API_SECRET    — from developer portal
  UPSTOX_REDIRECT_URI  — registered redirect URI (default: http://127.0.0.1:8765/callback)

For fully headless daily auth (no browser required):
  UPSTOX_MOBILE        — your Upstox login mobile number
  UPSTOX_PASSWORD      — your Upstox account password
  UPSTOX_PIN           — your 6-digit Upstox MPIN / trading PIN
  UPSTOX_TOTP_SECRET   — base32 TOTP secret from Upstox 2FA setup

  Uses the upstox-totp library (pip install upstox-totp) for robust headless auth.
  If UPSTOX_PASSWORD is omitted, UPSTOX_PIN is used for both fields (works if
  your account password and trading PIN are the same value).

First-time setup:
  1. Set env vars above
  2. python main.py --auth        (opens browser for one-time login, saves token)
  3. python main.py --scheduler   (auto-refreshes token daily from here on)
"""

import sys
import subprocess
import importlib.util
import os


def _ensure_deps():
    """
    Auto-installs any missing packages from requirements.txt before the bot starts.
    Runs silently when all deps are present (fast importlib check, no pip call).
    Only triggers pip when something is actually missing.
    """
    # Map: import name → pip package name (only where they differ)
    _pip_name = {
        "upstox_client": "upstox-python-sdk",
        "upstox_totp":   "upstox-totp",
        "sklearn":       "scikit-learn",
    }
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(req_file):
        return

    with open(req_file) as f:
        packages = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    missing = []
    for pkg in packages:
        # Derive the importable module name from the pip package name
        import_name = pkg.replace("-", "_").split("==")[0].split(">=")[0].split("<=")[0]
        import_name = _pip_name.get(import_name, import_name)
        if importlib.util.find_spec(import_name) is None:
            missing.append(pkg)

    if not missing:
        return

    print(f"[setup] Installing {len(missing)} missing package(s): {', '.join(missing)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *missing],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[setup] pip install failed:\n{result.stderr}")
        sys.exit(1)
    print("[setup] All dependencies installed. Continuing...\n")


_ensure_deps()

from config import Config       # noqa: E402 — must come after deps are installed
from system import TradingSystem  # noqa: E402


def run_auth_setup():
    """
    Forces an Upstox token refresh. Use this on first run, or whenever the stored
    token is missing/corrupted. After this succeeds, the scheduler refreshes daily.
    """
    from database import DatabaseManager
    from upstox_auth import UpstoxAuth
    db    = DatabaseManager()
    auth  = UpstoxAuth(db)
    # Clear any stale token so get_valid_token() is forced to re-authenticate
    db.set_state("upstox_token",       "")
    db.set_state("upstox_token_expiry", "")
    token = auth.get_valid_token()
    if token:
        print(f"\nUpstox auth successful. Token saved to DB.")
        print("You can now run: python main.py --scheduler")
    else:
        print("\nAuth failed — check env vars and try again.")
        print("Required: UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_REDIRECT_URI")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "--auth":
        run_auth_setup()
        return

    system = TradingSystem()

    if mode == "--live":
        print("LIVE MODE — Real orders will be placed at Upstox!")
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
