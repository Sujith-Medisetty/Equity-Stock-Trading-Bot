"""
upstox_auth.py — Automated Upstox OAuth2 token management for bot operation.

Upstox uses OAuth2. The access token expires daily (~3:30 AM next day).
A bot can't do a manual browser login every morning, so this module automates it.

Two-step operation:
  1. Check DB for a stored, non-expired token → return it immediately.
  2. If expired / missing → run the automated login flow:
       a) upstox-totp library logs in with mobile + password + PIN + TOTP
       b) Library returns the access token directly — no manual code exchange needed
       c) Save the new token to DB (valid until next 3:30 AM)

Fallback: if headless login fails (wrong creds, package issue), a local HTTP server
starts and prints a URL — operator opens it in a browser once, logs in manually,
and the server catches the redirect automatically. After that the token is in DB
and automated refresh runs daily.

Required env vars:
  UPSTOX_API_KEY       — your app's API key (from Upstox developer portal)
  UPSTOX_API_SECRET    — your app's API secret
  UPSTOX_REDIRECT_URI  — registered redirect URI (default http://127.0.0.1:8765/callback)
  UPSTOX_MOBILE        — your Upstox login mobile number (for headless mode)
  UPSTOX_PASSWORD      — your Upstox account password (for headless mode)
  UPSTOX_PIN           — your 6-digit Upstox MPIN / trading PIN (for headless mode)
  UPSTOX_TOTP_SECRET   — base32 TOTP secret (scan Upstox 2FA QR → store the raw secret)

Install required packages:
  pip install upstox-totp upstox-python-sdk pyotp
  upstox-totp repo: https://github.com/batpool/upstox-totp

One-time first-run setup:
  python main.py --auth
  If headless creds aren't set, the printed URL needs one manual browser login.
  After that, daily refresh is fully automatic.
"""

import os
import urllib.parse
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Tuple

import requests

from config import log
from database import DatabaseManager

# upstox-totp: handles the complete Upstox OAuth TOTP flow (mobile + password + PIN + TOTP → token)
# Repo: https://github.com/batpool/upstox-totp  |  Install: pip install upstox-totp
try:
    from upstox_totp import UpstoxTOTP, UpstoxError, ConfigurationError
    from pydantic import SecretStr
    UPSTOX_TOTP_AVAILABLE = True
except ImportError:
    UPSTOX_TOTP_AVAILABLE = False
    log.warning(
        "upstox-totp not installed — headless login disabled. "
        "pip install upstox-totp  (https://github.com/batpool/upstox-totp)"
    )

_AUTH_BASE   = "https://api.upstox.com/v2"
_DIALOG_URL  = f"{_AUTH_BASE}/login/authorization/dialog"
_TOKEN_URL   = f"{_AUTH_BASE}/login/authorization/token"

_DB_TOKEN_KEY  = "upstox_token"
_DB_EXPIRY_KEY = "upstox_token_expiry"
_REFRESH_BUFFER_MINS = 30   # refresh this many minutes before actual expiry


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that captures the ?code= from Upstox's OAuth redirect."""
    captured_code: Optional[str] = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            _OAuthCallbackHandler.captured_code = params["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<h2>Upstox auth successful. You can close this tab.</h2>")
            log.info("OAuth callback received — auth code captured.")
        else:
            error = params.get("error", ["unknown"])[0]
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"<h2>Auth failed: {error}</h2>".encode())
            log.error(f"OAuth callback received error: {error}")

    def log_message(self, fmt, *args):
        pass  # silence default HTTP access log — our logger handles messaging


class UpstoxAuth:
    """
    Manages the Upstox access token lifecycle for the trading bot.
    One instance is created in DataCollector and OrderManager at startup.
    They share the same DatabaseManager so the token is written once and read by both.

    Usage:
        token = auth.get_valid_token()
        if not token:
            log.error("Cannot connect to Upstox — authentication failed")
            return
        configuration.access_token = token
    """

    def __init__(self, db: DatabaseManager):
        self.db           = db
        self.api_key      = os.getenv("UPSTOX_API_KEY",      "")
        self.api_secret   = os.getenv("UPSTOX_API_SECRET",   "")
        self.redirect_uri = os.getenv("UPSTOX_REDIRECT_URI", "http://127.0.0.1:8765/callback")
        self.mobile       = os.getenv("UPSTOX_MOBILE",       "")
        self.password     = os.getenv("UPSTOX_PASSWORD",     "")
        self.pin          = os.getenv("UPSTOX_PIN",          "")
        self.totp_secret  = os.getenv("UPSTOX_TOTP_SECRET",  "")
        self._port        = int(urllib.parse.urlparse(self.redirect_uri).port or 8765)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_valid_token(self) -> Optional[str]:
        """
        Returns a valid Upstox access token.
        Loads from DB if still fresh; triggers automated re-auth if expired.
        Token is refreshed REFRESH_BUFFER_MINS (30 min) before actual expiry
        so we never use a token in its last moments.
        Returns None if all auth attempts fail.
        """
        token, expires_at = self._load_token_from_db()

        if token and expires_at:
            if datetime.now() < expires_at - timedelta(minutes=_REFRESH_BUFFER_MINS):
                return token
            log.info(f"Upstox token expires at {expires_at:%H:%M} — refreshing now.")

        if not self.api_key or not self.api_secret:
            log.error(
                "UPSTOX_API_KEY / UPSTOX_API_SECRET not set in environment. "
                "Cannot authenticate with Upstox."
            )
            return None

        log.info("Upstox token missing or expired — starting automated login...")
        new_token = self._run_automated_login()

        if new_token:
            log.info("Upstox token refreshed and saved.")
        else:
            log.error(
                "Upstox automated login FAILED after all attempts.\n"
                "To manually set a token for today, run:\n"
                "  python -c \"from upstox_auth import UpstoxAuth; "
                "from database import DatabaseManager; "
                "UpstoxAuth(DatabaseManager()).save_token('YOUR_TOKEN')\""
            )
        return new_token

    def save_token(self, token: str, expires_in_seconds: int = 86400):
        """
        Persists a token to DB. Called automatically after successful login,
        or manually by the operator to inject a token obtained outside the bot.
        Upstox tokens last ~24h (expire at 3:30 AM next day).
        """
        expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        self.db.set_state(_DB_TOKEN_KEY,  token)
        self.db.set_state(_DB_EXPIRY_KEY, expires_at.isoformat())
        log.info(f"Upstox token saved to DB. Expires at: {expires_at:%Y-%m-%d %H:%M}")

    # -------------------------------------------------------------------------
    # Internal flow
    # -------------------------------------------------------------------------

    def _load_token_from_db(self) -> Tuple[Optional[str], Optional[datetime]]:
        """Returns (token, expiry_datetime) from DB, or (None, None) if not stored."""
        token      = self.db.get_state(_DB_TOKEN_KEY)
        expiry_str = self.db.get_state(_DB_EXPIRY_KEY)
        if not token or not expiry_str:
            return None, None
        try:
            return token, datetime.fromisoformat(expiry_str)
        except Exception:
            return None, None

    def _run_automated_login(self) -> Optional[str]:
        """
        Tries two login strategies in order:
        1. Headless (fully automated): uses upstox-totp library with
           MOBILE + PASSWORD + PIN + TOTP_SECRET env vars.
        2. Browser-assisted: starts a local HTTP server, prints a URL for the operator
           to open in a browser. Catches the redirect automatically.
           The operator only needs to do this once per day (token saved to DB).
        """
        headless_ready = (
            UPSTOX_TOTP_AVAILABLE
            and self.mobile
            and self.pin
            and self.totp_secret
        )
        if headless_ready:
            log.info("Attempting headless Upstox login via upstox-totp library...")
            token = self._headless_login_via_library()
            if token:
                return token
            log.warning("Headless login failed — falling back to browser-assisted login.")

        return self._browser_assisted_login()

    def _headless_login_via_library(self) -> Optional[str]:
        """
        Uses the upstox-totp library to perform the complete OAuth TOTP flow:
        mobile + password + PIN + TOTP → access token directly.

        UPSTOX_PASSWORD is the Upstox account password.
        UPSTOX_PIN is the 6-digit MPIN / trading PIN.
        These are different fields — set both in your env.
        If UPSTOX_PASSWORD is not set, falls back to UPSTOX_PIN for the password field,
        which works if your account password and trading PIN are the same.
        """
        password = self.password or self.pin   # backward compat if only PIN is set
        try:
            upx = UpstoxTOTP(
                username=self.mobile,
                password=SecretStr(password),
                pin_code=SecretStr(self.pin),
                totp_secret=SecretStr(self.totp_secret),
                client_id=self.api_key,
                client_secret=SecretStr(self.api_secret),
                redirect_uri=self.redirect_uri,
            )
            response = upx.app_token.get_access_token()
            if response.success:
                token = response.data.access_token
                self.save_token(token)
                return token

            log.error(f"upstox-totp login failed: {response}")
            return None

        except ConfigurationError as e:
            log.error(f"upstox-totp config error (check env vars): {e}")
            return None
        except UpstoxError as e:
            log.error(f"upstox-totp auth error: {e}")
            return None
        except Exception as e:
            log.error(f"upstox-totp unexpected error: {e}")
            return None

    def _browser_assisted_login(self) -> Optional[str]:
        """
        Starts a local HTTP server on self._port to catch the OAuth redirect.
        Operator must open the printed URL in a browser and log in with Upstox credentials.
        After login, Upstox redirects to redirect_uri?code=XXX — the server captures it.
        This is a one-time action per day; after the token is saved in DB, the
        headless flow runs automatically at next expiry.
        """
        _OAuthCallbackHandler.captured_code = None

        try:
            server = HTTPServer(("127.0.0.1", self._port), _OAuthCallbackHandler)
        except OSError as e:
            log.error(
                f"Cannot start local auth server on port {self._port}: {e}\n"
                f"Check that nothing else is using port {self._port}."
            )
            return None

        server.timeout = 300  # 5 minutes for the operator to log in

        auth_url = (
            f"{_DIALOG_URL}?response_type=code"
            f"&client_id={self.api_key}"
            f"&redirect_uri={urllib.parse.quote(self.redirect_uri)}"
            f"&state=bot"
        )
        print("\n" + "=" * 70)
        print("UPSTOX AUTH REQUIRED")
        print("Open this URL in your browser and log in with your Upstox credentials:")
        print(f"\n  {auth_url}\n")
        print(f"Waiting up to 5 minutes for the login redirect on port {self._port}...")
        print("=" * 70 + "\n")

        server.handle_request()
        server.server_close()

        code = _OAuthCallbackHandler.captured_code
        if not code:
            log.error("Browser-assisted login timed out or was cancelled — no auth code received.")
            return None

        return self._exchange_code_for_token(code)

    def _exchange_code_for_token(self, code: str) -> Optional[str]:
        """
        Exchanges the OAuth2 authorization code for an Upstox access token.
        Used only by the browser fallback path (library handles this internally for headless).
        """
        try:
            resp = requests.post(
                _TOKEN_URL,
                data={
                    "code":          code,
                    "client_id":     self.api_key,
                    "client_secret": self.api_secret,
                    "redirect_uri":  self.redirect_uri,
                    "grant_type":    "authorization_code",
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept":       "application/json",
                },
                timeout=15
            )
            resp.raise_for_status()
            data  = resp.json()
            token = data.get("access_token")
            exp   = int(data.get("expires_in", 86400))
            if not token:
                log.error(f"Token exchange response missing access_token field: {data}")
                return None
            self.save_token(token, exp)
            return token

        except requests.HTTPError as e:
            log.error(f"Token exchange HTTP error {e.response.status_code}: {e.response.text[:300]}")
            return None
        except requests.RequestException as e:
            log.error(f"Token exchange network error: {e}")
            return None
