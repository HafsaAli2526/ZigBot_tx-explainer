"""
Token Registry — Exponent-Aware Amount Formatting

Loads token metadata (name, symbol, exponent) and provides human-readable
amount formatting using the correct decimal places.

Data sources (priority order):
  1. Degenter API: https://dev-api.degenter.io/tokens?limit=1000
  2. ZigChain LCD (authoritative for exponent):
     https://zigchain-mainnet-lcd.zigscan.net/cosmos/bank/v1beta1/denoms_metadata/<denom>

Exponent rules:
  - exponent 0: indivisible token, raw units = display units
  - exponent 6: micro-unit token, 1,000,000 raw = 1 display (e.g., uzig → ZIG)
"""

import requests
from threading import Lock

# ══════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════

DEGENTER_API = "https://dev-api.degenter.io/tokens"
LCD_BASE = "https://zigchain-mainnet-lcd.zigscan.net"
FETCH_TIMEOUT = 15  # seconds

# Known fallbacks for core tokens (used if both APIs are unreachable)
_KNOWN_TOKENS = {
    "uzig": {"symbol": "ZIG", "exponent": 6, "name": "ZIG"},
}


# ══════════════════════════════════════════════
# Token Registry
# ══════════════════════════════════════════════

class TokenRegistry:
    """Thread-safe, lazily-populated token metadata cache.

    Usage:
        registry = TokenRegistry()
        registry.load()                             # bulk load from API
        formatted = registry.format_amount(5000000, "uzig")  # "5.000000 ZIG"
    """

    def __init__(self):
        self._cache: dict[str, dict] = dict(_KNOWN_TOKENS)
        self._lock = Lock()
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def token_count(self) -> int:
        return len(self._cache)

    # ──────────────────────────────────────────
    # Bulk load from Degenter API
    # ──────────────────────────────────────────

    def load(self, verbose: bool = False) -> int:
        """Bulk-load token metadata from the Degenter API.

        Returns the number of tokens loaded.
        """
        try:
            resp = requests.get(
                f"{DEGENTER_API}?limit=1000",
                timeout=FETCH_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            tokens = data if isinstance(data, list) else data.get("tokens", data.get("data", []))

            count = 0
            with self._lock:
                for token in tokens:
                    denom = token.get("denom") or token.get("base_denom") or token.get("id")
                    if not denom:
                        continue

                    entry = {
                        "symbol": token.get("symbol") or token.get("display", denom),
                        "exponent": token.get("exponent", token.get("decimals", 0)),
                        "name": token.get("name") or token.get("symbol", denom),
                    }

                    # Don't overwrite known tokens with potentially wrong data
                    if denom not in _KNOWN_TOKENS:
                        self._cache[denom] = entry
                    count += 1

                self._loaded = True

            if verbose:
                print(f"  ✓ Loaded {count} tokens from Degenter API")
            return count

        except Exception as e:
            if verbose:
                print(f"  ✗ Token load failed ({type(e).__name__}: {e})")
            self._loaded = True  # mark as loaded even on failure — we have fallbacks
            return 0

    # ──────────────────────────────────────────
    # LCD verification (on-demand per denom)
    # ──────────────────────────────────────────

    def _fetch_from_lcd(self, denom: str) -> dict | None:
        """Fetch and parse denom metadata from ZigChain LCD.

        The LCD is the authoritative source for exponent.
        Returns {"symbol": str, "exponent": int, "name": str} or None.
        """
        try:
            url = f"{LCD_BASE}/cosmos/bank/v1beta1/denoms_metadata/{denom}"
            resp = requests.get(url, timeout=FETCH_TIMEOUT)
            resp.raise_for_status()
            metadata = resp.json().get("metadata", {})

            denom_units = metadata.get("denom_units", [])
            if not denom_units:
                return None

            # Find the display unit: the one that does NOT start with "coin."
            # and has the highest exponent
            display_unit = None
            for unit in denom_units:
                unit_denom = unit.get("denom", "")
                if not unit_denom.startswith("coin.") and unit.get("exponent", 0) > 0:
                    display_unit = unit
                    break

            # If no display unit found, use exponent 0
            exponent = display_unit.get("exponent", 0) if display_unit else 0
            symbol = metadata.get("symbol") or metadata.get("display") or denom
            name = metadata.get("name") or symbol

            return {"symbol": symbol, "exponent": exponent, "name": name}

        except Exception:
            return None

    # ──────────────────────────────────────────
    # Token lookup
    # ──────────────────────────────────────────

    def get_token(self, denom: str) -> dict:
        """Get token metadata for a denom. Tries cache → LCD → fallback.

        Always returns a dict with {symbol, exponent, name}.
        """
        with self._lock:
            if denom in self._cache:
                return self._cache[denom]

        # Not in cache — try LCD
        lcd_data = self._fetch_from_lcd(denom)
        if lcd_data:
            with self._lock:
                self._cache[denom] = lcd_data
            return lcd_data

        # Fallback — unknown token, exponent 0
        fallback = {
            "symbol": _display_denom(denom),
            "exponent": 0,
            "name": denom,
        }
        with self._lock:
            self._cache[denom] = fallback
        return fallback

    def get_exponent(self, denom: str) -> int:
        """Get the exponent for a denom."""
        return self.get_token(denom)["exponent"]

    def get_symbol(self, denom: str) -> str:
        """Get the display symbol for a denom."""
        return self.get_token(denom)["symbol"]

    # ──────────────────────────────────────────
    # Amount formatting
    # ──────────────────────────────────────────

    def format_amount(self, raw_amount: int, denom: str) -> str:
        """Convert a raw on-chain amount to human-readable format.

        Examples:
            format_amount(33694976, "uzig")     → "33.694976 ZIG"
            format_amount(25, "somecoin")       → "25 somecoin"  (exponent 0)
            format_amount(1500000, "factory/...stZIG")  → depends on registry
        """
        token = self.get_token(denom)
        exponent = token["exponent"]
        symbol = token["symbol"]

        if exponent == 0:
            return f"{raw_amount:,} {symbol}"

        divisor = 10 ** exponent
        display_amount = raw_amount / divisor

        # Format with the right number of decimal places
        # Strip unnecessary trailing zeros but keep at least 2 decimals
        formatted = f"{display_amount:,.{exponent}f}"

        # Remove trailing zeros after decimal, keep min 2
        if "." in formatted:
            integer_part, decimal_part = formatted.split(".")
            decimal_part = decimal_part.rstrip("0")
            if len(decimal_part) < 2:
                decimal_part = decimal_part.ljust(2, "0")
            formatted = f"{integer_part}.{decimal_part}"

        return f"{formatted} {symbol}"


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _display_denom(denom: str) -> str:
    """Make a raw denom string more readable for display."""
    # Factory tokens: factory/zig1abc.../SYMBOL → SYMBOL
    if denom.startswith("factory/"):
        parts = denom.split("/")
        if len(parts) >= 3:
            return parts[-1]

    # IBC tokens: ibc/ABC123 → ibc/ABC1..
    if denom.startswith("ibc/") and len(denom) > 12:
        return f"ibc/{denom[4:10]}.."

    # u-prefixed base denoms: uzig → ZIG
    if denom.startswith("u") and len(denom) <= 8:
        return denom[1:].upper()

    # Truncate long denoms
    if len(denom) > 20:
        return f"{denom[:15]}.."

    return denom


# ══════════════════════════════════════════════
# Module-level singleton
# ══════════════════════════════════════════════

registry = TokenRegistry()
