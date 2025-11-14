# Source this file from your shell profile (~/.zshrc or ~/.bash_profile)
# to force marimo (and other Python tools that respect $BROWSER) to open in Google Chrome.
#
# Usage:
#   echo 'source $(pwd)/scripts/shell_snippet.sh' >> ~/.zshrc   # if using zsh (macOS default)
#   # OR for bash
#   echo 'source $(pwd)/scripts/shell_snippet.sh' >> ~/.bash_profile
#   # Then restart terminal or: source ~/.zshrc
#
# Optional alias provided below.

################################################################################
# Browser override logic (macOS)
# We prefer invoking the Chrome *binary* directly so Python's webbrowser module
# respects BROWSER and doesn't fall back to the system default (Safari).
################################################################################
CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

if [ -x "$CHROME_BIN" ]; then
  export BROWSER="$CHROME_BIN"
else
  # Fallback (will probably hit Safari if Chrome missing)
  export BROWSER="open"
fi

################################################################################
# Convenience function: force Chrome for a single marimo invocation even if
# the environment was sourced before Chrome was installed.
################################################################################
marimo_chrome() {
  local bin="$CHROME_BIN"
  if [ ! -x "$bin" ]; then
    echo "[marimo_chrome] Chrome binary not found at $bin; using default BROWSER=$BROWSER" >&2
    bin="$BROWSER"
  fi
  if command -v poetry >/dev/null 2>&1; then
    BROWSER="$bin" poetry run marimo "$@"
  else
    BROWSER="$bin" marimo "$@"
  fi
}
# Enable tab completion word splitting to treat file paths normally
compctl -K marimo_chrome marimo_chrome 2>/dev/null || true

# Example usage after sourcing this file:
#   marimo_chrome edit notebooks/market_backtest.py

################################################################################
# Optional: function to print the URL only (manual copy) if you set
#   export MARIMO_NO_AUTO=1
################################################################################
if [ "${MARIMO_NO_AUTO:-}" = "1" ]; then
  export BROWSER="false"  # Prevent auto-launch; marimo will just print URL
fi
