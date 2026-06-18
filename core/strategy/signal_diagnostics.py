"""DEPRECATED: absorbed into core.backtesting. Use core.backtesting.diagnostics."""
import warnings, sys

warnings.warn(
    "core.strategy.signal_diagnostics is deprecated. "
    "Use core.backtesting.diagnostics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from core.backtesting.diagnostics import main_cli  # noqa: E402, F401

if __name__ == "__main__":
    sys.exit(main_cli())
