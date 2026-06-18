"""DEPRECATED: absorbed into core.backtesting. Use core.backtesting.backtest_engine."""
import warnings

warnings.warn(
    "scripts.backtesting.backtest_engine is deprecated. "
    "Use core.backtesting.backtest_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

from core.backtesting.backtest_engine import *  # noqa: E402, F401, F403
