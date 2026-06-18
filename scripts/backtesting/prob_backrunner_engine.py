"""DEPRECATED: absorbed into core.backtesting. Use core.backtesting.BackrunnerEngine.

Note: automatic BTC data refresh (subprocess call to data_fetcher.py) has been
removed. Run 'python core/data/data_fetcher.py' manually before running backrunner.
"""
import warnings, sys

warnings.warn(
    "scripts.backtesting.prob_backrunner_engine is deprecated. "
    "Use core.backtesting.BackrunnerEngine instead.",
    DeprecationWarning,
    stacklevel=2,
)

from core.backtesting.backrunner import BackrunnerEngine  # noqa: E402, F401

if __name__ == "__main__":
    from core.backtesting.backrunner import main
    sys.exit(main())
