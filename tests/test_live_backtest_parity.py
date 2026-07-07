"""
test_live_backtest_parity.py

Tests for the Wave 2 pricing-fix plan (temp/pricing_fix_implementation_plan.md),
task T5 (H2: route live pipelines through calculate_probabilities via a single
engine-kwargs builder). All tests use synthetic data (no DATA/ or network
dependency) so they run everywhere.

The real deliverable of T5 is `core.pricing.engine_config.build_engine_kwargs`:
a single function used by BOTH core/backtesting/backrunner.py and the live
pipelines (scripts/pipelines/run_full_pipeline.py,
scripts/pipelines/batch_pricing_runner.py) to build the calculate_probabilities
kwarg bundle, so the two call sites cannot silently drift apart again. These
tests assert:
  1. The bundle returned for "backtest-style" construction (as_of=<snapshot>)
     and "live-style" construction (as_of=None) differ ONLY in the documented
     `as_of` key.
  2. calculate_probabilities produces IDENTICAL output when called with the
     bundle under identical inputs (as_of=None on both, per the plan's parity
     assertion guidance).
  3. All three call sites actually use build_engine_kwargs (source-level
     regression guard against re-introducing a hand-rolled kwarg bundle).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _synthetic_hourly_df(n: int = 3000, seed: int = 42, start: str = "2024-01-01") -> pd.DataFrame:
    """Synthetic hourly OHLC-ish frame: 'date' + 'close' columns, GBM-like
    diffusion (no jumps needed for a parity check). n=3000 (~125 days) is
    comfortably above the >=500-row GARCH-fit floor."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="h", tz="UTC")
    returns = rng.normal(0.0, 0.004, n)
    prices = 50000.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": dates, "close": prices})


def _synthetic_intraday_df(hourly_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Minimal intraday frame anchored at the hourly frame's last close/timestamp
    (S0 source for load_and_prep_data)."""
    last_close = float(hourly_df["close"].iloc[-1])
    last_ts = hourly_df["date"].iloc[-1]
    times = pd.date_range(last_ts, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"timestamp": times, "close": np.full(n, last_close)})


# ===========================================================================
# build_engine_kwargs: documented-keys-only divergence
# ===========================================================================

def test_build_engine_kwargs_differs_only_in_as_of():
    from core.pricing.engine_config import build_engine_kwargs
    from core.pricing.regime_detector import RegimeDetector
    from core.pricing.btc_pricing_engine import build_regime_jump_params

    detector = RegimeDetector()
    jump_params = {
        "lambda": 30.0, "crash_prob": 0.5, "eta_up": 40.0, "eta_down": 40.0,
        "mu_v": 1e-5, "rho_J": -0.1, "rho_j_slope": 0.0,
    }
    calibrated = {**jump_params, "lam": 30.0, "p_crash": 0.5, "fit_converged": True}
    regime_params = build_regime_jump_params(calibrated=calibrated)

    snapshot_ts = pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime()

    backtest_kwargs = build_engine_kwargs(
        advanced_features=True, detector=detector, regime_params=regime_params,
        jump_params=jump_params, n_sims=1000, seed=42, as_of=snapshot_ts,
    )
    live_kwargs = build_engine_kwargs(
        advanced_features=True, detector=detector, regime_params=regime_params,
        jump_params=jump_params, n_sims=1000, seed=42, as_of=None,
    )

    assert set(backtest_kwargs.keys()) == set(live_kwargs.keys())
    diff_keys = {k for k in backtest_kwargs if backtest_kwargs[k] != live_kwargs[k]}
    assert diff_keys == {"as_of"}, f"unexpected divergence: {diff_keys}"


def test_build_engine_kwargs_xgb_gating():
    """use_xgb_direction is only True when BOTH use_xgb and a trained model
    are supplied -- mirrors the pre-existing backrunner _process_one logic."""
    from core.pricing.engine_config import build_engine_kwargs

    kw_no_model = build_engine_kwargs(
        advanced_features=True, detector=None, regime_params=None,
        jump_params=None, n_sims=1000, use_xgb=True, xgb_model=None,
    )
    assert kw_no_model["use_xgb_direction"] is False

    kw_with_model = build_engine_kwargs(
        advanced_features=True, detector=None, regime_params=None,
        jump_params=None, n_sims=1000, use_xgb=True, xgb_model=object(),
    )
    assert kw_with_model["use_xgb_direction"] is True

    kw_disabled = build_engine_kwargs(
        advanced_features=True, detector=None, regime_params=None,
        jump_params=None, n_sims=1000, use_xgb=False, xgb_model=object(),
    )
    assert kw_disabled["use_xgb_direction"] is False


# ===========================================================================
# calculate_probabilities parity under the built bundle
# ===========================================================================

def test_calculate_probabilities_parity_backtest_vs_live_construction():
    """Identical inputs (as_of=None on both, per the plan's parity guidance)
    must produce identical probabilities regardless of whether the caller
    resembles the backtest or the live pipeline -- both now go through the
    same build_engine_kwargs() bundle."""
    from core.pricing.btc_pricing_engine import calculate_probabilities
    from core.pricing.engine_config import build_engine_kwargs

    hourly_df = _synthetic_hourly_df()
    intraday_df = _synthetic_intraday_df(hourly_df)
    strikes = [45000.0, 50000.0, 55000.0]

    engine_kwargs = build_engine_kwargs(
        advanced_features=False,  # keep it a plain-engine parity check
        detector=None, regime_params=None, jump_params=None,
        n_sims=2000, seed=7, as_of=None,
    )

    common = dict(
        strikes=strikes,
        hours_to_expiry=168.0,
        hourly_df=hourly_df,
        intraday_df=intraday_df,
        disable_staleness_check=True,  # synthetic dates are far from wall-clock
    )

    probs_backtest_style = calculate_probabilities(
        garch_cache={}, s0_override=None, **common, **engine_kwargs,
    )
    probs_live_style = calculate_probabilities(
        garch_cache={}, s0_override=None, **common, **engine_kwargs,
    )

    for k in strikes:
        assert probs_backtest_style[k] == pytest.approx(probs_live_style[k])


# ===========================================================================
# Source-level regression guard: all three call sites use the shared builder
# ===========================================================================

def test_all_call_sites_use_build_engine_kwargs():
    repo_root = Path(__file__).resolve().parent.parent
    call_sites = [
        "core/backtesting/backrunner.py",
        "scripts/pipelines/run_full_pipeline.py",
        "scripts/pipelines/batch_pricing_runner.py",
    ]
    for rel in call_sites:
        src = (repo_root / rel).read_text(encoding="utf-8")
        assert "build_engine_kwargs" in src, (
            f"{rel} does not reference build_engine_kwargs -- H2 regression risk"
        )
