"""
test_regime_layer.py

Tests for the Wave 2 pricing-fix plan (temp/pricing_fix_implementation_plan.md),
task T6 (H3: regime layer -- threshold-aware state labeling, weight
propagation over the horizon, emission-variance scaling). All tests use
synthetic data (no DATA/ or network dependency) so they run everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_hourly_intraday(n: int = 3000, seed: int = 1, start: str = "2024-01-01"):
    """Synthetic hourly + intraday frames for calculate_probabilities, plus the
    resulting S0 (last close) for building strikes relative to it."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="h", tz="UTC")
    returns = rng.normal(0.0, 0.003, n)
    prices = 50000.0 * np.exp(np.cumsum(returns))
    hourly_df = pd.DataFrame({"date": dates, "close": prices})

    s0 = float(prices[-1])
    times = pd.date_range(dates[-1], periods=5, freq="1min", tz="UTC")
    intraday_df = pd.DataFrame({"timestamp": times, "close": np.full(5, s0)})
    return hourly_df, intraday_df, s0


# ===========================================================================
# Threshold-aware state labeling
# ===========================================================================

def test_threshold_labeling_bear_sideways_bull_when_clearly_separated():
    from core.pricing.regime_detector import RegimeDetector

    rng = np.random.default_rng(11)
    bear = rng.normal(-0.020, 0.012, 200)
    side = rng.normal(0.000, 0.008, 200)
    bull = rng.normal(0.020, 0.010, 200)
    daily_returns = np.concatenate([bear, side, bull])

    det = RegimeDetector()
    result = det.fit(daily_returns, force=True)
    assert result is not None
    assert set(det._labels.state_labels) == {"bear", "sideways", "bull"}


def test_threshold_labeling_no_bear_when_all_positive_mean():
    """Three clusters, all with positive daily mean (+0.05%/+0.2%/+0.6%): the
    provisional (lowest-mean) 'bear' state's annualized mean clears
    bear_threshold (-10%), so it must be demoted to 'sideways' -- no state may
    be labeled 'bear' (PRICING_REVIEW.md H3 dead-threshold finding)."""
    from core.pricing.regime_detector import RegimeDetector

    rng = np.random.default_rng(23)
    c1 = rng.normal(0.0005, 0.004, 200)
    c2 = rng.normal(0.0020, 0.006, 200)
    c3 = rng.normal(0.0060, 0.010, 200)
    daily_returns = np.concatenate([c1, c2, c3])

    det = RegimeDetector()
    result = det.fit(daily_returns, force=True)
    assert result is not None
    assert "bear" not in det._labels.state_labels


# ===========================================================================
# Weight propagation over the horizon
# ===========================================================================

def test_predict_weights_propagates_toward_stationary_distribution():
    from core.pricing.regime_detector import RegimeDetector

    rng = np.random.default_rng(5)
    bear = rng.normal(-0.020, 0.012, 200)
    side = rng.normal(0.000, 0.008, 200)
    bull = rng.normal(0.020, 0.010, 200)
    daily_returns = np.concatenate([bear, side, bull])

    det = RegimeDetector()
    result = det.fit(daily_returns, force=True)
    assert result is not None

    stationary_state = det._stationary_distribution(det._last_transmat)
    stationary_by_label = {"bear": 0.0, "sideways": 0.0, "bull": 0.0}
    for state_idx in range(det.n_states):
        label = det._labels.state_labels[state_idx]
        stationary_by_label[label] += float(stationary_state[state_idx])

    w0 = det.predict_weights(0)
    w60 = det.predict_weights(60)

    def _l1(a, b):
        keys = set(a) | set(b)
        return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)

    l1_0 = _l1(w0, stationary_by_label)
    l1_60 = _l1(w60, stationary_by_label)
    assert l1_60 < l1_0, f"predict_weights(60) (L1={l1_60:.4f}) not closer to stationary than predict_weights(0) (L1={l1_0:.4f})"


# ===========================================================================
# Emission-variance scales
# ===========================================================================

def test_variance_scales_keys_bounds_and_ordering():
    from core.pricing.regime_detector import RegimeDetector

    rng = np.random.default_rng(9)
    bear = rng.normal(-0.020, 0.020, 200)   # high vol
    side = rng.normal(0.000, 0.006, 200)    # low vol
    bull = rng.normal(0.020, 0.010, 200)    # mid vol
    daily_returns = np.concatenate([bear, side, bull])

    det = RegimeDetector()
    result = det.fit(daily_returns, force=True)
    assert result is not None

    scales = det.get_regime_variance_scales()
    assert set(scales.keys()) == {"bear", "sideways", "bull"}
    for v in scales.values():
        assert 0.5 <= v <= 2.0

    # Bear was generated with visibly higher vol than sideways.
    assert scales["bear"] > scales["sideways"], scales


def test_variance_scales_unfitted_returns_all_ones():
    from core.pricing.regime_detector import RegimeDetector

    det = RegimeDetector()
    scales = det.get_regime_variance_scales()
    assert scales == {"bear": 1.0, "sideways": 1.0, "bull": 1.0}


# ===========================================================================
# Engine integration
# ===========================================================================

def test_calculate_probabilities_regime_switching_with_prefit_detector():
    from core.pricing.btc_pricing_engine import calculate_probabilities
    from core.pricing.regime_detector import RegimeDetector

    hourly_df, intraday_df, s0 = _synthetic_hourly_intraday()

    rng = np.random.default_rng(3)
    bear = rng.normal(-0.020, 0.012, 200)
    side = rng.normal(0.000, 0.008, 200)
    bull = rng.normal(0.020, 0.010, 200)
    daily_returns = np.concatenate([bear, side, bull])

    detector = RegimeDetector()
    detector.fit(daily_returns, force=True)

    strikes = [s0 * 0.8, s0 * 1.0, s0 * 1.2]
    res = calculate_probabilities(
        strikes=strikes,
        hours_to_expiry=336.0,  # 14 days
        hourly_df=hourly_df,
        intraday_df=intraday_df,
        n_sims=2000,
        seed=11,
        use_regime_switching=True,
        regime_detector=detector,
        disable_staleness_check=True,
    )

    for k in strikes:
        assert 0.0 <= res[k] <= 1.0

    # Monotone non-increasing in strike (survival function).
    assert res[strikes[0]] >= res[strikes[1]] >= res[strikes[2]]

    meta = res["_meta"]
    assert "regime_weights" in meta
    assert "regime_weights_used" in meta
    assert "regime_variance_scales" in meta
    assert set(meta["regime_variance_scales"].keys()) == {"bear", "sideways", "bull"}
