"""
test_jump_filtered_garch.py

Tests for the Wave 1 pricing-fix plan (temp/pricing_fix_implementation_plan.md),
tasks T1 (H1: jump-filtered GARCH fit), T2 (M3: Lee-Mykland contemporaneous-
return bug), and T3 (M2: mu_v confounded by the jump's own square). All tests
use synthetic data (no DATA/ dependency) so they run everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _simulate_garch11(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:
    """Simulate a GARCH(1,1) return series with Gaussian innovations, starting
    the variance recursion at the unconditional variance."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    returns = np.zeros(n)
    var = omega / (1.0 - alpha - beta)
    for t in range(n):
        returns[t] = np.sqrt(var) * z[t]
        var = omega + alpha * returns[t] ** 2 + beta * var
    return returns


def _inject_jumps(diffusion: np.ndarray, n_jumps: int, jump_mean: float, seed: int) -> np.ndarray:
    """Add n_jumps signed Exp(mean=jump_mean) jumps at random positions."""
    rng = np.random.default_rng(seed)
    n = len(diffusion)
    jumped = diffusion.copy()
    jump_idx = rng.choice(n, size=n_jumps, replace=False)
    jump_signs = rng.choice([-1, 1], size=n_jumps)
    jump_sizes = rng.exponential(jump_mean, size=n_jumps) * jump_signs
    jumped[jump_idx] += jump_sizes
    return jumped


# ===========================================================================
# T1 (H1) — filter_jump_returns and fit_garch_model(filter_jumps=...)
# ===========================================================================

# Known GARCH(1,1) DGP shared by the T1 tests.
_OMEGA, _ALPHA, _BETA = 1e-6, 0.05, 0.90
_TRUE_UNCOND_VAR = _OMEGA / (1.0 - _ALPHA - _BETA)


def test_t1_filter_reduces_variance_toward_diffusion():
    """filter_jump_returns should pull the sample variance of a jump-
    contaminated series back toward the jump-free (diffusion-only) variance."""
    from core.pricing.btc_pricing_engine import filter_jump_returns

    diffusion = _simulate_garch11(20000, _OMEGA, _ALPHA, _BETA, seed=123)
    jumped = _inject_jumps(diffusion, n_jumps=60, jump_mean=0.03, seed=123456)

    var_diffusion = np.var(diffusion)
    var_jumped = np.var(jumped)
    filtered = filter_jump_returns(pd.Series(jumped))
    var_filtered = np.var(filtered.to_numpy())

    ratio_raw = var_jumped / var_diffusion
    ratio_filtered = var_filtered / var_diffusion
    assert abs(ratio_filtered - 1.0) < abs(ratio_raw - 1.0), (
        f"filtered ratio {ratio_filtered:.4f} not closer to 1 than raw ratio {ratio_raw:.4f}"
    )
    # Non-jump bars must be untouched.
    non_jump_mask = jumped == diffusion
    assert np.array_equal(filtered.to_numpy()[non_jump_mask], diffusion[non_jump_mask])


def test_t1_fit_garch_filtered_closer_to_true_unconditional_variance():
    """fit_garch_model(filter_jumps=True) should recover the true diffusion
    unconditional variance more accurately than filter_jumps=False on a series
    with jumps stacked on top of a known GARCH(1,1) diffusion."""
    from core.pricing.btc_pricing_engine import fit_garch_model

    diffusion = _simulate_garch11(20000, _OMEGA, _ALPHA, _BETA, seed=123)
    jumped = _inject_jumps(diffusion, n_jumps=60, jump_mean=0.03, seed=123456)
    ret_series = pd.Series(jumped)

    params_filtered = fit_garch_model(ret_series, filter_jumps=True)
    params_raw = fit_garch_model(ret_series, filter_jumps=False)

    uncond_filtered = params_filtered['omega'] / (1 - params_filtered['alpha'] - params_filtered['beta'])
    uncond_raw = params_raw['omega'] / (1 - params_raw['alpha'] - params_raw['beta'])

    err_filtered = abs(uncond_filtered - _TRUE_UNCOND_VAR)
    err_raw = abs(uncond_raw - _TRUE_UNCOND_VAR)
    assert err_filtered < err_raw, (
        f"filtered error {err_filtered:.3e} not smaller than raw error {err_raw:.3e}"
    )


def test_t1_zero_jumps_filter_is_noop():
    """With a series too short for the bipower window (n < window+2), no jumps
    can be detected, so filter_jump_returns and fit_garch_model(filter_jumps=)
    must be bit-identical on vs off."""
    from core.pricing.btc_pricing_engine import filter_jump_returns, fit_garch_model
    from core.pricing.jump_calibration import detect_jumps_bipower

    rng = np.random.default_rng(5)
    short_ret = pd.Series(rng.normal(0, 0.01, 50))  # n=50 < window(78)+2

    # Detection itself must be all-False (a structural guarantee, not luck).
    mask = detect_jumps_bipower(short_ret.to_numpy())
    assert not np.any(mask)

    filtered = filter_jump_returns(short_ret)
    pd.testing.assert_series_equal(filtered, short_ret)

    params_on = fit_garch_model(short_ret, filter_jumps=True)
    params_off = fit_garch_model(short_ret, filter_jumps=False)
    for key in ("omega", "alpha", "beta", "nu", "mu", "last_variance"):
        assert params_on[key] == pytest.approx(params_off[key]), key


# ===========================================================================
# T2 (M3) — Lee-Mykland contemporaneous-return bug fix
# ===========================================================================

def test_t2_single_large_jump_detected():
    from core.pricing.jump_calibration import detect_jumps_bipower

    rng = np.random.default_rng(7)
    n = 5000
    sigma = 0.01
    ret = rng.normal(0, sigma, n)
    ret[2500] = 10 * sigma

    mask = detect_jumps_bipower(ret)
    assert mask[2500]


def test_t2_second_close_jump_still_detected():
    """A second 10-sigma jump 10 bars after the first must still be detected.
    Pre-fix, the contemporaneous-return leak let the first jump partially
    inflate the local sigma seen by the second, making closely spaced jumps
    harder to flag."""
    from core.pricing.jump_calibration import detect_jumps_bipower

    rng = np.random.default_rng(7)
    n = 5000
    sigma = 0.01
    ret = rng.normal(0, sigma, n)
    ret[2500] = 10 * sigma
    ret[2510] = 10 * sigma

    mask = detect_jumps_bipower(ret)
    assert mask[2500]
    assert mask[2510]


# ===========================================================================
# T3 (M2) — mu_v censored-at-zero mean vs the old J^2-confounded selective mean
# ===========================================================================

def test_t3_mu_v_censored_mean_much_smaller_than_old_method():
    """On a constant-diffusion series with injected jumps that carry NO true
    vol change, the new (jump-square-replaced, censored-at-zero) mu_v must be
    at least 3x smaller than the old method (rolling var of raw squared
    returns, mean of positive deltas only -- recomputed inline here for
    comparison) and below 2e-5."""
    from core.pricing.jump_calibration import detect_jumps_bipower, _estimate_vol_jump_params

    rng = np.random.default_rng(99)
    n = 6000
    sigma = 0.005
    returns = rng.normal(0, sigma, n)
    n_jumps = 40
    jump_positions = rng.choice(np.arange(200, n - 200), size=n_jumps, replace=False)
    jump_signs = rng.choice([-1, 1], size=n_jumps)
    returns[jump_positions] += 0.03 * jump_signs

    jump_mask, sigma_local = detect_jumps_bipower(returns, return_sigma=True)
    assert jump_mask.sum() > 10  # need enough events for both methods to be meaningful
    jump_returns = returns[jump_mask]

    new_mu_v, _rho_J, _rho_slope, n_events = _estimate_vol_jump_params(
        returns, jump_mask, jump_returns, window=24, sigma_local=sigma_local,
    )

    # --- Old method, recomputed inline (pre-fix behavior) ---
    squared_returns = returns ** 2
    rolling_var = pd.Series(squared_returns).rolling(24, min_periods=4).mean().to_numpy()
    jump_indices = np.where(jump_mask)[0]
    old_vol_changes = []
    for j_idx, full_idx in enumerate(jump_indices):
        if full_idx >= 2 and full_idx < len(rolling_var) - 1 and j_idx < len(jump_returns):
            pre_var = np.nan_to_num(rolling_var[max(0, full_idx - 2)], nan=0.0)
            post_var = np.nan_to_num(rolling_var[min(len(rolling_var) - 1, full_idx + 2)], nan=0.0)
            delta_var = max(0.0, post_var - pre_var)
            old_vol_changes.append(delta_var)
    old_positive = np.array([v for v in old_vol_changes if v > 0])
    if len(old_positive) > 5:
        old_mu_v = float(np.clip(np.mean(old_positive), 1e-6, 1e-3))
    else:
        old_mu_v = 0.000025

    assert new_mu_v < old_mu_v / 3.0, f"new_mu_v={new_mu_v:.3e} not < old_mu_v/3={old_mu_v / 3.0:.3e}"
    assert new_mu_v < 2e-5, f"new_mu_v={new_mu_v:.3e} not below 2e-5"


def test_t3_rho_j_slope_default_zero_when_insufficient_events():
    """Fewer than 10 usable jump events must yield rho_j_slope=0.0 (the
    documented fallback), not a noisy/undefined slope."""
    from core.pricing.jump_calibration import _estimate_vol_jump_params

    rng = np.random.default_rng(3)
    n = 500
    returns = rng.normal(0, 0.005, n)
    jump_mask = np.zeros(n, dtype=bool)
    jump_mask[[100, 150, 200]] = True  # only 3 events
    jump_returns = returns[jump_mask]

    mu_v, rho_J, rho_slope, n_events = _estimate_vol_jump_params(
        returns, jump_mask, jump_returns, window=24, sigma_local=None,
    )
    assert n_events == 3
    assert rho_slope == 0.0
