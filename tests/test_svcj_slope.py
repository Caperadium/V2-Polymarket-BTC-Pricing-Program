"""
test_svcj_slope.py

Test for the Wave 1 pricing-fix plan (temp/pricing_fix_implementation_plan.md),
task T4 (M1: SVCJ rho_J channel -- correlation used as slope).

Proves two things:
  1. rho_j_slope now actually moves the return equation (a sufficiently
     negative slope lowers the terminal log-return mean under SVCJ jumps).
  2. The legacy rho_J key (a Pearson correlation, reporting-only as of this
     fix) is INERT in the return equation -- rho_J alone must not change the
     simulated distribution at all.
"""

from __future__ import annotations

import numpy as np


def _base_garch_params() -> dict:
    return {
        "omega": 0.00001 / 24,
        "alpha": 0.1,
        "beta": 0.85,
        "nu": 5.0,
        "mu": 0.0,
        "last_variance": 0.0004 / 24,
    }


def test_negative_rho_j_slope_lowers_terminal_mean():
    from core.pricing.btc_pricing_engine import simulate_paths

    gp = _base_garch_params()
    jp_slope = {
        "lambda": 50.0, "crash_prob": 0.5, "eta_up": 50.0, "eta_down": 50.0,
        "mu_v": 1e-4, "rho_j_slope": -200.0,
    }
    jp_zero = {
        "lambda": 50.0, "crash_prob": 0.5, "eta_up": 50.0, "eta_down": 50.0,
        "mu_v": 1e-4, "rho_j_slope": 0.0,
    }

    paths_slope = simulate_paths(
        S0=100000.0, garch_params=gp, jump_params=jp_slope,
        hours_to_expiry=240.0, n_sims=20000, seed=42, use_svcj=True,
    )
    paths_zero = simulate_paths(
        S0=100000.0, garch_params=gp, jump_params=jp_zero,
        hours_to_expiry=240.0, n_sims=20000, seed=42, use_svcj=True,
    )

    mean_slope = np.mean(np.log(paths_slope / 100000.0))
    mean_zero = np.mean(np.log(paths_zero / 100000.0))
    assert mean_slope < mean_zero, (
        f"negative rho_j_slope did not lower the terminal mean "
        f"({mean_slope:.5f} vs {mean_zero:.5f})"
    )


def test_legacy_rho_j_correlation_is_inert_in_return_equation():
    """With rho_j_slope=0.0 (default), varying the legacy rho_J key alone must
    NOT change the simulated distribution -- proves the correlation key is no
    longer (mis)used as a slope in the return equation."""
    from core.pricing.btc_pricing_engine import simulate_paths

    gp = _base_garch_params()
    jp_rho_j = {
        "lambda": 50.0, "crash_prob": 0.5, "eta_up": 50.0, "eta_down": 50.0,
        "mu_v": 1e-4, "rho_J": -0.5, "rho_j_slope": 0.0,
    }
    jp_rho_j_zero = {
        "lambda": 50.0, "crash_prob": 0.5, "eta_up": 50.0, "eta_down": 50.0,
        "mu_v": 1e-4, "rho_J": 0.0, "rho_j_slope": 0.0,
    }

    paths_a = simulate_paths(
        S0=100000.0, garch_params=gp, jump_params=jp_rho_j,
        hours_to_expiry=240.0, n_sims=20000, seed=1, use_svcj=True,
    )
    paths_b = simulate_paths(
        S0=100000.0, garch_params=gp, jump_params=jp_rho_j_zero,
        hours_to_expiry=240.0, n_sims=20000, seed=1, use_svcj=True,
    )
    assert np.array_equal(paths_a, paths_b)


def test_svcj_rho_j_slope_module_constant_is_zero():
    from core.pricing.btc_pricing_engine import SVCJ_RHO_J_SLOPE

    assert SVCJ_RHO_J_SLOPE == 0.0


def test_build_regime_jump_params_carries_rho_j_slope():
    """Every regime dict must carry rho_j_slope, with the bear/sideways/bull
    multipliers applied to the slope (not just rho_J)."""
    from core.pricing.btc_pricing_engine import build_regime_jump_params

    calibrated = {
        "lam": 30.0, "p_crash": 0.5, "eta_up": 40.0, "eta_down": 40.0,
        "mu_v": 5e-5, "rho_J": -0.1, "rho_j_slope": -10.0, "fit_converged": True,
    }
    regimes = build_regime_jump_params(calibrated=calibrated)
    assert set(regimes.keys()) == {"bear", "sideways", "bull"}
    for label in regimes:
        assert "rho_j_slope" in regimes[label]

    assert regimes["bear"]["rho_j_slope"] == -10.0 * 1.5
    assert regimes["sideways"]["rho_j_slope"] == -10.0 * 1.0
    assert regimes["bull"]["rho_j_slope"] == -10.0 * 0.5
