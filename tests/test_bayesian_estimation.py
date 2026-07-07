"""
test_bayesian_estimation.py

Tests for the Wave 3 pricing-fix plan (temp/pricing_fix_implementation_plan.md),
task T8: core/pricing/bayesian_estimation.py. All synthetic (no DATA/
dependency); n_iter/n_draws kept small so the suite runs quickly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import gamma as gamma_dist, beta as beta_dist

from core.pricing.bayesian_estimation import (
    garch_posterior,
    jump_posterior,
    posterior_probability_bands,
    _lam_posterior_shape_rate,
    _p_crash_posterior_ab,
    _eta_posterior_shape_rate,
)


def _simulate_garch11_t(n: int, omega: float, alpha: float, beta: float, nu: float, seed: int) -> np.ndarray:
    """Simulate a GARCH(1,1)-Student-t return series with known parameters."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt((nu - 2.0) / nu)
    z = rng.standard_t(nu, size=n) * scale
    returns = np.zeros(n)
    var = omega / (1.0 - alpha - beta)
    for t in range(n):
        returns[t] = np.sqrt(var) * z[t]
        var = omega + alpha * returns[t] ** 2 + beta * var
    return returns


# ===========================================================================
# GARCH posterior (random-walk Metropolis)
# ===========================================================================

def test_garch_posterior_recovers_known_params():
    omega, alpha, beta, nu = 2e-6, 0.06, 0.90, 6.0
    returns = pd.Series(_simulate_garch11_t(8000, omega, alpha, beta, nu, seed=11))

    post = garch_posterior(returns, n_iter=1500, burn_in=500, seed=5)

    assert 0.05 <= post.acceptance_rate <= 0.6, f"acceptance_rate={post.acceptance_rate}"

    alpha_med = float(post.draws["alpha"].median())
    beta_med = float(post.draws["beta"].median())
    assert 0.02 <= alpha_med <= 0.12, f"alpha median={alpha_med}"
    assert 0.84 <= beta_med <= 0.95, f"beta median={beta_med}"

    assert ((post.draws["alpha"] + post.draws["beta"]) < 0.999).all()
    assert set(post.draws.columns) == {"omega", "alpha", "beta", "nu"}
    assert isinstance(post.point_estimate, dict) and "omega" in post.point_estimate
    assert set(post.rhat.keys()) == {"omega", "alpha", "beta", "nu"}


# ===========================================================================
# Jump parameter conjugacy (closed-form Gamma/Beta)
# ===========================================================================

def test_jump_posterior_conjugacy_matches_analytic():
    """Posterior means computed from the module's closed-form shape/rate (or
    a/b) helpers must match the corresponding scipy.stats distribution mean
    to numerical precision -- a regression guard against parametrization bugs
    (e.g. shape/rate vs shape/scale mixups)."""
    n_obs = 100_000
    n_jumps = 40
    n_up, n_down = 15, 25
    sum_up = 15 * 0.02
    sum_down = 25 * 0.03

    lam_shape, lam_rate = _lam_posterior_shape_rate(n_jumps, n_obs)
    analytic_lam_mean = lam_shape / lam_rate
    scipy_lam_mean = gamma_dist(a=lam_shape, scale=1.0 / lam_rate).mean()
    assert abs(analytic_lam_mean - scipy_lam_mean) < 1e-9

    a_c, b_c = _p_crash_posterior_ab(n_down, n_up)
    analytic_crash_mean = a_c / (a_c + b_c)
    scipy_crash_mean = beta_dist(a_c, b_c).mean()
    assert abs(analytic_crash_mean - scipy_crash_mean) < 1e-9

    up_shape, up_rate = _eta_posterior_shape_rate(n_up, sum_up, a0=2.0, b0=2.0 / 50.0)
    analytic_eta_up_mean = up_shape / up_rate
    scipy_eta_up_mean = gamma_dist(a=up_shape, scale=1.0 / up_rate).mean()
    assert abs(analytic_eta_up_mean - scipy_eta_up_mean) < 1e-9

    down_shape, down_rate = _eta_posterior_shape_rate(n_down, sum_down, a0=2.0, b0=2.0 / 25.0)
    analytic_eta_down_mean = down_shape / down_rate
    scipy_eta_down_mean = gamma_dist(a=down_shape, scale=1.0 / down_rate).mean()
    assert abs(analytic_eta_down_mean - scipy_eta_down_mean) < 1e-9


def test_jump_posterior_smoke_on_synthetic_jumps():
    """jump_posterior() end-to-end (bipower detection + conjugate draws) must
    run without exception and return arrays of the requested length, all in
    their respective valid ranges."""
    rng = np.random.default_rng(21)
    n = 6000
    returns = rng.normal(0, 0.004, n)
    jump_idx = rng.choice(np.arange(100, n - 100), size=30, replace=False)
    signs = rng.choice([-1, 1], size=30)
    returns[jump_idx] += 0.05 * signs

    post = jump_posterior(returns, n_draws=500, seed=3)

    assert len(post.lam_draws) == 500
    assert len(post.eta_up_draws) == 500
    assert len(post.eta_down_draws) == 500
    assert len(post.p_crash_draws) == 500
    assert (post.lam_draws > 0).all()
    assert (post.eta_up_draws > 0).all()
    assert (post.eta_down_draws > 0).all()
    assert ((post.p_crash_draws >= 0) & (post.p_crash_draws <= 1)).all()


# ===========================================================================
# Posterior probability bands
# ===========================================================================

def _make_synthetic_hourly_df(n: int, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, sigma, size=n)
    log_price = np.cumsum(log_rets) + np.log(50000.0)
    close = np.exp(log_price)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "close": close})


def _make_synthetic_hourly_garch_t_df(n: int, omega: float, alpha: float, beta: float,
                                       nu: float, seed: int) -> pd.DataFrame:
    """GARCH(1,1)-t hourly series (fat tails, like real BTC) rather than pure
    Gaussian GBM: with Gaussian-only data the Student-t MLE degree-of-freedom
    diverges toward infinity (thin tails), which makes the weakly-informative
    Exponential(mean=8) prior on nu dominate the posterior and systematically
    pulls posterior-draw tail probabilities away from the (thin-tailed) MLE
    point estimate for far-OTM strikes -- an edge case of pure-Gaussian input,
    not a property of real (fat-tailed) BTC-like data."""
    returns = _simulate_garch11_t(n, omega, alpha, beta, nu, seed)
    log_price = np.cumsum(returns) + np.log(50000.0)
    close = np.exp(log_price)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "close": close})


def test_posterior_probability_bands_synthetic():
    df = _make_synthetic_hourly_garch_t_df(n=6000, omega=2e-6, alpha=0.06, beta=0.90, nu=6.0, seed=17)
    S0 = float(df["close"].iloc[-1])
    strikes = [S0 * m for m in (0.85, 0.95, 1.0, 1.05, 1.15)]

    bands = posterior_probability_bands(
        strikes, hours_to_expiry=24 * 14, hourly_df=df,
        n_posterior=10, n_sims_per_draw=500, seed=9,
    )

    n_within = 0
    for K in strikes:
        b = bands[K]
        assert 0.0 <= b["q05"] <= 1.0
        assert 0.0 <= b["q50"] <= 1.0
        assert 0.0 <= b["q95"] <= 1.0
        assert 0.0 <= b["point"] <= 1.0
        assert b["q05"] <= b["q50"] <= b["q95"], f"quantiles not ordered for K={K}: {b}"
        if b["q05"] <= b["point"] <= b["q95"]:
            n_within += 1

    assert n_within / len(strikes) >= 0.8, "point estimate should fall within its own band for most strikes"
    assert "_meta" in bands and "garch_rhat" in bands["_meta"]
