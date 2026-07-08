"""Property-style tests for market_maker.logodds (plan Section 2.2)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.logodds import (
    DEFAULT_P_HI,
    DEFAULT_P_LO,
    floor_half_spread,
    half_spread_p_exact,
    half_spread_p_linear,
    logit,
    logit_bounds,
    s_double_prime,
    s_prime,
    sigmoid,
)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_round_trip_inside_band():
    for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)
        assert logit(sigmoid(logit(p))) == pytest.approx(logit(p), abs=1e-9)


def test_round_trip_array():
    p = np.array([0.02, 0.2, 0.5, 0.8, 0.98])
    out = sigmoid(logit(p))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, p, atol=1e-9)


# ---------------------------------------------------------------------------
# Clamp behavior
# ---------------------------------------------------------------------------

def test_logit_clamps_at_and_outside_band():
    x_lo, x_hi = logit_bounds()
    # exactly at edges
    assert logit(DEFAULT_P_LO) == pytest.approx(x_lo)
    assert logit(DEFAULT_P_HI) == pytest.approx(x_hi)
    # outside [0,1] and at hard 0/1
    assert logit(0.0) == pytest.approx(x_lo)
    assert logit(1.0) == pytest.approx(x_hi)
    assert logit(-3.0) == pytest.approx(x_lo)
    assert logit(5.0) == pytest.approx(x_hi)


def test_sigmoid_clamps_output_to_band():
    assert sigmoid(1e9) <= DEFAULT_P_HI
    assert sigmoid(1e9) == pytest.approx(DEFAULT_P_HI)
    assert sigmoid(-1e9) >= DEFAULT_P_LO
    assert sigmoid(-1e9) == pytest.approx(DEFAULT_P_LO)
    assert math.isinf(float(np.inf))
    assert sigmoid(np.inf) == pytest.approx(DEFAULT_P_HI)
    assert sigmoid(-np.inf) == pytest.approx(DEFAULT_P_LO)


def test_nan_inputs_do_not_escape():
    assert math.isfinite(logit(float("nan")))
    assert math.isfinite(sigmoid(float("nan")))
    assert 0.0 <= sigmoid(float("nan")) <= 1.0


def test_custom_clamp_arguments():
    lo, hi = 0.05, 0.95
    x_lo, x_hi = logit_bounds(lo, hi)
    assert logit(0.001, lo, hi) == pytest.approx(x_lo)
    assert sigmoid(1e9, lo, hi) == pytest.approx(hi)


# ---------------------------------------------------------------------------
# Jacobian S'(x) = p(1-p)
# ---------------------------------------------------------------------------

def test_s_prime_peak_and_symmetry():
    assert s_prime(0.0) == pytest.approx(0.25)
    for x in [0.3, 1.0, 2.5, 6.0]:
        assert s_prime(x) == pytest.approx(s_prime(-x), abs=1e-12)
        assert s_prime(x) < 0.25
    # monotone decreasing away from 0
    assert s_prime(0.5) > s_prime(1.0) > s_prime(2.0)


def test_s_prime_positive_everywhere():
    x = np.linspace(-50, 50, 501)
    assert np.all(np.asarray(s_prime(x)) > 0.0)


# ---------------------------------------------------------------------------
# Second derivative S''(x) sign flip at p=0.5
# ---------------------------------------------------------------------------

def test_s_double_prime_sign_flip():
    assert s_double_prime(0.0) == pytest.approx(0.0, abs=1e-12)
    assert s_double_prime(-1.0) > 0.0  # p<0.5 -> (1-2p)>0
    assert s_double_prime(1.0) < 0.0   # p>0.5 -> (1-2p)<0
    # antisymmetric
    for x in [0.4, 1.3, 3.0]:
        assert s_double_prime(x) == pytest.approx(-s_double_prime(-x), abs=1e-12)


# ---------------------------------------------------------------------------
# Spread conversion: exact two-point vs linearization
# ---------------------------------------------------------------------------

def test_exact_and_linear_agree_near_center():
    # small delta near x=0: linearization is a good local approximation
    x_c, d = 0.0, 0.01
    exact = half_spread_p_exact(x_c, d)
    lin = half_spread_p_linear(x_c, d)
    assert exact == pytest.approx(lin, rel=1e-3)


def test_exact_and_linear_diverge_near_clamp():
    # deep in the wing with a wide delta the two forms must diverge materially
    x_lo, _ = logit_bounds()
    x_c = x_lo + 0.2  # near the lower clamp
    d = 3.0  # wide enough that the lower leg clamps and linearization breaks down
    exact = half_spread_p_exact(x_c, d)
    lin = half_spread_p_linear(x_c, d)
    assert exact >= 0.0
    assert lin >= 0.0
    # exact is bounded by the band; linear (S' * delta) overshoots here
    assert abs(exact - lin) > 1e-3
    assert exact <= 1.0


def test_exact_half_spread_bounded_and_positive():
    x = np.linspace(-40, 40, 200)
    hs = np.asarray(half_spread_p_exact(x, 0.5))
    assert np.all(hs >= 0.0)
    assert np.all(hs <= 1.0)
    assert np.all(np.isfinite(hs))


# ---------------------------------------------------------------------------
# Floor enforcement
# ---------------------------------------------------------------------------

def test_floor_half_spread_enforces_tick():
    assert floor_half_spread(0.0001, 0.01) == pytest.approx(0.01)
    assert floor_half_spread(0.05, 0.01) == pytest.approx(0.05)
    assert floor_half_spread(0.0, 0.005) == pytest.approx(0.005)


def test_floor_half_spread_guards_nonfinite():
    assert floor_half_spread(float("nan"), 0.01) == pytest.approx(0.01)
    assert floor_half_spread(-0.5, 0.01) == pytest.approx(0.01)
    out = floor_half_spread(np.array([0.0001, 0.05, np.nan]), 0.01)
    np.testing.assert_allclose(out, [0.01, 0.05, 0.01])


# ---------------------------------------------------------------------------
# No nan/inf across the specified domains, scalar and array
# ---------------------------------------------------------------------------

def test_no_nan_inf_over_domain_arrays():
    p = np.linspace(1e-9, 1 - 1e-9, 1000)
    x = np.linspace(-50, 50, 1000)
    for arr in [logit(p), sigmoid(x), s_prime(x), s_double_prime(x),
                half_spread_p_exact(x, 0.3), half_spread_p_linear(x, 0.3)]:
        a = np.asarray(arr)
        assert np.all(np.isfinite(a))


def test_no_nan_inf_scalars():
    for p in [1e-9, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-9]:
        assert math.isfinite(logit(p))
    for x in [-50.0, -6.9, 0.0, 6.9, 50.0]:
        assert math.isfinite(sigmoid(x))
        assert math.isfinite(s_prime(x))
        assert math.isfinite(s_double_prime(x))
        assert math.isfinite(half_spread_p_exact(x, 0.3))
        assert math.isfinite(half_spread_p_linear(x, 0.3))


def test_scalar_returns_python_float():
    assert isinstance(logit(0.5), float)
    assert isinstance(sigmoid(0.0), float)
    assert isinstance(s_prime(0.0), float)
    assert isinstance(floor_half_spread(0.01, 0.005), float)
