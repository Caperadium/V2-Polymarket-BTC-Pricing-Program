"""Log-odds transform layer (plan Section 2.2).

Pure, dependency-free math. logit/sigmoid with hard p-band clamps, Jacobian
S'(x) = p(1-p) and S''(x) = p(1-p)(1-2p), exact two-point spread conversion,
displayed-half-spread floor. No inf/nan may escape any function. Every function
accepts scalars and numpy arrays symmetrically.
"""
from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np

Number = Union[float, np.ndarray]

# Default p-band clamps (plan Section 2.2). Passed as arguments everywhere so
# they are never hardcoded inline in callers.
DEFAULT_P_LO: float = 0.001
DEFAULT_P_HI: float = 0.999


def logit_bounds(p_lo: float = DEFAULT_P_LO, p_hi: float = DEFAULT_P_HI) -> Tuple[float, float]:
    """x-space bounds corresponding to the (p_lo, p_hi) probability band."""
    return (math.log(p_lo / (1.0 - p_lo)), math.log(p_hi / (1.0 - p_hi)))


def _as_array(v: Number) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(v, dtype=float)
    return arr, (arr.ndim == 0)


def _finish(arr: np.ndarray, scalar: bool, fill: float, lo: float, hi: float) -> Number:
    """Replace non-finite entries then clip to [lo, hi]; return scalar if input was."""
    out = np.where(np.isfinite(arr), arr, fill)
    out = np.clip(out, lo, hi)
    return float(out) if scalar else out


def logit(p: Number, p_lo: float = DEFAULT_P_LO, p_hi: float = DEFAULT_P_HI) -> Number:
    """log(p/(1-p)) with p hard-clamped to [p_lo, p_hi]. nan -> band midpoint."""
    arr, scalar = _as_array(p)
    a = np.where(np.isnan(arr), 0.5 * (p_lo + p_hi), arr)
    a = np.clip(a, p_lo, p_hi)  # clip also maps +/-inf to the band edges
    x = np.log(a / (1.0 - a))
    x_lo, x_hi = logit_bounds(p_lo, p_hi)
    return _finish(x, scalar, 0.0, x_lo, x_hi)


def sigmoid(x: Number, p_lo: float = DEFAULT_P_LO, p_hi: float = DEFAULT_P_HI) -> Number:
    """1/(1+exp(-x)) with output hard-clamped to [p_lo, p_hi]. nan x -> 0."""
    arr, scalar = _as_array(x)
    x_lo, x_hi = logit_bounds(p_lo, p_hi)
    a = np.where(np.isnan(arr), 0.0, arr)
    a = np.clip(a, x_lo, x_hi)  # bound the argument so exp never overflows
    p = 1.0 / (1.0 + np.exp(-a))
    return _finish(p, scalar, 0.5 * (p_lo + p_hi), p_lo, p_hi)


def s_prime(x: Number, p_lo: float = DEFAULT_P_LO, p_hi: float = DEFAULT_P_HI) -> Number:
    """Jacobian S'(x) = p(1-p), evaluated on the clamped sigmoid."""
    p = np.asarray(sigmoid(x, p_lo, p_hi), dtype=float)
    val = p * (1.0 - p)
    scalar = np.ndim(x) == 0
    # S' is strictly positive inside the band; floor at the clamp value.
    lo = p_lo * (1.0 - p_lo)
    return _finish(val, scalar, lo, lo, 0.25)


def s_double_prime(x: Number, p_lo: float = DEFAULT_P_LO, p_hi: float = DEFAULT_P_HI) -> Number:
    """Second derivative S''(x) = p(1-p)(1-2p). Sign flips at p=0.5 (x=0)."""
    p = np.asarray(sigmoid(x, p_lo, p_hi), dtype=float)
    val = p * (1.0 - p) * (1.0 - 2.0 * p)
    scalar = np.ndim(x) == 0
    return _finish(val, scalar, 0.0, -1.0, 1.0)


def half_spread_p_exact(
    x_center: Number,
    delta_x: Number,
    p_lo: float = DEFAULT_P_LO,
    p_hi: float = DEFAULT_P_HI,
) -> Number:
    """Displayed half-spread in p-units via EXACT two-point evaluation:
    (S(x_center + delta_x) - S(x_center - delta_x)) / 2. Correct at the clamps,
    unlike the Jacobian linearization.
    """
    up = np.asarray(sigmoid(np.asarray(x_center, dtype=float) + np.asarray(delta_x, dtype=float), p_lo, p_hi))
    dn = np.asarray(sigmoid(np.asarray(x_center, dtype=float) - np.asarray(delta_x, dtype=float), p_lo, p_hi))
    val = 0.5 * (up - dn)
    scalar = np.ndim(x_center) == 0 and np.ndim(delta_x) == 0
    return _finish(val, scalar, 0.0, 0.0, 1.0)


def half_spread_p_linear(
    x_center: Number,
    delta_x: Number,
    p_lo: float = DEFAULT_P_LO,
    p_hi: float = DEFAULT_P_HI,
) -> Number:
    """Linearized half-spread S'(x_center) * delta_x. Reference only; diverges
    from the exact form near the clamps where S' collapses.
    """
    sp = np.asarray(s_prime(x_center, p_lo, p_hi), dtype=float)
    val = sp * np.abs(np.asarray(delta_x, dtype=float))
    scalar = np.ndim(x_center) == 0 and np.ndim(delta_x) == 0
    return _finish(val, scalar, 0.0, 0.0, 1.0)


def floor_half_spread(half_spread_p: Number, tick: Number) -> Number:
    """Enforce displayed half-spread >= 1 tick. Guards non-finite/negative."""
    hs, hs_scalar = _as_array(half_spread_p)
    tk, tk_scalar = _as_array(tick)
    hs = np.where(np.isfinite(hs), hs, 0.0)
    tk = np.where(np.isfinite(tk), np.abs(tk), 0.0)
    val = np.maximum(np.maximum(hs, 0.0), tk)
    scalar = hs_scalar and tk_scalar
    return float(val) if scalar else val
