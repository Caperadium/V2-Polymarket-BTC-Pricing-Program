"""
test_jump_calibration_window.py

Tests for Package C / work package W1 (REVISED): trailing-window
(era-conditioned) jump calibration in core/pricing/jump_calibration.py
(temp/mm_package_c_plan.md section 2.2-REV5, "Authoritative design").

The original W1 design (windowing lam, p_crash, eta_up AND eta_down via a
SECOND fresh detection pass on the windowed slice) FAILED W3 acceptance and
was replaced. This file targets the shipped replacement:

- ONE detection pass on the full slice (already existed). NO second
  detection call anywhere in the windowed path.
- Only eta_up (the up-jump mean size) is windowed, via a MASK-SLICE of the
  full-slice jump mask (jump_mask[-window_hours:] / returns[-window_hours:])
  -- never a fresh detection on the windowed slice alone.
- lam, p_crash, eta_down and the SVCJ vol-jump leg (mu_v, rho_J,
  rho_j_slope) are ALWAYS full-slice, regardless of window_hours.

Two testing strategies are used deliberately:

- Hand-built jump_mask arrays, exercising `_blend_windowed_eta_up` directly
  with EXACT, caller-controlled in-window/out-of-window up/down jump
  counts and magnitudes -- no detection algorithm involved. This is the
  right tool for exact shrinkage-weight and mask-slice-correctness checks
  (the plan's own guidance: "pass a hand-built jump_mask").
- Real bipower detection through `calibrate_jumps`, for the integration-
  level regression pin, directional, invariance and leak-free checks.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.pricing.jump_calibration import (
    JUMP_CAL_WINDOW_HOURS,
    JUMP_CAL_WINDOW_TARGET_UP_JUMPS,
    JumpCalibrationResult,
    _blend_windowed_eta_up,
    calibrate_jumps,
    detect_jumps_bipower,
)


# ---------------------------------------------------------------------------
# Shared synthetic-data builders
# ---------------------------------------------------------------------------

def _golden_returns() -> np.ndarray:
    """
    Fixed seeded synthetic hourly returns used for the regression pin.
    Captured via temp/w1_golden_capture.py, run against the PRE-CHANGE
    calibrate_jumps() to produce the golden values baked into
    test_regression_pin_window_none below (regression pin procedure in
    temp/mm_package_c_plan.md). The window_hours=None path is untouched by
    the REV5 revision, so these golden values still apply unchanged.
    """
    rng = np.random.default_rng(123)
    n = 20000
    sigma = 0.005
    ret = rng.normal(0.0, sigma, n)

    n_spikes = 30
    spike_idx = rng.choice(n, size=n_spikes, replace=False)
    spike_signs = rng.choice([-1.0, 1.0], size=n_spikes)
    spike_mag = rng.uniform(0.03, 0.08, size=n_spikes)
    ret[spike_idx] = spike_signs * spike_mag
    return ret


def _hand_built_window_case(
    n_obs: int,
    window_hours: int,
    in_window_up_mags: tuple = (),
    in_window_down_mags: tuple = (),
    out_window_up_mags: tuple = (),
) -> tuple:
    """
    Build a (returns, jump_mask) pair with EXACTLY the caller-specified jump
    events, split between inside the trailing window (the last window_hours
    bars) and strictly outside it (before the window start). No detection
    algorithm runs -- jump_mask is set directly at the chosen positions --
    so tests built on this isolate `_blend_windowed_eta_up`'s own
    mask-slice arithmetic (jump_mask[-window_hours:] / returns[-window_hours:])
    from bipower/MAD detection behavior (exercised separately, at the
    calibrate_jumps integration level).

    All background (non-jump) bars are exactly 0.0 -- irrelevant here since
    the mask, not a threshold, decides what counts as a jump.
    """
    ret = np.zeros(n_obs)
    mask = np.zeros(n_obs, dtype=bool)
    window_start = n_obs - window_hours
    assert window_start >= len(out_window_up_mags), "not enough room before the window"
    assert window_hours >= len(in_window_up_mags) + len(in_window_down_mags), (
        "not enough room inside the window"
    )

    for i, mag in enumerate(out_window_up_mags):
        ret[i] = mag
        mask[i] = True

    pos = window_start
    for mag in in_window_up_mags:
        ret[pos] = mag
        mask[pos] = True
        pos += 1
    for mag in in_window_down_mags:
        ret[pos] = -mag
        mask[pos] = True
        pos += 1

    return ret, mask


# ---------------------------------------------------------------------------
# 1. Regression pin: window_hours=None reproduces pre-W1 output byte-identically
# ---------------------------------------------------------------------------

def test_regression_pin_window_none():
    """
    window_hours=None must short-circuit BEFORE any windowed blending and
    reproduce the pre-change calibrate_jumps() output exactly (rel tol
    1e-12) on the fixed golden array -- proving the None path never
    executes the eta_up blend code. Untouched by the REV5 revision.
    """
    ret = _golden_returns()
    result = calibrate_jumps(returns=ret, detection_method="bipower", window_hours=None)

    assert result.n_jumps_detected == 29
    assert result.n_obs == 20000
    assert result.fit_converged is True

    golden = {
        "lam": 12.702,
        "p_crash": 0.4827586206896552,
        "eta_up": 18.26054831213074,
        "eta_down": 20.697479477182334,
        "mu_v": 1.9456352215421424e-06,
        "rho_J": 0.13529924362868867,
        "rho_j_slope": 2483.7936605388136,
    }
    for field, expected in golden.items():
        actual = getattr(result, field)
        assert math.isclose(actual, expected, rel_tol=1e-12), (field, actual, expected)

    # "not windowed" additive-field defaults
    assert result.calibration_window_hours is None
    assert result.window_weight == 1.0
    assert result.n_window_jumps == 0


# ---------------------------------------------------------------------------
# 2. window_hours >= len(returns) -> no error, windowed slice == full slice
# ---------------------------------------------------------------------------

def test_window_ge_len_returns_no_error_matches_full_slice():
    """
    window_hours >= len(returns): jump_mask[-window_hours:] and
    returns[-window_hours:] naturally yield the entire arrays (Python/numpy
    slicing semantics), so the windowed up-sample IS the full-slice
    up-jump sample -> w=1.0 and eta_up must equal the window_hours=None
    result. lam/p_crash/eta_down/SVCJ are pinned regardless of window_hours
    and must match exactly too.
    """
    ret = _golden_returns()
    huge_window = len(ret) + 50000

    windowed = calibrate_jumps(returns=ret, detection_method="bipower", window_hours=huge_window)
    full = calibrate_jumps(returns=ret, detection_method="bipower", window_hours=None)

    assert windowed.window_weight == 1.0

    # n_window_jumps now counts UP-side jumps only (mask-slice) -- verify
    # independently against a fresh full-slice detection pass.
    jump_mask_full = detect_jumps_bipower(ret)
    n_full_up = int(np.sum(ret[jump_mask_full] > 0))
    assert windowed.n_window_jumps == n_full_up

    assert math.isclose(windowed.eta_up, full.eta_up, rel_tol=1e-9)
    for field in ("eta_down", "lam", "p_crash", "mu_v", "rho_J", "rho_j_slope"):
        assert math.isclose(getattr(windowed, field), getattr(full, field), rel_tol=1e-12), field


# ---------------------------------------------------------------------------
# 3. lam/p_crash/eta_down/SVCJ invariance: only eta_up may differ between
#    the windowed and window_hours=None runs.
# ---------------------------------------------------------------------------

def test_lam_p_crash_eta_down_svcj_invariant_between_windowed_and_none():
    """
    lam, p_crash, eta_down and the SVCJ vol-jump leg (mu_v, rho_J,
    rho_j_slope) must be IDENTICAL between the default windowed run and the
    window_hours=None run on the same array -- plan section 2.2-REV5 pins
    all of these to the full slice; only eta_up is windowed. SVCJ (mu_v,
    rho_J, rho_j_slope) specifically is always estimated on the FULL slice
    regardless of windowing because windowed SVCJ estimates at typical
    window jump counts are unstable / sign-flipping, so the vol-jump leg
    deliberately does not era-condition (folded in from the former
    test_svcj_params_pinned_to_full_slice, a strict subset of this test).
    """
    ret = _golden_returns()
    windowed = calibrate_jumps(returns=ret, detection_method="bipower")
    full = calibrate_jumps(returns=ret, detection_method="bipower", window_hours=None)

    assert windowed.lam == full.lam
    assert windowed.p_crash == full.p_crash
    assert windowed.eta_down == full.eta_down
    assert windowed.mu_v == full.mu_v
    assert windowed.rho_J == full.rho_J
    assert windowed.rho_j_slope == full.rho_j_slope


# ---------------------------------------------------------------------------
# 5. Directional: wild-up-jumps-early / calm-small-up-jumps-late array ->
#    windowed eta_up HIGHER than full-slice eta_up; down-side and lam
#    identical to full.
# ---------------------------------------------------------------------------

def test_directional_wild_up_early_calm_up_late_windowed_eta_up_higher():
    """
    The default trailing window (8760h) covers the RECENT (second) half of
    a 2*8760h array. A wild first half (large-magnitude spikes both signs)
    + calm second half (small-magnitude spikes both signs) must give a
    windowed eta_up HIGHER (thinner up-tail) than the full-slice eta_up,
    which is fattened by the wild era's large up-jumps mixed into the mean.
    eta_down, lam and p_crash are pinned -- unchanged between windowed and
    full regardless of the era split.
    """
    rng = np.random.default_rng(7)
    half = JUMP_CAL_WINDOW_HOURS

    wild = rng.normal(0.0, 0.006, half)
    n_wild_spikes = 50
    wild_idx = rng.choice(half, size=n_wild_spikes, replace=False)
    wild_signs = rng.choice([-1.0, 1.0], size=n_wild_spikes)
    wild_mag = rng.uniform(0.04, 0.09, size=n_wild_spikes)
    wild[wild_idx] = wild_signs * wild_mag

    calm = rng.normal(0.0, 0.003, half)
    n_calm_spikes = 20
    calm_idx = rng.choice(half, size=n_calm_spikes, replace=False)
    calm_signs = rng.choice([-1.0, 1.0], size=n_calm_spikes)
    calm_mag = rng.uniform(0.015, 0.025, size=n_calm_spikes)
    calm[calm_idx] = calm_signs * calm_mag

    combined = np.concatenate([wild, calm])

    windowed = calibrate_jumps(returns=combined, detection_method="bipower")
    full = calibrate_jumps(returns=combined, detection_method="bipower", window_hours=None)

    assert windowed.eta_up > full.eta_up
    assert windowed.eta_down == full.eta_down
    assert windowed.lam == full.lam
    assert windowed.p_crash == full.p_crash
    assert windowed.calibration_window_hours == JUMP_CAL_WINDOW_HOURS
    assert windowed.n_window_jumps > 0


# ---------------------------------------------------------------------------
# 6. Mask-slice correctness: the windowed up-sample must be exactly the
#    in-window up-spikes -- excluding out-of-window jumps and in-window
#    down jumps.
# ---------------------------------------------------------------------------

def test_mask_slice_correctness_uses_exactly_in_window_up_spikes():
    """
    The windowed up-sample must be EXACTLY the full-slice mask's up-jump
    positions restricted to the trailing window -- excluding both (a)
    jumps flagged outside the window and (b) in-window DOWN jumps. Six
    in-window up-jumps hits window_weight=1.0 (target=6), so eta_up is a
    pure function of the in-window up-jump sample, independent of
    eta_up_full and the excluded jumps present elsewhere in the mask.
    """
    in_window_up = (0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
    ret, mask = _hand_built_window_case(
        n_obs=100, window_hours=40,
        in_window_up_mags=in_window_up,
        in_window_down_mags=(0.05,),        # must be excluded from eta_up
        out_window_up_mags=(0.10, 0.20),    # must be excluded (outside window)
    )
    eta_up, n_window_up, w = _blend_windowed_eta_up(
        returns=ret, jump_mask=mask, window_hours=40, eta_up_full=40.0,
    )
    assert n_window_up == 6
    assert w == 1.0
    expected_mean = sum(in_window_up) / len(in_window_up)
    assert math.isclose(eta_up, 1.0 / expected_mean, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 7. Shrinkage w mapping on the up side: n_window_up in {0, 3, >=6} ->
#    w in {0.0, 0.5, 1.0}.
# ---------------------------------------------------------------------------

def test_shrinkage_weight_zero_pins_to_full_slice():
    """n_window_up == 0 (an empty in-window mask) -> w == 0.0 -> eta_up
    equals eta_up_full exactly (a genuinely empty in-window up-sample
    degrades to the full-slice value)."""
    ret, mask = _hand_built_window_case(n_obs=100, window_hours=40)
    eta_up, n_window_up, w = _blend_windowed_eta_up(
        returns=ret, jump_mask=mask, window_hours=40, eta_up_full=40.0,
    )
    assert n_window_up == 0
    assert w == 0.0
    assert eta_up == 40.0


def test_shrinkage_weight_half_lands_strictly_between():
    """n_window_up == 3 (target 6) -> w == 0.5 -> mean-space blend lands
    strictly between the raw windowed up-mean and the full-slice mean."""
    in_window_up = (0.09, 0.08, 0.07)
    eta_up_full = 40.0
    ret, mask = _hand_built_window_case(
        n_obs=100, window_hours=40, in_window_up_mags=in_window_up,
    )
    eta_up, n_window_up, w = _blend_windowed_eta_up(
        returns=ret, jump_mask=mask, window_hours=40, eta_up_full=eta_up_full,
    )
    assert n_window_up == 3
    assert math.isclose(w, 0.5, rel_tol=1e-12)

    mean_win_up = sum(in_window_up) / len(in_window_up)
    mean_full_up = 1.0 / eta_up_full
    lo, hi = sorted((mean_win_up, mean_full_up))
    assert lo < 1.0 / eta_up < hi


def test_shrinkage_weight_full_at_target_up_jumps():
    """n_window_up >= JUMP_CAL_WINDOW_TARGET_UP_JUMPS -> w == 1.0 -> eta_up
    is the pure windowed up-mean (no full-slice influence at all), whether
    exactly at target or above it."""
    ret6, mask6 = _hand_built_window_case(
        n_obs=100, window_hours=40, in_window_up_mags=(0.05,) * 6,
    )
    eta_up6, n6, w6 = _blend_windowed_eta_up(
        returns=ret6, jump_mask=mask6, window_hours=40, eta_up_full=40.0,
    )
    assert n6 == JUMP_CAL_WINDOW_TARGET_UP_JUMPS
    assert w6 == 1.0
    assert math.isclose(eta_up6, 1.0 / 0.05, rel_tol=1e-12)

    ret9, mask9 = _hand_built_window_case(
        n_obs=100, window_hours=40, in_window_up_mags=(0.04,) * 9,
    )
    eta_up9, n9, w9 = _blend_windowed_eta_up(
        returns=ret9, jump_mask=mask9, window_hours=40, eta_up_full=40.0,
    )
    assert n9 == 9
    assert w9 == 1.0
    assert math.isclose(eta_up9, 1.0 / 0.04, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 8. n_window_up == 0 via all-down in-window jumps -> w=0, no nan/warning.
# ---------------------------------------------------------------------------

def test_all_in_window_jumps_down_gives_w_zero_no_nan():
    """In-window jumps exist but are ALL on the down side -> the up-sample
    is still empty -> w=0, eta_up pins to full-slice exactly, no NaN (the
    empty-mean guard fires on n_window_up specifically, not on 'any jump
    present in the window')."""
    ret, mask = _hand_built_window_case(
        n_obs=100, window_hours=40,
        in_window_down_mags=(0.04, 0.05, 0.06, 0.07),
    )
    eta_up, n_window_up, w = _blend_windowed_eta_up(
        returns=ret, jump_mask=mask, window_hours=40, eta_up_full=55.0,
    )
    assert n_window_up == 0
    assert w == 0.0
    assert eta_up == 55.0
    assert not math.isnan(eta_up)


# ---------------------------------------------------------------------------
# 9. Continuity constructed IN THE n_up < 6 REGIME: one extra detected
#    in-window up-jump moves eta_up by exactly one 1/6 blend step.
# ---------------------------------------------------------------------------

def test_continuity_one_extra_up_jump_is_one_sixth_blend_step():
    """
    Two window masks differing by exactly ONE extra in-window up-jump
    (n=4 -> n=5, both < JUMP_CAL_WINDOW_TARGET_UP_JUMPS=6 so w is not yet
    saturated at 1.0 -- at n>=6 this test would be vacuous). All up-jump
    magnitudes are IDENTICAL across the two cases, so mean_win_up is
    unchanged and the entire eta_up movement is attributable to w stepping
    by exactly 1/6 -- letting the blend formula be checked exactly, not
    just bounded.
    """
    mag = 0.05
    eta_up_full = 40.0

    ret4, mask4 = _hand_built_window_case(
        n_obs=100, window_hours=40, in_window_up_mags=(mag,) * 4,
    )
    ret5, mask5 = _hand_built_window_case(
        n_obs=100, window_hours=40, in_window_up_mags=(mag,) * 5,
    )

    eta_up4, n4, w4 = _blend_windowed_eta_up(
        returns=ret4, jump_mask=mask4, window_hours=40, eta_up_full=eta_up_full,
    )
    eta_up5, n5, w5 = _blend_windowed_eta_up(
        returns=ret5, jump_mask=mask5, window_hours=40, eta_up_full=eta_up_full,
    )

    assert n4 == 4 and n5 == 5
    assert 0 < n4 < JUMP_CAL_WINDOW_TARGET_UP_JUMPS
    assert 0 < n5 < JUMP_CAL_WINDOW_TARGET_UP_JUMPS

    step = 1.0 / JUMP_CAL_WINDOW_TARGET_UP_JUMPS
    assert math.isclose(w5 - w4, step, rel_tol=1e-12)

    mean_full_up = 1.0 / eta_up_full
    expected_delta = step * (mag - mean_full_up)
    actual_delta = (1.0 / eta_up5) - (1.0 / eta_up4)
    assert math.isclose(actual_delta, expected_delta, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 10. Full slice thin (< 10 jumps) -> existing literature-defaults path
#     unchanged.
# ---------------------------------------------------------------------------

def test_full_slice_thin_literature_defaults_unchanged():
    """
    The full-slice n_jumps < 10 gate fires BEFORE any windowed blending,
    regardless of window_hours -- fit_converged stays False, the
    literature-default params (Teng 2025) are unchanged, and the window
    fields stay at their 'not windowed' defaults (the windowed branch never
    runs on this path).
    """
    rng = np.random.default_rng(42)
    ret = rng.normal(0.0, 0.001, 2000)  # tight noise, no injected spikes

    for wh in (JUMP_CAL_WINDOW_HOURS, None):
        result = calibrate_jumps(returns=ret, detection_method="bipower", window_hours=wh)
        assert result.n_jumps_detected < 10
        assert result.fit_converged is False
        assert result.lam == 25.0
        assert result.p_crash == 0.6
        assert result.eta_up == 50.0
        assert result.eta_down == 25.0
        assert result.mu_v == 0.000025
        assert result.rho_J == -0.08
        assert result.calibration_window_hours is None
        assert result.window_weight == 1.0
        assert result.n_window_jumps == 0


# ---------------------------------------------------------------------------
# 11. Leak-free: returns= path (windowed) must never touch the CSV.
# ---------------------------------------------------------------------------

def test_leak_free_windowed_path_no_file_io(monkeypatch):
    """Passing returns= must NEVER read the hourly CSV, including on the
    windowed eta_up-blend branch (existing leak-free property, extended to
    the mask-slice code path -- there is no second detection call to leak
    either)."""
    import core.pricing.jump_calibration as jc

    def _boom(*a, **k):
        raise AssertionError("read_csv called -- windowed calibrate_jumps leaked to file")

    monkeypatch.setattr(jc.pd, "read_csv", _boom)
    ret = _golden_returns()
    jc.calibrate_jumps(returns=ret, detection_method="bipower", window_hours=JUMP_CAL_WINDOW_HOURS)


# ---------------------------------------------------------------------------
# Package C review F7: window_hours must be validated BEFORE any use.
# window_hours=0 is not a valid "no windowing" sentinel (that's None) --
# `returns[-0:]` silently yields the WHOLE array (equivalent to no
# windowing, not an empty window), and a negative window_hours silently
# slices from the FRONT of the array instead of the trailing end. Both are
# wrong-semantics footguns rather than legitimate configurations, so
# calibrate_jumps must reject them outright instead of running to a
# silently-wrong result.
# ---------------------------------------------------------------------------

def test_window_hours_zero_raises_value_error():
    ret = _golden_returns()
    with pytest.raises(ValueError):
        calibrate_jumps(returns=ret, detection_method="bipower", window_hours=0)


def test_window_hours_negative_raises_value_error():
    ret = _golden_returns()
    with pytest.raises(ValueError):
        calibrate_jumps(returns=ret, detection_method="bipower", window_hours=-5)


# ---------------------------------------------------------------------------
# 12. Determinism: same inputs -> identical results.
# ---------------------------------------------------------------------------

def test_determinism_same_inputs_identical_results():
    ret = _golden_returns()
    r1 = calibrate_jumps(returns=ret, detection_method="bipower")
    r2 = calibrate_jumps(returns=ret, detection_method="bipower")

    fields = (
        "lam", "p_crash", "eta_up", "eta_down", "mu_v", "rho_J", "rho_j_slope",
        "n_jumps_detected", "n_obs", "fit_converged",
        "calibration_window_hours", "window_weight", "n_window_jumps",
    )
    for field in fields:
        assert getattr(r1, field) == getattr(r2, field), field


# ---------------------------------------------------------------------------
# 13. Old-style JumpCalibrationResult construction still works.
# ---------------------------------------------------------------------------

def test_old_style_jump_calibration_result_construction():
    """Constructing JumpCalibrationResult the pre-W1 way (no new fields
    supplied) must still work, with the new fields defaulting to the
    'not windowed' state."""
    r = JumpCalibrationResult(
        lam=25.0, p_crash=0.6, eta_up=50.0, eta_down=25.0,
        mu_v=0.000025, rho_J=-0.08, lam_v=25.0,
    )
    assert r.calibration_window_hours is None
    assert r.window_weight == 1.0
    assert r.n_window_jumps == 0
