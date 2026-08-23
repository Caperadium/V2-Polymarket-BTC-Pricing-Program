"""Tests for C1 belly drift-anchored Bayes scoring (temp/mm_c1_belly_drift_
plan.md v3), Part A: market_maker/fair_value_anchor.py's advance_weights()
helper and belly_lag_* kwargs, plus market_maker/state_store.py's
bayes_score_log table.

Covers (per the plan's "Tests" section, Part-A items only -- harness/
paper_runner items are Part B and live in a later test file):
  - the full-support drift factor's level-shift response vs the old
    belly-SUBSET form (regression guard),
  - the martingale invariant (v1's failing case),
  - the alpha-sweep zero-crossing at alpha == w_p,
  - s_tail_frac on the review's worked 5-strike example,
  - control factor == drift factor when the target vectors coincide,
  - advance_weights() parity with the pre-refactor applied-loop arithmetic
    and its s<=0/non-finite skip,
  - the mode matrix (legacy / shadow / live / unknown-string),
  - no-mutation of forecasts/lag arrays,
  - C1's own precondition (shape/finiteness skips) and the live-mode-fires-
    without-lag-1-history case,
  - state_store.bayes_score_log round-trip, auto-migration, and pruning.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import (
    BankrollState,
    ConfidenceTier,
    PricerSnapshot,
    Sigma2Source,
)
from market_maker.fair_value_anchor import (
    BELLY_REGION,
    MARKET_MODEL_ID,
    PRICER_MODEL_ID,
    WING_REGION,
    _apply_floor,
    _resolve_belly_score_mode,
    advance_weights,
    buckets_to_ladder,
    compute_fair_value,
    ladder_to_buckets,
)
from market_maker.state_store import BayesScoreRow, MMStateStore

TS = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def _snapshot(strikes, p_list, expiry_key="2026-08-20"):
    p_hat = {float(k): float(p) for k, p in zip(strikes, p_list)}
    sigma2 = {float(k): 1e-6 for k in strikes}
    return PricerSnapshot(
        ts=TS,
        expiry_key=expiry_key,
        tte_days=7.0,
        s0=100000.0,
        n_sims=15000,
        strikes=[float(k) for k in strikes],
        grid_strikes=[float(k) for k in strikes],
        p_hat=p_hat,
        p_grid=dict(p_hat),
        sigma2=sigma2,
        sigma2_ladder=1e-6,
        sigma2_source=Sigma2Source.MC,
        confidence_tier=ConfidenceTier.FULL,
        horizon_gate_active=False,
        stale=False,
    )


def _bankrolls(pricer=0.5, market=0.5, frozen=False, update_count=0):
    return BankrollState(
        model_ids=["pricer", "market"],
        bankrolls={"pricer": pricer, "market": market},
        last_update=TS,
        update_count=update_count,
        frozen=frozen,
    )


def _states(pricer=0.5, market=0.5, frozen=False):
    return {
        BELLY_REGION: _bankrolls(pricer, market, frozen),
        WING_REGION: _bankrolls(pricer, market, frozen),
    }


def _mids(strikes, p_list):
    return {float(k): float(p) for k, p in zip(strikes, p_list)}


# ---------------------------------------------------------------------------
# Level-shift response: full-support drift factor responds; a belly-SUBSET
# (v1) form, computed inline as the regression guard, does not.
# ---------------------------------------------------------------------------


def test_level_shift_response_full_support_vs_subset_regression_guard():
    # market_now_bucket / c_lag chosen with an ASYMMETRIC ratio profile
    # (market_now/c_lag differs at bucket 0 vs bucket n) so a level shift
    # concentrated in the two tail buckets produces a nonzero net response
    # (hand-derived and cross-checked against this exact implementation).
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0]
    market_now_bucket = np.array([0.05, 0.20, 0.25, 0.25, 0.15, 0.10])
    c_lag = np.array([0.10, 0.20, 0.20, 0.20, 0.20, 0.10])
    market_now_p = buckets_to_ladder(market_now_bucket)

    m_lag_bucket = c_lag.copy()  # any valid bucket vector; unused by the pricer-only assertions below
    p_base_p = np.array([0.90, 0.65, 0.45, 0.30, 0.15])

    cfg = MMConfig(belly_score_mode="shadow", belly_drift_bucket_eps=0.0)  # exact-golden: unsmoothed algebra
    factors = {}
    subset_idx = [1, 2, 3, 4]  # belly-only interior buckets (v1 form)
    subset = {}
    for c in (0.0, 0.05):
        p_lag_bucket = ladder_to_buckets(p_base_p + c)
        res = compute_fair_value(
            _snapshot(strikes, market_now_p), _mids(strikes, market_now_p), _states(), cfg,
            belly_lag_forecasts={MARKET_MODEL_ID: m_lag_bucket, PRICER_MODEL_ID: p_lag_bucket},
            belly_lag_consensus=c_lag,
        )
        assert res.belly_score_skip is None
        factors[c] = res.belly_drift_factors["pricer"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_p = (market_now_bucket * p_lag_bucket / c_lag)[subset_idx]
        subset[c] = float(np.sum(ratio_p))

    # Full-support factor RESPONDS to the level shift.
    assert factors[0.05] == pytest.approx(1.025, abs=1e-9)
    assert factors[0.0] == pytest.approx(1.0, abs=1e-9)
    assert abs(factors[0.05] - factors[0.0]) > 0.01

    # The belly-SUBSET (v1) form does NOT respond -- interior buckets are
    # identical regardless of c (the level shift cancels there).
    assert subset[0.05] == pytest.approx(subset[0.0], abs=1e-9)


# ---------------------------------------------------------------------------
# Martingale: M_now == M_lag -> factor_market >= factor_pricer (v1's failing
# case is impossible over full support).
# ---------------------------------------------------------------------------


def test_martingale_market_never_loses_to_pricer():
    strikes = [1.0, 2.0, 3.0]
    mids_lag = [0.60, 0.40, 0.20]
    pricer_lag = [0.75, 0.50, 0.28]
    m_lag = ladder_to_buckets(np.array(mids_lag))
    p_lag = ladder_to_buckets(np.array(pricer_lag))
    c_lag = 0.5 * (m_lag + p_lag)

    cfg = MMConfig(belly_score_mode="shadow", belly_drift_bucket_eps=0.0)  # exact-golden: unsmoothed algebra
    res = compute_fair_value(
        _snapshot(strikes, mids_lag), _mids(strikes, mids_lag), _states(), cfg,
        belly_lag_forecasts={MARKET_MODEL_ID: m_lag, PRICER_MODEL_ID: p_lag},
        belly_lag_consensus=c_lag,
    )
    assert res.belly_score_skip is None
    assert res.belly_drift_factors["market"] >= res.belly_drift_factors["pricer"]
    # Hand-derived exact values.
    assert res.belly_drift_factors["market"] == pytest.approx(1.0272283272283274, abs=1e-9)
    assert res.belly_drift_factors["pricer"] == pytest.approx(0.9727716727716729, abs=1e-9)


# ---------------------------------------------------------------------------
# Alpha sweep: factor gap (market - pricer) crosses zero exactly at
# alpha == w_p (positive below, ~0 at, negative above).
# ---------------------------------------------------------------------------


def test_alpha_sweep_crosses_zero_at_w_p():
    strikes = [1.0, 2.0, 3.0]
    m_lag = np.array([0.1, 0.3, 0.4, 0.2])
    d = np.array([0.02, -0.01, -0.03, 0.02])
    p_lag = m_lag - d
    w_p = 0.5
    c_lag = m_lag - w_p * d
    cfg = MMConfig(belly_score_mode="shadow")

    gaps = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        m_now_bucket = m_lag - alpha * d
        m_now_p = buckets_to_ladder(m_now_bucket)
        res = compute_fair_value(
            _snapshot(strikes, m_now_p), _mids(strikes, m_now_p), _states(), cfg,
            belly_lag_forecasts={MARKET_MODEL_ID: m_lag, PRICER_MODEL_ID: p_lag},
            belly_lag_consensus=c_lag,
        )
        assert res.belly_score_skip is None
        gaps[alpha] = res.belly_drift_factors["market"] - res.belly_drift_factors["pricer"]

    assert gaps[0.0] > 0.0
    assert gaps[0.25] > 0.0
    assert gaps[0.5] == pytest.approx(0.0, abs=1e-9)
    assert gaps[0.75] < 0.0
    assert gaps[1.0] < 0.0
    # Monotone decreasing in alpha, symmetric around the w_p=0.5 crossing.
    assert gaps[0.0] == pytest.approx(-gaps[1.0], abs=1e-9)
    assert gaps[0.25] == pytest.approx(-gaps[0.75], abs=1e-9)


# ---------------------------------------------------------------------------
# s_tail_frac on the review's 5-strike worked example (~0.81).
# ---------------------------------------------------------------------------


def test_s_tail_frac_worked_example():
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0]
    mids_lag = [0.95, 0.80, 0.50, 0.20, 0.05]
    pricer_lag = [0.96, 0.83, 0.53, 0.24, 0.08]
    m_lag = ladder_to_buckets(np.array(mids_lag))
    p_lag = ladder_to_buckets(np.array(pricer_lag))
    c_lag = 0.5 * (m_lag + p_lag)

    cfg = MMConfig(belly_score_mode="shadow")
    res = compute_fair_value(
        _snapshot(strikes, mids_lag), _mids(strikes, mids_lag), _states(), cfg,
        belly_lag_forecasts={MARKET_MODEL_ID: m_lag, PRICER_MODEL_ID: p_lag},
        belly_lag_consensus=c_lag,
    )
    assert res.belly_score_skip is None
    assert res.belly_s_tail_frac == pytest.approx(0.81, abs=0.02)


def test_s_tail_frac_zero_when_s_zero():
    # Identical lag forecasts (d == 0 everywhere) -> S == 0 -> s_tail_frac
    # defined as exactly 0.0 (not NaN/undefined).
    strikes = [1.0, 2.0, 3.0]
    p = [0.6, 0.4, 0.2]
    lag_bucket = ladder_to_buckets(np.array(p))
    c_lag = lag_bucket.copy()
    cfg = MMConfig(belly_score_mode="shadow")
    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _states(), cfg,
        belly_lag_forecasts={MARKET_MODEL_ID: lag_bucket, PRICER_MODEL_ID: lag_bucket},
        belly_lag_consensus=c_lag,
    )
    assert res.belly_score_skip is None
    assert res.belly_s_tail_frac == 0.0


# ---------------------------------------------------------------------------
# Control factor == drift factor when consensus_new == market_now (both
# regions weighted 100% market so the PRE-update ladder-space consensus
# equals the market's own sanitized ladder exactly, for every strike).
# ---------------------------------------------------------------------------


def test_control_equals_drift_when_consensus_new_equals_market_now():
    strikes = [1.0, 2.0, 3.0]
    pricer_now = [0.70, 0.55, 0.40]  # differs from market_now -- genuine blend inputs
    market_now = [0.65, 0.50, 0.35]  # all strikes inside belly_band -> wing weight irrelevant
    states = {BELLY_REGION: _bankrolls(0.0, 1.0), WING_REGION: _bankrolls(0.0, 1.0)}

    mids_lag = [0.60, 0.40, 0.20]
    pricer_lag = [0.75, 0.50, 0.28]
    m_lag = ladder_to_buckets(np.array(mids_lag))
    p_lag = ladder_to_buckets(np.array(pricer_lag))
    c_lag = 0.5 * (m_lag + p_lag)

    cfg = MMConfig(belly_score_mode="shadow")
    res = compute_fair_value(
        _snapshot(strikes, pricer_now), _mids(strikes, market_now), states, cfg,
        belly_lag_forecasts={MARKET_MODEL_ID: m_lag, PRICER_MODEL_ID: p_lag},
        belly_lag_consensus=c_lag,
    )
    assert res.belly_score_skip is None
    assert res.belly_drift_factors == pytest.approx(res.belly_control_factors, abs=1e-12)


def test_control_differs_from_drift_when_consensus_new_differs_from_market_now():
    # Contrast case: default region-uniform 0.5/0.5 weights make consensus_new
    # a genuine BLEND of pricer_now and market_now (not equal to either), so
    # drift and control targets differ and the factors must differ too.
    strikes = [1.0, 2.0, 3.0]
    pricer_now = [0.70, 0.55, 0.40]
    market_now = [0.65, 0.50, 0.35]
    mids_lag = [0.60, 0.40, 0.20]
    pricer_lag = [0.75, 0.50, 0.28]
    m_lag = ladder_to_buckets(np.array(mids_lag))
    p_lag = ladder_to_buckets(np.array(pricer_lag))
    c_lag = 0.5 * (m_lag + p_lag)

    cfg = MMConfig(belly_score_mode="shadow")
    res = compute_fair_value(
        _snapshot(strikes, pricer_now), _mids(strikes, market_now), _states(), cfg,
        belly_lag_forecasts={MARKET_MODEL_ID: m_lag, PRICER_MODEL_ID: p_lag},
        belly_lag_consensus=c_lag,
    )
    assert res.belly_score_skip is None
    assert res.belly_drift_factors["pricer"] != pytest.approx(
        res.belly_control_factors["pricer"], abs=1e-9)


# ---------------------------------------------------------------------------
# advance_weights(): parity with the pre-refactor applied-loop arithmetic;
# None on the s<=0/non-finite skip.
# ---------------------------------------------------------------------------


def test_advance_weights_parity_with_hand_computed_sequence():
    w_pre = np.array([0.3, 0.7])
    factors = np.array([2.0, 0.5])
    temper = 0.4
    floor = 0.02

    f = factors ** temper
    updated = w_pre * f
    s = updated.sum()
    expected = _apply_floor(updated / s, floor)

    got = advance_weights(w_pre, factors, temper, floor)
    assert got == pytest.approx(expected, abs=1e-15)


def test_advance_weights_temper_one_skips_power_op():
    # temper == 1.0 must skip the ** entirely (not merely equal x**1.0),
    # matching the pre-refactor `if temper < 1.0:` gate exactly.
    w_pre = np.array([0.5, 0.5])
    factors = np.array([3.0, 1.0])
    got = advance_weights(w_pre, factors, 1.0, 0.02)
    expected = _apply_floor((w_pre * factors) / (w_pre * factors).sum(), 0.02)
    assert got == pytest.approx(expected, abs=1e-15)


def test_advance_weights_none_on_zero_factor_sum():
    w_pre = np.array([0.5, 0.5])
    factors = np.array([0.0, 0.0])
    assert advance_weights(w_pre, factors, 1.0, 0.02) is None


def test_advance_weights_none_on_non_finite_factor_sum():
    w_pre = np.array([0.5, 0.5])
    factors = np.array([np.inf, 1.0])
    assert advance_weights(w_pre, factors, 1.0, 0.02) is None

    factors_nan = np.array([np.nan, 1.0])
    assert advance_weights(w_pre, factors_nan, 1.0, 0.02) is None


def test_advance_weights_none_on_negative_sum():
    # w_pre nonneg but a large negative factor can still drive the weighted
    # sum negative.
    w_pre = np.array([0.9, 0.1])
    factors = np.array([-5.0, 1.0])
    assert advance_weights(w_pre, factors, 1.0, 0.02) is None


# ---------------------------------------------------------------------------
# Mode matrix.
# ---------------------------------------------------------------------------


def test_resolve_belly_score_mode_unknown_strings_and_missing_attr():
    assert _resolve_belly_score_mode(MMConfig(belly_score_mode="legacy")) == "legacy"
    assert _resolve_belly_score_mode(MMConfig(belly_score_mode="shadow")) == "shadow"
    assert _resolve_belly_score_mode(MMConfig(belly_score_mode="live")) == "live"
    assert _resolve_belly_score_mode(MMConfig(belly_score_mode="bogus")) == "legacy"

    class _NoAttr:
        pass

    assert _resolve_belly_score_mode(_NoAttr()) == "legacy"


def _lag_kwargs():
    m_lag = ladder_to_buckets(np.array([0.6, 0.4, 0.2]))
    p_lag = ladder_to_buckets(np.array([0.75, 0.5, 0.28]))
    c_lag = 0.5 * (m_lag + p_lag)
    return dict(
        belly_lag_forecasts={MARKET_MODEL_ID: m_lag, PRICER_MODEL_ID: p_lag},
        belly_lag_consensus=c_lag,
    )


def test_mode_legacy_explicit_byte_identical_to_default_shadow_no_lag_kwargs():
    # belly_score_mode="legacy" (explicit) vs the config default ("shadow",
    # field omitted): with NO belly_lag_* kwargs supplied, shadow's applied
    # path never touches belly at all -- both configs must produce identical
    # fair_value/bankroll_states outputs. The skip reasons themselves
    # legitimately differ: legacy never enters the C1 block at all (skip
    # stays None), while shadow enters it and reports "no_lag" -- both are
    # journaling-only and neither perturbs the applied path asserted above.
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.8, 0.5, 0.2]
    market = [0.7, 0.5, 0.3]

    res_legacy = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4),
        MMConfig(belly_score_mode="legacy"),
    )
    res_default = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4),
        MMConfig(),
    )
    assert MMConfig().belly_score_mode == "shadow"
    assert res_legacy.fair_value.consensus_p == pytest.approx(res_default.fair_value.consensus_p)
    for region in (BELLY_REGION, WING_REGION):
        assert res_legacy.bankroll_states[region].bankrolls == pytest.approx(
            res_default.bankroll_states[region].bankrolls)
        assert (res_legacy.bankroll_states[region].update_count
                == res_default.bankroll_states[region].update_count)
    assert res_legacy.belly_score_skip is None
    assert res_default.belly_score_skip == "no_lag"
    assert res_legacy.belly_drift_factors is None
    assert res_default.belly_drift_factors is None


def test_mode_shadow_applied_path_byte_identical_to_legacy_while_carrying_factors():
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.8, 0.5, 0.2]
    market = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()

    res_legacy = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4),
        MMConfig(belly_score_mode="legacy"),
    )
    res_shadow = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4),
        MMConfig(belly_score_mode="shadow"),
        **lag,
    )
    # Applied path byte-identical (lag kwargs are journaling-only in shadow).
    assert res_shadow.fair_value.consensus_p == pytest.approx(res_legacy.fair_value.consensus_p)
    for region in (BELLY_REGION, WING_REGION):
        assert res_shadow.bankroll_states[region].bankrolls == pytest.approx(
            res_legacy.bankroll_states[region].bankrolls)
        assert (res_shadow.bankroll_states[region].update_count
                == res_legacy.bankroll_states[region].update_count)
    # ... but shadow's AnchorResult carries the drift/control factors.
    assert res_shadow.belly_score_skip is None
    assert res_shadow.belly_drift_factors is not None
    assert res_shadow.belly_control_factors is not None
    assert res_shadow.belly_s_tail_frac is not None


def test_mode_live_applies_drift_to_belly_only_wing_pin_unaffected():
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()
    cfg = MMConfig(belly_score_mode="live", wing_pricer_weight_pin=0.5)

    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _states(), cfg, **lag,
    )
    assert res.belly_score_skip is None
    assert res.bankroll_states[BELLY_REGION].update_count == 1
    assert res.bankroll_states[BELLY_REGION].bankrolls != pytest.approx(
        {"pricer": 0.5, "market": 0.5})
    # Wing stays exactly at the pin -- completely untouched by live mode.
    assert res.bankroll_states[WING_REGION].update_count == 0
    assert res.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})


def test_mode_live_without_lag_kwargs_skips_belly_update_entirely():
    # "refreshes without an event skip the belly update" -- live mode with
    # NO belly_lag_* this tick must leave belly weights/update_count
    # unchanged (not fall through to the legacy per-refresh update either).
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    cfg = MMConfig(belly_score_mode="live", wing_pricer_weight_pin=0.5)
    res = compute_fair_value(_snapshot(strikes, p), _mids(strikes, p), _states(), cfg)
    assert res.belly_score_skip == "no_lag"
    assert res.bankroll_states[BELLY_REGION].update_count == 0
    assert res.bankroll_states[BELLY_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})


def test_unknown_mode_string_behaves_as_legacy():
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.8, 0.5, 0.2]
    market = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()

    res_bogus = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4),
        MMConfig(belly_score_mode="bogus"), **lag,
    )
    res_legacy = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4),
        MMConfig(belly_score_mode="legacy"), **lag,
    )
    assert res_bogus.fair_value.consensus_p == pytest.approx(res_legacy.fair_value.consensus_p)
    assert res_bogus.belly_score_skip is None
    assert res_bogus.belly_drift_factors is None
    assert res_bogus.belly_control_factors is None
    assert res_bogus.belly_s_tail_frac is None


# ---------------------------------------------------------------------------
# No mutation of forecasts/lag arrays.
# ---------------------------------------------------------------------------


def test_no_mutation_of_lag_arrays():
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()
    m_lag_copy = lag["belly_lag_forecasts"][MARKET_MODEL_ID].copy()
    p_lag_copy = lag["belly_lag_forecasts"][PRICER_MODEL_ID].copy()
    c_lag_copy = lag["belly_lag_consensus"].copy()

    cfg = MMConfig(belly_score_mode="live", wing_pricer_weight_pin=0.5)
    res = compute_fair_value(_snapshot(strikes, p), _mids(strikes, p), _states(), cfg, **lag)

    assert res.belly_score_skip is None
    assert np.array_equal(lag["belly_lag_forecasts"][MARKET_MODEL_ID], m_lag_copy)
    assert np.array_equal(lag["belly_lag_forecasts"][PRICER_MODEL_ID], p_lag_copy)
    assert np.array_equal(lag["belly_lag_consensus"], c_lag_copy)


# ---------------------------------------------------------------------------
# Own precondition: shape/finiteness skips; applied path unaffected; live +
# pinned wing + prev_forecasts/prev_consensus None + valid lag inputs ->
# belly drift update still fires.
# ---------------------------------------------------------------------------


def test_precondition_shape_mismatch_wrong_consensus_shape():
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()
    lag["belly_lag_consensus"] = np.array([0.5, 0.5])  # wrong length (n+1=4 expected)
    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _states(), MMConfig(belly_score_mode="shadow"),
        **lag,
    )
    assert res.belly_score_skip == "shape_mismatch"
    assert res.belly_drift_factors is None


def test_precondition_shape_mismatch_missing_model_key():
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()
    del lag["belly_lag_forecasts"][PRICER_MODEL_ID]
    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _states(), MMConfig(belly_score_mode="shadow"),
        **lag,
    )
    assert res.belly_score_skip == "shape_mismatch"
    assert res.belly_drift_factors is None


def test_precondition_non_finite_lag_forecast():
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()
    bad = lag["belly_lag_forecasts"][MARKET_MODEL_ID].copy()
    bad[0] = np.nan
    lag["belly_lag_forecasts"] = {MARKET_MODEL_ID: bad, PRICER_MODEL_ID: lag["belly_lag_forecasts"][PRICER_MODEL_ID]}
    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _states(), MMConfig(belly_score_mode="shadow"),
        **lag,
    )
    assert res.belly_score_skip == "non_finite"
    assert res.belly_drift_factors is None


def test_precondition_non_finite_lag_consensus():
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()
    bad_c = lag["belly_lag_consensus"].copy()
    bad_c[1] = np.inf
    lag["belly_lag_consensus"] = bad_c
    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _states(), MMConfig(belly_score_mode="shadow"),
        **lag,
    )
    assert res.belly_score_skip == "non_finite"
    assert res.belly_drift_factors is None


def test_precondition_skip_does_not_affect_applied_path():
    # A malformed lag kwarg must not perturb the applied fair_value/
    # bankroll_states at all -- compare against the same call with no lag
    # kwargs supplied.
    strikes = [1.0, 2.0, 3.0]
    pricer = [0.8, 0.5, 0.2]
    market = [0.7, 0.5, 0.3]
    cfg = MMConfig(belly_score_mode="shadow")

    res_baseline = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4), cfg,
    )
    lag = _lag_kwargs()
    lag["belly_lag_consensus"] = np.array([0.5, 0.5])  # malformed
    res_malformed = compute_fair_value(
        _snapshot(strikes, pricer), _mids(strikes, market), _states(0.6, 0.4), cfg, **lag,
    )
    assert res_malformed.belly_score_skip == "shape_mismatch"
    assert res_malformed.fair_value.consensus_p == pytest.approx(res_baseline.fair_value.consensus_p)
    for region in (BELLY_REGION, WING_REGION):
        assert res_malformed.bankroll_states[region].bankrolls == pytest.approx(
            res_baseline.bankroll_states[region].bankrolls)


def test_live_mode_fires_without_lag1_history_and_with_pinned_wing():
    # prev_forecasts/prev_consensus both None (no lag-1 history at all) --
    # the legacy per-refresh update never even runs -- yet the C1 drift
    # block, gated only on its OWN belly_lag_* precondition, still fires.
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()
    cfg = MMConfig(belly_score_mode="live", wing_pricer_weight_pin=0.5)

    res = compute_fair_value(
        _snapshot(strikes, p), _mids(strikes, p), _states(),
        cfg, prev_forecasts=None, prev_consensus=None, **lag,
    )
    assert res.belly_score_skip is None
    assert res.bankroll_states[BELLY_REGION].update_count == 1
    assert res.bankroll_states[WING_REGION].update_count == 0
    assert res.bankroll_states[WING_REGION].bankrolls == pytest.approx(
        {"pricer": 0.5, "market": 0.5})


def test_frozen_belly_live_mode_skips_with_reason():
    strikes = [1.0, 2.0, 3.0]
    p = [0.7, 0.5, 0.3]
    lag = _lag_kwargs()
    states = {BELLY_REGION: _bankrolls(frozen=True), WING_REGION: _bankrolls()}
    cfg = MMConfig(belly_score_mode="live", wing_pricer_weight_pin=0.5)
    res = compute_fair_value(_snapshot(strikes, p), _mids(strikes, p), states, cfg, **lag)
    assert res.belly_score_skip == "frozen"
    assert res.belly_drift_factors is None
    assert res.bankroll_states[BELLY_REGION].update_count == 0
    assert res.bankroll_states[BELLY_REGION].frozen is True


# ---------------------------------------------------------------------------
# _fallback: new AnchorResult fields None; belly_score_skip="fallback" only
# when lag kwargs were supplied under shadow/live.
# ---------------------------------------------------------------------------


def test_fallback_sets_belly_score_skip_fallback_when_lag_provided():
    strikes = [1.0, 2.0, 3.0]
    lag = _lag_kwargs()
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        {1.0: 0.8, 2.0: float("nan"), 3.0: 0.2},
        _states(0.5, 0.5),
        MMConfig(belly_score_mode="shadow"),
        **lag,
    )
    from market_maker.contracts import AnchorMethod
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.belly_score_skip == "fallback"
    assert res.belly_drift_factors is None
    assert res.belly_control_factors is None
    assert res.belly_s_tail_frac is None


def test_fallback_belly_score_skip_none_when_no_lag_kwargs():
    strikes = [1.0, 2.0, 3.0]
    res = compute_fair_value(
        _snapshot(strikes, [0.9, 0.6, 0.3]),
        {1.0: 0.8, 2.0: float("nan"), 3.0: 0.2},
        _states(0.5, 0.5),
        MMConfig(belly_score_mode="shadow"),
    )
    from market_maker.contracts import AnchorMethod
    assert res.fair_value.anchor_method == AnchorMethod.FIXED_BLEND_FALLBACK
    assert res.belly_score_skip is None


# ===========================================================================
# state_store.bayes_score_log
# ===========================================================================


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "mm_state.db")
    s = MMStateStore(db_path)
    yield s
    s.close()


def _row(ts, expiry_key="2026-08-20", model_id="pricer", **overrides):
    base = dict(
        ts=ts,
        expiry_key=expiry_key,
        mode="shadow",
        model_id=model_id,
        factor_legacy=1.01,
        factor_drift=1.02,
        factor_control=1.015,
        skip_reason=None,
        anchor_method="BEUOY",
        lag_s=3612.0,
        belly_divergence=0.03,
        s_tail_frac=0.42,
        belly_snapshot=[[60000.0, 0.55, 0.52], [62000.0, 0.30, 0.28]],
        weight_applied_after=0.55,
        weight_drift_after=0.57,
        weight_control_after=0.56,
    )
    base.update(overrides)
    return BayesScoreRow(**base)


def test_bayes_score_log_round_trip(store):
    rows = [
        _row(TS, model_id="pricer"),
        _row(TS, model_id="market", factor_legacy=0.99, factor_drift=0.98, factor_control=0.985),
    ]
    n = store.append_bayes_scores(rows)
    assert n == 2
    got = store.get_bayes_scores()
    assert len(got) == 2
    assert got[0].model_id == "pricer"
    assert got[0].ts == TS
    assert got[0].expiry_key == "2026-08-20"
    assert got[0].mode == "shadow"
    assert got[0].factor_legacy == pytest.approx(1.01)
    assert got[0].factor_drift == pytest.approx(1.02)
    assert got[0].factor_control == pytest.approx(1.015)
    assert got[0].skip_reason is None
    assert got[0].anchor_method == "BEUOY"
    assert got[0].lag_s == pytest.approx(3612.0)
    assert got[0].belly_divergence == pytest.approx(0.03)
    assert got[0].s_tail_frac == pytest.approx(0.42)
    assert got[0].belly_snapshot == [[60000.0, 0.55, 0.52], [62000.0, 0.30, 0.28]]
    assert got[0].weight_applied_after == pytest.approx(0.55)
    assert got[0].weight_drift_after == pytest.approx(0.57)
    assert got[0].weight_control_after == pytest.approx(0.56)
    assert got[1].model_id == "market"

    # empty list is a no-op
    assert store.append_bayes_scores([]) == 0


def test_bayes_score_log_skip_row_model_id_empty(store):
    # Event identity: a skipped event writes exactly ONE row with model_id=''
    # and NULL factor columns.
    row = _row(
        TS, model_id="", factor_legacy=None, factor_drift=None, factor_control=None,
        skip_reason="no_lag", weight_applied_after=None, weight_drift_after=None,
        weight_control_after=None, belly_snapshot=None,
    )
    store.append_bayes_scores([row])
    got = store.get_bayes_scores()
    assert len(got) == 1
    assert got[0].model_id == ""
    assert got[0].skip_reason == "no_lag"
    assert got[0].factor_legacy is None
    assert got[0].factor_drift is None
    assert got[0].factor_control is None
    assert got[0].weight_applied_after is None
    assert got[0].belly_snapshot is None


def test_bayes_score_log_event_key_distinctness(store):
    # (ts, expiry_key) is the event key -- two distinct events (different
    # ts) each write their own row set; get_bayes_scores preserves insertion
    # order (id ASC) so rows group by event.
    store.append_bayes_scores([_row(TS, model_id="pricer"), _row(TS, model_id="market")])
    ts2 = TS + timedelta(seconds=900)
    store.append_bayes_scores([_row(ts2, model_id="", skip_reason="stale_lag",
                                     factor_legacy=None, factor_drift=None, factor_control=None)])
    got = store.get_bayes_scores()
    assert [(r.ts, r.model_id) for r in got] == [
        (TS, "pricer"), (TS, "market"), (ts2, ""),
    ]
    distinct_events = {(r.ts, r.expiry_key) for r in got}
    assert distinct_events == {(TS, "2026-08-20"), (ts2, "2026-08-20")}


def test_bayes_score_log_get_since_ts_filters(store):
    ts0 = TS
    ts1 = TS + timedelta(seconds=900)
    ts2 = TS + timedelta(seconds=1800)
    store.append_bayes_scores([_row(ts0, model_id="pricer")])
    store.append_bayes_scores([_row(ts1, model_id="pricer")])
    store.append_bayes_scores([_row(ts2, model_id="pricer")])
    got = store.get_bayes_scores(since_ts=ts1)
    assert [r.ts for r in got] == [ts1, ts2]


def test_bayes_score_log_prune_honors_cutoff(store):
    old_ts = TS - timedelta(days=40)
    recent_ts = TS - timedelta(days=1)
    store.append_bayes_scores([_row(old_ts, model_id="pricer")])
    store.append_bayes_scores([_row(recent_ts, model_id="pricer")])

    deleted = store.prune_bayes_score_log(TS - timedelta(days=28))
    assert deleted == 1
    got = store.get_bayes_scores()
    assert len(got) == 1
    assert got[0].ts == recent_ts


def test_bayes_score_log_migration_on_pre_existing_db(tmp_path):
    # A pre-C1 db without the table gains it on reopen (CREATE TABLE IF NOT
    # EXISTS in _init_schema, same pattern as fill_markouts).
    db_path = str(tmp_path / "mm.db")
    s = MMStateStore(db_path)
    s._conn.execute("DROP TABLE bayes_score_log")
    s._conn.commit()
    s.close()

    s2 = MMStateStore(db_path)
    s2.append_bayes_scores([_row(TS, model_id="pricer")])
    got = s2.get_bayes_scores()
    assert len(got) == 1
    assert got[0].model_id == "pricer"
    s2.close()


# ---------------------------------------------------------------------------
# Bucket-smoothing epsilon (2026-08-21 acceptance-review fix): zero buckets
# no longer skip the event; legacy sentinel preserved; law invariants hold.
# ---------------------------------------------------------------------------


def _zero_bucket_inputs():
    """Lag pair with a genuinely ZERO bucket (flat adjacent ladder values),
    the measured cause of the 31% non_finite skip rate on the VPS."""
    strikes = [1.0, 2.0, 3.0, 4.0]
    # flat segment: p[0] == p[1] -> interior bucket 1 is exactly 0 in both
    # the lag consensus and the lagged forecasts.
    p_flat = np.array([0.90, 0.90, 0.40, 0.10])
    c_lag = ladder_to_buckets(p_flat)
    assert float(c_lag[1]) == 0.0  # precondition of the scenario
    lag = {MARKET_MODEL_ID: ladder_to_buckets(p_flat),
           PRICER_MODEL_ID: ladder_to_buckets(np.array([0.92, 0.91, 0.45, 0.12]))}
    mids_p = p_flat
    return strikes, mids_p, lag, c_lag


def test_smooth_buckets_unit():
    from market_maker.fair_value_anchor import _smooth_buckets
    v = np.array([0.0, 0.5, 0.5, 0.0])
    s = _smooth_buckets(v, 1e-6)
    assert s.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(s > 0.0)
    # sentinel: eps <= 0 (and NaN) return the input unchanged
    assert _smooth_buckets(v, 0.0) is v
    assert _smooth_buckets(v, -1.0) is v
    assert _smooth_buckets(v, float("nan")) is v


def test_zero_bucket_event_scores_with_default_eps():
    strikes, mids_p, lag, c_lag = _zero_bucket_inputs()
    cfg = MMConfig(belly_score_mode="shadow")  # default eps 1e-6
    res = compute_fair_value(
        _snapshot(strikes, mids_p), _mids(strikes, mids_p), _states(), cfg,
        belly_lag_forecasts=lag, belly_lag_consensus=c_lag,
    )
    assert res.belly_score_skip is None
    for f in res.belly_drift_factors.values():
        assert np.isfinite(f) and 1e-3 <= f <= 1e3
    for f in res.belly_control_factors.values():
        assert np.isfinite(f)
    assert res.belly_s_tail_frac is not None and np.isfinite(res.belly_s_tail_frac)


def test_zero_bucket_event_skips_with_eps_sentinel_off():
    strikes, mids_p, lag, c_lag = _zero_bucket_inputs()
    cfg = MMConfig(belly_score_mode="shadow", belly_drift_bucket_eps=0.0)
    res = compute_fair_value(
        _snapshot(strikes, mids_p), _mids(strikes, mids_p), _states(), cfg,
        belly_lag_forecasts=lag, belly_lag_consensus=c_lag,
    )
    assert res.belly_score_skip == "non_finite"
    assert res.belly_drift_factors is None


def test_martingale_invariant_survives_smoothing_with_zero_buckets():
    # M_now == M_lag with a zero bucket present: market factor must still
    # be >= pricer factor (the dMass == 0 cancellation must survive the
    # smoothing -- all vectors smoothed, all stay normalized).
    strikes, mids_p, lag, c_lag = _zero_bucket_inputs()
    cfg = MMConfig(belly_score_mode="shadow")
    res = compute_fair_value(
        _snapshot(strikes, mids_p), _mids(strikes, mids_p), _states(), cfg,
        belly_lag_forecasts=lag, belly_lag_consensus=c_lag,
    )
    assert res.belly_score_skip is None
    assert res.belly_drift_factors["market"] >= res.belly_drift_factors["pricer"]


def test_smoothing_distortion_negligible_without_zero_buckets():
    # On a ladder with no zero buckets, eps 1e-6 must move the factors by
    # less than 1e-4 vs the unsmoothed computation.
    strikes = [1.0, 2.0, 3.0, 4.0, 5.0]
    p_mid = np.array([0.90, 0.65, 0.45, 0.30, 0.15])
    lag = {MARKET_MODEL_ID: ladder_to_buckets(p_mid),
           PRICER_MODEL_ID: ladder_to_buckets(p_mid + 0.03)}
    c_lag = ladder_to_buckets(p_mid + 0.015)
    out = {}
    for eps in (0.0, 1e-6):
        cfg = MMConfig(belly_score_mode="shadow", belly_drift_bucket_eps=eps)
        res = compute_fair_value(
            _snapshot(strikes, p_mid), _mids(strikes, p_mid), _states(), cfg,
            belly_lag_forecasts=lag, belly_lag_consensus=c_lag,
        )
        assert res.belly_score_skip is None
        out[eps] = res.belly_drift_factors
    for mid_ in ("pricer", "market"):
        assert out[1e-6][mid_] == pytest.approx(out[0.0][mid_], abs=1e-4)
