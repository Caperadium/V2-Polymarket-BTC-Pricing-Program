"""
test_xgb_drift_shift.py

Tests for the re-enabled XGBoost directional drift shift (FIX 3 / H2,
temp/xgb_activation_plan.md §7). The shift replaces the old invalid per-strike
additive blend with a single monotonicity-preserving distribution shift.

Pure-helper tests run without DATA/. Integration tests skip cleanly when the
BTC CSVs are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.pricing.btc_pricing_engine import (
    apply_xgb_drift_shift,
    dte_bucket_horizon,
    XGB_MAX_SHIFT_FRAC,
)
from core.pricing.directional_xgb import (
    DirectionalXGB,
    build_features,
    to_daily_log_return_series,
)

_REPO = Path(__file__).resolve().parent.parent
_HOURLY = _REPO / "DATA" / "btc_hourly.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lognormal_paths(S0=100.0, sigma=0.1, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    return S0 * np.exp(rng.normal(0.0, sigma, n))


def _jumpy_paths(S0=100.0, n=200_000, seed=1):
    """Non-Gaussian terminal distribution: diffusion + occasional down-jumps."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.08, n)
    jump = np.where(rng.random(n) < 0.05, rng.normal(-0.3, 0.1, n), 0.0)
    return S0 * np.exp(base + jump)


def _p_up(paths, S0):
    return float(np.mean(paths >= S0))


# ---------------------------------------------------------------------------
# 1. Identity short-circuits
# ---------------------------------------------------------------------------

def test_identity_neutral_p_up():
    paths = _lognormal_paths()
    out, d, meta = apply_xgb_drift_shift(paths, 100.0, 0.5, 0.5)
    assert d == 0.0 and meta["applied"] is False
    assert out is paths  # untouched reference


def test_identity_lambda_zero():
    paths = _lognormal_paths()
    out, d, meta = apply_xgb_drift_shift(paths, 100.0, 0.8, 0.0)
    assert d == 0.0 and meta["applied"] is False


def test_identity_sigma_floor():
    # All paths equal → sigma_H == 0 → skip.
    paths = np.full(1000, 100.0)
    out, d, meta = apply_xgb_drift_shift(paths, 100.0, 0.8, 0.5)
    assert d == 0.0 and meta["applied"] is False


# ---------------------------------------------------------------------------
# 2. Monotonicity preserved after shift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_up", [0.2, 0.5, 0.8])
def test_monotonicity_preserved(p_up):
    S0 = 100.0
    paths = _jumpy_paths(S0)
    shifted, _, _ = apply_xgb_drift_shift(paths, S0, p_up, 0.5)
    strikes = np.linspace(60, 160, 25)
    probs = [np.mean(shifted >= k) for k in strikes]
    assert all(probs[i] >= probs[i + 1] - 1e-12 for i in range(len(probs) - 1))


# ---------------------------------------------------------------------------
# 3. Direction of the tilt
# ---------------------------------------------------------------------------

def test_direction_bullish_raises_p_up():
    S0 = 100.0
    paths = _lognormal_paths(S0)
    base = _p_up(paths, S0)
    shifted, d, _ = apply_xgb_drift_shift(paths, S0, 0.85, 0.5)
    assert d > 0
    assert _p_up(shifted, S0) > base


def test_direction_bearish_lowers_p_up():
    S0 = 100.0
    paths = _lognormal_paths(S0)
    base = _p_up(paths, S0)
    shifted, d, _ = apply_xgb_drift_shift(paths, S0, 0.15, 0.5)
    assert d < 0
    assert _p_up(shifted, S0) < base


# ---------------------------------------------------------------------------
# 4. Safety cap binds
# ---------------------------------------------------------------------------

def test_shift_cap_binds():
    S0 = 100.0
    paths = _lognormal_paths(S0, sigma=0.1)
    sigma_H = float(np.std(np.log(paths / S0)))
    # Tiny cap forces saturation; full-strength bullish tilt.
    shifted, d, meta = apply_xgb_drift_shift(
        paths, S0, 0.85, 1.0, max_shift_frac=0.01
    )
    assert abs(d) == pytest.approx(0.01 * sigma_H, rel=1e-6)


# ---------------------------------------------------------------------------
# 9. ECDF accuracy + monotone in p_up (non-Gaussian distribution)
# ---------------------------------------------------------------------------

def test_ecdf_hits_target_on_jumpy_dist():
    S0 = 100.0
    paths = _jumpy_paths(S0)
    # lam=1 so p_target == clipped p_up; generous cap so it isn't truncated.
    for p_up, expected in [(0.3, 0.3), (0.7, 0.7)]:
        shifted, _, meta = apply_xgb_drift_shift(
            paths, S0, p_up, 1.0, max_shift_frac=5.0
        )
        assert _p_up(shifted, S0) == pytest.approx(expected, abs=0.01)


def test_achieved_p_up_monotone_in_p_up():
    S0 = 100.0
    paths = _jumpy_paths(S0)
    achieved = []
    for p_up in np.linspace(0.15, 0.85, 8):
        shifted, _, _ = apply_xgb_drift_shift(paths, S0, float(p_up), 0.5,
                                              max_shift_frac=5.0)
        achieved.append(_p_up(shifted, S0))
    assert all(achieved[i] <= achieved[i + 1] + 1e-9 for i in range(len(achieved) - 1))


# ---------------------------------------------------------------------------
# 10. p_base extremes → guard skips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift_const", [+2.0, -2.0])
def test_p_base_extreme_guard(shift_const):
    # Shift a tight distribution far above/below S0 so base P(up) ~ 1 or ~ 0.
    S0 = 100.0
    rng = np.random.default_rng(3)
    paths = S0 * np.exp(rng.normal(shift_const, 0.05, 50_000))
    out, d, meta = apply_xgb_drift_shift(paths, S0, 0.2, 0.5)
    assert d == 0.0 and meta["applied"] is False
    assert meta["p_base"] <= 0.02 or meta["p_base"] >= 0.98


# ---------------------------------------------------------------------------
# 11. Macro date-join is past-only (C3)
# ---------------------------------------------------------------------------

def test_macro_date_join_past_only():
    # BTC daily index spans calendar days (incl weekends); macro is business-day.
    btc_idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    btc_ret = pd.Series(np.random.default_rng(4).normal(0, 0.02, len(btc_idx)),
                        index=btc_idx, name="btc_ret")
    macro_idx = pd.bdate_range("2022-01-01", periods=400, tz="UTC")
    # 'gold' encodes each macro row's own day-ordinal so we can detect leakage.
    ordinals = (macro_idx.asi8 // 86_400_000_000_000)
    macro = pd.DataFrame({"gold": ordinals.astype(float)}, index=macro_idx)

    feats = build_features(btc_ret, macro, horizon_days=22, include_target=False)
    assert "gold_level" in feats.columns
    btc_ord = btc_idx.asi8 // 86_400_000_000_000
    # Surviving rows keep their original positional index → map back to btc dates.
    for pos, gold in feats["gold_level"].items():
        assert gold <= btc_ord[pos] + 1e-9, (
            f"row {pos}: macro day-ordinal {gold} > btc day-ordinal {btc_ord[pos]} (leak)"
        )


# ---------------------------------------------------------------------------
# 12. DTE bucket horizon mapping (C2-a)
# ---------------------------------------------------------------------------

def test_dte_bucket_horizon_mapping():
    assert dte_bucket_horizon(0.0) == 3.5
    assert dte_bucket_horizon(3.0) == 3.5
    assert dte_bucket_horizon(7.0) == 10.5      # interior edge → higher bucket
    assert dte_bucket_horizon(13.9) == 10.5
    assert dte_bucket_horizon(14.0) == 22.0
    assert dte_bucket_horizon(29.9) == 22.0
    assert dte_bucket_horizon(30.0) is None     # gated off beyond buckets
    assert dte_bucket_horizon(45.0) is None


def test_short_bucket_not_floored_to_7():
    # The ≤7 bucket horizon (3.5→4) must reach the predictor, not the legacy
    # max(7, …) floor. predict_direction_adjustment(horizon_days=...) overrides it.
    if not _HOURLY.exists():
        pytest.skip("DATA/btc_hourly.csv absent")
    s = to_daily_log_return_series(pd.read_csv(_HOURLY))
    m = DirectionalXGB()
    assert m.train_from_slice(s, None, 4)
    assert m._horizon_days == 4  # trained at the bucket horizon, not floored


# ---------------------------------------------------------------------------
# 6. Untrained model → neutral → no shift
# ---------------------------------------------------------------------------

def test_untrained_model_returns_neutral():
    m = DirectionalXGB()
    p = m.predict_direction_adjustment(hours_to_expiry=11 * 24,
                                       btc_returns=np.zeros(500))
    assert p == 0.5


# ---------------------------------------------------------------------------
# Integration tests (need DATA/)
# ---------------------------------------------------------------------------

def _engine_hourly_df():
    df = pd.read_csv(_HOURLY)
    cmap = {c.lower(): c for c in df.columns}
    dc = cmap.get("date", cmap.get("timestamp"))
    df[dc] = pd.to_datetime(df[dc], utc=True)
    return df.reset_index(drop=True)


@pytest.mark.skipif(not _HOURLY.exists(), reason="DATA/btc_hourly.csv absent")
def test_flag_off_bit_for_bit_regression():
    from core.pricing.btc_pricing_engine import calculate_probabilities
    h = _engine_hourly_df()
    strikes = [80000, 95000, 110000]
    kw = dict(strikes=strikes, hours_to_expiry=11 * 24, hourly_df=h,
              n_sims=15000, seed=7, disable_staleness_check=True)
    base = calculate_probabilities(**kw)
    # XGB flag on but lambda 0 → must be byte-identical to flag-off baseline.
    m = DirectionalXGB()
    m.train_from_slice(to_daily_log_return_series(h), None, 11)
    on0 = calculate_probabilities(use_xgb_direction=True, xgb_model=m,
                                  xgb_tilt_lambda=0.0, **kw)
    for s in strikes:
        assert base[s] == on0[s]


@pytest.mark.skipif(not _HOURLY.exists(), reason="DATA/btc_hourly.csv absent")
def test_martingale_anchor_skips_xgb():
    from core.pricing.btc_pricing_engine import calculate_probabilities
    h = _engine_hourly_df()
    m = DirectionalXGB()
    m.train_from_slice(to_daily_log_return_series(h), None, 11)
    r = calculate_probabilities(
        strikes=[95000], hours_to_expiry=11 * 24, hourly_df=h, n_sims=15000,
        seed=7, disable_staleness_check=True, use_xgb_direction=True,
        xgb_model=m, xgb_tilt_lambda=0.5, martingale_anchor=True,
    )
    assert r["_meta"]["xgb_applied"] is False


# ---------------------------------------------------------------------------
# 7. Verifier macro arm: strict-< truncation excludes >= snapshot (leak guard)
# ---------------------------------------------------------------------------

def test_verifier_macro_truncation_strict_lt(tmp_path):
    from core.backtesting.in_sample_oos import _load_btc_max_index, _strict_lt
    # Minimal macro CSV (date-indexed) in an isolated DATA dir.
    idx = pd.date_range("2022-01-01", periods=50, freq="D", tz="UTC")
    pd.DataFrame({"gold": np.arange(50.0)}, index=idx).to_csv(tmp_path / "macro_daily.csv")
    indexes = _load_btc_max_index(tmp_path)
    assert "macro" in indexes
    snap = idx[30]
    sl = _strict_lt(indexes["macro"], snap)
    assert len(sl) == 30 and sl.max() < snap


# ---------------------------------------------------------------------------
# 13. XGB-ON is NOT bit-reproducible across n_sims (documented; not an assertion
#     of equality). Confirms stochastic p_base so calibration must fix n_sims.
# ---------------------------------------------------------------------------

def test_xgb_on_stochastic_across_n_sims():
    S0 = 100.0
    p1 = _lognormal_paths(S0, n=50_000, seed=9)
    p2 = _lognormal_paths(S0, n=200_000, seed=9)
    _, d1, _ = apply_xgb_drift_shift(p1, S0, 0.7, 0.5)
    _, d2, _ = apply_xgb_drift_shift(p2, S0, 0.7, 0.5)
    # Different sample sizes → different empirical quantile → different Δ_H.
    assert d1 != d2
