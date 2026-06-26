"""
test_pricing_engine_fixes.py

Regression + edge-case tests for the BTC pricing-engine fix plan
(temp/PRICING_ENGINE_FIX_PLAN.md). One section per FIX. Data-dependent tests
skip cleanly when DATA/ CSVs are absent.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
_HOURLY = _REPO / "DATA" / "btc_hourly.csv"
_INTRADAY = _REPO / "DATA" / "btc_intraday_1m.csv"


def _load_hourly_returns(n: int | None = None) -> np.ndarray:
    df = pd.read_csv(_HOURLY)
    cmap = {c.lower(): c for c in df.columns}
    cc = cmap["close"]
    ret = np.log(df[cc] / df[cc].shift(1)).dropna().to_numpy()
    return ret if n is None else ret[:n]


def _hourly_df(before: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(_HOURLY)
    cmap = {c.lower(): c for c in df.columns}
    dc = cmap.get("date", cmap.get("timestamp"))
    df[dc] = pd.to_datetime(df[dc], utc=True)
    if before is not None:
        df = df[df[dc] < pd.Timestamp(before, tz="UTC")]
    return df.reset_index(drop=True)


def _intraday_df(before: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(_INTRADAY)
    cmap = {c.lower(): c for c in df.columns}
    tc = cmap.get("timestamp", cmap.get("date"))
    df[tc] = pd.to_datetime(df[tc], utc=True)
    if before is not None:
        df = df[df[tc] < pd.Timestamp(before, tz="UTC")]
    return df.reset_index(drop=True)


needs_data = pytest.mark.skipif(
    not _HOURLY.exists(), reason="DATA/btc_hourly.csv not available"
)


# ===========================================================================
# FIX 1 (C1) — FIGARCH actually fits when requested
# ===========================================================================

@needs_data
def test_fix1_fit_garch_returns_figarch_weights():
    from core.pricing.btc_pricing_engine import fit_garch_model

    ret = pd.Series(_load_hourly_returns())
    params = fit_garch_model(ret, use_figarch=True)
    # On real BTC hourly data the FIGARCH fit converges; the dict MUST carry the
    # ARCH(inf) weights or simulate_paths(use_figarch=True) silently degrades to GARCH.
    assert "figarch_weights" in params
    assert params.get("use_figarch") is True
    assert len(params["figarch_weights"]) > 1


# ===========================================================================
# FIX 2 (M1 / M4) — jump calibration: Lee-Mykland bipower, leak-free
# ===========================================================================

@needs_data
def test_fix2_bipower_detects_sane_jump_rate():
    from core.pricing.jump_calibration import detect_jumps_bipower

    ret = _load_hourly_returns()
    mask = detect_jumps_bipower(ret)
    rate = mask.mean()
    # Old global-gated test flagged 0; MAD over-flags ~14%. Lee-Mykland should land
    # in a sensible jump-rate band.
    assert 0.0 < rate < 0.05


@needs_data
def test_fix2_calibrate_jumps_converges_and_is_bounded():
    from core.pricing.jump_calibration import calibrate_jumps

    ret = _load_hourly_returns()
    cal = calibrate_jumps(returns=ret, detection_method="bipower")
    assert cal.fit_converged
    assert 5.0 <= cal.lam <= 100.0
    assert 0.0 <= cal.p_crash <= 1.0
    assert 5.0 <= cal.eta_up <= 200.0
    assert 5.0 <= cal.eta_down <= 200.0


def test_fix2_default_detection_is_bipower():
    import inspect
    from core.pricing.jump_calibration import calibrate_jumps

    sig = inspect.signature(calibrate_jumps)
    assert sig.parameters["detection_method"].default == "bipower"


def test_fix2_calibrate_jumps_is_leak_free_with_returns_arg(tmp_path, monkeypatch):
    """Passing returns= must NEVER read the hourly CSV (would leak future data)."""
    import core.pricing.jump_calibration as jc

    def _boom(*a, **k):
        raise AssertionError("read_csv called — calibrate_jumps leaked to the full file")

    monkeypatch.setattr(jc.pd, "read_csv", _boom)
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, 5000)
    # Should run purely on the array; if it touched read_csv the monkeypatch fires.
    jc.calibrate_jumps(returns=ret, detection_method="bipower")


# ===========================================================================
# FIX 3 (H2) — XGBoost directional drift shift (RE-ENABLED).
# The old per-strike additive blend was removed; the engine now applies a single
# distribution drift shift. A malformed model must degrade gracefully (no raise),
# and the dedicated suite tests/test_xgb_drift_shift.py covers the math.
# ===========================================================================

@needs_data
def test_fix3_xgb_malformed_model_degrades_gracefully():
    from core.pricing.btc_pricing_engine import calculate_probabilities

    class DummyModel:  # lacks predict_direction_adjustment
        pass

    # Must NOT raise — the engine catches the failure and uses unshifted paths.
    res = calculate_probabilities(
        strikes=[100000.0],
        hours_to_expiry=48.0,
        hourly_df=_hourly_df(before="2026-05-01"),
        intraday_df=_intraday_df(before="2026-05-01"),
        n_sims=500,
        seed=1,
        use_xgb_direction=True,
        xgb_model=DummyModel(),
        disable_staleness_check=True,
    )
    assert 0.0 <= res[100000.0] <= 1.0
    assert res["_meta"]["xgb_applied"] is False


@needs_data
def test_fix3_xgb_off_runs_normally():
    from core.pricing.btc_pricing_engine import calculate_probabilities

    probs = calculate_probabilities(
        strikes=[100000.0],
        hours_to_expiry=48.0,
        hourly_df=_hourly_df(before="2026-05-01"),
        intraday_df=_intraday_df(before="2026-05-01"),
        n_sims=500,
        seed=1,
        use_xgb_direction=False,
        disable_staleness_check=True,
    )
    assert 0.0 <= probs[100000.0] <= 1.0


# ===========================================================================
# FIX 4 (H1) — regime switching wired leak-free via as_of
# ===========================================================================

def test_fix4_regime_detector_deterministic_and_bear_on_bear_series():
    from core.pricing.regime_detector import RegimeDetector

    rng = np.random.default_rng(7)
    # Three separable regimes so the 3-state HMM can actually fit, ENDING in a
    # strongly-bear segment (regime detection reports the last observation's state).
    bull = rng.normal(0.020, 0.010, 150)
    side = rng.normal(0.000, 0.008, 150)
    bear = rng.normal(-0.020, 0.012, 150)
    series = np.concatenate([bull, side, bear])
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    d1 = RegimeDetector()
    w1, dom1 = d1.fit_predict(series, now=now, force=True)
    d2 = RegimeDetector()
    w2, dom2 = d2.fit_predict(series, now=now, force=True)

    assert dom1 == "bear"
    assert w1["bear"] > 0.5
    # Determinism given fixed now + seed.
    assert dom1 == dom2
    assert w1 == pytest.approx(w2)


@needs_data
def test_fix4_as_of_threaded_no_wallclock_leak():
    """calculate_probabilities must refit the detector at as_of, not wall-clock."""
    from core.pricing.btc_pricing_engine import calculate_probabilities
    from core.pricing.regime_detector import RegimeDetector

    as_of = pd.Timestamp("2026-05-01", tz="UTC")
    det = RegimeDetector()
    calculate_probabilities(
        strikes=[100000.0],
        hours_to_expiry=72.0,
        hourly_df=_hourly_df(before="2026-05-01"),
        intraday_df=_intraday_df(before="2026-05-01"),
        n_sims=400,
        seed=3,
        use_regime_switching=True,
        regime_detector=det,
        as_of=as_of.to_pydatetime(),
        disable_staleness_check=True,
    )
    # The detector's fit timestamp is the SNAPSHOT, never real wall-clock now.
    assert det._last_fit_date == as_of.to_pydatetime()
    assert det._last_fit_date < datetime.now(timezone.utc)


# ===========================================================================
# FIX 5 (H3) — SVCJ vol-jump persistence under FIGARCH
# ===========================================================================

def _figarch_params():
    from core.pricing.btc_pricing_engine import _compute_figarch_weights

    return {
        "omega": 0.00001 / 24,
        "beta": 0.4558,
        "nu": 5.0,
        "mu": 0.0,
        "last_variance": 0.0004 / 24,
        "use_figarch": True,
        "figarch_weights": _compute_figarch_weights(d=0.3889, phi=0.3056, beta=0.4558, trunc_k=200),
        "figarch_d": 0.3889,
        "figarch_phi": 0.3056,
    }


def test_fix5_svcj_adds_variance_under_figarch():
    from core.pricing.btc_pricing_engine import simulate_paths

    gp = _figarch_params()
    no_svcj = simulate_paths(
        S0=100000.0, garch_params=gp,
        jump_params={"lambda": 25.0, "crash_prob": 0.6, "eta_up": 50.0,
                     "eta_down": 25.0, "mu_v": 0.0, "rho_J": 0.0},
        hours_to_expiry=720.0, n_sims=5000, seed=42, use_svcj=False, use_figarch=True,
    )
    svcj = simulate_paths(
        S0=100000.0, garch_params=gp,
        jump_params={"lambda": 25.0, "crash_prob": 0.6, "eta_up": 50.0,
                     "eta_down": 25.0, "mu_v": 0.0001, "rho_J": -0.3},
        hours_to_expiry=720.0, n_sims=5000, seed=42, use_svcj=True, use_figarch=True,
    )
    std_no = np.std(np.log(no_svcj / 100000))
    std_yes = np.std(np.log(svcj / 100000))
    # The H3 bug made these ~equal; persistence must lift terminal std measurably.
    assert std_yes > std_no * 1.03


def test_fix5_paths_finite_and_bounded_over_long_horizon():
    from core.pricing.btc_pricing_engine import simulate_paths

    gp = _figarch_params()
    paths = simulate_paths(
        S0=100000.0, garch_params=gp,
        jump_params={"lambda": 50.0, "crash_prob": 0.6, "eta_up": 50.0,
                     "eta_down": 25.0, "mu_v": 0.0005, "rho_J": -0.3},
        hours_to_expiry=720.0, n_sims=3000, seed=11, use_svcj=True, use_figarch=True,
    )
    assert np.all(np.isfinite(paths))
    assert np.all(paths > 0)


def test_fix5_persist_clamped_no_blowup():
    """svcj_persist >= 1 must be clamped so variance can't compound unboundedly."""
    from core.pricing.btc_pricing_engine import simulate_paths

    gp = _figarch_params()
    paths = simulate_paths(
        S0=100000.0, garch_params=gp,
        jump_params={"lambda": 50.0, "crash_prob": 0.6, "eta_up": 50.0,
                     "eta_down": 25.0, "mu_v": 0.0005, "rho_J": -0.3,
                     "svcj_persist": 5.0},  # pathological — must be clamped to (0,1)
        hours_to_expiry=720.0, n_sims=2000, seed=5, use_svcj=True, use_figarch=True,
    )
    assert np.all(np.isfinite(paths)) and np.all(paths > 0)


def test_fix5_garch_path_unchanged_determinism():
    """GARCH+SVCJ stream must be byte-identical (RNG draw order preserved)."""
    from core.pricing.btc_pricing_engine import simulate_paths

    gp = {"omega": 0.00001 / 24, "alpha": 0.1, "beta": 0.85, "nu": 5.0,
          "mu": 0.0, "last_variance": 0.0004 / 24}
    jp = {"lambda": 25.0, "crash_prob": 0.6, "eta_up": 50.0, "eta_down": 25.0,
          "mu_v": 0.0001, "rho_J": -0.3}
    a = simulate_paths(S0=100000.0, garch_params=gp, jump_params=jp,
                       hours_to_expiry=240.0, n_sims=2000, seed=42, use_svcj=True)
    b = simulate_paths(S0=100000.0, garch_params=gp, jump_params=jp,
                       hours_to_expiry=240.0, n_sims=2000, seed=42, use_svcj=True)
    assert np.array_equal(a, b)


# ===========================================================================
# FIX 6 (C2) — Basel MC shape bug
# ===========================================================================

def test_fix6_simulate_paths_is_1d():
    from core.pricing.btc_pricing_engine import simulate_paths

    gp = {"omega": 0.00001 / 24, "alpha": 0.1, "beta": 0.85, "nu": 5.0,
          "mu": 0.0, "last_variance": 0.0004 / 24}
    paths = simulate_paths(S0=100000.0, garch_params=gp, jump_params=None,
                           hours_to_expiry=24.0, n_sims=1000, seed=1)
    # The crash was `paths[:, -1]`; simulate_paths returns 1-D terminal prices.
    assert paths.ndim == 1
    # np.log(paths / S0) must work (this is the corrected line).
    assert np.all(np.isfinite(np.log(paths / 100000.0)))


@needs_data
def test_fix6_basel_mc_runs_to_completion():
    from core.validation.basel_backtest import run_basel_backtest

    # GARCH variant keeps the unit test fast; the shape fix (paths[:, -1] → paths)
    # is variant-independent, and the deployed FIGARCH variant is exercised
    # separately (it fits FIGARCH per refit and is too slow for a unit test).
    ret = _load_hourly_returns(3000)
    res = run_basel_backtest(
        ret, horizons=[1, 24], alphas=[0.05, 0.01], mode="mc",
        refit_every=1200, num_sims=600,
        use_figarch=False, use_svcj=True, use_skewed_t=True,
    )
    # A traffic-light summary string must be produced without raising (the bug
    # raised IndexError inside compute_mc_var before any summary).
    assert isinstance(res.summary, str) and "Green:" in res.summary


# ===========================================================================
# FIX 7 (M2) — outcome-based recalibration (default OFF)
# ===========================================================================

def test_fix7_logit_shift_corrects_inflated_probs():
    from core.pricing.fit_probability_curves import calibrate_logit_shift

    rng = np.random.default_rng(0)
    n = 4000
    true_p = rng.uniform(0.1, 0.9, n)
    outcomes = (rng.uniform(size=n) < true_p).astype(float)
    # Deliberately INFLATE the model prob by +0.1 → calibration should pull it back
    # (negative B).
    inflated = np.clip(true_p + 0.1, 1e-4, 1 - 1e-4)
    shift = calibrate_logit_shift(inflated, outcomes)
    assert shift is not None
    assert shift["B_fitted"] < 0.0


def test_fix7_apply_shift_identity_when_zero():
    from core.pricing.fit_probability_curves import apply_calibration_shift

    p = np.array([0.2, 0.5, 0.8])
    assert np.array_equal(apply_calibration_shift(p, 0.0), p)
    shifted = apply_calibration_shift(p, -1.0)
    assert np.all(shifted < p)  # negative B lowers probabilities


def test_fix7_fit_calibration_walk_forward(tmp_path):
    from core.pricing.fit_probability_curves import fit_calibration

    rng = np.random.default_rng(1)
    n = 1200
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    times = [t0 + timedelta(hours=int(i)) for i in range(n)]
    p = rng.uniform(0.1, 0.9, n)
    outcomes = (rng.uniform(size=n) < p).astype(float)
    dte = rng.choice([1.0, 5.0, 20.0], size=n)  # 0-2, 2-7, 7-30 buckets
    df = pd.DataFrame({
        "snapshot_time": times,
        "model_prob_used": p,
        "outcome_yes": outcomes,
        "dte_days": dte,
    })
    out_csv = tmp_path / "calibration_shift.csv"
    table = fit_calibration(df, output_path=str(out_csv), train_frac=0.7, min_obs=50)
    assert out_csv.exists()
    # Buckets present and each carries an n_obs from the TRAIN span only.
    assert set(table.keys()) >= {"0-2", "2-7", "7-30"}
    total_train = sum(v["n_obs"] for v in table.values())
    assert total_train <= int(n * 0.7) + 5  # train span only (leak guard)


def test_fix7_calibration_default_off_and_toggle(monkeypatch):
    import core.strategy.common as common

    df = pd.DataFrame({
        "p_model_cal": [0.10, 0.20, 0.30],
        "p_model_fit": [0.40, 0.50, 0.60],
    })
    # Default OFF: must use p_model_fit, NOT p_model_cal (no silent activation).
    assert common.USE_CALIBRATED_PROB is False
    res_off = common.resolve_model_prob(df)
    assert list(res_off) == [0.40, 0.50, 0.60]

    # Flip ON: now prefers the calibrated column.
    monkeypatch.setattr(common, "USE_CALIBRATED_PROB", True)
    res_on = common.resolve_model_prob(df)
    assert list(res_on) == [0.10, 0.20, 0.30]


def test_fix7_dte_bucket_edges():
    from core.pricing.fit_probability_curves import dte_bucket

    assert dte_bucket(0.5) == "0-2"
    assert dte_bucket(2.0) == "0-2"
    assert dte_bucket(2.1) == "2-7"
    assert dte_bucket(7.0) == "2-7"
    assert dte_bucket(30.0) == "7-30"
    assert dte_bucket(45.0) == "30+"
    assert dte_bucket(float("nan")) == "0-2"


# ===========================================================================
# FIX 9 (M3) — p_rn_fit renamed to p_market_fit (alias kept)
# ===========================================================================

def test_fix9_process_batch_emits_market_and_alias(tmp_path):
    from core.pricing.fit_probability_curves import process_batch

    # Minimal monotone batch: one expiry, 6 strikes (>=4 needed for the logistic fit).
    strikes = [90000, 95000, 100000, 105000, 110000, 115000]
    p_mc = [0.92, 0.78, 0.55, 0.34, 0.18, 0.08]
    mkt = [0.90, 0.76, 0.54, 0.33, 0.17, 0.07]
    df = pd.DataFrame({
        "strike": strikes,
        "T_days": [5.0] * 6,
        "market_price": mkt,
        "p_real_mc": p_mc,
        "expiry_date": ["2026-06-01"] * 6,
    })
    inp = tmp_path / "batch.csv"
    df.to_csv(inp, index=False)
    out_batch = tmp_path / "batch_with_fits.csv"
    out_curve = tmp_path / "curve_params.csv"
    process_batch(str(inp), str(out_batch), str(out_curve))

    res = pd.read_csv(out_batch)
    assert "p_market_fit" in res.columns
    assert "p_rn_fit" in res.columns  # deprecated alias retained
    # Alias must be identical to the primary column.
    pd.testing.assert_series_equal(
        res["p_market_fit"], res["p_rn_fit"], check_names=False
    )


# ===========================================================================
# FIX 11 (L6) — FIGARCH positivity guard falls back to GARCH
# ===========================================================================

@needs_data
def test_fix11_negative_weights_fall_back_to_garch(monkeypatch):
    import core.pricing.btc_pricing_engine as eng

    real = eng._compute_figarch_weights

    def _bad_weights(*a, **k):
        w = real(*a, **k)
        w = w.copy()
        w[5] = -1.0  # inject a B-M positivity violation
        return w

    monkeypatch.setattr(eng, "_compute_figarch_weights", _bad_weights)
    params = eng.fit_garch_model(pd.Series(_load_hourly_returns(8000)), use_figarch=True)
    # Guard must reject FIGARCH and fall back to GARCH (alpha present, no weights).
    assert "figarch_weights" not in params
    assert "alpha" in params
