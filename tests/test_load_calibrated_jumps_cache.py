"""
test_load_calibrated_jumps_cache.py

Tests for Package C / work package W2: cache schema versioning + windowed-
calibration diagnostic columns in
core/pricing/btc_pricing_engine.py::load_calibrated_jumps
(temp/mm_package_c_plan.md section 2.3 "load_calibrated_jumps" bullet +
section 3 W2 bullet list), PLUS the Package C code-review follow-up fixes
(F1-F6):

- F1: any failure reading/parsing the cache (zero-byte file, header-only
  CSV, malformed CSV) is caught and treated as stale -- self-healing
  recalibration + rewrite instead of a crash loop.
- F2: NaN-safe parsing (via module-private _cache_opt_int/_cache_opt_float)
  for the optional diagnostic columns (schema_version,
  calibration_window_hours, window_weight, n_window_jumps) -- a NaN/blank
  cell never raises.
- F3: the cache-miss write path is atomic (write to a .tmp sibling, then
  os.replace).
- F4: schema_version must match JUMP_CAL_SCHEMA_VERSION EXACTLY (not just
  >=) -- a future version is also stale (rollback safety).
- F5: calibration_window_hours must also match the current
  JUMP_CAL_WINDOW_HOURS constant exactly; missing/NaN/mismatched -> stale.
- F6: the cache-hit dict is symmetric with the cache-miss dict -- both
  carry schema_version.

All cache files and hourly CSV fixtures live under tmp_path; no test reads
or writes the repo's DATA/ directory.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import core.pricing.btc_pricing_engine as eng
import core.pricing.jump_calibration as jc
from core.pricing.jump_calibration import JumpCalibrationResult

# A fixed stub result returned by the monkeypatched calibrate_jumps -- never
# runs the real fit for the cache-behavior tests (only the end-to-end test
# at the bottom exercises the real calibration).
STUB_RESULT = JumpCalibrationResult(
    lam=17.5,
    p_crash=0.51,
    eta_up=43.0,
    eta_down=38.0,
    mu_v=3.2e-6,
    rho_J=0.11,
    lam_v=17.5,
    rho_j_slope=12.3,
    n_jumps_detected=124,
    n_obs=43792,
    jump_threshold=0.0,
    detection_method="bipower",
    fit_converged=True,
    calibration_window_hours=8760,
    window_weight=1.0,
    n_window_jumps=16,
)


def _stub_calibrate_jumps(monkeypatch, calls):
    """Monkeypatch core.pricing.jump_calibration.calibrate_jumps with a stub
    that records each call and always returns STUB_RESULT -- never runs the
    real fit. load_calibrated_jumps does `from core.pricing.jump_calibration
    import calibrate_jumps` INSIDE the function, so patching the attribute on
    the jump_calibration module (not a re-import) is what the call-time
    lookup resolves to."""
    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return STUB_RESULT
    monkeypatch.setattr(jc, "calibrate_jumps", _stub)


def _old_style_cache_row() -> dict:
    """Pre-Package-C cache row: no schema_version, no window columns (this is
    exactly what DATA/jump_calibration.csv looks like before this fix)."""
    return {
        "lam": 22.0,
        "p_crash": 0.58,
        "eta_up": 33.0,
        "eta_down": 29.0,
        "mu_v": 2.1e-6,
        "rho_J": -0.05,
        "rho_j_slope": 5.0,
        "lam_v": 22.0,
        "n_jumps_detected": 110,
        "fit_converged": 1,
        "calibration_date": "2026-06-01T00:00:00+00:00",
    }


def _current_cache_row(**overrides) -> dict:
    row = {
        "lam": 30.0,
        "p_crash": 0.60,
        "eta_up": 40.0,
        "eta_down": 35.0,
        "mu_v": 4.0e-6,
        "rho_J": 0.02,
        "rho_j_slope": -3.0,
        "lam_v": 30.0,
        "n_jumps_detected": 130,
        "fit_converged": 1,
        "calibration_date": "2026-07-15T00:00:00+00:00",
        "schema_version": eng.JUMP_CAL_SCHEMA_VERSION,
        "calibration_window_hours": 8760,
        "window_weight": 1.0,
        "n_window_jumps": 20,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# 1. Pre-fix cache fixture (no schema_version column) -> treated stale,
#    recalibrated, file rewritten with schema_version + window columns.
# ---------------------------------------------------------------------------

def test_prefix_cache_missing_schema_version_is_stale(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    pd.DataFrame([_old_style_cache_row()]).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 1  # cache treated stale -> recalibrated
    assert result["lam"] == STUB_RESULT.lam
    assert result["eta_up"] == STUB_RESULT.eta_up
    assert result["schema_version"] == eng.JUMP_CAL_SCHEMA_VERSION
    assert result["calibration_window_hours"] == STUB_RESULT.calibration_window_hours
    assert result["window_weight"] == STUB_RESULT.window_weight
    assert result["n_window_jumps"] == STUB_RESULT.n_window_jumps

    # File rewritten with the new columns present.
    rewritten = pd.read_csv(cache_path)
    assert "schema_version" in rewritten.columns
    assert "calibration_window_hours" in rewritten.columns
    assert "window_weight" in rewritten.columns
    assert "n_window_jumps" in rewritten.columns
    assert int(rewritten.iloc[-1]["schema_version"]) == eng.JUMP_CAL_SCHEMA_VERSION

    # F3: atomic write via .tmp + os.replace -- no leftover tmp sibling.
    assert not cache_path.with_name(cache_path.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# 2. Current-version cache honored within max age -> no recalibration,
#    values returned from cache. Also covers (f): the warm-cache dict
#    carries schema_version (F6 -- symmetric with the cache-miss dict).
# ---------------------------------------------------------------------------

def test_current_version_cache_honored_no_recalibration(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    row = _current_cache_row()
    pd.DataFrame([row]).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 0  # calibrate_jumps must NOT be called
    assert result["lam"] == row["lam"]
    assert result["eta_up"] == row["eta_up"]
    assert result["p_crash"] == row["p_crash"]
    assert result["calibration_window_hours"] == row["calibration_window_hours"]
    assert result["window_weight"] == row["window_weight"]
    assert result["n_window_jumps"] == row["n_window_jumps"]
    # (f) F6: schema_version present on the cache-HIT dict too.
    assert result["schema_version"] == eng.JUMP_CAL_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 3. Older-version rejected -> recalibrates. F4 makes this an EXACT match
#    requirement (not >=) -- see test_future_schema_version_is_stale below
#    for the future-version half.
# ---------------------------------------------------------------------------

def test_older_schema_version_rejected(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    row = _current_cache_row(schema_version=1)
    pd.DataFrame([row]).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 1
    assert result["lam"] == STUB_RESULT.lam
    assert result["schema_version"] == eng.JUMP_CAL_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# (d) Future-version rejected too -> recalibrates. F4: schema_version must
#     match JUMP_CAL_SCHEMA_VERSION EXACTLY, not >= -- an older code
#     checkout must not blindly trust a cache written by a newer schema it
#     doesn't know how to interpret (rollback safety).
# ---------------------------------------------------------------------------

def test_future_schema_version_is_stale(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    row = _current_cache_row(schema_version=eng.JUMP_CAL_SCHEMA_VERSION + 1)
    pd.DataFrame([row]).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 1
    assert result["schema_version"] == eng.JUMP_CAL_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 4. A version-2 row missing calibration_window_hours entirely is now STALE
#    (F5), not a cache hit with .get() fallbacks -- this is a behavior
#    CHANGE from the pre-review version of this test (which asserted
#    len(calls) == 0 / fallback values on a hit). F5 requires
#    calibration_window_hours to match the current JUMP_CAL_WINDOW_HOURS
#    constant exactly; missing counts as a mismatch, so a schema-version
#    bump is no longer required to invalidate a cache written before the
#    window columns existed at all.
# ---------------------------------------------------------------------------

def test_v2_row_missing_window_hours_column_is_stale(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    row = _old_style_cache_row()
    row["schema_version"] = eng.JUMP_CAL_SCHEMA_VERSION
    # Deliberately no calibration_window_hours / window_weight / n_window_jumps.
    pd.DataFrame([row]).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 1  # schema matches but window_hours missing -> stale (F5)
    assert result["schema_version"] == eng.JUMP_CAL_SCHEMA_VERSION
    assert result["calibration_window_hours"] == STUB_RESULT.calibration_window_hours


# ---------------------------------------------------------------------------
# (e) A version-2 row with a calibration_window_hours VALUE present but not
#     matching the current JUMP_CAL_WINDOW_HOURS constant is also stale --
#     e.g. a cache written under a since-retuned window (4380 = 6mo vs the
#     current 8760 = 12mo).
# ---------------------------------------------------------------------------

def test_calibration_window_hours_mismatch_is_stale(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    row = _current_cache_row(calibration_window_hours=4380)
    pd.DataFrame([row]).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 1
    assert result["calibration_window_hours"] == STUB_RESULT.calibration_window_hours


# ---------------------------------------------------------------------------
# (a) NaN/blank cells on the window_weight / n_window_jumps DIAGNOSTIC-ONLY
#     columns of an otherwise-valid v2 row (current schema_version AND
#     matching calibration_window_hours) must not raise. F2's
#     _cache_opt_float/_cache_opt_int substitute the documented defaults
#     (1.0 / 0) for those two columns specifically -- staleness (F4/F5) is
#     gated ONLY on schema_version and calibration_window_hours, both valid
#     here, so the cache is still HONORED (not recalibrated): a corrupt/
#     blank cell on an unrelated diagnostic-only column must degrade
#     gracefully to a default rather than force a perfectly good cached fit
#     to be thrown away.
# ---------------------------------------------------------------------------

def test_nan_window_diagnostics_fallback_no_exception_cache_still_hit(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    row = _current_cache_row(window_weight=float("nan"), n_window_jumps=None)
    pd.DataFrame([row]).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 0  # NaN diagnostics degrade to defaults, no exception, no staleness
    assert result["window_weight"] == 1.0
    assert result["n_window_jumps"] == 0
    assert result["schema_version"] == eng.JUMP_CAL_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# (b) F1: a zero-byte cache file (e.g. left by a writer that crashed
#     mid-write, the exact scenario F3's atomic write now prevents going
#     forward) raises pandas.errors.EmptyDataError from pd.read_csv --
#     caught and treated as stale, so the function self-heals: recalibrates
#     and rewrites a valid cache instead of crashing on every call for up
#     to max_cache_age_days.
# ---------------------------------------------------------------------------

def test_zero_byte_cache_file_self_heals(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    cache_path.write_bytes(b"")  # zero-byte file, fresh mtime (just created)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 1
    assert result["lam"] == STUB_RESULT.lam
    assert result["schema_version"] == eng.JUMP_CAL_SCHEMA_VERSION

    # File self-healed: now a valid, readable cache with the current schema.
    rewritten = pd.read_csv(cache_path)
    assert len(rewritten) == 1
    assert int(rewritten.iloc[-1]["schema_version"]) == eng.JUMP_CAL_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# (c) F1: a header-only CSV (0 data rows -- e.g. a writer that wrote only
#     the header before crashing) makes iloc[-1] raise IndexError -- caught
#     and treated as stale, same self-healing outcome as the zero-byte case.
# ---------------------------------------------------------------------------

def test_header_only_cache_file_self_heals(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    pd.DataFrame(columns=list(_current_cache_row().keys())).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 1
    assert result["lam"] == STUB_RESULT.lam

    rewritten = pd.read_csv(cache_path)
    assert len(rewritten) == 1


# ---------------------------------------------------------------------------
# Stale-by-age still works (age > max_cache_age_days -> recalibrate) --
# existing behavior preserved.
# ---------------------------------------------------------------------------

def test_stale_by_age_still_recalibrates(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    row = _current_cache_row()
    pd.DataFrame([row]).to_csv(cache_path, index=False)

    # Backdate the file's mtime by 40 days (> default max_cache_age_days=30).
    old_time = time.time() - 40 * 86400
    os.utime(cache_path, (old_time, old_time))

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path),
    )

    assert len(calls) == 1
    assert result["lam"] == STUB_RESULT.lam


# ---------------------------------------------------------------------------
# force_recalibrate=True still bypasses cache.
# ---------------------------------------------------------------------------

def test_force_recalibrate_bypasses_valid_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "jump_calibration.csv"
    row = _current_cache_row()
    pd.DataFrame([row]).to_csv(cache_path, index=False)

    calls = []
    _stub_calibrate_jumps(monkeypatch, calls)

    result = eng.load_calibrated_jumps(
        hourly_csv="unused.csv", cache_path=str(cache_path), force_recalibrate=True,
    )

    assert len(calls) == 1
    assert result["lam"] == STUB_RESULT.lam


# ---------------------------------------------------------------------------
# End-to-end integration (plan round-1 finding 9): real calibrate_jumps
#    runs against a small synthetic hourly fixture -> windowed params
#    written (schema_version present, calibration_window_hours == 8760,
#    window_weight in [0,1], n_window_jumps >= 0); a second call WITHOUT
#    force_recalibrate returns the cached values without recalibrating.
# ---------------------------------------------------------------------------

def _write_synthetic_hourly_csv(path):
    """~20000 seeded hourly log-returns with ~30 injected jump spikes,
    reconstructed into a close-price series -- same generative shape as the
    W1 golden fixture (tests/test_jump_calibration_window.py), so bipower
    detection reliably clears the n_jumps < 10 literature-default gate."""
    rng = np.random.default_rng(321)
    n = 20000
    sigma = 0.005
    ret = rng.normal(0.0, sigma, n)

    n_spikes = 30
    spike_idx = rng.choice(n, size=n_spikes, replace=False)
    spike_signs = rng.choice([-1.0, 1.0], size=n_spikes)
    spike_mag = rng.uniform(0.03, 0.08, size=n_spikes)
    ret[spike_idx] = spike_signs * spike_mag

    log_prices = np.concatenate([[np.log(50000.0)], np.log(50000.0) + np.cumsum(ret)])
    closes = np.exp(log_prices)
    timestamps = pd.date_range("2024-01-01", periods=n + 1, freq="h", tz="UTC")

    df = pd.DataFrame({"timestamp": timestamps, "close": closes})
    df.to_csv(path, index=False)


def test_end_to_end_real_calibration_writes_and_caches(tmp_path, monkeypatch):
    hourly_csv = tmp_path / "synthetic_hourly.csv"
    cache_path = tmp_path / "jump_calibration.csv"
    _write_synthetic_hourly_csv(hourly_csv)

    first = eng.load_calibrated_jumps(
        hourly_csv=str(hourly_csv), cache_path=str(cache_path), force_recalibrate=True,
    )

    assert first["schema_version"] == eng.JUMP_CAL_SCHEMA_VERSION
    assert first["calibration_window_hours"] == 8760
    assert 0.0 <= first["window_weight"] <= 1.0
    assert first["n_window_jumps"] >= 0
    assert first["fit_converged"] is True  # 30 injected spikes clear the n<10 gate

    written = pd.read_csv(cache_path)
    assert "schema_version" in written.columns
    assert int(written.iloc[-1]["schema_version"]) == eng.JUMP_CAL_SCHEMA_VERSION

    # Second call, no force_recalibrate -> served from cache, no recalibration.
    def _boom(*args, **kwargs):
        raise AssertionError("calibrate_jumps called on second load -- cache not honored")
    monkeypatch.setattr(jc, "calibrate_jumps", _boom)

    second = eng.load_calibrated_jumps(
        hourly_csv=str(hourly_csv), cache_path=str(cache_path),
    )

    assert second["lam"] == first["lam"]
    assert second["calibration_window_hours"] == first["calibration_window_hours"]
    assert second["window_weight"] == first["window_weight"]
    assert second["n_window_jumps"] == first["n_window_jumps"]
