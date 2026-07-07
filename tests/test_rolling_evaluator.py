"""
test_rolling_evaluator.py

Tests for the Wave 3 pricing-fix plan (temp/pricing_fix_implementation_plan.md),
task T7: core/validation/rolling_evaluator.py. Uses synthetic GBM hourly data
(no DATA/ dependency) with a small n_sims / max_windows so the test runs fast.
All tests pass out_dir=tmp_path so pytest never writes into the real
DATA/rolling_eval/ directory.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from core.validation.rolling_evaluator import RollingEvaluator, WindowResult


def _make_synthetic_hourly_gbm(n: int, sigma: float, seed: int) -> pd.DataFrame:
    """n hourly bars of GBM (no drift, no jumps) with a known hourly sigma."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, sigma, size=n)
    log_price = np.cumsum(log_rets) + np.log(50000.0)
    close = np.exp(log_price)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "close": close})


def test_rolling_evaluator_gbm_no_jumps_runs_and_calibrates(tmp_path):
    """On synthetic GBM (no jumps) with a small MC budget, the evaluator
    should run without exception and produce a well-formed, roughly
    calibrated (model close to naive) output frame."""
    df = _make_synthetic_hourly_gbm(n=3000, sigma=0.004, seed=7)

    evaluator = RollingEvaluator(
        hourly_df=df,
        window_days=30,
        step_days=30,
        horizons=(1,),
        n_sims=1000,
        max_windows=3,
        seed=123,
        out_dir=str(tmp_path),
    )

    result_df = evaluator.run()

    # Expected columns match the WindowResult dataclass fields.
    expected_cols = set(WindowResult.__dataclass_fields__.keys())
    assert not result_df.empty, "expected at least one (window, horizon) forecast"
    assert set(result_df.columns) == expected_cols

    # A non-empty run must have written exactly one CSV into out_dir.
    written = [f for f in os.listdir(tmp_path) if f.startswith("rolling_eval_")]
    assert len(written) == 1

    assert (result_df["brier_model"] >= 0).all() and (result_df["brier_model"] <= 1).all()
    assert (result_df["brier_naive"] >= 0).all() and (result_df["brier_naive"] <= 1).all()

    # Both should be near-calibrated on pure GBM (no jumps, no regime signal).
    brier_model_mean = result_df["brier_model"].mean()
    brier_naive_mean = result_df["brier_naive"].mean()
    assert abs(brier_model_mean - brier_naive_mean) < 0.15, (
        f"model Brier {brier_model_mean:.4f} vs naive Brier {brier_naive_mean:.4f} "
        "diverge more than expected on jump-free GBM data"
    )

    # var_hit columns are 0/1 indicators (or absent if skipped upstream).
    for col in ("var_hit_5", "var_hit_1"):
        vals = result_df[col].dropna().unique()
        assert set(vals.tolist()).issubset({0, 1})

    # summary() must not raise and should report the sign test + VaR backtest.
    summ = evaluator.summary(result_df)
    assert summ["n_rows"] == len(result_df)
    assert "sign_test_p_value" in summ
    assert "var_backtest" in summ


def test_rolling_evaluator_max_windows_keeps_most_recent_anchors(tmp_path):
    """max_windows must keep the MOST RECENT N anchors, not the first N.
    3000 hourly bars = 125 days; window=30d, step=30d gives candidate anchors
    at day 30, 60, 90, 120. max_windows=3 must therefore keep 60/90/120 and
    drop the EARLIEST anchor (day 30)."""
    df = _make_synthetic_hourly_gbm(n=3000, sigma=0.004, seed=7)
    first_ts = df["timestamp"].iloc[0]

    evaluator = RollingEvaluator(
        hourly_df=df,
        window_days=30,
        step_days=30,
        horizons=(1,),
        n_sims=200,
        max_windows=3,
        seed=123,
        out_dir=str(tmp_path),
    )
    result_df = evaluator.run()

    assert not result_df.empty
    anchors = pd.to_datetime(result_df["window_end"]).unique()
    earliest_candidate = first_ts + pd.Timedelta(days=30)
    assert earliest_candidate not in anchors, (
        "earliest anchor should have been dropped by max_windows tail-taking"
    )
    # The latest candidate anchor (day 120) must be present.
    assert (first_ts + pd.Timedelta(days=120)) in anchors
    assert len(anchors) == 3


def test_rolling_evaluator_empty_data_returns_empty_frame(tmp_path):
    """Degenerate input (too little data for even one window) must not raise,
    and -- critically -- must NOT write an empty CSV to out_dir."""
    df = _make_synthetic_hourly_gbm(n=50, sigma=0.004, seed=1)
    evaluator = RollingEvaluator(
        hourly_df=df, window_days=90, step_days=7, horizons=(1,), n_sims=100,
        out_dir=str(tmp_path),
    )
    result_df = evaluator.run()
    assert result_df.empty
    assert os.listdir(tmp_path) == [], "zero-forecast run must not write any CSV"
    summ = evaluator.summary(result_df)
    assert summ["n_rows"] == 0
