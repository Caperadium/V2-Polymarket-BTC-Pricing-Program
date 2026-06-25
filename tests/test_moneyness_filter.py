"""Tests for signed/abs moneyness filter and related helpers."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.strategy.common import filter_by_moneyness, latest_spot_as_of


# ---------------------------------------------------------------------------
# filter_by_moneyness
# ---------------------------------------------------------------------------

def _df(moneyness_values):
    return pd.DataFrame({"moneyness": moneyness_values, "val": range(len(moneyness_values))})


def test_abs_mode_symmetric_drop():
    df = _df([-0.10, -0.03, 0.0, 0.03, 0.10])
    result = filter_by_moneyness(df, lower=0.02, upper=0.08, mode="abs")
    assert set(result["moneyness"].tolist()) == {-0.03, 0.03, -0.10, 0.10} - {-0.10, 0.10}
    # |m| between 0.02 and 0.08 → -0.03 and 0.03 survive
    assert sorted(result["moneyness"].tolist()) == [-0.03, 0.03]


def test_signed_mode_otm_only():
    df = _df([-0.10, -0.03, 0.0, 0.03, 0.10])
    result = filter_by_moneyness(df, lower=0.0, upper=None, mode="signed")
    assert sorted(result["moneyness"].tolist()) == [0.0, 0.03, 0.10]


def test_signed_mode_negative_lower():
    df = _df([-0.10, -0.03, 0.0, 0.03, 0.10])
    result = filter_by_moneyness(df, lower=-0.05, upper=0.05, mode="signed")
    assert sorted(result["moneyness"].tolist()) == [-0.03, 0.0, 0.03]


def test_signed_mode_itm_only():
    df = _df([-0.10, -0.03, 0.0, 0.03, 0.10])
    result = filter_by_moneyness(df, lower=None, upper=0.0, mode="signed")
    assert sorted(result["moneyness"].tolist()) == [-0.10, -0.03, 0.0]


def test_none_bounds_no_filter():
    df = _df([-0.10, 0.0, 0.10])
    result = filter_by_moneyness(df, lower=None, upper=None)
    assert len(result) == 3


def test_missing_column_passthrough():
    df = pd.DataFrame({"val": [1, 2, 3]})
    result = filter_by_moneyness(df, lower=0.0, upper=0.1)
    assert len(result) == 3


def test_nan_moneyness_dropped_when_bound_active():
    df = _df([0.03, float("nan"), 0.10])
    result = filter_by_moneyness(df, lower=0.0, upper=0.20, mode="signed")
    assert float("nan") not in result["moneyness"].tolist()
    assert len(result) == 2


def test_nan_moneyness_kept_when_no_bounds():
    df = _df([0.03, float("nan"), 0.10])
    result = filter_by_moneyness(df, lower=None, upper=None)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# latest_spot_as_of
# ---------------------------------------------------------------------------

def _btc_df(timestamps, closes):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, utc=True),
        "close": closes,
    })


def test_strict_lt_cutoff_no_leak():
    btc = _btc_df(
        ["2024-01-01 12:00:00", "2024-01-01 12:01:00", "2024-01-01 12:02:00"],
        [100.0, 101.0, 102.0],
    )
    as_of = datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc)
    spot = latest_spot_as_of(btc, as_of)
    # Bar stamped at 12:01 must NOT be included (strict <), last bar before is 12:00
    assert spot == 100.0


def test_returns_none_for_empty_df():
    assert latest_spot_as_of(pd.DataFrame(), datetime.now(timezone.utc)) is None


def test_returns_none_for_none_df():
    assert latest_spot_as_of(None, datetime.now(timezone.utc)) is None


def test_returns_none_when_no_prior_bars():
    btc = _btc_df(["2024-01-02 00:00:00"], [50000.0])
    as_of = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert latest_spot_as_of(btc, as_of) is None


# ---------------------------------------------------------------------------
# build_targets integration — dead-filter regression guard
# ---------------------------------------------------------------------------

def test_build_targets_signed_filter_drops_itm():
    """moneyness filter must actually run; ITM rows must be excluded."""
    from core.strategy.auto_reco import build_targets
    from core.strategy.common import RebalanceConfig
    from core.strategy.vol_gate import VolGateResult

    batch = pd.DataFrame({
        "slug": ["s1", "s2", "s3"],
        "strike": [90000.0, 100000.0, 110000.0],
        "market_price": [0.80, 0.50, 0.20],
        "p_model_fit": [0.70, 0.60, 0.30],
        "moneyness": [-0.10, 0.0, 0.10],  # s1 ITM, s2 ATM, s3 OTM
        "expiry_date": ["2024-12-31", "2024-12-31", "2024-12-31"],
    })

    vol_gate = VolGateResult(
        now_utc="2024-01-01T00:00:00+00:00",
        regime="normal",
        vol15=None,
        vol60=None,
        vol15_pct=None,
        shock=False,
        allow_new_entries=True,
        edge_add_cents=0.0,
        kelly_mult=1.0,
        reason="test",
    )
    config = RebalanceConfig(
        bankroll=1000.0,
        min_edge_entry=0.02,
        min_moneyness=0.0,   # OTM only
        max_moneyness=None,
        moneyness_mode="signed",
        require_active=False,
    )

    targets = build_targets(batch, {}, vol_gate, config)
    slugs_in_targets = {t.slug for t in targets.values()}
    assert "s1" not in slugs_in_targets, "ITM contract (m=-0.10) must be filtered out"
    assert "s3" in slugs_in_targets, "OTM contract (m=0.10) must survive"
