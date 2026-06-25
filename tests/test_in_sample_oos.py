#!/usr/bin/env python3
"""
test_in_sample_oos.py — tests for the IS/OOS evaluation window module.

Covers: default-cutoff split, partitioning, settlement-time derivation (REVIEW B1),
settlement-based M2 training, cache write isolation from the global file (REVIEW B2),
fingerprint invalidation (REVIEW S5), OOS load-only / no-refit, the §7 hygiene guard,
small-sample suppression, trade-panel windowing, the inert-banner signal (REVIEW S1),
and the §9 leak verifier (pass + label-leak + BTC-truncation-violation).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.backtesting import in_sample_oos as iso
from core.backtesting.in_sample_oos import (
    WindowMode,
    WindowSpec,
    OOSHygieneError,
    OOSLeakError,
)

UTC = "UTC"
DAY0 = pd.Timestamp("2026-01-01", tz=UTC)


def _priced_rows(
    n_contracts,
    *,
    start_day=0,
    expiry_offset_days=2,
    dte_days=5.0,
    snapshots_per_contract=1,
    outcome_alternate=True,
    model_prob=0.5,
    slug_prefix="m",
):
    """Build all_priced_df-shaped rows: one contract per day from start_day."""
    rows = []
    for i in range(n_contracts):
        snap_day = DAY0 + pd.Timedelta(days=start_day + i)
        snap = snap_day.floor("D")
        expiry = (snap_day + pd.Timedelta(days=expiry_offset_days)).strftime("%Y-%m-%d")
        outcome = float(i % 2) if outcome_alternate else 1.0
        for s in range(snapshots_per_contract):
            rows.append({
                "snapshot_time": snap + pd.Timedelta(days=s),
                "expiry_date": expiry,
                "strike": 100000.0 + i,
                "slug": f"{slug_prefix}{i}",
                "market_yes_price": 0.5,
                "model_prob_raw": model_prob,
                "model_prob_used": model_prob,
                "moneyness": 0.0,
                "dte_days": dte_days,
                "outcome_yes": outcome,
            })
    return rows


def _make_priced(n_contracts, **kw):
    return pd.DataFrame(_priced_rows(n_contracts, **kw))


# ---------------------------------------------------------------------------
# Cutoff / partition
# ---------------------------------------------------------------------------

def test_default_cutoff_70_30():
    df = _make_priced(10)  # one contract per day, days 0..9
    cutoff = iso.compute_default_cutoff(df, target_is_frac=0.7)
    assert cutoff is not None
    is_eval, oos_eval, _ = iso.partition_contracts(df, cutoff)
    n_is = iso.contract_ids(is_eval).nunique()
    n_oos = iso.contract_ids(oos_eval).nunique()
    assert n_is + n_oos == 10
    # ~70/30 within ±1 contract
    assert abs(n_is - 7) <= 1
    assert cutoff == cutoff.floor("D")  # midnight aligned (REVIEW N2)


def test_partition_disjoint():
    df = _make_priced(10)
    cutoff = iso.compute_default_cutoff(df, target_is_frac=0.7)
    is_eval, oos_eval, straddlers = iso.partition_contracts(df, cutoff)
    is_idx, oos_idx = set(is_eval.index), set(oos_eval.index)
    assert is_idx.isdisjoint(oos_idx)
    assert is_idx | oos_idx == set(df.index)        # straddlers ⊂ is_eval
    assert set(straddlers.index).issubset(is_idx)


def test_settlement_time_derivation():
    # 12:00 ET ≈ 17:00 UTC (standard time in January)
    st = iso.derive_settlement_time("2026-01-15")
    assert st is not None
    assert st.tzname() == "UTC"
    assert st.hour in (16, 17)  # 17 EST / 16 EDT — January is EST → 17
    # NaT expiry → None, no crash (REVIEW B1)
    assert iso.derive_settlement_time(pd.NaT) is None
    df = pd.DataFrame({"expiry_date": ["2026-01-15", None]})
    out = iso.add_settlement_time(df)
    assert out["settlement_time"].notna().sum() == 1


# ---------------------------------------------------------------------------
# M2 training population (settlement axis)
# ---------------------------------------------------------------------------

def test_m2_training_uses_settlement():
    # Contract priced day 5, expires day 7 → settles day 7 noon ET. A cutoff of
    # day 6 makes it a STRADDLER (priced IS, settles OOS) → excluded from training
    # though it is IS by snapshot.
    df = _make_priced(1, start_day=5, expiry_offset_days=2)
    cutoff = (DAY0 + pd.Timedelta(days=6)).floor("D")
    train = iso.m2_training_set(df, cutoff)
    assert train.empty  # settles after cutoff → not eligible
    # Later cutoff (day 20) → settlement is before it → eligible.
    cutoff2 = (DAY0 + pd.Timedelta(days=20)).floor("D")
    train2 = iso.m2_training_set(df, cutoff2)
    assert len(train2) == 1


# ---------------------------------------------------------------------------
# train_pipeline / cache
# ---------------------------------------------------------------------------

def _trainable_df():
    """Enough same-bucket resolved contracts (both classes) to fit a bucket."""
    # 40 contracts days 0..39, expire +2 → settle before a day-60 cutoff.
    return _make_priced(40, start_day=0, expiry_offset_days=2, dte_days=5.0)


def test_train_pipeline_caches(tmp_path):
    df = _trainable_df()
    cutoff = (DAY0 + pd.Timedelta(days=60)).floor("D")
    iso.train_pipeline(cutoff, df, cache_root=tmp_path, min_obs=10)
    cache_dir = tmp_path / f"cutoff_{cutoff:%Y-%m-%d}"
    assert (cache_dir / "manifest.json").exists()
    assert (cache_dir / "calibration_shift.csv").exists()
    # Re-load via load_artifacts
    loaded = iso.load_artifacts(cutoff, cache_root=tmp_path)
    assert loaded is not None
    assert loaded["manifest"]["components"]["m2"]["min_obs"] == 10


def test_train_pipeline_does_not_touch_global(tmp_path):
    global_path = Path(iso._PROJECT_ROOT) / "DATA" / "calibration_shift.csv"
    before = global_path.read_bytes() if global_path.exists() else None
    df = _trainable_df()
    cutoff = (DAY0 + pd.Timedelta(days=60)).floor("D")
    iso.train_pipeline(cutoff, df, cache_root=tmp_path, min_obs=10)
    after = global_path.read_bytes() if global_path.exists() else None
    assert before == after  # global file untouched (REVIEW B2)


def test_cache_fingerprint_invalidation(tmp_path):
    df = _trainable_df()
    cutoff = (DAY0 + pd.Timedelta(days=60)).floor("D")
    iso.load_or_train(cutoff, df, mode=WindowMode.IS, cache_root=tmp_path, min_obs=10)
    # Same data, OOS mode → cache hit, no raise.
    iso.load_or_train(cutoff, df, mode=WindowMode.OOS, cache_root=tmp_path, min_obs=10)
    # Mutate training population (more contracts) → fingerprint changes → OOS raises.
    df2 = pd.concat([df, _make_priced(5, start_day=40, slug_prefix="x")], ignore_index=True)
    with pytest.raises(OOSLeakError):
        iso.load_or_train(cutoff, df2, mode=WindowMode.OOS, cache_root=tmp_path, min_obs=10)


def test_oos_load_only_no_refit(tmp_path, monkeypatch):
    calls = {"n": 0}
    real_fit = iso.fit_calibration

    def _spy(*a, **k):
        calls["n"] += 1
        return real_fit(*a, **k)

    monkeypatch.setattr(iso, "fit_calibration", _spy)

    df = _trainable_df()
    cutoff = (DAY0 + pd.Timedelta(days=60)).floor("D")
    iso.load_or_train(cutoff, df, mode=WindowMode.IS, cache_root=tmp_path, min_obs=10)
    assert calls["n"] == 1  # trained once
    iso.load_or_train(cutoff, df, mode=WindowMode.OOS, cache_root=tmp_path, min_obs=10)
    assert calls["n"] == 1  # OOS cache hit → no refit
    # OOS miss (empty cache root) → raise, still no fit.
    with pytest.raises(OOSLeakError):
        iso.load_or_train(cutoff, df, mode=WindowMode.OOS, cache_root=tmp_path / "empty", min_obs=10)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Hygiene / small-sample / windowing
# ---------------------------------------------------------------------------

def test_oos_hygiene_guard():
    df = _make_priced(3)
    spec_oos = WindowSpec(cutoff=DAY0, mode=WindowMode.OOS)
    spec_is = WindowSpec(cutoff=DAY0, mode=WindowMode.IS)
    spec_all = WindowSpec(cutoff=DAY0, mode=WindowMode.ALL)
    with pytest.raises(OOSHygieneError):
        iso.guarded_filter(df, spec_oos, conditions_on_outcome=True, desc="winners only")
    # OOS but not outcome-conditioning → allowed.
    assert iso.guarded_filter(df, spec_oos, conditions_on_outcome=False) is df
    # IS / ALL short-circuit even when outcome-conditioning (REVIEW N3).
    assert iso.guarded_filter(df, spec_is, conditions_on_outcome=True) is df
    assert iso.guarded_filter(df, spec_all, conditions_on_outcome=True) is df


def test_small_sample_suppression():
    assert iso.small_sample_state(199)["suppress"] is True
    assert iso.small_sample_state(199)["banner"]
    s = iso.small_sample_state(200)
    assert s["suppress"] is False and s["banner"] is None


def test_window_filters_trades():
    cutoff = (DAY0 + pd.Timedelta(days=5)).floor("D")
    trades = pd.DataFrame({
        "pricing_date": [DAY0 + pd.Timedelta(days=d) for d in [1, 3, 6, 8]] + [pd.NaT],
        "pnl": [1, 2, 3, 4, 5],
    })
    equity = pd.DataFrame({
        "pricing_date": [DAY0 + pd.Timedelta(days=d) for d in [1, 6]],
        "bankroll": [1000, 1100],
    })
    t_is, e_is = iso.apply_window_trades(trades, equity, WindowSpec(cutoff, WindowMode.IS))
    t_oos, e_oos = iso.apply_window_trades(trades, equity, WindowSpec(cutoff, WindowMode.OOS))
    assert len(t_is) == 2 and len(t_oos) == 2          # NaT excluded from both splits
    assert len(e_is) == 1 and len(e_oos) == 1
    t_all, _ = iso.apply_window_trades(trades, equity, WindowSpec(cutoff, WindowMode.ALL))
    assert len(t_all) == 5                             # NaT survives under ALL


def test_m2_all_buckets_inert_banner(tmp_path):
    # min_obs huge → no bucket reaches threshold → all inert.
    df = _trainable_df()
    cutoff = (DAY0 + pd.Timedelta(days=60)).floor("D")
    art = iso.train_pipeline(cutoff, df, cache_root=tmp_path, min_obs=10_000)
    assert iso.is_m2_inert(art["shift_table"]) is True
    assert art["manifest"]["components"]["m2"]["inert"] is True


# ---------------------------------------------------------------------------
# §9 verification
# ---------------------------------------------------------------------------

def _leakfree_setup(tmp_path):
    """IS contracts settle before cutoff; OOS priced after. Returns (df, cutoff, art)."""
    is_df = _make_priced(30, start_day=0, expiry_offset_days=2, dte_days=5.0)   # days 0..29
    oos_df = _make_priced(10, start_day=40, expiry_offset_days=2, slug_prefix="o")  # days 40..49
    df = pd.concat([is_df, oos_df], ignore_index=True)
    cutoff = (DAY0 + pd.Timedelta(days=35)).floor("D")
    art = iso.train_pipeline(cutoff, df, cache_root=tmp_path, min_obs=10)
    return df, cutoff, art


def test_verify_oos_leak_free_pass(tmp_path):
    df, cutoff, art = _leakfree_setup(tmp_path)
    # Should pass on >=3 random OOS contracts.
    iso.verify_oos_leak_free(df, cutoff, art, n_samples=3, seed=1)


def test_verify_oos_leak_free_detects_label_leak(tmp_path):
    df, cutoff, art = _leakfree_setup(tmp_path)
    # Poison the manifest: a training label that resolves AFTER an OOS pricing time.
    leaked = (DAY0 + pd.Timedelta(days=45)).isoformat()
    art["manifest"]["is_label_max_ts"] = leaked
    with pytest.raises(OOSLeakError):
        iso.verify_oos_leak_free(df, cutoff, art, n_samples=5, seed=2)


def test_verify_detects_btc_truncation_violation(tmp_path):
    df, cutoff, art = _leakfree_setup(tmp_path)
    # Build a fake DATA dir whose hourly bar lands AT/AFTER an OOS snapshot time.
    data_dir = tmp_path / "DATA"
    data_dir.mkdir()
    # Bars up to and INCLUDING day 60 (>= every OOS snapshot at days 40..49).
    ts = pd.date_range(DAY0, periods=61, freq="D", tz=UTC)
    pd.DataFrame({"timestamp": ts, "close": np.arange(61)}).to_csv(
        data_dir / "btc_hourly.csv", index=False
    )
    # Inject the regression this guards: a `<=` truncation leaks the midnight bar.
    with pytest.raises(OOSLeakError):
        iso.verify_oos_leak_free(
            df, cutoff, art, n_samples=5, seed=3,
            include_btc_arm=True, data_dir=data_dir,
            btc_truncate=lambda idx, ts: idx[idx <= ts],
        )
    # Strict `<` (production rule) must NOT raise on the same data.
    iso.verify_oos_leak_free(
        df, cutoff, art, n_samples=5, seed=3,
        include_btc_arm=True, data_dir=data_dir,
    )
