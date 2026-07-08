"""Pricer adapter tests (plan Section 2.1 / 4.1, task P1).

All tests use a STUBBED engine_fn -- never the real
`core.pricing.btc_pricing_engine.calculate_probabilities` (it fits GARCH and
is slow). The stub mimics the real signature: `(strikes, hours_to_expiry,
**kwargs) -> {strike: prob, ..., '_meta': {...}}`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from market_maker.config import MMConfig
from market_maker.contracts import ConfidenceTier, Sigma2Source
from market_maker.pricer_adapter import build_snapshot

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def _no_posterior(strikes, hours_to_expiry, **kwargs):
    """No-op PARAM_POSTERIOR stub: tests must NEVER hit the real slow channel."""
    return {}


def _linear_stub(n_sims=15000, s0=100000.0, horizon_gate_active=False, extra_meta=None):
    """A stub whose probability decreases linearly with strike (strictly
    monotone), spanning p=0.9 at the lowest strike to p=0.1 at the highest.
    """
    def stub(strikes, hours_to_expiry, **kwargs):
        lo, hi = min(strikes), max(strikes)
        span = hi - lo if hi > lo else 1.0
        results = {}
        for k in strikes:
            frac = (k - lo) / span
            results[k] = 0.9 - 0.8 * frac
        meta = {"n_sims": n_sims, "S0": s0, "horizon_gate_active": horizon_gate_active}
        if extra_meta:
            meta.update(extra_meta)
        results["_meta"] = meta
        return results
    return stub


def test_sigma2_math_and_ladder_max_and_grid_layout():
    strikes = [90000.0, 100000.0, 110000.0]
    stub = _linear_stub(n_sims=15000)
    snap = build_snapshot(strikes, "2026-07-20", hours_to_expiry=336.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW)

    assert snap.n_sims == 15000
    assert snap.strikes == [90000.0, 100000.0, 110000.0]
    assert snap.grid_strikes == [90000.0, 95000.0, 100000.0, 105000.0, 110000.0]
    # midpoints present
    assert 95000.0 in snap.grid_strikes and 105000.0 in snap.grid_strikes
    # quoted/grid split correct
    assert set(snap.p_hat.keys()) == set(strikes)
    assert set(snap.p_grid.keys()) == set(snap.grid_strikes)

    for k, p in snap.p_hat.items():
        expected_sigma2 = p * (1.0 - p) / 15000
        assert snap.sigma2[k] == pytest.approx(expected_sigma2)

    assert snap.sigma2_ladder == pytest.approx(max(snap.sigma2.values()))
    assert snap.sigma2_source is Sigma2Source.MC


@pytest.mark.parametrize(
    "tte_days,expected_tier",
    [
        (14.0, ConfidenceTier.FULL),
        (14.1, ConfidenceTier.DEGRADED),
        (28.5, ConfidenceTier.MINIMAL),
        (30.5, ConfidenceTier.NAIVE_GATED),
    ],
)
def test_confidence_tier_boundaries(tte_days, expected_tier):
    stub = _linear_stub()
    snap = build_snapshot(
        [100000.0], "2026-07-20", hours_to_expiry=tte_days * 24.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW,
    )
    assert snap.confidence_tier is expected_tier


def test_horizon_gate_active_passthrough_true():
    stub = _linear_stub(horizon_gate_active=True)
    snap = build_snapshot([100000.0], "2026-07-20", hours_to_expiry=48 * 24.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW)
    assert snap.horizon_gate_active is True


def test_horizon_gate_active_passthrough_false():
    stub = _linear_stub(horizon_gate_active=False)
    snap = build_snapshot([100000.0], "2026-07-20", hours_to_expiry=5 * 24.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW)
    assert snap.horizon_gate_active is False


def test_stale_passthrough_from_engine_meta():
    stub = _linear_stub(extra_meta={"stale": True})
    snap = build_snapshot([100000.0], "2026-07-20", hours_to_expiry=5 * 24.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW)
    assert snap.stale is True


def test_stale_from_snapshot_max_age():
    cfg = MMConfig()
    stub = _linear_stub()
    old_ts = NOW - timedelta(seconds=cfg.pricer_max_age_s + 60.0)
    snap = build_snapshot(
        [100000.0], "2026-07-20", hours_to_expiry=5 * 24.0, engine_fn=stub, posterior_fn=_no_posterior, ts=old_ts, now=NOW, config=cfg,
    )
    assert snap.stale is True


def test_not_stale_when_fresh_and_no_engine_flag():
    stub = _linear_stub()
    snap = build_snapshot([100000.0], "2026-07-20", hours_to_expiry=5 * 24.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW)
    assert snap.stale is False


def test_non_monotone_stub_triggers_warning(caplog):
    def bad_stub(strikes, hours_to_expiry, **kwargs):
        # Deliberately non-monotone: probability goes UP with strike partway through.
        sorted_strikes = sorted(strikes)
        results = {}
        for i, k in enumerate(sorted_strikes):
            results[k] = 0.5 if i != len(sorted_strikes) - 1 else 0.9  # last strike jumps up
        results["_meta"] = {"n_sims": 15000, "S0": 100000.0}
        return results

    with caplog.at_level(logging.WARNING, logger="market_maker.pricer_adapter"):
        build_snapshot(
            [90000.0, 100000.0, 110000.0], "2026-07-20", hours_to_expiry=336.0,
            engine_fn=bad_stub, posterior_fn=_no_posterior, ts=NOW, now=NOW,
        )
    assert any("non-monotone" in rec.message for rec in caplog.records)


def test_monotone_stub_does_not_warn(caplog):
    stub = _linear_stub()
    with caplog.at_level(logging.WARNING, logger="market_maker.pricer_adapter"):
        build_snapshot(
            [90000.0, 100000.0, 110000.0], "2026-07-20", hours_to_expiry=336.0,
            engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW,
        )
    assert not any("non-monotone" in rec.message for rec in caplog.records)


def test_engine_kwargs_passthrough():
    captured = {}

    def stub(strikes, hours_to_expiry, **kwargs):
        captured.update(kwargs)
        results = {k: 0.5 for k in strikes}
        results["_meta"] = {"n_sims": 15000, "S0": 100000.0}
        return results

    garch_cache = {}
    build_snapshot(
        [100000.0], "2026-07-20", hours_to_expiry=336.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW,
        garch_cache=garch_cache, s0_override=99000.0, seed=42, use_figarch=True,
    )
    assert captured["garch_cache"] is garch_cache
    assert captured["s0_override"] == 99000.0
    assert captured["seed"] == 42
    assert captured["use_figarch"] is True


def test_missing_n_sims_raises():
    def stub(strikes, hours_to_expiry, **kwargs):
        results = {k: 0.5 for k in strikes}
        results["_meta"] = {"S0": 100000.0}  # no n_sims
        return results

    with pytest.raises(ValueError):
        build_snapshot([100000.0], "2026-07-20", hours_to_expiry=336.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW)


def test_single_strike_grid_has_no_midpoints():
    stub = _linear_stub()
    snap = build_snapshot([100000.0], "2026-07-20", hours_to_expiry=336.0, engine_fn=stub, posterior_fn=_no_posterior, ts=NOW, now=NOW)
    assert snap.grid_strikes == [100000.0]
    assert snap.p_hat.keys() == snap.p_grid.keys()


# ---------------------------------------------------------------------------
# Decision D2: PARAM_POSTERIOR wing channel
# ---------------------------------------------------------------------------

def _band_posterior(width):
    """Posterior stub returning symmetric q05/q95 bands of the given width."""
    def stub(strikes, hours_to_expiry, **kwargs):
        out = {}
        for k in strikes:
            out[k] = {"q05": 0.5 - width / 2.0, "q50": 0.5, "q95": 0.5 + width / 2.0, "point": 0.5}
        out["_meta"] = {}
        return out
    return stub


def test_wing_posterior_applied_only_to_wings():
    import market_maker.pricer_adapter as pa
    pa._wing_posterior_cache.clear()
    stub = _linear_stub(n_sims=15000)  # p spans 0.9 (wing) .. 0.1 (wing), mid 0.5 (belly)
    cfg = MMConfig(use_param_posterior_wings=True)
    width = 0.10  # sigma = 0.10/3.29 -> sigma2 ~ 9.2e-4 >> MC ~ 6e-6
    snap = build_snapshot(
        [90000.0, 100000.0, 110000.0], "2026-07-21", hours_to_expiry=336.0,
        engine_fn=stub, posterior_fn=_band_posterior(width), config=cfg, ts=NOW, now=NOW,
    )
    exp_wing_sigma2 = (width / 3.29) ** 2
    assert snap.sigma2[90000.0] == pytest.approx(exp_wing_sigma2)  # p=0.9 wing
    assert snap.sigma2[110000.0] == pytest.approx(exp_wing_sigma2)  # p=0.1 wing
    # belly strike keeps MC
    assert snap.sigma2[100000.0] == pytest.approx(0.5 * 0.5 / 15000)
    assert snap.sigma2_source is Sigma2Source.PARAM_POSTERIOR
    assert snap.sigma2_ladder == pytest.approx(exp_wing_sigma2)
    assert snap.engine_meta["param_posterior_strikes"] == [90000.0, 110000.0]


def test_wing_posterior_never_reduces_below_mc():
    import market_maker.pricer_adapter as pa
    pa._wing_posterior_cache.clear()
    stub = _linear_stub(n_sims=10)  # tiny n_sims -> MC sigma2 large
    cfg = MMConfig(use_param_posterior_wings=True)
    snap = build_snapshot(
        [90000.0, 100000.0, 110000.0], "2026-07-22", hours_to_expiry=336.0,
        engine_fn=stub, posterior_fn=_band_posterior(0.001), config=cfg, ts=NOW, now=NOW,
    )
    # posterior sigma2 ~ 9.2e-8 << MC 0.9*0.1/10 = 9e-3 -> MC stands, source stays MC
    assert snap.sigma2[90000.0] == pytest.approx(0.9 * 0.1 / 10)
    assert snap.sigma2_source is Sigma2Source.MC


def test_wing_posterior_failure_falls_back_to_mc():
    import market_maker.pricer_adapter as pa
    pa._wing_posterior_cache.clear()

    def broken(strikes, hours_to_expiry, **kwargs):
        raise RuntimeError("posterior exploded")

    stub = _linear_stub(n_sims=15000)
    cfg = MMConfig(use_param_posterior_wings=True)
    snap = build_snapshot(
        [90000.0, 110000.0], "2026-07-23", hours_to_expiry=336.0,
        engine_fn=stub, posterior_fn=broken, config=cfg, ts=NOW, now=NOW,
    )
    assert snap.sigma2_source is Sigma2Source.MC
    for k, p in snap.p_hat.items():
        assert snap.sigma2[k] == pytest.approx(p * (1 - p) / 15000)


def test_wing_posterior_cache_ttl():
    import market_maker.pricer_adapter as pa
    pa._wing_posterior_cache.clear()
    calls = []

    def counting(strikes, hours_to_expiry, **kwargs):
        calls.append(1)
        return _band_posterior(0.10)(strikes, hours_to_expiry, **kwargs)

    stub = _linear_stub(n_sims=15000)
    cfg = MMConfig(use_param_posterior_wings=True, posterior_refresh_s=3600.0)
    for i in range(3):
        build_snapshot(
            [90000.0, 110000.0], "2026-07-24", hours_to_expiry=336.0,
            engine_fn=stub, posterior_fn=counting, config=cfg,
            ts=NOW + timedelta(seconds=60 * i), now=NOW + timedelta(seconds=60 * i),
        )
    assert len(calls) == 1  # cached within TTL
    build_snapshot(
        [90000.0, 110000.0], "2026-07-24", hours_to_expiry=336.0,
        engine_fn=stub, posterior_fn=counting, config=cfg,
        ts=NOW + timedelta(seconds=7200), now=NOW + timedelta(seconds=7200),
    )
    assert len(calls) == 2  # recomputed after TTL


def test_wing_posterior_disabled_by_config():
    import market_maker.pricer_adapter as pa
    pa._wing_posterior_cache.clear()

    def must_not_call(strikes, hours_to_expiry, **kwargs):
        raise AssertionError("posterior_fn called with channel disabled")

    stub = _linear_stub(n_sims=15000)
    cfg = MMConfig(use_param_posterior_wings=False)
    snap = build_snapshot(
        [90000.0, 110000.0], "2026-07-25", hours_to_expiry=336.0,
        engine_fn=stub, posterior_fn=must_not_call, config=cfg, ts=NOW, now=NOW,
    )
    assert snap.sigma2_source is Sigma2Source.MC
