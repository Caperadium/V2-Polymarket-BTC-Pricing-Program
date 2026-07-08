"""Workstream 2 tests (plan i-m-preparing-to-launch-sharded-snail.md, section
2.2 auto event selection / 2.5 GARCH cache expiry / 2.7 retry) for
market_maker/shadow_runner.py.
"""
from __future__ import annotations

import sys
import time as time_mod
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker import shadow_runner
from market_maker.config import MMConfig

NOW = datetime(2026, 7, 7, 10, 0, tzinfo=timezone.utc)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url="x", code=code, msg="err", hdrs=None, fp=None)


def _event_payload(end_date: str, strikes=(98000.0, 102000.0)) -> List[Dict[str, Any]]:
    markets = []
    for i, k in enumerate(strikes):
        markets.append({
            "slug": f"m-{int(k)}",
            "question": f"Will the price of Bitcoin be above ${k:,.0f} on ...",
            "clobTokenIds": f'["tok-{i}"]',
        })
    return [{"endDate": end_date, "markets": markets}]


# ---------------------------------------------------------------------------
# 2.7 -- _get_retry backoff / SystemExit / 404 passthrough
# ---------------------------------------------------------------------------


def test_get_retry_succeeds_after_transient_failures(monkeypatch):
    calls = {"n": 0}
    sleeps: List[float] = []

    def _flaky(url):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return {"ok": True}

    monkeypatch.setattr(shadow_runner, "_get", _flaky)
    monkeypatch.setattr(shadow_runner.time, "sleep", lambda s: sleeps.append(s))

    result = shadow_runner._get_retry("http://x")
    assert result == {"ok": True}
    assert calls["n"] == 3
    assert sleeps == [2.0, 4.0]  # exponential backoff 2s -> 4s before the 2 retries


def test_get_retry_404_not_retried(monkeypatch):
    calls = {"n": 0}

    def _always_404(url):
        calls["n"] += 1
        raise _http_error(404)

    monkeypatch.setattr(shadow_runner, "_get", _always_404)
    monkeypatch.setattr(shadow_runner.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")))

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        shadow_runner._get_retry("http://x")
    assert exc_info.value.code == 404
    assert calls["n"] == 1  # no retry on 404


def test_get_retry_exhausts_all_attempts_then_systemexit(monkeypatch):
    calls = {"n": 0}
    sleeps: List[float] = []

    def _always_fails(url):
        calls["n"] += 1
        raise ConnectionError("still down")

    monkeypatch.setattr(shadow_runner, "_get", _always_fails)
    monkeypatch.setattr(shadow_runner.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(SystemExit):
        shadow_runner._get_retry("http://x")
    assert calls["n"] == 5  # _RETRY_ATTEMPTS
    assert sleeps == [2.0, 4.0, 8.0, 16.0]  # capped only if it would exceed 30s


# ---------------------------------------------------------------------------
# 2.2 -- resolve_next_event: date probing, padded/unpadded forms, retry
# ---------------------------------------------------------------------------


def test_resolve_next_event_picks_correct_date_and_form(monkeypatch):
    """Only the padded slug for day+3 (2026-07-10) exists; every earlier
    date/form (both padded and unpadded) 404s."""
    target_slug = "bitcoin-above-on-july-10-2026"
    target_expiry = "2026-07-10"
    seen_urls: List[str] = []

    def _fake_get(url):
        seen_urls.append(url)
        if f"slug={target_slug}" in url:
            return _event_payload(target_expiry + "T12:00:00Z")
        raise _http_error(404)

    monkeypatch.setattr(shadow_runner, "_get", _fake_get)
    monkeypatch.setattr(shadow_runner.time, "sleep", lambda s: None)

    expiry_key, ladder = shadow_runner.resolve_next_event(NOW, lead_days=3)

    assert expiry_key == target_expiry
    assert len(ladder) == 2
    # both unpadded (july-8) and padded (july-08) forms must have been tried
    # for at least the earlier in-range dates.
    assert any("july-8-2026" in u for u in seen_urls)
    assert any(f"slug={target_slug}" in u for u in seen_urls)


def test_resolve_next_event_skips_events_too_close_to_settlement(monkeypatch):
    """An event that exists but settles too soon (inside near_resolution_pull_
    hours + 12h of `now`) must be skipped in favor of a later one."""
    near_slug = "bitcoin-above-on-july-8-2026"   # settles ~26h after NOW -- too soon
    far_slug = "bitcoin-above-on-july-12-2026"    # settles comfortably later

    def _fake_get(url):
        if f"slug={near_slug}" in url:
            return _event_payload("2026-07-08T12:00:00Z")
        if f"slug={far_slug}" in url:
            return _event_payload("2026-07-12T12:00:00Z")
        raise _http_error(404)

    monkeypatch.setattr(shadow_runner, "_get", _fake_get)
    monkeypatch.setattr(shadow_runner.time, "sleep", lambda s: None)

    expiry_key, _ladder = shadow_runner.resolve_next_event(NOW, lead_days=5)
    assert expiry_key == "2026-07-12"


def test_resolve_next_event_retries_transient_failures(monkeypatch):
    target_slug = "bitcoin-above-on-july-10-2026"
    target_expiry = "2026-07-10"
    call_counts: Dict[str, int] = {}
    sleeps: List[float] = []

    def _fake_get(url):
        call_counts[url] = call_counts.get(url, 0) + 1
        if f"slug={target_slug}" in url:
            if call_counts[url] < 3:
                raise ConnectionError("transient")
            return _event_payload(target_expiry + "T12:00:00Z")
        raise _http_error(404)

    monkeypatch.setattr(shadow_runner, "_get", _fake_get)
    monkeypatch.setattr(shadow_runner.time, "sleep", lambda s: sleeps.append(s))

    expiry_key, _ladder = shadow_runner.resolve_next_event(NOW, lead_days=3)
    assert expiry_key == target_expiry
    assert len(sleeps) >= 2  # the transient-failure retries actually slept


def test_resolve_next_event_raises_when_nothing_found(monkeypatch):
    monkeypatch.setattr(shadow_runner, "_get", lambda url: (_ for _ in ()).throw(_http_error(404)))
    monkeypatch.setattr(shadow_runner.time, "sleep", lambda s: None)

    with pytest.raises(SystemExit):
        shadow_runner.resolve_next_event(NOW, lead_days=3)


# ---------------------------------------------------------------------------
# 2.5 -- CachedEngine GARCH cache expiry
# ---------------------------------------------------------------------------


def test_garch_cache_expires_and_refits(monkeypatch):
    calls: List[Dict[str, Any]] = []

    def _stub_calc_probs(strikes, hours_to_expiry, **kwargs):
        cache = kwargs.get("garch_cache")
        was_empty = cache is not None and not cache
        if was_empty:
            cache["fit_marker"] = True
        calls.append({"was_empty": was_empty})
        out = {float(k): 0.5 for k in strikes}
        out["_meta"] = {"n_sims": 100, "S0": 100000.0, "horizon_gate_active": False}
        return out

    import core.pricing.btc_pricing_engine as pricing_mod
    monkeypatch.setattr(pricing_mod, "calculate_probabilities", _stub_calc_probs)

    clock = {"t": 0.0}
    monkeypatch.setattr(shadow_runner.time, "time", lambda: clock["t"])

    engine = shadow_runner.CachedEngine(reprice_s=1.0, garch_refit_s=100.0)

    # call 1 @ t=0: cold cache -> fit happens.
    engine([98000.0], 100.0)
    assert calls[-1]["was_empty"] is True
    assert engine._garch_fitted_at == pytest.approx(0.0)
    assert engine._garch_cache == {"fit_marker": True}

    # call 2 @ t=2 (past reprice_s, well before garch_refit_s): re-prices but
    # does NOT clear/refit the garch cache.
    clock["t"] = 2.0
    engine([98000.0], 100.0)
    assert calls[-1]["was_empty"] is False
    assert engine._garch_fitted_at == pytest.approx(0.0)  # unchanged -- no new fit
    assert engine._garch_cache == {"fit_marker": True}

    # call 3 @ t=150 (>= garch_refit_s past the tick-1 fit): cache cleared and
    # refit.
    clock["t"] = 150.0
    engine([98000.0], 100.0)
    assert calls[-1]["was_empty"] is True
    assert engine._garch_fitted_at == pytest.approx(150.0)
    assert engine._garch_cache == {"fit_marker": True}
    assert len(calls) == 3
