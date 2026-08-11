"""Tests for market_maker.quote_engine (plan task Q1, Section 2.4)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker import logodds
from market_maker.config import MMConfig
from market_maker.contracts import QuoteProposal
from market_maker.quote_engine import (
    estimate_sigma_b,
    glft_side_deltas,
    make_quote,
    make_quote_from_config,
    per_share_skew_x,
)


# ---------------------------------------------------------------------------
# Golden values (Dalen AS in x)
# ---------------------------------------------------------------------------

def test_dalen_golden_values():
    # x_fair=0, q=0, gamma=0.5, sigma_b=0.3, tte=14, k=2
    qp = make_quote("m", x_fair=0.0, q=0.0, tte_days=14.0, sigma_b=0.3,
                    gamma=0.5, k=2.0, A=1.0, variant="dalen")
    # r_x = 0 - 0 = 0 ; skew = 0
    assert qp.r_x == pytest.approx(0.0, abs=1e-12)
    assert qp.skew_x == pytest.approx(0.0, abs=1e-12)
    # delta_x = 0.5*(0.5*0.09*14 + (2/2)*ln(1.25))
    expected = 0.5 * (0.5 * 0.09 * 14.0 + 1.0 * math.log(1.25))
    assert qp.delta_x == pytest.approx(expected, abs=1e-12)
    assert qp.delta_x == pytest.approx(0.426571775657, abs=1e-9)
    assert qp.x_bid == pytest.approx(-expected, abs=1e-12)
    assert qp.x_ask == pytest.approx(expected, abs=1e-12)


def test_dalen_skew_sign_long_shifts_down():
    base = make_quote("m", 0.0, 0.0, 14.0, 0.3, 0.5, 2.0, 1.0, variant="dalen")
    lng = make_quote("m", 0.0, 1.0, 14.0, 0.3, 0.5, 2.0, 1.0, variant="dalen")
    shrt = make_quote("m", 0.0, -1.0, 14.0, 0.3, 0.5, 2.0, 1.0, variant="dalen")
    # q>0 shifts both quotes DOWN
    assert lng.x_bid < base.x_bid
    assert lng.x_ask < base.x_ask
    assert lng.skew_x < 0.0
    # q<0 shifts both quotes UP
    assert shrt.x_bid > base.x_bid
    assert shrt.x_ask > base.x_ask
    assert shrt.skew_x > 0.0


def test_dalen_spread_widens_with_sigma_b():
    prev = None
    for sb in [0.1, 0.2, 0.4, 0.8]:
        qp = make_quote("m", 0.0, 0.0, 14.0, sb, 0.5, 2.0, 1.0, variant="dalen")
        if prev is not None:
            assert qp.delta_x > prev
        prev = qp.delta_x


def test_dalen_spread_widens_with_tte():
    prev = None
    for tte in [1.0, 7.0, 14.0, 28.0]:
        qp = make_quote("m", 0.0, 0.0, tte, 0.3, 0.5, 2.0, 1.0, variant="dalen")
        if prev is not None:
            assert qp.delta_x > prev
        prev = qp.delta_x


def test_p_outputs_respect_clamp_and_ordering():
    p_lo, p_hi = 0.001, 0.999
    for x_fair in [-8.0, -1.0, 0.0, 1.0, 8.0]:
        for q in [-3.0, 0.0, 3.0]:
            qp = make_quote("m", x_fair, q, 14.0, 0.5, 0.5, 2.0, 1.0,
                            p_lo=p_lo, p_hi=p_hi, variant="dalen")
            assert p_lo <= qp.p_bid_raw <= p_hi
            assert p_lo <= qp.p_ask_raw <= p_hi
            assert qp.p_bid_raw < qp.p_ask_raw
            assert math.isfinite(qp.x_bid) and math.isfinite(qp.x_ask)


# ---------------------------------------------------------------------------
# GLFT variant
# ---------------------------------------------------------------------------

def test_glft_golden_symmetric_at_zero_inventory():
    gamma, k, A, sb = 0.5, 2.0, 1.0, 0.3
    qp = make_quote("m", 0.0, 0.0, 14.0, sb, gamma, k, A, variant="glft")
    base = (1.0 / gamma) * math.log(1.0 + gamma / k)
    C = math.sqrt(sb * sb * gamma / (2.0 * k * A)) * (1.0 + gamma / k) ** (1.0 + k / gamma)
    # q=0 -> symmetric: skew 0, delta_x = base + 0.5*C
    assert qp.skew_x == pytest.approx(0.0, abs=1e-12)
    assert qp.delta_x == pytest.approx(base + 0.5 * C, abs=1e-12)
    db, da = glft_side_deltas(0.0, sb, gamma, k, A)
    assert db == pytest.approx(da, abs=1e-12)


def test_glft_long_inventory_skews_down():
    gamma, k, A, sb = 0.5, 2.0, 1.0, 0.3
    base = make_quote("m", 0.0, 0.0, 14.0, sb, gamma, k, A, variant="glft")
    lng = make_quote("m", 0.0, 2.0, 14.0, sb, gamma, k, A, variant="glft")
    assert lng.skew_x < 0.0
    assert lng.x_bid < base.x_bid
    assert lng.x_ask < base.x_ask
    # delta_x is inventory-independent under GLFT
    assert lng.delta_x == pytest.approx(base.delta_x, abs=1e-12)


def test_glft_spread_widens_with_sigma_b():
    prev = None
    for sb in [0.1, 0.2, 0.4, 0.8]:
        qp = make_quote("m", 0.0, 0.0, 14.0, sb, 0.5, 2.0, 1.0, variant="glft")
        if prev is not None:
            assert qp.delta_x > prev
        prev = qp.delta_x


# ---------------------------------------------------------------------------
# sigma_b estimator
# ---------------------------------------------------------------------------

def test_estimate_sigma_b_constant_series_returns_floor():
    x = np.full(500, 1.234)
    sb = estimate_sigma_b(x, dt_days=1.0, sigma_b_floor=0.05, sigma_b_cap=5.0)
    assert sb == pytest.approx(0.05, abs=1e-12)


def test_estimate_sigma_b_recovers_known_vol():
    rng = np.random.default_rng(42)
    sigma_true = 0.35
    dt = 0.25
    n = 60000
    steps = rng.normal(0.0, sigma_true * math.sqrt(dt), size=n)
    x = np.concatenate([[0.0], np.cumsum(steps)])
    sb = estimate_sigma_b(x, dt_days=dt, sigma_b_floor=0.01, sigma_b_cap=5.0, lam=0.995)
    assert sb == pytest.approx(sigma_true, rel=0.15)


def test_estimate_sigma_b_clamps_to_cap():
    rng = np.random.default_rng(1)
    steps = rng.normal(0.0, 3.0, size=5000)
    x = np.concatenate([[0.0], np.cumsum(steps)])
    sb = estimate_sigma_b(x, dt_days=1.0, sigma_b_floor=0.05, sigma_b_cap=1.0)
    assert sb == pytest.approx(1.0, abs=1e-12)


def test_estimate_sigma_b_floor_when_too_short():
    assert estimate_sigma_b([1.0], 1.0, 0.05, 5.0) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# params_id fingerprint
# ---------------------------------------------------------------------------

def test_params_id_changes_with_parameter():
    a = make_quote("m", 0.0, 0.0, 14.0, 0.3, 0.5, 2.0, 1.0, variant="dalen")
    b = make_quote("m", 0.0, 0.0, 14.0, 0.3, 0.6, 2.0, 1.0, variant="dalen")  # gamma
    c = make_quote("m", 0.0, 0.0, 14.0, 0.3, 0.5, 2.0, 1.0, variant="glft")   # variant
    assert a.params_id != b.params_id
    assert a.params_id != c.params_id
    # identical params -> identical id
    d = make_quote("m", 0.0, 0.0, 14.0, 0.3, 0.5, 2.0, 1.0, variant="dalen")
    assert a.params_id == d.params_id


def test_returns_quote_proposal_type():
    qp = make_quote("mkt", 0.5, 0.0, 10.0, 0.3, 0.5, 2.0, 1.0)
    assert isinstance(qp, QuoteProposal)
    assert qp.market_id == "mkt"


# ---------------------------------------------------------------------------
# Decision D3: arrival-denominator switch (Dalen 2/k vs classical AS 2/gamma)
# ---------------------------------------------------------------------------

def test_arrival_denominator_switch():
    import math
    from market_maker.quote_engine import make_quote

    gamma, k, sigma_b, tte = 0.5, 2.0, 0.3, 14.0
    var_term = gamma * sigma_b * sigma_b * tte
    qd = make_quote("m", 0.0, 0.0, tte, sigma_b, gamma, k, 1.0,
                    arrival_denominator="k")
    qa = make_quote("m", 0.0, 0.0, tte, sigma_b, gamma, k, 1.0,
                    arrival_denominator="gamma")
    exp_k = 0.5 * (var_term + (2.0 / k) * math.log(1 + gamma / k))
    exp_g = 0.5 * (var_term + (2.0 / gamma) * math.log(1 + gamma / k))
    assert qd.delta_x == pytest.approx(exp_k)
    assert qa.delta_x == pytest.approx(exp_g)
    assert qa.delta_x > qd.delta_x  # gamma < k here -> AS wider
    assert qd.params_id != qa.params_id  # fingerprint distinguishes the setting


# ---------------------------------------------------------------------------
# Inventory-skew displacement cap (2026-08-10 skew-explosion fix)
# ---------------------------------------------------------------------------

def test_skew_x_cap_disabled_by_default():
    # make_quote's own default (skew_x_cap=0.0) is the legacy unbounded path;
    # only make_quote_from_config threads MMConfig's default-ON cap.
    gamma, k, sb, tte = 0.5, 2.0, 3.0, 28.0
    qp = make_quote("m", 0.0, 50.0, tte, sb, gamma, k, 1.0, variant="dalen")
    var_term = gamma * sb * sb * tte
    assert qp.skew_x == pytest.approx(-50.0 * var_term, abs=1e-9)
    assert abs(qp.skew_x) > 1.0


def test_skew_x_cap_binds_symmetric_dalen():
    gamma, k, sb, tte, cap = 0.5, 2.0, 2.0, 14.0, 1.0
    long_q = make_quote("m", 0.0, 50.0, tte, sb, gamma, k, 1.0,
                         variant="dalen", skew_x_cap=cap)
    short_q = make_quote("m", 0.0, -50.0, tte, sb, gamma, k, 1.0,
                          variant="dalen", skew_x_cap=cap)
    assert long_q.skew_x == pytest.approx(-cap, abs=1e-12)
    assert short_q.skew_x == pytest.approx(cap, abs=1e-12)
    # identity holds exactly (x_fair == 0.0 here, no band clamp involved)
    assert long_q.r_x - long_q.skew_x == pytest.approx(0.0, abs=1e-12)
    assert short_q.r_x - short_q.skew_x == pytest.approx(0.0, abs=1e-12)


def test_skew_x_cap_binds_symmetric_glft():
    gamma, k, A, sb, cap = 0.5, 2.0, 1.0, 2.0, 1.0
    long_q = make_quote("m", 0.0, 50.0, 14.0, sb, gamma, k, A,
                         variant="glft", skew_x_cap=cap)
    short_q = make_quote("m", 0.0, -50.0, 14.0, sb, gamma, k, A,
                          variant="glft", skew_x_cap=cap)
    assert long_q.skew_x == pytest.approx(-cap, abs=1e-12)
    assert short_q.skew_x == pytest.approx(cap, abs=1e-12)
    assert long_q.r_x - long_q.skew_x == pytest.approx(0.0, abs=1e-12)
    assert short_q.r_x - short_q.skew_x == pytest.approx(0.0, abs=1e-12)


def test_skew_x_cap_inert_for_small_skew_dalen():
    # |skew| well under cap -> byte-identical to legacy (cap disabled).
    gamma, k, sb, tte, x_fair, q = 0.5, 2.0, 0.3, 14.0, 0.0, 1.0
    capped = make_quote("m", x_fair, q, tte, sb, gamma, k, 1.0,
                         variant="dalen", skew_x_cap=1.0)
    legacy = make_quote("m", x_fair, q, tte, sb, gamma, k, 1.0,
                         variant="dalen", skew_x_cap=0.0)
    assert abs(legacy.skew_x) < 1.0  # sanity: this case does not bind the cap
    assert capped.skew_x == pytest.approx(legacy.skew_x, abs=1e-12)
    assert capped.r_x == pytest.approx(legacy.r_x, abs=1e-12)
    assert capped.x_bid == pytest.approx(legacy.x_bid, abs=1e-12)
    assert capped.x_ask == pytest.approx(legacy.x_ask, abs=1e-12)


def test_skew_x_cap_inert_for_small_skew_glft():
    gamma, k, A, sb, x_fair, q = 0.5, 2.0, 1.0, 0.1, 0.0, 1.0
    capped = make_quote("m", x_fair, q, 14.0, sb, gamma, k, A,
                         variant="glft", skew_x_cap=1.0)
    legacy = make_quote("m", x_fair, q, 14.0, sb, gamma, k, A,
                         variant="glft", skew_x_cap=0.0)
    assert abs(legacy.skew_x) < 1.0
    assert capped.skew_x == pytest.approx(legacy.skew_x, abs=1e-12)
    assert capped.r_x == pytest.approx(legacy.r_x, abs=1e-12)


def test_skew_x_cap_disabled_matches_legacy_band_clamp_path():
    # cap <= 0 must be an EXACT legacy revert, INCLUDING the case where the
    # pre-existing band clamp binds -- legacy deliberately keeps the
    # band-clamp identity OFFSET (skew_x stays the raw, un-re-derived term).
    gamma, k, sb, tte, x_fair, q = 0.5, 2.0, 3.0, 28.0, 6.5, 50.0
    x_lo, x_hi = logodds.logit_bounds()
    var_term = gamma * sb * sb * tte
    expected_skew_raw = -q * var_term
    expected_r_x = float(min(max(x_fair + expected_skew_raw, x_lo), x_hi))
    for cap in (0.0, -1.0):
        qp = make_quote("m", x_fair, q, tte, sb, gamma, k, 1.0,
                         variant="dalen", skew_x_cap=cap)
        assert qp.skew_x == pytest.approx(expected_skew_raw, abs=1e-6)
        assert qp.r_x == pytest.approx(expected_r_x, abs=1e-9)
        assert qp.r_x == pytest.approx(x_lo, abs=1e-9)  # band clamp did bind
        # legacy keeps the band-clamp OFFSET -- identity does NOT hold here
        assert (qp.r_x - qp.skew_x) != pytest.approx(x_fair, abs=1e-3)


def test_skew_x_cap_identity_exact_including_deep_wing_band_clamp():
    # Identity x_fair == r_x - skew_x must be EXACT whenever the cap is
    # enabled, even on the deep-wing path where the band clamp binds AFTER
    # the cap clamp (same-signed skew pushes r_x past the band edge);
    # |skew_x| must come back < cap there, never phantom-shifted.
    gamma, k, sb, tte, cap = 0.5, 2.0, 2.0, 14.0, 1.0
    x_lo, x_hi = logodds.logit_bounds()
    cases = [
        (0.0, 5.0, "dalen"),
        (0.0, -5.0, "dalen"),
        (0.0, 5.0, "glft"),
        (0.0, -5.0, "glft"),
        # deep wing, same-signed skew as the direction that would push r_x
        # further past the edge -> band clamp binds after the cap clamp
        (x_hi - 0.3, -50.0, "dalen"),  # short -> positive skew, r_x pushed past x_hi
        (x_lo + 0.3, 50.0, "dalen"),   # long -> negative skew, r_x pushed past x_lo
        (x_hi - 0.3, -50.0, "glft"),
        (x_lo + 0.3, 50.0, "glft"),
    ]
    for x_fair, q, variant in cases:
        A = 1.0
        qp = make_quote("m", x_fair, q, tte, sb, gamma, k, A,
                         variant=variant, skew_x_cap=cap)
        assert qp.r_x - qp.skew_x == pytest.approx(x_fair, abs=1e-9)
        assert abs(qp.skew_x) <= cap + 1e-9


def test_skew_x_cap_deep_wing_skew_shrinks_not_phantom_shifted():
    # Explicit check of the "never phantom-shifted" acceptance criterion:
    # at the band edge, the re-derived skew_x comes back STRICTLY smaller
    # in magnitude than the cap (not equal to it, and not larger).
    gamma, k, sb, tte, cap = 0.5, 2.0, 2.0, 14.0, 1.0
    x_lo, x_hi = logodds.logit_bounds()
    qp = make_quote("m", x_hi - 0.3, -50.0, tte, sb, gamma, k, 1.0,
                     variant="dalen", skew_x_cap=cap)
    assert qp.r_x == pytest.approx(x_hi, abs=1e-9)
    assert qp.skew_x == pytest.approx(0.3, abs=1e-9)
    assert 0.0 < qp.skew_x < cap


def test_make_quote_from_config_threads_skew_x_cap():
    cfg = MMConfig()  # default skew_x_cap=1.0 (default ON)
    q_big = make_quote_from_config(
        cfg, "m", x_fair=0.0, q=1.0e6, tte_days=14.0, sigma_b=2.0, variant="dalen"
    )
    assert abs(q_big.skew_x) == pytest.approx(cfg.skew_x_cap, abs=1e-9)

    class _BareConfig:
        p_clamp = (0.001, 0.999)
        gamma = 0.5
        k_arrival = 2.0
        arrival_scale_A = 1.0
        # no skew_x_cap attribute -> getattr-guarded fallback to 0.0 (disabled)

    bare_q = make_quote_from_config(
        _BareConfig(), "m", x_fair=0.0, q=1.0e6, tte_days=14.0, sigma_b=2.0,
        variant="dalen",
    )
    assert abs(bare_q.skew_x) > cfg.skew_x_cap  # legacy unbounded on a bare config


def test_incident_regression_skew_cap_prevents_fire_sale():
    # VPS incident 2026-08-10: q=13.42, sigma_b=2.46, gamma=0.1, tte=1.08 at
    # the live k_arrival=12.8 default produced skew_x=-8.77 uncapped,
    # pinning the reservation near the p_clamp floor and posting a fire-sale
    # ask. With skew_x_cap=1.0 the displacement is bound to exactly -1.0 and
    # the posted ask lands within roughly a half-spread of x_fair (~0.77
    # area), not the fire-sale zone.
    q, sigma_b, gamma, k, tte, x_fair = 13.42, 2.46, 0.1, 12.8, 1.08, 1.9
    capped = make_quote("m", x_fair, q, tte, sigma_b, gamma, k, 1.0,
                         variant="dalen", skew_x_cap=1.0)
    legacy = make_quote("m", x_fair, q, tte, sigma_b, gamma, k, 1.0,
                         variant="dalen", skew_x_cap=0.0)

    assert capped.skew_x == pytest.approx(-1.0, abs=1e-12)
    assert capped.p_ask_raw == pytest.approx(0.7734, abs=0.01)
    assert capped.p_ask_raw > 0.6  # NOT the 0.14-0.23 fire-sale zone

    assert legacy.p_ask_raw < 0.05  # uncapped: fire-sale ask, deep below fair
    assert capped.p_ask_raw - legacy.p_ask_raw > 0.5


# ---------------------------------------------------------------------------
# per_share_skew_x -- shared helper (consumed by item 2's sizing cap)
# ---------------------------------------------------------------------------

def test_per_share_skew_x_matches_make_quote_dalen():
    gamma, k, sb, tte, q = 0.5, 2.0, 0.3, 14.0, 3.0
    qp = make_quote("m", 0.0, q, tte, sb, gamma, k, 1.0, variant="dalen")
    per_share = per_share_skew_x("dalen", sb, gamma, k, 1.0, tte)
    assert qp.skew_x == pytest.approx(-q * per_share, abs=1e-12)
    # matches the raw Dalen skew term formula directly too
    assert per_share == pytest.approx(gamma * sb * sb * tte, abs=1e-12)


def test_per_share_skew_x_matches_make_quote_glft():
    gamma, k, A, sb, q = 0.5, 2.0, 1.0, 0.3, 3.0
    qp = make_quote("m", 0.0, q, 14.0, sb, gamma, k, A, variant="glft")
    per_share = per_share_skew_x("glft", sb, gamma, k, A, 14.0)
    assert qp.skew_x == pytest.approx(-q * per_share, abs=1e-12)
    expected_C = math.sqrt(sb * sb * gamma / (2.0 * k * A)) * (1.0 + gamma / k) ** (
        1.0 + k / gamma
    )
    assert per_share == pytest.approx(expected_C, abs=1e-12)


def test_per_share_skew_x_no_clamping():
    # Pure function -- no cap, no floor, matches the raw variant term even
    # for absurd inputs (the caller, not this helper, is responsible for
    # any clamping).
    huge = per_share_skew_x("dalen", sigma_b=100.0, gamma=5.0, k=2.0, A=1.0,
                             tte_days=365.0)
    assert huge == pytest.approx(5.0 * 100.0 * 100.0 * 365.0, abs=1e-6)


def test_per_share_skew_x_unknown_variant_raises():
    with pytest.raises(ValueError):
        per_share_skew_x("bogus", sigma_b=0.3, gamma=0.5, k=2.0, A=1.0, tte_days=14.0)
