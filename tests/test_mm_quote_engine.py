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
from market_maker.contracts import QuoteProposal
from market_maker.quote_engine import estimate_sigma_b, glft_side_deltas, make_quote


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
