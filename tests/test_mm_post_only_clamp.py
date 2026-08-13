"""Tests for the post-only book clamp (2026-08-13 bleed-2 fix, item 2;
temp/mm_bleed2_fix_plan.md). A resting bid above the venue's best ask (or a
resting ask below the venue's best bid) is modelled by paper_fill_sim as
filling at OUR OWN crossed price with queue_ahead=0 -- a real post-only order
would instead be rejected/repriced by the venue. `spread_builder.
post_only_clamp` bounds each side of the desired ladder to stay inside the
opposite venue touch by `MMConfig.post_only_margin_ticks` before the QuoteSet
is journaled/sent to the lifecycle; the harness wires it into `tick()`
gated on that sentinel.

Section 1: pure-function tests on `post_only_clamp` directly (no harness).
Section 2: harness-level integration, scripted synthetic feeds only,
following tests/test_mm_harness_ws1.py's / tests/test_mm_integration.py's
conventions (fixed clock, deterministic scripted books).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from market_maker.config import MMConfig
from market_maker.contracts import (
    HedgeReason,
    HedgeRecommendation,
    QuoteMode,
    QuoteSet,
    Side,
)
from market_maker.harness import PaperTradingLoop
from market_maker.order_lifecycle import SimClock
from market_maker.spread_builder import post_only_clamp
from market_maker.state_store import MMStateStore

TS = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
TICK = 0.01
BAND = (0.001, 0.999)

START = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
S0 = 100000.0
EXPIRY = "2026-07-06"
MARKETS = [("m-100k", 100000.0), ("m-102k", 102000.0)]


# ---------------------------------------------------------------------------
# Section 1: pure-function tests
# ---------------------------------------------------------------------------


def _qs(bid_price=0.40, ask_price=0.50, bid_size=10.0, ask_size=10.0, terms=None):
    return QuoteSet(
        ts=TS, market_id="m1", bid_price=bid_price, ask_price=ask_price,
        bid_size=bid_size, ask_size=ask_size, terms=dict(terms or {}),
        risk_mode=QuoteMode.TWO_SIDED, noarb_checked=True, source_seq=1,
    )


def test_bid_crossing_venue_ask_clamped_to_best_ask_minus_margin():
    qs = _qs(bid_price=0.65, ask_price=0.75)
    out = post_only_clamp(qs, best_bid=0.60, best_ask=0.62, tick=TICK, band=BAND, margin_ticks=1)
    assert out.bid_price == pytest.approx(0.61)  # best_ask - 1 tick
    assert out.ask_price == pytest.approx(0.75)  # unchanged, no crossing on this side
    assert out.bid_size == 10.0
    assert out.terms["post_only_bid"] == pytest.approx(0.65 - 0.61)
    assert out.terms["post_only_bid"] > 0.0
    assert "post_only_ask" not in out.terms


def test_ask_below_venue_bid_clamped_to_best_bid_plus_margin():
    qs = _qs(bid_price=0.20, ask_price=0.35)
    out = post_only_clamp(qs, best_bid=0.40, best_ask=0.42, tick=TICK, band=BAND, margin_ticks=1)
    assert out.ask_price == pytest.approx(0.41)  # best_bid + 1 tick
    assert out.bid_price == pytest.approx(0.20)  # unchanged
    assert out.ask_size == 10.0
    assert out.terms["post_only_ask"] == pytest.approx(0.41 - 0.35)
    assert out.terms["post_only_ask"] > 0.0
    assert "post_only_bid" not in out.terms


def test_non_crossing_returns_same_object_and_no_terms_keys():
    qs = _qs(bid_price=0.40, ask_price=0.50)
    out = post_only_clamp(qs, best_bid=0.10, best_ask=0.90, tick=TICK, band=BAND, margin_ticks=1)
    assert out is qs
    assert "post_only_bid" not in out.terms
    assert "post_only_ask" not in out.terms


def test_idempotent():
    qs = _qs(bid_price=0.65, ask_price=0.75)
    once = post_only_clamp(qs, best_bid=0.60, best_ask=0.62, tick=TICK, band=BAND, margin_ticks=1)
    twice = post_only_clamp(once, best_bid=0.60, best_ask=0.62, tick=TICK, band=BAND, margin_ticks=1)
    assert twice is once  # nothing left to bind -> same object, strongest form
    assert twice.bid_price == pytest.approx(once.bid_price)
    assert twice.terms == once.terms


@pytest.mark.parametrize("bad_ref", [None, float("nan"), 0.0, 1.0, -0.1, float("inf")])
def test_unusable_reference_leaves_side_untouched(bad_ref):
    # Bid WOULD cross a usable best_ask of 0.62; best_bid is also None so the
    # ask side is untouched too -> the whole QuoteSet must come back unchanged.
    qs = _qs(bid_price=0.65, ask_price=0.75)
    out = post_only_clamp(qs, best_bid=None, best_ask=bad_ref, tick=TICK, band=BAND, margin_ticks=1)
    assert out is qs


def test_margin_ticks_two_honored():
    qs = _qs(bid_price=0.65, ask_price=0.75)
    out = post_only_clamp(qs, best_bid=0.60, best_ask=0.62, tick=TICK, band=BAND, margin_ticks=2)
    assert out.bid_price == pytest.approx(0.60)  # best_ask - 2 ticks


def test_degenerate_bid_zeroed_near_band_floor_price_reverted():
    # threshold = best_ask - tick = 0.015 - 0.01 = 0.005 -> floors to 0.0 ->
    # band-clamps to band_lo (0.001) -> below max(tick, band_lo) = 0.01 -> zero.
    qs = _qs(bid_price=0.50, ask_price=0.60)
    out = post_only_clamp(qs, best_bid=None, best_ask=0.015, tick=TICK, band=BAND, margin_ticks=1)
    assert out.bid_size == 0.0
    assert out.bid_price == pytest.approx(0.50)  # old valid value retained
    assert "post_only_bid" not in out.terms  # zeroing is not a price move
    assert out.ask_size == 10.0  # never resurrects/touches the other side


def test_degenerate_ask_zeroed_near_band_ceiling_price_reverted():
    # threshold = best_bid + tick = 0.985 + 0.01 = 0.995 -> ceils to 1.0 ->
    # band-clamps to band_hi (0.999) -> above min(1-tick, band_hi) = 0.99 -> zero.
    qs = _qs(bid_price=0.40, ask_price=0.50)
    out = post_only_clamp(qs, best_bid=0.985, best_ask=None, tick=TICK, band=BAND, margin_ticks=1)
    assert out.ask_size == 0.0
    assert out.ask_price == pytest.approx(0.50)  # old valid value retained
    assert "post_only_ask" not in out.terms
    assert out.bid_size == 10.0


def test_off_grid_reference_rounds_outward_not_nearest():
    # threshold = 0.2377 - 0.01 = 0.2277 -> floor (outward) gives 0.22, NOT
    # nearest (which would give 0.23, since .77 rounds up).
    qs = _qs(bid_price=0.50, ask_price=0.60)
    out = post_only_clamp(qs, best_bid=None, best_ask=0.2377, tick=TICK, band=BAND, margin_ticks=1)
    assert out.bid_price == pytest.approx(0.22)


def test_band_clamp_respected_without_degenerate_zero():
    band = (0.001, 0.95)
    qs = _qs(bid_price=0.20, ask_price=0.50)
    out = post_only_clamp(qs, best_bid=0.97, best_ask=None, tick=TICK, band=band, margin_ticks=1)
    assert out.ask_price == pytest.approx(0.95)  # clamped into the narrower band
    assert out.ask_size == 10.0  # NOT zeroed -- clamped value is not degenerate
    assert out.terms["post_only_ask"] == pytest.approx(0.95 - 0.50)


def test_zero_size_side_never_resurrected_bid():
    # bid_size already 0 (e.g. directive-suppressed) even though the price
    # would otherwise have crossed -- must stay untouched entirely.
    qs = _qs(bid_price=0.90, bid_size=0.0, ask_price=0.95, ask_size=10.0)
    out = post_only_clamp(qs, best_bid=None, best_ask=0.10, tick=TICK, band=BAND, margin_ticks=1)
    assert out is qs
    assert out.bid_size == 0.0
    assert out.bid_price == pytest.approx(0.90)


def test_zero_size_side_never_resurrected_ask():
    qs = _qs(bid_price=0.05, bid_size=10.0, ask_price=0.10, ask_size=0.0)
    out = post_only_clamp(qs, best_bid=0.90, best_ask=None, tick=TICK, band=BAND, margin_ticks=1)
    assert out is qs
    assert out.ask_size == 0.0
    assert out.ask_price == pytest.approx(0.10)


def test_incident_replay_consensus_rich_bid_clamped_to_venue_touch():
    """temp/mm_bleed2_fix_plan.md Tests(item 2): "consensus p 0.35, book
    0.21/0.23, desired bid 0.28 -> posted 0.22" -- the venue book from the
    mm-skew-oscillation-2026-08-13 diagnosis memory. The consensus/pricer
    math that produces a 0.28 desired bid from a 0.35 consensus is out of
    this pure function's scope (see the harness-level incident scenario in
    Section 2 for the full pipeline); here the clamp's own contract is
    checked directly against those exact numbers."""
    qs = _qs(bid_price=0.28, ask_price=0.40)
    out = post_only_clamp(qs, best_bid=0.21, best_ask=0.23, tick=TICK, band=BAND, margin_ticks=1)
    assert out.bid_price == pytest.approx(0.22)


def test_ladder_property_no_exploitable_arb_even_when_ask_monotonicity_breaks():
    """Cross-strike no-arb property (spread_builder.py:65 update): clamping
    a lower-strike bid down (or a higher-strike ask up) independently at
    each strike can reintroduce an ASK-ladder monotonicity wobble -- a
    lower-strike ask left unclamped while a higher-strike ask clamps UP past
    it -- but can NEVER create the EXPLOITABLE ask_K < bid_{K+1} crossing,
    because the clamp only ever moves prices OUTWARD (bid down, ask up) from
    an already bid < ask pair."""
    qs_98k = _qs(bid_price=0.70, ask_price=0.74)  # lower strike -> higher probability
    qs_100k = _qs(bid_price=0.50, ask_price=0.54)  # higher strike -> lower probability
    assert qs_98k.ask_price >= qs_100k.bid_price  # pre-clamp ladder is already sane

    # Force the HIGHER-strike (100k) ask up past the LOWER-strike (98k) ask,
    # left untouched -- an accepted ask-monotonicity wobble.
    clamped_100k = post_only_clamp(
        qs_100k, best_bid=0.76, best_ask=None, tick=TICK, band=BAND, margin_ticks=1,
    )
    assert clamped_100k.ask_price > qs_98k.ask_price  # monotonicity wobble (accepted)
    assert clamped_100k.bid_price < clamped_100k.ask_price  # per-strike sanity holds
    # The exploitable direction never breaks: the (unmoved) 98k ask still
    # dominates the (unmoved) 100k bid.
    assert qs_98k.ask_price >= qs_100k.bid_price


# ---------------------------------------------------------------------------
# Section 2: harness integration (scripted feeds)
# ---------------------------------------------------------------------------


def _engine(s0=S0, scale=2000.0, n_sims=15000):
    def fn(strikes, hours_to_expiry, **kwargs):
        out = {float(k): float(1.0 / (1.0 + np.exp((float(k) - s0) / scale))) for k in strikes}
        out["_meta"] = {"n_sims": n_sims, "S0": s0, "horizon_gate_active": False}
        return out
    return fn


class _VG:
    def __init__(self, regime="normal", shock=False, kelly_mult=1.0, edge_add_cents=0.0):
        self.regime = regime
        self.shock = shock
        self.kelly_mult = kelly_mult
        self.edge_add_cents = edge_add_cents


def _vol_gate():
    return lambda: _VG()


def _snapshot_msg(p, prints=None):
    bid = round(max(0.01, p - 0.03), 4)
    ask = round(min(0.99, p + 0.03), 4)
    msgs = [{
        "type": "snapshot",
        "bids": [(bid, 100.0), (round(bid - 0.01, 4), 100.0)],
        "asks": [(ask, 100.0), (round(ask + 0.01, 4), 100.0)],
    }]
    for pr in (prints or []):
        msgs.append({"type": "trade", "price": pr[0], "size": pr[1]})
    return msgs


def _thin_wing_book_msg(bid, ask, prints=None):
    """A hand-fixed, ultra-tight book, independent of the pricer's own
    probability -- used to force our own (much wider, consensus-centered)
    desired quote to cross the venue touch without needing to reverse-
    engineer the Beuoy consensus math."""
    msgs = [{
        "type": "snapshot",
        "bids": [(bid, 100.0)],
        "asks": [(ask, 100.0)],
    }]
    for pr in (prints or []):
        msgs.append({"type": "trade", "price": pr[0], "size": pr[1]})
    return msgs


@pytest.fixture
def store(tmp_path):
    s = MMStateStore(str(tmp_path / "mm.db"))
    yield s
    s.close()


def test_harness_clamp_applied_post_repair_last_quote_sets_reflects_clamp(store):
    """A thin wing book (best_bid=0.005/best_ask=0.015) far below the
    ATM-centered desired quote forces the bid side to clamp into the
    degenerate range; last_quote_sets (post-repair, post-clamp, post-skew)
    must show the zeroed size, and the (unchanged) price is the pre-clamp
    valid value (rule 5)."""
    cfg = MMConfig()  # post_only_margin_ticks default 1 (ACTIVE)
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=[("m-100k", 100000.0)],
        engine_fn=_engine(), config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    loop.tick({"m-100k": _thin_wing_book_msg(0.005, 0.015)})

    qs = loop.last_quote_sets["m-100k"]
    assert qs.bid_size == 0.0  # clamp pushed it into the degenerate range
    assert "post_only_bid" not in qs.terms  # zeroing, not a price move -- see pure-fn test
    # ask side was never near the venue touch -> untouched.
    assert qs.ask_size > 0.0


def test_harness_clamp_zeroed_side_produces_no_resting_order(store):
    cfg = MMConfig()
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=[("m-100k", 100000.0)],
        engine_fn=_engine(), config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    loop.tick({"m-100k": _thin_wing_book_msg(0.005, 0.015)})
    assert loop.last_quote_sets["m-100k"].bid_size == 0.0

    live_bids = store.get_live_orders("m-100k", Side.BUY_YES)
    assert live_bids == []


def test_zeroed_side_stays_zeroed_through_apply_hedge_skew(store):
    """The clamp's own degenerate zeroing composes with the PRE-EXISTING
    hedge-skew suppressed-side precedence (harness._apply_hedge_skew,
    'never resurrect a suppressed side'): a QuoteSet the clamp already
    zeroed must not be resurrected by a hedge recommendation targeting that
    exact side."""
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=MARKETS, engine_fn=_engine(),
        config=MMConfig(), clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    now = loop.clock.now()
    market_id, other_market = "m-100k", "m-102k"

    from dataclasses import replace

    clamp_zeroed = post_only_clamp(
        _qs(bid_price=0.50, ask_price=0.60), best_bid=None, best_ask=0.015,
        tick=TICK, band=BAND, margin_ticks=1,
    )
    clamp_zeroed = replace(clamp_zeroed, market_id=market_id)  # _qs() defaults to "m1"
    assert clamp_zeroed.bid_size == 0.0

    rec = HedgeRecommendation(
        ts=now, expiry_key=EXPIRY, target_market_id=market_id, side=Side.BUY_YES,
        size=5.0, max_price=0.95, reason=HedgeReason.VERTICAL_OFFSET,
        paired_market_id=other_market, beta=None, expires=now + timedelta(seconds=300.0),
    )
    loop._pending_hedge_recs = [rec]
    out = loop._apply_hedge_skew([clamp_zeroed], now)

    assert out[0].bid_size == 0.0
    assert loop.hedge_journal[-1]["applied"] is False
    assert loop.hedge_journal[-1]["reason"] == "suppressed_side"


def test_multi_loop_clamp_reads_only_its_own_market_states(store, tmp_path):
    """Multi-expiry note (plan Tests(item 2)): the clamp reads market_states
    from the CURRENT tick's local dict built by THIS loop -- two independent
    loops (standing in for two LadderSlots) with different scripted books
    for the same market_id string must clamp independently, proving no
    cross-loop leakage."""
    store_b = MMStateStore(str(tmp_path / "mm_b.db"))
    try:
        cfg = MMConfig()
        loop_thin = PaperTradingLoop(
            store=store, expiry_key=EXPIRY, markets=[("m-100k", 100000.0)],
            engine_fn=_engine(), config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
        )
        loop_wide = PaperTradingLoop(
            store=store_b, expiry_key=EXPIRY, markets=[("m-100k", 100000.0)],
            engine_fn=_engine(), config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
        )
        loop_thin.tick({"m-100k": _thin_wing_book_msg(0.005, 0.015)})
        loop_wide.tick({"m-100k": _snapshot_msg(0.5)})

        assert loop_thin.last_quote_sets["m-100k"].bid_size == 0.0
        assert loop_wide.last_quote_sets["m-100k"].bid_size > 0.0
    finally:
        store_b.close()


def test_regression_margin_zero_never_invokes_clamp(store, monkeypatch):
    """post_only_margin_ticks=0 (disabled sentinel) must be byte-identical
    to the pre-item-2 pipeline: the clamp function is never even called."""
    import market_maker.harness as harness_mod

    calls = {"n": 0}
    orig = harness_mod.post_only_clamp

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(harness_mod, "post_only_clamp", spy)

    cfg = MMConfig(post_only_margin_ticks=0)
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=[("m-100k", 100000.0)],
        engine_fn=_engine(), config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    loop.tick({"m-100k": _thin_wing_book_msg(0.005, 0.015)})

    assert calls["n"] == 0
    # And the (legacy) crossed price is exactly what sizing/build_quote_set
    # produced -- proving the pipeline ran the old, un-clamped path (at
    # margin_ticks=1 this same book zeroes bid_size entirely, see
    # test_harness_clamp_zeroed_side_produces_no_resting_order).
    qs = loop.last_quote_sets["m-100k"]
    assert qs.bid_size > 0.0
    assert qs.bid_price > 0.015  # crosses the thin book's best_ask -- legacy behavior


def test_fill_sim_smoke_clamped_bid_does_not_fill_on_print_above_clamped_price(store):
    """With the clamp active, a print strictly between the CLAMPED resting
    price and the pre-clamp DESIRED (crossing) price must NOT fill -- that
    print would have touched the pre-item-2 (unclamped, crossed) resting
    order, reproducing the aug-16-shape incident this item removes. A print
    that actually touches the clamped resting price fills normally."""
    cfg = MMConfig()
    loop = PaperTradingLoop(
        store=store, expiry_key=EXPIRY, markets=[("m-100k", 100000.0)],
        engine_fn=_engine(), config=cfg, clock=SimClock(START), vol_gate_fn=_vol_gate(),
    )
    # A moderately (not degenerately) thin book: our own ATM-centered desired
    # bid crosses it, but the clamped result still lands well inside the
    # venue band (non-degenerate), so the order actually rests and can fill.
    book = _thin_wing_book_msg(0.30, 0.32)

    # Warm-up ticks (mirrors test_mm_integration.py's happy-path convention):
    # placement_latency_ms elapses between decision and a fillable order.
    loop.tick({"m-100k": book})
    loop.tick({"m-100k": book})
    qs = loop.last_quote_sets["m-100k"]
    assert qs.bid_size > 0.0, "expected a non-degenerate clamp for this scenario"
    assert "post_only_bid" in qs.terms, "expected the clamp to actually fire"

    clamped_bid = qs.bid_price
    would_be_unclamped_bid = clamped_bid + qs.terms["post_only_bid"]
    probe_price = (clamped_bid + would_be_unclamped_bid) / 2.0
    assert clamped_bid < probe_price < would_be_unclamped_bid

    # A print strictly ABOVE the resting (clamped) price does not touch it.
    loop.tick({"m-100k": _thin_wing_book_msg(0.30, 0.32, prints=[(probe_price, 50.0)])})
    fills_probe = [f for f in loop.last_fills if f.market_id == "m-100k" and f.side == Side.BUY_YES]
    assert fills_probe == []

    # A print AT the resting (clamped) price fills normally, at that price.
    loop.tick({"m-100k": _thin_wing_book_msg(0.30, 0.32, prints=[(clamped_bid, 50.0)])})
    fills_touch = [f for f in loop.last_fills if f.market_id == "m-100k" and f.side == Side.BUY_YES]
    assert len(fills_touch) >= 1
    assert fills_touch[0].price == pytest.approx(clamped_bid)


def test_state_store_quotes_round_trip_with_post_only_terms(tmp_path):
    """Mirrors tests/test_mm_state_store.py::test_quotes_round_trip, with
    post_only_bid landing in the JSON terms column (plan Tests(item 2))."""
    s = MMStateStore(str(tmp_path / "mm.db"))
    try:
        qs = post_only_clamp(
            _qs(bid_price=0.65, ask_price=0.75), best_bid=0.60, best_ask=0.62,
            tick=TICK, band=BAND, margin_ticks=1,
        )
        assert "post_only_bid" in qs.terms
        row_id = s.append_quote(
            qs, r_x=0.1, delta_x=0.05, skew_x=0.0, sigma_b=0.2,
            params_id="cfg-v1", x_bid=0.05, x_ask=0.15, p_bid_raw=0.65, p_ask_raw=0.75,
        )
        assert row_id > 0
        got = s.get_quotes("m1")
        assert len(got) == 1
        assert got[0].quote_set == qs
        assert got[0].quote_set.terms["post_only_bid"] == pytest.approx(qs.terms["post_only_bid"])
    finally:
        s.close()
