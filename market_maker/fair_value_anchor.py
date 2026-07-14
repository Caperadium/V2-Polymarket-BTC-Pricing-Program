"""Fair-value anchor (plan task F1, Section 2.3) -- Beuoy bankroll-credibility.

Produces the ladder-consistent consensus fair value the quote engine centers on,
plus a live pricer-credibility score, by treating each model (pricer, market,
extensible) as a Kelly bettor with a bankroll = credibility.

DESIGN CHOICE (documented per the plan's V2 gate). The plan transcribes Beuoy's
eigenvector construction as INFORMATIVE but marks it non-normative and BINDS five
invariants instead (2.3). This module implements the simplest construction that
provably satisfies all five: the bankroll-weighted consensus is, per mutually
exclusive bucket, `consensus_bucket = sum_i(w_i * p_i_bucket)` with w_i the
normalized bankrolls. This is exactly the fixed point the plan's caution note
derives (mp/(1+mp) = sum(p_i w_i) under normalized bankrolls): a convex
combination of the per-model bucket distributions, which is bounded per bucket by
[min_i, max_i], nonnegative, sums to 1, and reproduces unanimity exactly. A user
review of the Beuoy primary (V2) may refine the algebra later; the invariants and
the FairValue / BankrollState interface will NOT change.

Bankroll update (Beuoy/Bayes, no resolution needed). On each refresh, each model's
bankroll is marked to market against the NEW consensus distribution (the realized
proxy) under its PREVIOUS forecast:
    b_i_new proportional to b_i * sum_buckets(
        consensus_new_bucket * p_i_prev_bucket / consensus_prev_bucket)
then normalized and floored (module default bankroll_floor = 0.02) to prevent
permanent zero-credibility lock-in. Callers thread the previous forecasts /
consensus and the BankrollState across refreshes (returned in AnchorResult).

Degeneracy fallback (risk 8.8). On any failure -- non-finite inputs, all-zero
bankrolls, or a per-strike sanity-bound violation (consensus must lie between the
SANITIZED pricer and mid ladders, i.e. each raw ladder round-tripped through
ladder_to_buckets/buckets_to_ladder -- the bucket transform repairs non-monotone
inputs, e.g. crossed venue mids, rather than triggering the fallback; the band
check is now a numeric safety net) -- the anchor falls back to a fixed w=0.5
blend of pricer and mid per strike, freezes the bankrolls
(BankrollState.frozen=True), tags AnchorMethod.FIXED_BLEND_FALLBACK, and logs a
warning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from market_maker import logodds
from market_maker.contracts import AnchorMethod, BankrollState, FairValue

logger = logging.getLogger(__name__)

# Bankroll floor if MMConfig does not supply one (plan open question 9).
DEFAULT_BANKROLL_FLOOR: float = 0.02

# Model identity for the credibility score.
PRICER_MODEL_ID: str = "pricer"
MARKET_MODEL_ID: str = "market"

# Per-strike sanity-bound tolerance before falling back.
_SANITY_TOL: float = 1.0e-6


@dataclass
class AnchorResult:
    """Anchor output plus the state a caller threads to the next refresh."""

    fair_value: FairValue
    bankroll_state: BankrollState
    forecasts: Dict[str, np.ndarray]  # per-model bucket vectors (feed as prev next)
    consensus_bucket: np.ndarray  # consensus bucket vector (feed as prev next)


# ---------------------------------------------------------------------------
# Ladder <-> bucket helpers
# ---------------------------------------------------------------------------


def ladder_to_buckets(p_ladder: np.ndarray) -> np.ndarray:
    """CDF-complement ladder p(K_1..K_n) -> n+1 mutually exclusive buckets.

    buckets = [1-p(K_1), p(K_1)-p(K_2), ..., p(K_{n-1})-p(K_n), p(K_n)].
    Negatives (from non-monotone inputs) are clipped to 0 and the result is
    renormalized to sum to 1.
    """
    p = np.asarray(p_ladder, dtype=float)
    n = p.size
    buckets = np.empty(n + 1, dtype=float)
    buckets[0] = 1.0 - p[0]
    for j in range(1, n):
        buckets[j] = p[j - 1] - p[j]
    buckets[n] = p[n - 1]
    buckets = np.clip(buckets, 0.0, None)
    total = buckets.sum()
    if total <= 0.0 or not np.isfinite(total):
        # Degenerate ladder -> uniform (caller's sanity check handles the rest).
        return np.full(n + 1, 1.0 / (n + 1))
    return buckets / total


def buckets_to_ladder(buckets: np.ndarray) -> np.ndarray:
    """n+1 buckets -> monotone non-increasing CDF-complement p(K_1..K_n).

    p(K_j) = sum(buckets[j+1:]) for j=0..n-1. Enforced non-increasing and clipped
    to [0,1] against floating-point drift.
    """
    b = np.asarray(buckets, dtype=float)
    n = b.size - 1
    p = np.empty(n, dtype=float)
    for j in range(n):
        p[j] = b[j + 1:].sum()
    # Enforce monotone non-increasing + [0,1].
    p = np.clip(p, 0.0, 1.0)
    for j in range(1, n):
        if p[j] > p[j - 1]:
            p[j] = p[j - 1]
    return p


# ---------------------------------------------------------------------------
# Bankroll bookkeeping
# ---------------------------------------------------------------------------


def _normalized_weights(model_ids: List[str], bankrolls: Dict[str, float]) -> Optional[np.ndarray]:
    """Return normalized weight vector over model_ids, or None if degenerate."""
    vals = np.array([float(bankrolls.get(m, 0.0)) for m in model_ids], dtype=float)
    if not np.all(np.isfinite(vals)) or np.any(vals < 0.0):
        return None
    total = vals.sum()
    if total <= 0.0:
        return None
    return vals / total


def _apply_floor(weights: np.ndarray, floor: float) -> np.ndarray:
    """Hard-clip floor: raise sub-floor shares to `floor`, take the deficit from
    the above-floor shares pro rata, and leave shares that already clear the floor
    untouched (so the floor perturbs weights only when it actually binds). Result
    is nonnegative and sums to 1.
    """
    w = np.asarray(weights, dtype=float).copy()
    m = w.size
    floor = max(0.0, floor)
    if m == 0 or m * floor >= 1.0:
        return np.full(m, 1.0 / m) if m > 0 else w
    below = w < floor
    if not below.any():
        return w
    above = ~below
    remaining = 1.0 - floor * float(below.sum())
    above_sum = float(w[above].sum())
    out = np.empty_like(w)
    out[below] = floor
    if above_sum > 0.0 and remaining > 0.0:
        out[above] = w[above] * (remaining / above_sum)
    else:
        out[above] = remaining / max(int(above.sum()), 1)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_fair_value(
    snapshot,
    mids: Dict[float, float],
    bankroll_state: BankrollState,
    config,
    market_ts: Optional[datetime] = None,
    prev_forecasts: Optional[Dict[str, np.ndarray]] = None,
    prev_consensus: Optional[np.ndarray] = None,
    ts: Optional[datetime] = None,
) -> AnchorResult:
    """Compute the Beuoy consensus FairValue and updated BankrollState.

    snapshot: PricerSnapshot (uses .strikes, .p_hat, .ts, .expiry_key).
    mids: market mid probability per strike.
    prev_forecasts / prev_consensus: previous refresh's model bucket vectors and
    consensus bucket vector; when present (and not frozen) the bankrolls are
    marked to market before the consensus is formed.
    """
    now = ts if ts is not None else datetime.now(timezone.utc)
    m_ts = market_ts if market_ts is not None else getattr(snapshot, "ts", now)
    floor = float(getattr(config, "bankroll_floor", DEFAULT_BANKROLL_FLOOR))

    strikes = sorted(float(s) for s in snapshot.strikes)
    pricer_p = np.array([snapshot.p_hat.get(K, np.nan) for K in strikes], dtype=float)
    market_p = np.array([mids.get(K, np.nan) for K in strikes], dtype=float)

    model_ids = list(bankroll_state.model_ids)
    if PRICER_MODEL_ID not in model_ids or MARKET_MODEL_ID not in model_ids:
        model_ids = [PRICER_MODEL_ID, MARKET_MODEL_ID]
    raw_ladders = {PRICER_MODEL_ID: pricer_p, MARKET_MODEL_ID: market_p}

    # --- degeneracy pre-checks (inputs) ---
    inputs_bad = (
        len(strikes) == 0
        or not np.all(np.isfinite(pricer_p))
        or not np.all(np.isfinite(market_p))
        or np.any(pricer_p < 0.0)
        or np.any(pricer_p > 1.0)
        or np.any(market_p < 0.0)
        or np.any(market_p > 1.0)
    )

    weights = _normalized_weights(model_ids, bankroll_state.bankrolls)
    if inputs_bad or weights is None:
        return _fallback(
            snapshot, strikes, raw_ladders, model_ids, bankroll_state,
            weights, now, m_ts, reason="non-finite inputs or degenerate bankrolls",
        )

    # --- per-model bucket forecasts ---
    forecasts = {mid: ladder_to_buckets(raw_ladders[mid]) for mid in model_ids}

    # --- bankroll mark-to-market (Bayes) using previous forecasts/consensus ---
    new_bankrolls = dict(bankroll_state.bankrolls)
    update_count = bankroll_state.update_count
    if (
        not bankroll_state.frozen
        and prev_forecasts is not None
        and prev_consensus is not None
    ):
        consensus_new = np.zeros_like(forecasts[model_ids[0]])
        for i, mid in enumerate(model_ids):
            consensus_new = consensus_new + weights[i] * forecasts[mid]
        cprev = np.asarray(prev_consensus, dtype=float)
        if (
            consensus_new.shape == cprev.shape
            and np.all(cprev > 0.0)
            and np.all(np.isfinite(cprev))
        ):
            factors = np.ones(len(model_ids), dtype=float)
            ok = True
            for i, mid in enumerate(model_ids):
                pi_prev = np.asarray(prev_forecasts.get(mid), dtype=float)
                if pi_prev.shape != cprev.shape or not np.all(np.isfinite(pi_prev)):
                    ok = False
                    break
                factors[i] = float(np.sum(consensus_new * pi_prev / cprev))
            if ok and np.all(np.isfinite(factors)) and np.all(factors > 0.0):
                updated = weights * factors
                s = updated.sum()
                if s > 0.0 and np.isfinite(s):
                    w_upd = _apply_floor(updated / s, floor)
                    for i, mid in enumerate(model_ids):
                        new_bankrolls[mid] = float(w_upd[i])
                    weights = w_upd
                    update_count += 1

    # --- consensus in bucket space (post-update weights) ---
    consensus_bucket = np.zeros_like(forecasts[model_ids[0]])
    for i, mid in enumerate(model_ids):
        consensus_bucket = consensus_bucket + weights[i] * forecasts[mid]
    s = consensus_bucket.sum()
    if s > 0.0 and np.isfinite(s):
        consensus_bucket = consensus_bucket / s

    consensus_p = buckets_to_ladder(consensus_bucket)

    # --- sanity bound: consensus must lie between the SANITIZED (bucket-round-
    # tripped) pricer and mid ladders, not the raw inputs -- ladder_to_buckets
    # clips/renormalizes non-monotone inputs (e.g. crossed venue mids), and the
    # consensus is provably a convex combination of these sanitized ladders, so
    # this check is now a numeric safety net rather than a real trigger path.
    recon = {mid: buckets_to_ladder(forecasts[mid]) for mid in model_ids}
    for mid in model_ids:
        deviation = float(np.max(np.abs(raw_ladders[mid] - recon[mid])))
        if deviation > 1e-9:
            logger.debug(
                "fair_value_anchor: model %s input ladder was non-monotone and "
                "sanitized by the bucket transform (max deviation %.6g)",
                mid, deviation,
            )
    lo_band = np.minimum.reduce([recon[m] for m in model_ids]) - _SANITY_TOL
    hi_band = np.maximum.reduce([recon[m] for m in model_ids]) + _SANITY_TOL
    if (
        not np.all(np.isfinite(consensus_p))
        or np.any(consensus_p < lo_band)
        or np.any(consensus_p > hi_band)
    ):
        return _fallback(
            snapshot, strikes, raw_ladders, model_ids, bankroll_state,
            weights, now, m_ts, reason="per-strike sanity-bound violation",
        )

    credibility = float(weights[model_ids.index(PRICER_MODEL_ID)])
    fv = _build_fair_value(
        snapshot, strikes, consensus_p, credibility,
        AnchorMethod.BEUOY, now, m_ts, config,
    )
    new_state = BankrollState(
        model_ids=model_ids,
        bankrolls=new_bankrolls,
        last_update=now,
        update_count=update_count,
        frozen=bankroll_state.frozen,
    )
    return AnchorResult(fv, new_state, forecasts, consensus_bucket)


# ---------------------------------------------------------------------------
# Fallback + FairValue construction
# ---------------------------------------------------------------------------


def _fallback(
    snapshot, strikes, raw_ladders, model_ids, bankroll_state,
    weights, now, m_ts, reason,
) -> AnchorResult:
    """FIXED_BLEND w=0.5 per strike; freeze bankrolls; warn (risk 8.8)."""
    logger.warning("fair_value_anchor degeneracy -> FIXED_BLEND_FALLBACK: %s", reason)
    pricer_p = raw_ladders[PRICER_MODEL_ID]
    market_p = raw_ladders[MARKET_MODEL_ID]

    def _san(a):
        a = np.asarray(a, dtype=float)
        return np.where(np.isfinite(a), np.clip(a, 0.0, 1.0), 0.5)

    blend = 0.5 * _san(pricer_p) + 0.5 * _san(market_p)
    # Enforce monotone non-increasing for a valid ladder.
    for j in range(1, blend.size):
        if blend[j] > blend[j - 1]:
            blend[j] = blend[j - 1]

    if weights is not None:
        credibility = float(weights[model_ids.index(PRICER_MODEL_ID)])
    else:
        credibility = 0.5

    fv = _build_fair_value(
        snapshot, strikes, blend, credibility,
        AnchorMethod.FIXED_BLEND_FALLBACK, now, m_ts, None,
    )
    frozen_state = BankrollState(
        model_ids=list(bankroll_state.model_ids),
        bankrolls=dict(bankroll_state.bankrolls),
        last_update=now,
        update_count=bankroll_state.update_count,
        frozen=True,
    )
    n = len(strikes)
    forecasts = {m: ladder_to_buckets(raw_ladders.get(m, np.zeros(n))) for m in model_ids}
    return AnchorResult(fv, frozen_state, forecasts, ladder_to_buckets(blend))


def _build_fair_value(
    snapshot, strikes, consensus_p, credibility, method, now, m_ts, config,
) -> FairValue:
    p_lo, p_hi = (config.p_clamp if config is not None else (logodds.DEFAULT_P_LO, logodds.DEFAULT_P_HI))
    cons_p = {}
    cons_x = {}
    for K, p in zip(strikes, np.asarray(consensus_p, dtype=float)):
        pv = float(np.clip(p, 0.0, 1.0)) if np.isfinite(p) else 0.5
        cons_p[K] = pv
        cons_x[K] = float(logodds.logit(pv, p_lo, p_hi))
    return FairValue(
        ts=now,
        expiry_key=snapshot.expiry_key,
        consensus_p=cons_p,
        consensus_x=cons_x,
        credibility=float(min(max(credibility, 0.0), 1.0)),
        anchor_method=method,
        inputs_ts=(getattr(snapshot, "ts", now), m_ts),
    )
