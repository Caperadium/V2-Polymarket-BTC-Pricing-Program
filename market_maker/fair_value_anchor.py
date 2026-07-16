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
blend of pricer and mid per strike, freezes BOTH region bankrolls
(BankrollState.frozen=True on each), tags AnchorMethod.FIXED_BLEND_FALLBACK, and
logs a warning.

PER-REGION BANKROLLS (package B2, 2026-07-15). Model skill is region-dependent
(belly decent, tails measurably rich -- see temp/mm_pnl_fix_plan.md section 0/2).
A single scalar credibility per expiry let a belly win give the pricer unchecked
wing authority. This module now tracks TWO independent BankrollStates per expiry,
keyed "belly"/"wing" (module constants BELLY_REGION/WING_REGION), so the market
can regain wing weight where the pricer keeps being marked down -- a mitigation
of the fair-value skew, not a root-cause fix (pricer tail recalibration, package
C direction, is out of scope here).

Region assignment (plan step 1/2): each STRIKE is classified belly/wing from the
SANITIZED MARKET ladder (the market's own mid, bucket-round-tripped) via
`config.in_belly_band` -- never from the consensus being built (avoids
self-reference) and never from the pricer (a rich pricer tail must not be able to
reclassify a strike into the region where it holds more credibility). n strikes
give n+1 mutually-exclusive buckets; buckets 0 and n (the open tails) are ALWAYS
"wing" regardless of the extreme strikes' own classification -- interior bucket j
(1<=j<=n-1) takes the region of its LEFT strike (strike j-1).

Two-phase bankroll update per refresh (plan step 3, binding sequence):
  1. Build the PRE-update consensus (ladder space, PRE-update per-region weights,
     see `_ladder_space_consensus`) -> `ladder_to_buckets` -> consensus_new_bucket.
  2. Per region R, per model i, a REGION-RESTRICTED Bayes factor: the same ratio
     test as the single-region algorithm, summed only over region R's buckets.
  3. Empty/degenerate region rule (BLOCKER-resolution, do NOT deviate): a region
     with zero assigned buckets, or whose factor sum s_R <= 0 or non-finite,
     SKIPS ITS OWN UPDATE -- weights and update_count unchanged for that region,
     NO fallback, NO freeze. (Tail buckets are always wing, so `wing` is never
     bucket-empty; `belly` can be, on an all-wing-strike ladder.)
  4. Update each non-skipped region's weights: normalize within region, apply
     `_apply_floor` per region, increment that region's update_count.
  5. Build the FINAL consensus with POST-update weights (same ladder-space
     pipeline); this is the FairValue. Its bucket form is threaded as the next
     refresh's `prev_consensus`.

Ladder-space consensus construction (plan step 4, `_ladder_space_consensus`):
per-strike weighted blend of the SANITIZED ladders, THEN (a) cummin repair
(enforce non-increasing, the `_fallback`-style loop), THEN (b) pointwise clamp
into [lo_band, hi_band] = [min, max] of the two sanitized ladders +- _SANITY_TOL.
Order is binding: clamp-then-repair is wrong (repair can push values back out of
band); repair-then-clamp preserves monotonicity (clamp is monotone in all three
args and both envelopes are non-increasing). The existing sanity-band check
(below) therefore passes by construction and remains only a numeric safety net.

FairValue.credibility is the strike-count-weighted average of the two regions'
pricer credibilities (FairValue.credibility_by_region carries both, additively).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from market_maker import logodds
from market_maker.config import in_belly_band
from market_maker.contracts import AnchorMethod, BankrollState, FairValue

logger = logging.getLogger(__name__)

# Bankroll floor if MMConfig does not supply one (plan open question 9).
DEFAULT_BANKROLL_FLOOR: float = 0.02

# Model identity for the credibility score.
PRICER_MODEL_ID: str = "pricer"
MARKET_MODEL_ID: str = "market"

# Region identities (package B2). AnchorResult/harness key their per-region
# bankroll dicts on these two strings; no other region names are supported.
BELLY_REGION: str = "belly"
WING_REGION: str = "wing"
REGIONS: Tuple[str, str] = (BELLY_REGION, WING_REGION)

# Per-strike sanity-bound tolerance before falling back.
_SANITY_TOL: float = 1.0e-6


@dataclass
class AnchorResult:
    """Anchor output plus the state a caller threads to the next refresh."""

    fair_value: FairValue
    bankroll_states: Dict[str, BankrollState]  # keyed BELLY_REGION/WING_REGION
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


def _bucket_region_map(region_of_strike: List[str]) -> List[str]:
    """n strikes -> n+1 buckets; bucket 0 and bucket n are ALWAYS wing (the
    open tails); interior bucket j (1<=j<=n-1) takes the region of its LEFT
    strike (strike j-1). Plan step 2, tail buckets pinned."""
    n = len(region_of_strike)
    bucket_region = [WING_REGION] * (n + 1)
    for j in range(1, n):
        bucket_region[j] = region_of_strike[j - 1]
    return bucket_region


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


def _weight_dict(w_arr: np.ndarray, model_ids: List[str]) -> Dict[str, float]:
    return {mid: float(w_arr[i]) for i, mid in enumerate(model_ids)}


# ---------------------------------------------------------------------------
# Ladder-space consensus (plan step 4)
# ---------------------------------------------------------------------------


def _ladder_space_consensus(
    weight_dicts_by_region: Dict[str, Dict[str, float]],
    region_of_strike: List[str],
    sanitized: Dict[str, np.ndarray],
    model_ids: List[str],
) -> np.ndarray:
    """Per-strike weighted blend of the SANITIZED ladders using each strike's
    region weight, THEN (a) cummin repair (enforce non-increasing), THEN
    (b) pointwise clamp into [lo_band, hi_band] = [min, max] of the two
    sanitized ladders +- _SANITY_TOL. ORDER IS BINDING (plan step 4): clamp-
    then-repair is wrong (repair can push values back out of band);
    repair-then-clamp preserves monotonicity."""
    n = len(region_of_strike)
    raw = np.empty(n, dtype=float)
    for k in range(n):
        w = weight_dicts_by_region[region_of_strike[k]]
        raw[k] = sum(w[mid] * sanitized[mid][k] for mid in model_ids)
    for j in range(1, n):
        if raw[j] > raw[j - 1]:
            raw[j] = raw[j - 1]
    stacked = np.stack([sanitized[mid] for mid in model_ids]) if model_ids else np.zeros((0, n))
    lo = np.min(stacked, axis=0) - _SANITY_TOL
    hi = np.max(stacked, axis=0) + _SANITY_TOL
    return np.clip(raw, lo, hi)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_fair_value(
    snapshot,
    mids: Dict[float, float],
    bankroll_states: Dict[str, BankrollState],
    config,
    market_ts: Optional[datetime] = None,
    prev_forecasts: Optional[Dict[str, np.ndarray]] = None,
    prev_consensus: Optional[np.ndarray] = None,
    ts: Optional[datetime] = None,
) -> AnchorResult:
    """Compute the Beuoy consensus FairValue and updated per-region BankrollStates.

    snapshot: PricerSnapshot (uses .strikes, .p_hat, .ts, .expiry_key).
    mids: market mid probability per strike.
    bankroll_states: dict keyed BELLY_REGION/WING_REGION (module constants).
    prev_forecasts / prev_consensus: previous refresh's model bucket vectors and
    consensus bucket vector (SHARED across regions -- forecasts are per-model,
    not per-region); when present (and a region is not frozen) that region's
    bankroll is marked to market before the final consensus is formed.
    """
    now = ts if ts is not None else datetime.now(timezone.utc)
    m_ts = market_ts if market_ts is not None else getattr(snapshot, "ts", now)
    floor = float(getattr(config, "bankroll_floor", DEFAULT_BANKROLL_FLOOR))
    belly_band = getattr(config, "belly_band", (0.2, 0.8))

    strikes = sorted(float(s) for s in snapshot.strikes)
    n = len(strikes)
    pricer_p = np.array([snapshot.p_hat.get(K, np.nan) for K in strikes], dtype=float)
    market_p = np.array([mids.get(K, np.nan) for K in strikes], dtype=float)

    belly_state = bankroll_states[BELLY_REGION]
    wing_state = bankroll_states[WING_REGION]

    model_ids = list(belly_state.model_ids)
    if PRICER_MODEL_ID not in model_ids or MARKET_MODEL_ID not in model_ids:
        model_ids = [PRICER_MODEL_ID, MARKET_MODEL_ID]
    raw_ladders = {PRICER_MODEL_ID: pricer_p, MARKET_MODEL_ID: market_p}

    # --- degeneracy pre-checks (inputs) ---
    inputs_bad = (
        n == 0
        or not np.all(np.isfinite(pricer_p))
        or not np.all(np.isfinite(market_p))
        or np.any(pricer_p < 0.0)
        or np.any(pricer_p > 1.0)
        or np.any(market_p < 0.0)
        or np.any(market_p > 1.0)
    )

    weights_belly = _normalized_weights(model_ids, belly_state.bankrolls)
    weights_wing = _normalized_weights(model_ids, wing_state.bankrolls)
    weights_by_region_arr = {BELLY_REGION: weights_belly, WING_REGION: weights_wing}
    if inputs_bad or weights_belly is None or weights_wing is None:
        return _fallback(
            snapshot, strikes, raw_ladders, model_ids, bankroll_states,
            weights_by_region_arr, now, m_ts, reason="non-finite inputs or degenerate bankrolls",
        )

    # --- per-model bucket forecasts (shared across regions) ---
    forecasts = {mid: ladder_to_buckets(raw_ladders[mid]) for mid in model_ids}
    recon = {mid: buckets_to_ladder(forecasts[mid]) for mid in model_ids}
    for mid in model_ids:
        deviation = float(np.max(np.abs(raw_ladders[mid] - recon[mid])))
        if deviation > 1e-9:
            logger.debug(
                "fair_value_anchor: model %s input ladder was non-monotone and "
                "sanitized by the bucket transform (max deviation %.6g)",
                mid, deviation,
            )

    # --- Step 1/2: region per strike from the SANITIZED MARKET ladder; then
    # the bucket -> region map (tail buckets always wing). ---
    market_san = recon[MARKET_MODEL_ID]
    region_of_strike = [
        BELLY_REGION if in_belly_band(float(market_san[i]), belly_band) else WING_REGION
        for i in range(n)
    ]
    bucket_region = _bucket_region_map(region_of_strike)

    weight_dicts_pre = {
        BELLY_REGION: _weight_dict(weights_belly, model_ids),
        WING_REGION: _weight_dict(weights_wing, model_ids),
    }
    # Default: unchanged unless a region's update fires below (Step 3.3/3.4).
    weight_dicts_post = dict(weight_dicts_pre)

    new_bankrolls_by_region = {
        BELLY_REGION: dict(belly_state.bankrolls),
        WING_REGION: dict(wing_state.bankrolls),
    }
    update_counts = {
        BELLY_REGION: belly_state.update_count,
        WING_REGION: wing_state.update_count,
    }

    # --- Step 3: two-phase bankroll mark-to-market (Bayes) using previous
    # forecasts/consensus, only when history is available. ---
    if prev_forecasts is not None and prev_consensus is not None:
        # 3.1: PRE-update consensus (ladder space, PRE-update per-region weights).
        consensus_new_ladder = _ladder_space_consensus(
            weight_dicts_pre, region_of_strike, recon, model_ids,
        )
        consensus_new_bucket = ladder_to_buckets(consensus_new_ladder)
        cprev = np.asarray(prev_consensus, dtype=float)
        # Shape is a hard data-integrity precondition (gates BOTH regions --
        # a ladder-width change between ticks means the threaded history is
        # simply not comparable). Divisor positivity/finiteness is checked
        # PER-REGION below (restricted to that region's own bucket indices,
        # `idxs`): a zero/degenerate bucket belonging to the OTHER region
        # must not block this region's otherwise-valid update (plan step
        # 3.3 is a per-region rule, not a whole-ladder one). A zero or
        # non-finite cprev[idxs] entry naturally produces a non-finite
        # factor below (0/0 -> nan, x/0 -> inf), which the existing
        # non-finite-factor check then routes to the same per-region skip.
        if consensus_new_bucket.shape == cprev.shape:
            for region in REGIONS:
                state = bankroll_states[region]
                if state.frozen:
                    continue
                idxs = [j for j in range(n + 1) if bucket_region[j] == region]
                if not idxs:
                    continue  # 3.3: zero assigned buckets -> skip, no fallback
                factors = np.ones(len(model_ids), dtype=float)
                ok = True
                for i, mid in enumerate(model_ids):
                    pi_prev = np.asarray(prev_forecasts.get(mid), dtype=float)
                    if pi_prev.shape != cprev.shape or not np.all(np.isfinite(pi_prev)):
                        ok = False
                        break
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = consensus_new_bucket[idxs] * pi_prev[idxs] / cprev[idxs]
                    factors[i] = float(np.sum(ratio))
                if not ok or not np.all(np.isfinite(factors)):
                    continue  # 3.3: non-finite factor (incl. a zero/degenerate
                    # divisor restricted to this region's own buckets) -> skip
                w_pre_arr = np.array([weight_dicts_pre[region][mid] for mid in model_ids])
                updated = w_pre_arr * factors
                s = updated.sum()
                if not (s > 0.0 and np.isfinite(s)):
                    continue  # 3.3: s_R <= 0 or non-finite -> skip
                w_upd = _apply_floor(updated / s, floor)
                new_bankrolls = _weight_dict(w_upd, model_ids)
                new_bankrolls_by_region[region] = new_bankrolls
                weight_dicts_post[region] = new_bankrolls
                update_counts[region] += 1

    # --- Step 3.5: FINAL consensus with POST-update weights (same pipeline). ---
    final_ladder = _ladder_space_consensus(weight_dicts_post, region_of_strike, recon, model_ids)
    consensus_bucket = ladder_to_buckets(final_ladder)
    consensus_p = final_ladder

    # --- Sanity bound: numeric safety net only. By construction (the clamp
    # inside _ladder_space_consensus) this cannot fire under normal operation --
    # kept as a defensive check in case a future refactor breaks that guarantee.
    lo_band = np.minimum.reduce([recon[m] for m in model_ids]) - _SANITY_TOL
    hi_band = np.maximum.reduce([recon[m] for m in model_ids]) + _SANITY_TOL
    if (
        not np.all(np.isfinite(consensus_p))
        or np.any(consensus_p < lo_band)
        or np.any(consensus_p > hi_band)
    ):
        return _fallback(
            snapshot, strikes, raw_ladders, model_ids, bankroll_states,
            weights_by_region_arr, now, m_ts, reason="per-strike sanity-bound violation",
        )

    n_belly = sum(1 for r in region_of_strike if r == BELLY_REGION)
    n_wing = n - n_belly
    cred_belly = float(weight_dicts_post[BELLY_REGION][PRICER_MODEL_ID])
    cred_wing = float(weight_dicts_post[WING_REGION][PRICER_MODEL_ID])
    credibility = ((n_belly * cred_belly + n_wing * cred_wing) / n) if n > 0 else 0.5
    credibility_by_region = {BELLY_REGION: cred_belly, WING_REGION: cred_wing}

    fv = _build_fair_value(
        snapshot, strikes, consensus_p, credibility,
        AnchorMethod.BEUOY, now, m_ts, config,
        credibility_by_region=credibility_by_region,
    )
    new_states = {
        BELLY_REGION: BankrollState(
            model_ids=model_ids, bankrolls=new_bankrolls_by_region[BELLY_REGION],
            last_update=now, update_count=update_counts[BELLY_REGION],
            frozen=belly_state.frozen,
        ),
        WING_REGION: BankrollState(
            model_ids=model_ids, bankrolls=new_bankrolls_by_region[WING_REGION],
            last_update=now, update_count=update_counts[WING_REGION],
            frozen=wing_state.frozen,
        ),
    }
    return AnchorResult(fv, new_states, forecasts, consensus_bucket)


# ---------------------------------------------------------------------------
# Fallback + FairValue construction
# ---------------------------------------------------------------------------


def _fallback(
    snapshot, strikes, raw_ladders, model_ids, bankroll_states,
    weights_by_region, now, m_ts, reason,
) -> AnchorResult:
    """FIXED_BLEND w=0.5 per strike; freeze BOTH region bankrolls (fallback is
    a whole-ladder event, plan step 8); warn (risk 8.8)."""
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

    # Cosmetic scalar credibility for the fallback FairValue (no region split
    # is meaningful here -- the whole ladder is a fixed 50/50 blend); prefer
    # whichever region's weights survived degeneracy, else neutral 0.5.
    credibility = 0.5
    w_belly = weights_by_region.get(BELLY_REGION)
    w_wing = weights_by_region.get(WING_REGION)
    if w_belly is not None:
        credibility = float(w_belly[model_ids.index(PRICER_MODEL_ID)])
    elif w_wing is not None:
        credibility = float(w_wing[model_ids.index(PRICER_MODEL_ID)])

    fv = _build_fair_value(
        snapshot, strikes, blend, credibility,
        AnchorMethod.FIXED_BLEND_FALLBACK, now, m_ts, None,
    )
    frozen_states: Dict[str, BankrollState] = {}
    for region in REGIONS:
        st = bankroll_states[region]
        frozen_states[region] = BankrollState(
            model_ids=list(st.model_ids),
            bankrolls=dict(st.bankrolls),
            last_update=now,
            update_count=st.update_count,
            frozen=True,
        )
    n = len(strikes)
    forecasts = {m: ladder_to_buckets(raw_ladders.get(m, np.zeros(n))) for m in model_ids}
    return AnchorResult(fv, frozen_states, forecasts, ladder_to_buckets(blend))


def _build_fair_value(
    snapshot, strikes, consensus_p, credibility, method, now, m_ts, config,
    credibility_by_region: Optional[Dict[str, float]] = None,
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
        credibility_by_region=credibility_by_region,
    )
