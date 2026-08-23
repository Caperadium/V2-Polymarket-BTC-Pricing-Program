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

WING PRICER WEIGHT PIN (2026-08-08 wing-bleed fix). The wing region's Bayes
update is a self-confirmation loop: factors score against a consensus built
from the PRE-update weights, so a pricer-heavy wing consensus keeps awarding
the pricer more wing weight (VPS 2026-08-08: wing pricer weight re-learned to
~0.978 while every wing YES fill settled worthless). When
`MMConfig.wing_pricer_weight_pin` is in [0, 1], the wing weights are PINNED --
pricer at the pin (clamped into [bankroll_floor, 1-bankroll_floor]), remainder
to the other models pro rata -- and the wing Bayes update is skipped entirely:
wing update_count never advances and every NON-FALLBACK return path persists
the pinned bankrolls (the fallback still copies the raw stored dicts; the next
clean tick re-pins and overwrites). Negative pin disables (legacy Bayes).
COVERAGE LIMIT: the region map is the anchor's own (strike region from the
sanitized MARKET ladder vs belly_band), so market-mid 0.20-0.28 strikes are
belly and the pin does not touch them. Belly PER-STRIKE consensus is unchanged
by the pin; belly BANKROLL trajectories can shift slightly (the whole-ladder
consensus feeds belly factors and the boundary bucket carries wing-weighted
values).

BANKROLL UPDATE TEMPERING (2026-08-10 skew-fix wave item 3). Per-tick Bayes
factors at 15s cadence can flip a region's weights full range (0.02 <-> 0.98)
within hours -- far more weight movement than one tick of mid movement can
justify; the resulting pricer-rich phases post rich bids and inflate Kelly's
phantom edge, the belly-side co-driver of the 2026-08-10 skew-explosion
incident (temp/mm_skew_fix_plan.md item 3). `MMConfig.bankroll_update_temper`
(t, default 0.1) tempers each region's per-tick Bayes-factor vector,
`factors = factors ** t`, applied AFTER the existing non-finite-factor check
(3.3) -- tempering must never mask that skip -- and BEFORE the weight update,
for every UNPINNED region (belly always; wing only when
`wing_pricer_weight_pin` is disabled; the pinned wing skip precedes factor
computation entirely and is untouched). Factors are non-negative by
construction (`ladder_to_buckets` clips at 0), so `factor**t` is always real
and a zero factor stays zero. t=1.0 -- or any non-finite/non-positive/>1
value, clamped -- is legacy untempered Bayes; 0<t<1 slows learning: t=0.1
makes a full 0.02<->0.98 flip take ~10x as many ticks (~5-7.5h of consistent
evidence instead of ~30-45min), clearing the 6h acceptance bar. Tempering
changes the RATE, not the attractor -- the 0.98/0.02 self-confirmation corner
is still where the dynamic points; this only bounds the damage rate. The
skip rules (3.3), floor, normalization, and update_count semantics are all
unchanged by tempering -- it only shrinks the per-tick step size.

C1 -- BELLY DRIFT-ANCHORED BAYES SCORING (2026-08-13, temp/mm_c1_belly_
drift_plan.md v3). The legacy belly Bayes update above is a self-
confirmation loop: factors score a model's PREVIOUS forecast against a
consensus built from that SAME model's own PRE-update weights, so the
dominant model keeps winning (measured, temp/mm_belly_divergence_
experiment.md, 7d/66k ticks: a RICH belly consensus loses to the mid at
settlement in EVERY divergence bucket, Brier gap +0.034 -> +0.116, and the
mid drifts AWAY from fair at 5-20c divergence, frac-toward 0.38-0.41). C1
replaces that target, for the belly weight only, with the CURRENT
sanitized market ladder some `belly_drift_horizon_s` (h, default 3600s)
LATER than the scored forecasts -- a FULL-SUPPORT (all n+1 buckets, not a
belly-only subset) drift factor:

    factor_i = sum_j( market_now_bucket[j] * p_i_lag[j] / c_lag[j] )

Why full support: a belly-only SUBSET factor cannot see a pure LEVEL
divergence (the measured faucet) -- it lives in the always-wing tail
buckets 0/n (`ladder_to_buckets` makes interior bucket j = p[j-1]-p[j], so
a uniform level shift cancels there identically) -- and it carries a
drift-independent static bias on pure martingale data. Over full support
both bucket vectors sum to 1, so that bias vanishes identically.

Update law (with d = M_lag - P_lag, c_lag = M_lag - w_p*d, target
M_now = M_lag - alpha*d + e; alpha = fraction of the lagged divergence
closed toward the pricer by market_now; e = martingale noise):

    factor_market - factor_pricer = (w_p - alpha)*S + sum_j(e_j*d_j/c_lag_j)
    S = sum_j( d_j^2 / c_lag_j ) >= 0

Martingale data (alpha=0, e=0) always favors the market by construction --
the legacy subset factor's failure mode (crediting the pricer on pure
martingale data) is impossible here. The pricer gains weight only when
alpha > w_p: the update's fixed point is the S-WEIGHTED average fractional
divergence-close, NOT assumed equal to the raw experiment's frac-toward
composition -- with a tail-dominated S (see `s_tail_frac` below) the
equilibrium composition must be MEASURED in shadow before any flip. Noise
runs ~2x the per-event drift signal but accumulates linearly in n vs
noise's sqrt(n) (daily SNR ~5 at ~96 events/day): the belly pricer weight
is therefore a mean-reverting walk AROUND the fixed point under live mode,
not a convergent constant.

HONEST NAME: this is LADDER-WIDE drift scoring applied to the belly
weight. On a realistic ladder with the model's known OTM upper-tail
richness, most of the signal mass S can sit in the two open-tail buckets
(0 and n) -- `s_tail_frac` is their share of S. If shadow measurement
shows this is PERSISTENT (median s_tail_frac > 0.6), C1 is measuring tail
richness, not belly drift -- the recorded fallback design is LADDER-SPACE
belly scoring (score p-space distances on belly strikes directly, no
bucket transform, no tail leakage), NOT implemented this wave.

If the wing pricer weight pin is ever retired, this full-support form is
not automatically safe to extend to the wing region as-is -- the correct
per-region form is CONDITIONAL RENORMALIZATION (renormalize the target,
lagged forecasts, and divisor over the region's own bucket idxs);
recorded, not implemented.

MODES (`MMConfig.belly_score_mode`; the harness, Part B, validates once at
__init__ and warns; this module itself treats anything not in
{"shadow","live"} as "legacy" silently, `_resolve_belly_score_mode`):
  - "legacy": today's behavior exactly; `belly_lag_*` kwargs ignored.
  - "shadow" (default this wave): the applied belly update stays legacy,
    on its normal per-refresh cadence -- byte-identical to legacy. The
    drift factor above, plus a RATE-MATCHED CONTROL factor (identical
    formula, target = consensus_new_bucket -- the PRE-update whole-ladder
    consensus -- instead of market_now_bucket; isolates precisely the one
    variable C1 changes) and `s_tail_frac`, are computed and returned on
    `AnchorResult` whenever `belly_lag_forecasts`/`belly_lag_consensus`
    are supplied and pass their own precondition (shape (n+1,), all
    finite -- independent of, and not gated by, the legacy update's lag-1
    guard or shape gate).
  - "live": the drift factor IS the belly update -- applied via the
    shared `advance_weights()` helper with `belly_drift_temper`, at
    scoring-event cadence only (a tick with no `belly_lag_*` supplied
    skips the belly update entirely: weights/update_count unchanged; the
    legacy per-refresh loop's own belly branch is unconditionally skipped
    in this mode -- a new 3.3 gate reason -- so the two paths never both
    write belly on the same tick). Wing is completely untouched by every
    mode.

`AnchorResult.belly_drift_factors` / `belly_control_factors` /
`belly_s_tail_frac` are populated together on a successful score (mode
shadow or live, valid lag inputs, and -- live mode only -- an unfrozen
belly region with a non-degenerate `advance_weights()` result); on ANY
skip (mode legacy, lag absent/malformed, a frozen belly in live mode, or a
degenerate `advance_weights()` result -- or the whole-anchor `_fallback`)
all three are None and `belly_score_skip` carries the reason (`no_lag` |
`shape_mismatch` | `non_finite` | `frozen` | `s_le_0` | `fallback`) --
matching the `bayes_score_log` "NULL on skip" convention (state_store.py)
so a caller can journal these fields directly.

C2 (recorded, out of scope this wave): settlement-anchored scoring -- the
persisted `belly_snapshot` history in `state_store.bayes_score_log`
accumulates exactly the forecast history it would need.
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
    # C1 belly drift-anchored Bayes scoring (2026-08-13, temp/
    # mm_c1_belly_drift_plan.md; module docstring "C1" section). Additive,
    # default None -- existing 4-positional-arg AnchorResult(...) call
    # sites are unaffected. Populated together on a successful score (mode
    # shadow or live, valid belly_lag_* inputs, and -- live mode only -- a
    # non-degenerate advance_weights() result on an unfrozen belly region);
    # all four are None on ANY skip (mode legacy, absent/malformed lag
    # kwargs, a frozen belly in live mode, a degenerate advance_weights()
    # result, or the whole-anchor fallback), with `belly_score_skip`
    # carrying the reason -- matching the `bayes_score_log` "NULL on skip"
    # convention (state_store.py) so a caller can journal these fields
    # directly.
    belly_drift_factors: Optional[Dict[str, float]] = None
    belly_control_factors: Optional[Dict[str, float]] = None
    belly_s_tail_frac: Optional[float] = None
    belly_score_skip: Optional[str] = None


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


def advance_weights(
    w_pre: np.ndarray, factors: np.ndarray, temper: float, floor: float,
) -> Optional[np.ndarray]:
    """Shared Bayes weight-advance step (C1 review NEW-7): ONE
    implementation for every path that turns (previous weights, Bayes
    factors) into (new weights) -- the applied per-region loop below, the
    C1 live-mode belly drift update, and (harness-owned, Part B) the
    shadow drift/control hypothetical trajectories. One implementation, no
    drift between disciplines.

    `factors` must already be validated finite by the CALLER -- every call
    site runs its own non-finite-factor check first; tempering here must
    never mask that check (matches the pre-refactor applied-loop sequence
    exactly, byte-identity instruction c). Applies `factors ** temper`
    only when `temper < 1.0` (an exact `temper == 1.0` skips the `**`
    entirely, matching the pre-refactor code rather than merely relying on
    `x ** 1.0 == x`), multiplies onto `w_pre`, then floors+normalizes via
    `_apply_floor`. Returns None on the s<=0/non-finite skip (3.3) -- the
    caller's own weights/update_count are then left unchanged, exactly as
    the pre-refactor applied loop's own `continue` did.
    """
    f = np.asarray(factors, dtype=float)
    if temper < 1.0:
        f = f ** temper
    updated = np.asarray(w_pre, dtype=float) * f
    s = updated.sum()
    if not (s > 0.0 and np.isfinite(s)):
        return None
    return _apply_floor(updated / s, floor)


def _weight_dict(w_arr: np.ndarray, model_ids: List[str]) -> Dict[str, float]:
    return {mid: float(w_arr[i]) for i, mid in enumerate(model_ids)}


def _bankroll_update_temper(config) -> float:
    """Resolve `MMConfig.bankroll_update_temper`, robustly clamped into
    (0, 1] (module docstring, "BANKROLL UPDATE TEMPERING"). 1.0 = legacy
    untempered Bayes. Any garbage value -- missing attribute, non-numeric,
    non-finite, <= 0, or > 1 -- falls back to 1.0 (legacy) rather than
    raising or silently misbehaving."""
    raw = getattr(config, "bankroll_update_temper", 1.0)
    try:
        t = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(t) or t <= 0.0 or t > 1.0:
        return 1.0
    return t


def _pinned_weights(model_ids: List[str], pin: float) -> np.ndarray:
    """Deterministic wing weight vector (Fix 1): pricer at `pin`, remainder
    split pro rata across the other models (2-model case: market = 1-pin);
    pricer absent -> uniform. Caller passes `pin` already clamped into
    [floor, 1-floor], so the module's floor invariant holds by construction."""
    m = len(model_ids)
    if m == 0:
        return np.zeros(0, dtype=float)
    if PRICER_MODEL_ID not in model_ids:
        return np.full(m, 1.0 / m)
    if m == 1:
        return np.ones(1, dtype=float)
    w = np.empty(m, dtype=float)
    share = (1.0 - pin) / (m - 1)
    for i, mid in enumerate(model_ids):
        w[i] = pin if mid == PRICER_MODEL_ID else share
    return w


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
# C1 belly drift-anchored Bayes scoring helpers (module docstring "C1"
# section; temp/mm_c1_belly_drift_plan.md)
# ---------------------------------------------------------------------------


def _resolve_belly_score_mode(config) -> str:
    """Resolve `MMConfig.belly_score_mode`. Anything other than
    "shadow"/"live" is treated as "legacy" silently (plan S7 -- the
    harness, Part B, validates once at __init__ and warns; this module's
    own fallback is silent by design)."""
    mode = getattr(config, "belly_score_mode", "legacy")
    return mode if mode in ("shadow", "live") else "legacy"


def _belly_drift_temper(config) -> float:
    """Resolve `MMConfig.belly_drift_temper` for the C1 live-mode belly
    drift update. Same defensive contract as `_bankroll_update_temper`:
    missing attribute, non-numeric, non-finite, <= 0, or > 1 all fall back
    to 1.0 (untempered full-strength drift factors) rather than raising."""
    raw = getattr(config, "belly_drift_temper", 1.0)
    try:
        t = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(t) or t <= 0.0 or t > 1.0:
        return 1.0
    return t


def _smooth_buckets(v: np.ndarray, eps: float) -> np.ndarray:
    """Additive-smoothing regularizer for a bucket probability vector:
    (v + eps) / (1 + len(v)*eps). 2026-08-21 acceptance-review fix for the
    measured 31% non_finite scoring-event skip rate (zero buckets from flat
    ladder segments zeroed the c_lag divisor). Preserves full support (no
    zero buckets -> finite Bayes ratios), preserves sum == 1, and keeps the
    full-support mass-cancellation property intact: smoothed normalized
    vectors still each sum to 1, so dMass == 0 and martingale data still
    always favors the market model (module docstring C1 law). eps <= 0
    returns `v` unchanged (legacy: a zero bucket -> non_finite skip).
    Consumed ONLY by the C1 drift/control block -- never by the legacy
    applied Bayes loop, whose step-3.3 divisor skip rules are load-bearing.
    """
    e = _finite_scalar(eps)
    if e <= 0.0:
        return v
    return (v + e) / (1.0 + v.size * e)


def _finite_scalar(v) -> float:
    """Coerce to a finite float (0.0 on garbage) -- local guard for the
    smoothing epsilon so a NaN config value disables rather than poisons."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


def _belly_lag_precondition(
    belly_lag_forecasts: Optional[Dict[str, np.ndarray]],
    belly_lag_consensus: Optional[np.ndarray],
    model_ids: List[str],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[str]]:
    """C1's OWN precondition (S3, byte-identity instruction c) --
    independent of, and NOT gated by, the legacy per-refresh update's lag-1
    guard or its shape gate: `belly_lag_consensus.shape == (n+1,)`, each
    `belly_lag_forecasts[mid].shape == (n+1,)`, all finite.

    Returns `(c_lag, per_model_lag, skip_reason)`; `skip_reason` is None on
    success, in which case the other two are non-None. Never mutates its
    inputs -- `np.asarray(..., dtype=float)` may return the same object
    when it is already a float64 array, but nothing here writes back into
    it (byte-identity instruction b).
    """
    expected = (n + 1,)
    c_lag = np.asarray(belly_lag_consensus, dtype=float)
    if c_lag.shape != expected:
        return None, None, "shape_mismatch"
    per_model: Dict[str, np.ndarray] = {}
    for mid in model_ids:
        raw = belly_lag_forecasts.get(mid) if belly_lag_forecasts is not None else None
        if raw is None:
            return None, None, "shape_mismatch"
        arr = np.asarray(raw, dtype=float)
        if arr.shape != expected:
            return None, None, "shape_mismatch"
        per_model[mid] = arr
    if not np.all(np.isfinite(c_lag)):
        return None, None, "non_finite"
    for arr in per_model.values():
        if not np.all(np.isfinite(arr)):
            return None, None, "non_finite"
    return c_lag, per_model, None


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
    belly_lag_forecasts: Optional[Dict[str, np.ndarray]] = None,
    belly_lag_consensus: Optional[np.ndarray] = None,
    belly_lag_ts: Optional[datetime] = None,
) -> AnchorResult:
    """Compute the Beuoy consensus FairValue and updated per-region BankrollStates.

    snapshot: PricerSnapshot (uses .strikes, .p_hat, .ts, .expiry_key).
    mids: market mid probability per strike.
    bankroll_states: dict keyed BELLY_REGION/WING_REGION (module constants).
    prev_forecasts / prev_consensus: previous refresh's model bucket vectors and
    consensus bucket vector (SHARED across regions -- forecasts are per-model,
    not per-region); when present (and a region is not frozen) that region's
    bankroll is marked to market before the final consensus is formed.
    belly_lag_forecasts / belly_lag_consensus / belly_lag_ts: C1 belly
    drift-anchored Bayes scoring (module docstring "C1" section; default
    None = legacy). Harness-owned lag buffer entry from ~belly_
    drift_horizon_s ago (per-model bucket vectors, the consensus bucket
    vector, and its timestamp). Read only when `config.belly_score_mode`
    resolves to "shadow"/"live" (`_resolve_belly_score_mode`); ignored
    entirely otherwise. `belly_lag_ts` is accepted for call-site symmetry
    with the harness's lag-buffer entry shape (ts, forecasts, consensus)
    but is not itself consumed by this function's own math.
    """
    now = ts if ts is not None else datetime.now(timezone.utc)
    m_ts = market_ts if market_ts is not None else getattr(snapshot, "ts", now)
    floor = float(getattr(config, "bankroll_floor", DEFAULT_BANKROLL_FLOOR))
    belly_band = getattr(config, "belly_band", (0.2, 0.8))
    # Bayes-factor tempering (2026-08-10 skew-fix wave item 3; module
    # docstring "BANKROLL UPDATE TEMPERING"). 1.0 = legacy untempered.
    temper = _bankroll_update_temper(config)
    # Wing pricer weight PIN (Fix 1, 2026-08-08 wing-bleed fix; module
    # docstring). Negative (or out-of-[0,1]) disables -> legacy Bayes.
    pin_raw = float(getattr(config, "wing_pricer_weight_pin", -1.0))
    pinned = 0.0 <= pin_raw <= 1.0
    pin = min(max(pin_raw, floor), 1.0 - floor) if pinned else pin_raw
    # C1 belly drift-anchored Bayes scoring mode (module docstring "C1"
    # section). "legacy" (today's behavior) unless belly_score_mode
    # resolves to "shadow"/"live".
    score_mode = _resolve_belly_score_mode(config)
    lag_provided = belly_lag_forecasts is not None or belly_lag_consensus is not None

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
    wing_was_degenerate = weights_wing is None
    if pinned:
        # Fix 1: the pin replaces the stored wing weights outright -- BEFORE
        # the fallback path could consult weights_by_region_arr (so even a
        # fallback's cosmetic credibility reports the pin).
        weights_wing = _pinned_weights(model_ids, pin)
    weights_by_region_arr = {BELLY_REGION: weights_belly, WING_REGION: weights_wing}
    if inputs_bad or weights_belly is None or (weights_wing is None and not pinned):
        return _fallback(
            snapshot, strikes, raw_ladders, model_ids, bankroll_states,
            weights_by_region_arr, now, m_ts, reason="non-finite inputs or degenerate bankrolls",
            score_mode=score_mode, lag_provided=lag_provided,
        )
    if pinned and wing_was_degenerate:
        # Self-healing must not be silent: degenerate STORED wing bankrolls
        # would have fallen back pre-pin; the pin rescues the tick and the
        # non-fallback return below overwrites the stored row with the pin.
        logger.warning(
            "fair_value_anchor: wing bankrolls degenerate; pin rescued "
            "(stored row will be overwritten)"
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
    # While pinned, weight_dicts_pre[WING] IS the pinned dict (weights_wing
    # was replaced above), so pre and post wing dicts both carry the pin.
    weight_dicts_post = dict(weight_dicts_pre)

    new_bankrolls_by_region = {
        BELLY_REGION: dict(belly_state.bankrolls),
        # Fix 1: seed wing from the PINNED dict (not the stored bankrolls) so
        # every NON-FALLBACK return path -- frozen wing, first tick (no
        # threaded history), per-region skip branches -- persists the pinned
        # state and overwrites a stale stored row on the next append.
        WING_REGION: (dict(weight_dicts_pre[WING_REGION]) if pinned
                      else dict(wing_state.bankrolls)),
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
                if region == WING_REGION and pinned:
                    # Fix 1: wing Bayes update skipped entirely while pinned
                    # (self-confirmation loop); wing update_count unchanged.
                    # Applies regardless of the frozen flag (carried through
                    # untouched either way).
                    continue
                if region == BELLY_REGION and score_mode == "live":
                    # C1 (2026-08-13): in live mode belly's applied update
                    # comes exclusively from the SEPARATE drift block below
                    # (scoring-event cadence, gated on the belly_lag_*
                    # kwargs) -- skip here unconditionally (new 3.3 gate
                    # reason) so the per-refresh legacy path and the
                    # scoring-event drift path never both write belly on
                    # the same tick. legacy/shadow never set score_mode to
                    # "live", so this branch is dead there -- byte-identity
                    # for those two modes is preserved.
                    continue
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
                    # 3.3: non-finite factor (incl. a zero/degenerate divisor
                    # restricted to this region's own buckets) -> skip. Logged
                    # (review 2026-07-15): a persistently-skipping region means
                    # its credibility never updates -- exactly the wing-learning
                    # failure B2 exists to fix -- and must be visible.
                    logger.debug(
                        "fair_value_anchor: region %s bankroll update skipped "
                        "(non-finite factor; a prev-consensus bucket in this "
                        "region is zero or degenerate)", region,
                    )
                    continue
                # Tempering runs AFTER the non-finite-factor check (must never
                # mask that skip) and applies to every UNPINNED region update
                # reaching this point (belly always; wing only when the pin is
                # disabled -- the pinned wing `continue` above precedes factor
                # computation entirely). Factors are non-negative by
                # construction (`ladder_to_buckets` clips at 0), so
                # factor**t is always real and a zero factor stays zero;
                # tempering changes the RATE of the weight update, not the
                # attractor it converges toward. Delegates to the shared
                # `advance_weights` helper (C1 review NEW-7) -- behavior-
                # identical to the pre-refactor inline sequence (golden
                # tests).
                w_pre_arr = np.array([weight_dicts_pre[region][mid] for mid in model_ids])
                w_upd = advance_weights(w_pre_arr, factors, temper, floor)
                if w_upd is None:
                    logger.debug(
                        "fair_value_anchor: region %s bankroll update skipped "
                        "(factor sum s_R <= 0 or non-finite)", region,
                    )
                    continue  # 3.3: s_R <= 0 or non-finite -> skip
                new_bankrolls = _weight_dict(w_upd, model_ids)
                new_bankrolls_by_region[region] = new_bankrolls
                weight_dicts_post[region] = new_bankrolls
                update_counts[region] += 1

    # --- C1 belly drift-anchored Bayes scoring (module docstring "C1"
    # section; temp/mm_c1_belly_drift_plan.md). A SEPARATE block, after the
    # per-region applied loop above, sharing none of its loop-local
    # variables (byte-identity instruction a); no in-place numpy ops on
    # `forecasts[...]` or the `belly_lag_*` arrays (instruction b); its own
    # precondition below (S3, `_belly_lag_precondition`) -- NOT gated by
    # the lag-1 guard just above or its shape check (instruction c), so a
    # live-mode belly drift update can fire even with no lag-1 history
    # (prev_forecasts/prev_consensus both None).
    belly_drift_factors: Optional[Dict[str, float]] = None
    belly_control_factors: Optional[Dict[str, float]] = None
    belly_s_tail_frac: Optional[float] = None
    belly_score_skip: Optional[str] = None
    if score_mode in ("shadow", "live"):
        if belly_lag_forecasts is None or belly_lag_consensus is None:
            belly_score_skip = "no_lag"
        else:
            c_lag, lag_per_model, shape_reason = _belly_lag_precondition(
                belly_lag_forecasts, belly_lag_consensus, model_ids, n,
            )
            if shape_reason is not None:
                belly_score_skip = shape_reason
            else:
                # 2026-08-21 acceptance-review fix: additive-smooth every
                # bucket vector entering the drift/control ratios (see
                # _smooth_buckets; config.belly_drift_bucket_eps, <= 0 =
                # legacy skip-on-zero). Smoothing ALL vectors -- divisor
                # and numerators alike -- keeps each normalized, so the
                # full-support dMass == 0 cancellation survives.
                eps = getattr(config, "belly_drift_bucket_eps", 0.0)
                c_lag_s = _smooth_buckets(c_lag, eps)
                lag_s = {mid: _smooth_buckets(lag_per_model[mid], eps) for mid in model_ids}
                market_now_bucket = _smooth_buckets(forecasts[MARKET_MODEL_ID], eps)
                # PRE-update whole-ladder consensus (control track target,
                # review NEW-2): its OWN computation here, deliberately not
                # reusing the applied loop's `consensus_new_bucket` local
                # (which may not even have run this tick) -- instruction a.
                control_target = _smooth_buckets(ladder_to_buckets(
                    _ladder_space_consensus(weight_dicts_pre, region_of_strike, recon, model_ids)
                ), eps)
                with np.errstate(divide="ignore", invalid="ignore"):
                    drift_arr = np.array([
                        float(np.sum(market_now_bucket * lag_s[mid] / c_lag_s))
                        for mid in model_ids
                    ])
                    control_arr = np.array([
                        float(np.sum(control_target * lag_s[mid] / c_lag_s))
                        for mid in model_ids
                    ])
                if not (np.all(np.isfinite(drift_arr)) and np.all(np.isfinite(control_arr))):
                    # A zero/degenerate c_lag entry naturally produces a
                    # non-finite factor here (0/0 -> nan, x/0 -> inf), same
                    # discipline as the legacy per-region loop above.
                    belly_score_skip = "non_finite"
                else:
                    # s_tail_frac (review NEW-1): d_j = M_lag[j] - P_lag[j]
                    # over the FULL lag pair (not this tick's market/
                    # pricer); S = sum_j(d_j^2 / c_lag_j); share of S from
                    # the two open-tail buckets (0 and n). Uses the same
                    # smoothed vectors as the factors (consistency: the
                    # diagnostic must describe the S actually scored).
                    d = lag_s[MARKET_MODEL_ID] - lag_s[PRICER_MODEL_ID]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        s_terms = d * d / c_lag_s
                    S = float(np.sum(s_terms))
                    tail_frac = 0.0 if S == 0.0 else float((s_terms[0] + s_terms[-1]) / S)

                    if score_mode == "live":
                        belly_state_now = bankroll_states[BELLY_REGION]
                        if belly_state_now.frozen:
                            belly_score_skip = "frozen"
                        else:
                            w_pre_belly = np.array(
                                [weight_dicts_pre[BELLY_REGION][mid] for mid in model_ids]
                            )
                            w_upd_belly = advance_weights(
                                w_pre_belly, drift_arr, _belly_drift_temper(config), floor,
                            )
                            if w_upd_belly is None:
                                belly_score_skip = "s_le_0"
                            else:
                                new_bankrolls = _weight_dict(w_upd_belly, model_ids)
                                new_bankrolls_by_region[BELLY_REGION] = new_bankrolls
                                weight_dicts_post[BELLY_REGION] = new_bankrolls
                                update_counts[BELLY_REGION] += 1

                    if belly_score_skip is None:
                        belly_drift_factors = _weight_dict(drift_arr, model_ids)
                        belly_control_factors = _weight_dict(control_arr, model_ids)
                        belly_s_tail_frac = tail_frac

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
            score_mode=score_mode, lag_provided=lag_provided,
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
    return AnchorResult(
        fv, new_states, forecasts, consensus_bucket,
        belly_drift_factors=belly_drift_factors,
        belly_control_factors=belly_control_factors,
        belly_s_tail_frac=belly_s_tail_frac,
        belly_score_skip=belly_score_skip,
    )


# ---------------------------------------------------------------------------
# Fallback + FairValue construction
# ---------------------------------------------------------------------------


def _fallback(
    snapshot, strikes, raw_ladders, model_ids, bankroll_states,
    weights_by_region, now, m_ts, reason,
    score_mode: str = "legacy", lag_provided: bool = False,
) -> AnchorResult:
    """FIXED_BLEND w=0.5 per strike; freeze BOTH region bankrolls (fallback is
    a whole-ladder event, plan step 8); warn (risk 8.8).

    C1 (module docstring "C1" section): the new AnchorResult belly_drift_*/
    belly_control_*/belly_s_tail_frac fields are always None here (the
    whole-anchor fallback is not a belly-specific event); `belly_score_skip`
    is "fallback" when the caller's belly_score_mode resolved to "shadow"/
    "live" AND belly_lag_* kwargs were supplied this tick (score_mode/
    lag_provided, threaded from the caller), else None (nothing was ever
    requested to skip)."""
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
    belly_score_skip = "fallback" if (score_mode in ("shadow", "live") and lag_provided) else None
    return AnchorResult(
        fv, frozen_states, forecasts, ladder_to_buckets(blend),
        belly_score_skip=belly_score_skip,
    )


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
