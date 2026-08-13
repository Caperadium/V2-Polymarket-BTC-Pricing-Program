"""Quote engine (plan task Q1, Section 2.4) -- Dalen Avellaneda-Stoikov in log-odds.

Per contract, compute a reservation log-odds and half-spread in x-space, then map
back to probability via the logodds layer. Two variants:

  * "dalen"  -- Dalen Eqs 8-9 (default):
        r_x     = x_fair - q * gamma * sigma_b^2 * tte
        delta_x = 0.5 * (gamma * sigma_b^2 * tte + (2/k) * log(1 + gamma/k))
        skew_x  = -q * gamma * sigma_b^2 * tte   (= r_x - x_fair)
  * "glft"   -- GLFT stationary closed-form (synthesis 2.3), skew embedded in the
        per-side deltas:
        base      = (1/gamma) * ln(1 + gamma/k)
        C         = sqrt(sigma_b^2 * gamma / (2*k*A)) * (1 + gamma/k)^(1 + k/gamma)
        delta_b(q)= base + ((2q+1)/2) * C
        delta_a(q)= base - ((2q-1)/2) * C
        x_bid = x_fair - delta_b, x_ask = x_fair + delta_a
     which reduces to the equivalent (r_x, delta_x) pair:
        r_x     = x_fair - q * C          skew_x = -q * C
        delta_x = base + 0.5 * C          (inventory-independent)

Units: tte in DAYS; q is a float (caller normalizes shares by a config unit --
`MMConfig.skew_q_norm`, wired by the harness since 2026-08-13: q_eff =
q_shares / skew_q_norm, see temp/mm_bleed2_fix_plan.md item 1).
sigma_b is per-sqrt-day belief vol of the log-odds series and is taken as an
ARGUMENT so the estimator (`estimate_sigma_b`) is separable from the quote math.

All probability conversions route through `market_maker.logodds`; no nan/inf may
escape, and `delta_x` is floored at a small positive value.

Inventory-skew displacement cap (2026-08-10 skew-explosion fix): `skew_x` above
is UNBOUNDED in q for both variants. `make_quote`'s `skew_x_cap` argument
(threaded from `MMConfig.skew_x_cap` via `make_quote_from_config`) clamps the
reservation displacement to `|skew_x| <= skew_x_cap` x-units, applied AFTER the
variant terms are computed and BEFORE the pre-existing band clamp on `r_x`;
`skew_x` is then re-derived from the (possibly band-clamped) `r_x` so the
identity `x_fair == r_x - skew_x` stays exact. `skew_x_cap <= 0` disables the
cap (legacy unbounded behavior, byte-identical). See `per_share_skew_x` for the
shared per-share displacement formula (consumed by sizing's skew-aware entry
cap, item 2 of the same fix wave).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

from market_maker import logodds
from market_maker.contracts import QuoteProposal

logger = logging.getLogger(__name__)

# Minimum half-spread in x-units (keeps x_bid < x_ask strictly).
DELTA_X_FLOOR: float = 1.0e-9

# EWMA decay default for sigma_b estimation (RiskMetrics-style).
DEFAULT_SIGMA_B_LAMBDA: float = 0.94


def _finite(v: float, fallback: float = 0.0) -> float:
    """Coerce to a finite float."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(f):
        return fallback
    return f


# ---------------------------------------------------------------------------
# sigma_b estimation (separable from the quote math)
# ---------------------------------------------------------------------------


def estimate_sigma_b(
    x_series,
    dt_days: float,
    sigma_b_floor: float,
    sigma_b_cap: float,
    lam: float = DEFAULT_SIGMA_B_LAMBDA,
) -> float:
    """EWMA realized volatility of a log-odds series, scaled to per-sqrt-day.

    Increments dx = diff(x) have EWMA mean-square E[dx^2]; the per-day variance is
    E[dx^2]/dt_days and sigma_b = sqrt(.), clamped to [floor, cap]. A constant
    series has dx == 0 -> sigma_b == floor.
    """
    lo = _finite(sigma_b_floor, 0.0)
    hi = _finite(sigma_b_cap, lo)
    if hi < lo:
        hi = lo
    dt = _finite(dt_days, 0.0)
    if dt <= 0.0:
        return lo

    arr = np.asarray(x_series, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return lo

    dx = np.diff(arr)
    sq = dx * dx
    lam = min(max(_finite(lam, DEFAULT_SIGMA_B_LAMBDA), 0.0), 0.999999)

    var = float(sq[0])
    for s in sq[1:]:
        var = lam * var + (1.0 - lam) * float(s)

    var_per_day = var / dt
    if not np.isfinite(var_per_day) or var_per_day < 0.0:
        var_per_day = 0.0
    sigma = float(np.sqrt(var_per_day))
    return float(min(max(sigma, lo), hi))


# ---------------------------------------------------------------------------
# Core reservation / spread math
# ---------------------------------------------------------------------------


def _dalen_terms(
    x_fair: float,
    q: float,
    tte_days: float,
    sigma_b: float,
    gamma: float,
    k: float,
    arrival_denominator: str = "k",
) -> Tuple[float, float, float]:
    """Return (r_x, delta_x, skew_x) for the Dalen AS-in-x variant.

    arrival_denominator (decision D3): "k" = Dalen Eq 9 verbatim,
    (2/k)*ln(1+gamma/k), VERIFIED against arXiv 2510.15205v2; "gamma" =
    classical Avellaneda-Stoikov 2008 / GLFT, (2/gamma)*ln(1+gamma/k).
    Dalen is the mandated source and the default; the AS setting exists for
    the Stage-A side-by-side comparison.
    """
    var_term = gamma * sigma_b * sigma_b * tte_days
    skew = -q * var_term
    r_x = x_fair + skew
    denom = gamma if arrival_denominator == "gamma" else k
    delta_x = 0.5 * (var_term + (2.0 / denom) * np.log1p(gamma / k))
    return r_x, delta_x, skew


def _glft_terms(
    x_fair: float, q: float, sigma_b: float, gamma: float, k: float, A: float
) -> Tuple[float, float, float]:
    """Return (r_x, delta_x, skew_x) for the GLFT stationary closed-form.

    The inventory skew is embedded in the per-side deltas; the equivalent
    (r_x, delta_x) pair is r_x = x_fair - q*C, delta_x = base + 0.5*C.
    """
    base = (1.0 / gamma) * np.log1p(gamma / k)
    C = np.sqrt(sigma_b * sigma_b * gamma / (2.0 * k * A)) * (1.0 + gamma / k) ** (
        1.0 + k / gamma
    )
    skew = -q * C
    r_x = x_fair + skew
    delta_x = base + 0.5 * C
    return r_x, delta_x, skew


def glft_side_deltas(
    q: float, sigma_b: float, gamma: float, k: float, A: float
) -> Tuple[float, float]:
    """Raw GLFT per-side deltas (delta_b, delta_a) in x-units around x_fair.

    Deliberately unclamped: this is a raw-math helper with no production
    caller and does not go through `make_quote`'s skew_x_cap (2026-08-10
    skew-explosion fix). A second clamp here would either double-apply the
    cap or diverge from the re-derived r_x that make_quote journals -- the
    cap lives in exactly one place, make_quote.
    """
    base = (1.0 / gamma) * np.log1p(gamma / k)
    C = np.sqrt(sigma_b * sigma_b * gamma / (2.0 * k * A)) * (1.0 + gamma / k) ** (
        1.0 + k / gamma
    )
    delta_b = base + ((2.0 * q + 1.0) / 2.0) * C
    delta_a = base - ((2.0 * q - 1.0) / 2.0) * C
    return float(delta_b), float(delta_a)


def per_share_skew_x(
    variant: str,
    sigma_b: float,
    gamma: float,
    k: float,
    A: float,
    tte_days: float,
) -> float:
    """Per-share reservation displacement (skew_x / q) for the given variant.

    Shared by `make_quote` (via the skew_x_cap bind-point sizing) and
    `robustness_sizing`'s skew-aware entry cap (2026-08-10 skew-fix wave item
    2), so the two agree on the exact same displacement-per-share the quote
    engine would produce. Pure function, no clamping -- returns the raw
    per-share term:

      * "dalen": gamma * sigma_b^2 * tte_days  (Dalen Eq 9 skew term / q)
      * "glft":  C = sqrt(sigma_b^2 * gamma / (2*k*A)) * (1 + gamma/k)^(1 + k/gamma)
                 (the GLFT skew term / q; no tte dependence -- GLFT is
                 stationary)

    Unknown variant raises ValueError, mirroring make_quote's dispatch (a bad
    variant must fail loudly rather than silently returning 0.0 and making
    the sizing cap inert).
    """
    variant = str(variant).lower()
    sigma_b = max(_finite(sigma_b, 0.0), 0.0)
    gamma = _finite(gamma, 0.0)
    k = _finite(k, 0.0)
    # Mirror make_quote's positivity guards so a misconfigured caller fails
    # with the same legible ValueError instead of a bare ZeroDivisionError
    # (the harness calls this helper BEFORE make_quote_from_config each tick).
    if gamma <= 0.0 or k <= 0.0:
        raise ValueError("gamma and k must be positive")
    if variant == "dalen":
        tte_days = max(_finite(tte_days, 0.0), 0.0)
        return float(gamma * sigma_b * sigma_b * tte_days)
    elif variant == "glft":
        A = _finite(A, 0.0)
        if A <= 0.0:
            raise ValueError("A must be positive for the glft variant")
        C = np.sqrt(sigma_b * sigma_b * gamma / (2.0 * k * A)) * (
            1.0 + gamma / k
        ) ** (1.0 + k / gamma)
        return float(C)
    else:
        raise ValueError("unknown variant: " + repr(variant))


def _params_id(
    variant: str,
    gamma: float,
    k: float,
    A: float,
    sigma_b: float,
    tte_days: float,
    p_lo: float,
    p_hi: float,
    arrival_denominator: str = "k",
) -> str:
    """Short deterministic fingerprint of the parameter values used."""
    key = "|".join(
        [
            variant,
            "g=%.10g" % gamma,
            "k=%.10g" % k,
            "A=%.10g" % A,
            "sb=%.10g" % sigma_b,
            "tte=%.10g" % tte_days,
            "plo=%.10g" % p_lo,
            "phi=%.10g" % p_hi,
            "ad=%s" % arrival_denominator,
        ]
    )
    return hashlib.sha1(key.encode("ascii")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def make_quote(
    market_id: str,
    x_fair: float,
    q: float,
    tte_days: float,
    sigma_b: float,
    gamma: float,
    k: float,
    A: float,
    p_lo: float = logodds.DEFAULT_P_LO,
    p_hi: float = logodds.DEFAULT_P_HI,
    variant: str = "dalen",
    ts: Optional[datetime] = None,
    arrival_denominator: str = "k",
    skew_x_cap: float = 0.0,
) -> QuoteProposal:
    """Build a `QuoteProposal` (contract 4.4) for one contract.

    variant: "dalen" (default) or "glft". arrival_denominator (dalen variant
    only, decision D3): "k" = Dalen Eq 9 verbatim; "gamma" = classical AS.
    All conversions route through logodds; delta_x is floored positive so
    x_bid < x_ask strictly.

    skew_x_cap (2026-08-10 skew-explosion fix): caps the reservation
    displacement at |skew_x| <= skew_x_cap x-units (see module docstring).
    <= 0 (the default here) disables the cap -- legacy unbounded skew_x,
    byte-identical to the pre-fix behavior. `make_quote_from_config` threads
    `MMConfig.skew_x_cap` (default 1.0, i.e. ON) through this argument.
    """
    x_fair = _finite(x_fair, 0.0)
    q = _finite(q, 0.0)
    tte_days = max(_finite(tte_days, 0.0), 0.0)
    sigma_b = max(_finite(sigma_b, 0.0), 0.0)
    gamma = _finite(gamma, 0.0)
    k = _finite(k, 0.0)
    A = _finite(A, 0.0)
    skew_x_cap = _finite(skew_x_cap, 0.0)

    if gamma <= 0.0 or k <= 0.0:
        raise ValueError("gamma and k must be positive")

    # Step 0 (2026-08-10 skew-explosion fix, round-2 nit 5): hoist the
    # x-band bounds and clamp x_fair into them, UNCONDITIONALLY (not gated
    # on skew_x_cap) and BEFORE the variant dispatch. Production x_fair
    # always arrives already band-clamped through logodds.logit, so this is
    # a no-op in practice; it exists so the |skew_x| <= skew_x_cap invariant
    # below holds even for an out-of-band x_fair passed directly to this
    # public function. Reused (not recomputed) at the step-2 band clamp.
    x_lo, x_hi = logodds.logit_bounds(p_lo, p_hi)
    x_fair = float(min(max(x_fair, x_lo), x_hi))

    variant = str(variant).lower()
    if variant == "glft":
        if A <= 0.0:
            raise ValueError("GLFT variant requires arrival scale A > 0")
        r_x, delta_x, skew_x = _glft_terms(x_fair, q, sigma_b, gamma, k, A)
    elif variant == "dalen":
        r_x, delta_x, skew_x = _dalen_terms(
            x_fair, q, tte_days, sigma_b, gamma, k,
            arrival_denominator=arrival_denominator,
        )
    else:
        raise ValueError("unknown variant: " + repr(variant))

    r_x = _finite(r_x, x_fair)
    skew_x = _finite(skew_x, 0.0)
    delta_x = _finite(delta_x, DELTA_X_FLOOR)
    delta_x = max(delta_x, DELTA_X_FLOOR)

    # Step 1 (2026-08-10 skew-explosion fix): cap the reservation
    # displacement BEFORE the band clamp. cap <= 0 skips this entirely --
    # legacy unbounded skew_x.
    if skew_x_cap > 0.0:
        skew_x_bound = max(-skew_x_cap, min(skew_x_cap, skew_x))
        if skew_x_bound != skew_x:
            logger.debug(
                "quote_engine: skew_x_cap bound skew_x %.6g -> %.6g "
                "(variant=%s, cap=%.6g)",
                skew_x, skew_x_bound, variant, skew_x_cap,
            )
        skew_x = skew_x_bound
        r_x = x_fair + skew_x

    # Step 2: bound reservation to the x-band so the quotes stay inside the
    # p-clamp (pre-existing; reuses the (x_lo, x_hi) pair from step 0 rather
    # than recomputing it).
    r_x = float(min(max(r_x, x_lo), x_hi))

    # Step 3 (2026-08-10 skew-explosion fix), GATED on skew_x_cap > 0:
    # re-derive skew_x from the (possibly band-clamped) r_x so the identity
    # x_fair == r_x - skew_x is EXACT whenever the cap is enabled -- the
    # band clamp only ever shrinks |r_x - x_fair|, so |skew_x| <= skew_x_cap
    # still holds given step 0's x_fair clamp. cap <= 0 must be an EXACT
    # legacy revert -- legacy deliberately keeps the pre-existing
    # band-clamp identity offset, so this step is skipped in that case.
    if skew_x_cap > 0.0:
        skew_x = r_x - x_fair

    x_bid = r_x - delta_x
    x_ask = r_x + delta_x
    p_bid_raw = float(logodds.sigmoid(x_bid, p_lo, p_hi))
    p_ask_raw = float(logodds.sigmoid(x_ask, p_lo, p_hi))

    return QuoteProposal(
        ts=ts if ts is not None else datetime.now(timezone.utc),
        market_id=str(market_id),
        r_x=r_x,
        delta_x=delta_x,
        skew_x=skew_x,
        sigma_b=sigma_b,
        params_id=_params_id(
            variant, gamma, k, A, sigma_b, tte_days, p_lo, p_hi,
            arrival_denominator=arrival_denominator,
        ),
        x_bid=x_bid,
        x_ask=x_ask,
        p_bid_raw=p_bid_raw,
        p_ask_raw=p_ask_raw,
    )


def make_quote_from_config(
    config,
    market_id: str,
    x_fair: float,
    q: float,
    tte_days: float,
    sigma_b: float,
    variant: str = "dalen",
    ts: Optional[datetime] = None,
) -> QuoteProposal:
    """Convenience wrapper pulling gamma/k/A, the p-clamp, the arrival-
    denominator setting (decision D3), and the skew_x_cap (2026-08-10
    skew-explosion fix) from an MMConfig."""
    p_lo, p_hi = config.p_clamp
    return make_quote(
        market_id=market_id,
        x_fair=x_fair,
        q=q,
        tte_days=tte_days,
        sigma_b=sigma_b,
        gamma=config.gamma,
        k=config.k_arrival,
        A=config.arrival_scale_A,
        p_lo=p_lo,
        p_hi=p_hi,
        variant=variant,
        ts=ts,
        arrival_denominator=getattr(config, "arrival_denominator", "k"),
        skew_x_cap=getattr(config, "skew_x_cap", 0.0),
    )
