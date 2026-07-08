"""Pricer adapter (plan Section 2.1, task P1).

Sole boundary to `core/pricing/btc_pricing_engine.py::calculate_probabilities`.
The pricing engine itself is NEVER modified by this module. This module:

- Calls the engine ONCE per (expiry, refresh) with the quoted strikes PLUS a
  densified grid (quoted strikes + midpoints between adjacent quoted strikes),
  then splits the single result dict into `p_hat` (quoted only) and `p_grid`
  (full grid).
- Derives per-strike `sigma2_mc = p_hat*(1-p_hat)/n_sims` (plan Section 1.1
  finding: the per-path indicator is Bernoulli, so its sample variance is
  exactly p*(1-p); n_sims comes from the engine's own `_meta['n_sims']`,
  never assumed/hardcoded here).
- Derives `confidence_tier` from `tte_days` against the MMConfig day
  boundaries, and passes through the engine's own `horizon_gate_active` flag.
- Derives a `stale` flag from two independent sources: whatever the engine
  itself reports in `_meta` (future-proofing; the live engine does not set
  this key today, so it defaults to not-stale) OR'd with a snapshot-age check
  against `MMConfig.pricer_max_age_s` (the `ts` the snapshot is stamped with,
  compared to "now").
- Accepts an injectable `engine_fn` (defaults to the real
  `calculate_probabilities`) so tests can stub it out entirely -- tests must
  NEVER invoke the real engine (it fits GARCH; slow).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from market_maker.config import MMConfig
from market_maker.contracts import ConfidenceTier, PricerSnapshot, Sigma2Source

logger = logging.getLogger(__name__)

# Real engine entry point; imported lazily-by-reference only (never called at
# import time), so importing this module has no GARCH/data-loading cost and
# tests can freely stub `engine_fn` without ever touching this default.
from core.pricing.btc_pricing_engine import calculate_probabilities as _real_calculate_probabilities

# Monotonicity tolerance (float noise guard); the engine's ladder is monotone
# by construction (all strikes drawn from the same simulated path set), so any
# violation beyond this tolerance indicates a caller/engine-boundary bug worth
# surfacing, not a modeling nuance.
_MONOTONE_TOL = 1e-9


def _densify_grid(strikes: List[float]) -> List[float]:
    """Quoted strikes plus midpoints between adjacent quoted strikes, sorted
    ascending, deduplicated. A single-strike ladder returns just that strike.
    """
    quoted_sorted = sorted(set(float(s) for s in strikes))
    midpoints = [
        (quoted_sorted[i] + quoted_sorted[i + 1]) / 2.0
        for i in range(len(quoted_sorted) - 1)
    ]
    grid = sorted(set(quoted_sorted) | set(midpoints))
    return grid


def _confidence_tier(tte_days: float, config: MMConfig) -> ConfidenceTier:
    if tte_days <= config.tier_full_max_days:
        return ConfidenceTier.FULL
    if tte_days <= config.tier_degraded_max_days:
        return ConfidenceTier.DEGRADED
    if tte_days <= config.tier_minimal_max_days:
        return ConfidenceTier.MINIMAL
    return ConfidenceTier.NAIVE_GATED


def _check_monotone(grid_strikes: List[float], p_grid: Dict[float, float], expiry_key: str) -> None:
    """Warn (never raise) if P(S_T >= K) is not non-increasing in K over the
    dense grid. The engine guarantees this per call (same path set for every
    strike), so a violation here signals a boundary bug, not model behavior.
    """
    prev_k = None
    prev_p = None
    for k in grid_strikes:
        p = p_grid.get(k)
        if p is None:
            continue
        if prev_p is not None and p > prev_p + _MONOTONE_TOL:
            logger.warning(
                "pricer_adapter: non-monotone grid CDF for expiry %s: "
                "p(K=%s)=%s > p(K=%s)=%s (expected non-increasing in K)",
                expiry_key, k, p, prev_k, prev_p,
            )
        prev_k, prev_p = k, p


def _real_posterior_bands(*args: Any, **kwargs: Any) -> dict:
    """Lazy import of the slow PARAM_POSTERIOR channel (minutes of runtime)."""
    from core.pricing.bayesian_estimation import posterior_probability_bands

    return posterior_probability_bands(*args, **kwargs)


# Wing-posterior cache: {(expiry_key, wing_strikes): (expires_at, {K: sigma2})}.
# The posterior moves on parameter-estimation timescales, not tick timescales,
# so one compute per posterior_refresh_s is enough (decision D2).
_wing_posterior_cache: Dict[Any, Any] = {}


def _wing_sigma2_from_posterior(
    wing_strikes: List[float],
    hours_to_expiry: float,
    expiry_key: str,
    now_dt: datetime,
    cfg: MMConfig,
    posterior_fn: Callable[..., dict],
    engine_kwargs: Dict[str, Any],
) -> Dict[float, float]:
    """Per-strike parameter-uncertainty variance for wing strikes, cached.

    Variance from the credible band: sigma ~= (q95 - q05) / 3.29 (the
    normal-equivalent width of a 90% interval). Returns {} on any failure --
    quoting must never block on the slow channel (falls back to MC sigma2).
    """
    key = (expiry_key, tuple(round(k, 8) for k in wing_strikes))
    cached = _wing_posterior_cache.get(key)
    if cached is not None and now_dt < cached[0]:
        return dict(cached[1])
    try:
        pb_kwargs: Dict[str, Any] = {}
        for fwd in ("hourly_df", "hourly_csv"):
            if fwd in engine_kwargs:
                pb_kwargs[fwd] = engine_kwargs[fwd]
        bands = posterior_fn(wing_strikes, hours_to_expiry, **pb_kwargs)
        out: Dict[float, float] = {}
        for k in wing_strikes:
            band = bands.get(k) or bands.get(float(k))
            if not isinstance(band, dict):
                continue
            q_hi = band.get("q95")
            q_lo = band.get("q05")
            if q_hi is None or q_lo is None:
                continue
            sigma = (float(q_hi) - float(q_lo)) / 3.29
            if sigma > 0.0:
                out[k] = sigma * sigma
        expires = now_dt + timedelta(seconds=float(cfg.posterior_refresh_s))
        _wing_posterior_cache[key] = (expires, dict(out))
        return out
    except Exception:
        logger.warning(
            "PARAM_POSTERIOR wing channel failed for %s; keeping MC sigma2",
            expiry_key,
            exc_info=True,
        )
        return {}


def build_snapshot(
    strikes: List[float],
    expiry_key: str,
    hours_to_expiry: float,
    *,
    engine_fn: Callable[..., Dict[Any, Any]] = _real_calculate_probabilities,
    config: Optional[MMConfig] = None,
    ts: Optional[datetime] = None,
    now: Optional[datetime] = None,
    posterior_fn: Callable[..., dict] = _real_posterior_bands,
    **engine_kwargs: Any,
) -> PricerSnapshot:
    """Build one immutable `PricerSnapshot` for a single expiry's strike ladder.

    Args:
        strikes: quoted strikes for this expiry (arbitrary order; will be
            sorted ascending in the returned snapshot).
        expiry_key: YYYY-MM-DD ladder identity.
        hours_to_expiry: hours to the 12:00 ET settlement instant.
        engine_fn: callable with the same call signature as
            `calculate_probabilities(strikes, hours_to_expiry, **kwargs)`,
            returning `{strike: prob, ..., '_meta': {...}}`. Defaults to the
            real engine; tests MUST pass a stub.
        config: MMConfig; a fresh default instance is used if omitted.
        ts: snapshot creation timestamp (UTC). Defaults to `now` (or real
            wall-clock if `now` is also omitted).
        now: wall-clock reference for the snapshot-age staleness check.
            Defaults to real wall-clock; tests can pin it for determinism.
        **engine_kwargs: forwarded verbatim to `engine_fn` (e.g. `garch_cache`,
            `s0_override`, `seed`, feature flags). NOT inspected here.

    Returns:
        PricerSnapshot (contract 4.1).
    """
    if not strikes:
        raise ValueError("build_snapshot requires at least one quoted strike")

    cfg = config or MMConfig()
    now_dt = now if now is not None else datetime.now(timezone.utc)
    ts_dt = ts if ts is not None else now_dt

    quoted_sorted = sorted(set(float(s) for s in strikes))
    grid_strikes = _densify_grid(quoted_sorted)

    # ---- Single engine call over the densified grid (quoted + midpoints) ----
    raw = engine_fn(grid_strikes, hours_to_expiry, **engine_kwargs)
    raw = dict(raw)  # defensive copy; do not mutate caller/stub state
    meta = dict(raw.pop("_meta", {}))

    if "n_sims" not in meta:
        raise ValueError(
            "engine_fn did not return _meta['n_sims']; sigma2_mc cannot be "
            "derived without it (plan Section 1.1: never assume n_sims)"
        )
    n_sims = int(meta["n_sims"])
    if n_sims <= 0:
        raise ValueError(f"engine_fn returned non-positive n_sims={n_sims!r}")

    p_grid: Dict[float, float] = {k: float(v) for k, v in raw.items()}
    p_hat: Dict[float, float] = {k: p_grid[k] for k in quoted_sorted if k in p_grid}
    missing = [k for k in quoted_sorted if k not in p_grid]
    if missing:
        raise ValueError(f"engine_fn result is missing quoted strikes: {missing}")

    _check_monotone(grid_strikes, p_grid, expiry_key)

    sigma2: Dict[float, float] = {
        k: (p * (1.0 - p)) / n_sims for k, p in p_hat.items()
    }
    sigma2_source = Sigma2Source.MC

    # Decision D2 (verification pass 2026-07-07): Baker-McHale's sigma^2 is the
    # TOTAL estimator error, and MC-SE is near-zero exactly at the wings where
    # model/tail error is largest. Fill WING strikes (p outside belly_band)
    # from the slow PARAM_POSTERIOR channel, cached per posterior_refresh_s;
    # belly strikes keep MC-SE. Failure or empty result -> MC values stand.
    if getattr(cfg, "use_param_posterior_wings", False):
        lo_b, hi_b = cfg.belly_band
        wing_strikes = [k for k, p in p_hat.items() if p < lo_b or p > hi_b]
        if wing_strikes:
            wing_sigma2 = _wing_sigma2_from_posterior(
                sorted(wing_strikes), hours_to_expiry, expiry_key, now_dt, cfg,
                posterior_fn, engine_kwargs,
            )
            applied = []
            for k, v in wing_sigma2.items():
                # never REDUCE uncertainty below the MC floor
                if v > sigma2.get(k, 0.0):
                    sigma2[k] = v
                    applied.append(k)
            if applied:
                sigma2_source = Sigma2Source.PARAM_POSTERIOR
                meta["param_posterior_strikes"] = sorted(applied)

    sigma2_ladder = max(sigma2.values()) if sigma2 else 0.0

    tte_days = hours_to_expiry / 24.0
    confidence_tier = _confidence_tier(tte_days, cfg)
    horizon_gate_active = bool(meta.get("horizon_gate_active", False))

    engine_stale = bool(meta.get("stale", False))
    age_s = (now_dt - ts_dt).total_seconds()
    age_stale = age_s > cfg.pricer_max_age_s
    stale = engine_stale or age_stale

    s0 = float(meta["S0"]) if meta.get("S0") is not None else float("nan")

    return PricerSnapshot(
        ts=ts_dt,
        expiry_key=expiry_key,
        tte_days=tte_days,
        s0=s0,
        n_sims=n_sims,
        strikes=quoted_sorted,
        grid_strikes=grid_strikes,
        p_hat=p_hat,
        p_grid=p_grid,
        sigma2=sigma2,
        sigma2_ladder=sigma2_ladder,
        sigma2_source=sigma2_source,
        confidence_tier=confidence_tier,
        horizon_gate_active=horizon_gate_active,
        stale=stale,
        engine_meta=meta,
    )
