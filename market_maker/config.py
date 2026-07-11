"""Launch configuration (MMConfig).

Single source of truth for the market-maker launch parameters named across the
plan. Values marked "launch default (pending Stage A/B calibration)" are
deliberately conservative placeholders (plan 8.10: quote wide, size small) and
are expected to be re-estimated from paper-trading fill data. Values with a
plan-stated default carry that default and cite it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from market_maker.contracts import ConfidenceTier


@dataclass
class MMConfig:
    # --- p-band and belly/wing bands ---
    p_clamp: Tuple[float, float] = (0.001, 0.999)  # plan Section 2.2 / 4 header
    belly_band: Tuple[float, float] = (0.2, 0.8)  # wing term active outside this (plan 2.5)

    # --- adverse-selection buffer ---
    eps_base: float = 0.0085  # crypto adverse-selection baseline (+0.85pp, prob units, plan 2.5)

    # --- quote-engine parameters (GLFT/Dalen in x) ---
    # All launch defaults (pending Stage A/B calibration): no fill history exists
    # to estimate arrival decay or risk aversion (plan 8.10, 10.3).
    gamma: float = 0.10  # risk aversion (launch default, pending calibration)
    k_arrival: float = 1.0  # kappa, arrival decay (launch default, pending calibration)
    arrival_scale_A: float = 1.0  # arrival scale A (launch default, pending calibration)
    sigma_b_floor: float = 0.05  # belief log-odds vol floor (launch default)
    sigma_b_cap: float = 5.0  # belief log-odds vol cap (launch default)
    # Minimum sampling interval for the sigma_b estimator. Stage-A shadow run
    # 2026-07-07 finding: estimating on raw 30s consensus diffs annualizes
    # REST mid jitter / spread bounce into sigma_b ~ 2.5-3.5 per sqrt-day
    # (100x vol-term inflation, 63c belly spreads vs 1c market touch).
    # Subsampling the consensus-x history to >= this interval filters the
    # microstructure noise; belief vol is a minutes-scale quantity (plan 10.2).
    sigma_b_sample_s: float = 300.0
    phi_running_penalty: float = 0.01  # running inventory penalty (launch default)

    # --- sizing / robustness ---
    fractional_kelly_c: float = 0.5  # HARD ceiling, never full-Kelly (plan 2.8)
    q_max_scale: float = 100.0  # scale on the q_max rule below (launch default)
    # q_max rule (verification decision D1, 2026-07-07): "shrinking" (default,
    # conservative: q_max = q_max_scale * max(S'(x), s_prime_floor), cap
    # shrinks at the wings) or "dalen" (primary's verbatim form: q_max =
    # q_max_scale / max(S'(x), s_prime_floor), cap grows at the wings bounded
    # by 1/s_prime_floor). Dalen mode is dormant, selectable later.
    q_max_mode: str = "shrinking"
    s_prime_floor: float = 1.0e-3  # eps_cap on S'(x) for q_max/beta (launch default)
    beta_max: float = 5.0  # cross-strike hedge clamp |beta| <= beta_max (launch default)

    # --- wing/tail widening scale per confidence tier (plan 2.5) ---
    wing_widen_scale: Dict[ConfidenceTier, float] = field(
        default_factory=lambda: {
            ConfidenceTier.FULL: 1.0,
            ConfidenceTier.DEGRADED: 1.5,
            ConfidenceTier.MINIMAL: 2.0,
            ConfidenceTier.NAIVE_GATED: 3.0,
        }
    )  # multipliers on the wing term (launch defaults, pending calibration)

    # --- belly widening (temp/suitability.md: belly is the model's softest
    # region, +4.8c bias at 1-2d growing to +8.6c at 5-7d; launch defaults
    # pending Stage-B calibration) ---
    belly_widen_base_p: float = 0.005  # flat extra half-spread inside belly_band (1-2d bias mostly shared with market -> small flat guard)
    belly_widen_slope_p_per_day: float = 0.0075  # belly bias grows ~0.8c/day past belly_widen_free_days -- charge roughly the un-shared half
    belly_widen_free_days: float = 2.0  # no slope inside the 1-2d MM sweet spot; flat base only

    # --- latency / execution (plan 2.12 / 6.3.4 defaults) ---
    placement_latency_ms: int = 2000  # plan default
    cancel_latency_ms: int = 2000  # plan default

    # --- liquidity ---
    volume_discount: float = 2.5  # headline-volume discount (plan 2.9 default)

    # --- staleness / feed health ---
    pricer_max_age_s: float = 300.0  # PricerSnapshot max age -> widen then pull (launch default)
    feed_gap_threshold_s: float = 5.0  # feed gap -> exposed/pull (launch default)

    # --- quote-engine arrival term (verification decision D3, 2026-07-07) ---
    # Dalen Eq 9 (verified from primary) uses (2/k)*ln(1+gamma/k); classical
    # AS/GLFT use (2/gamma)*ln(1+gamma/k). "k" = Dalen (default); "gamma" = AS.
    # Stage-A shadow mode runs both side by side (see verification/
    # spread_settings_comparison.md).
    arrival_denominator: str = "k"

    # --- wing-strike parameter-uncertainty sigma2 (verification decision D2) ---
    # When True, the pricer adapter fills sigma2 for WING strikes (consensus p
    # outside belly_band) from the slow PARAM_POSTERIOR channel
    # (core/pricing/bayesian_estimation.posterior_probability_bands) instead of
    # the near-zero MC standard error, per Baker-McHale's total-estimator-error
    # semantics. Cached; refreshed at most every posterior_refresh_s.
    use_param_posterior_wings: bool = True
    posterior_refresh_s: float = 3600.0  # posterior recompute cadence (slow channel)

    # --- near-resolution / settlement ---
    near_resolution_pull_hours: float = 24.0  # plan 2.10 / 10.8 default
    settlement_retry_window_hours: float = 6.0  # unsettleable retry window (plan 2.13; 24h -> 6h 2026-07-11, escalation is journal-only)

    # --- confidence-tier day boundaries (plan 2.1; tightened per
    # temp/suitability.md -- no backtest coverage past 7 DTE, 28d tail defect) ---
    tier_full_max_days: float = 7.0  # FULL for tte <= 7d (suitability envelope: no coverage past 7 DTE)
    tier_degraded_max_days: float = 14.0  # DEGRADED for 7-14d
    tier_minimal_max_days: float = 30.0  # MINIMAL for 14-30d; NAIVE_GATED above

    # --- ladder machinery gate (plan 10.7) ---
    min_ladder_width_for_ladder_machinery: int = 3  # min strikes to enable ladder-level machinery (launch default)

    # --- requote tolerances (plan 2.11): no re-quote inside tolerance ---
    requote_price_tol: float = 0.005  # price units (launch default)
    requote_size_tol: float = 0.10  # fractional size change (launch default)

    # --- state-store retention (plan Section 5 / Wave 0 W0.2) ---
    quotes_retention_s: float = 14 * 86400.0  # prune `quotes` rows older than this (14d default)

    # --- fair-value staleness (plan Wave 1 W1.2) ---
    fv_max_age_s: float = 300.0  # FairValue max age -> widen then pull (same value as pricer_max_age_s)

    # --- bankroll auto-unfreeze (plan Wave 1 W1.3) ---
    bankroll_unfreeze_clean_ticks: int = 20  # consecutive clean BEUOY ticks before an auto-unfreeze

    # --- promoted phantom config (plan Wave 1 W1.4): these previously lived
    # only as getattr-probed module defaults (fair_value_anchor.
    # DEFAULT_BANKROLL_FLOOR, risk_controller._DEFAULT_LATCH_SECONDS); now
    # real MMConfig fields so the getattr fallback is inert on a fresh
    # config (values match the old module defaults exactly). ---
    bankroll_floor: float = 0.02
    risk_latch_seconds: float = 60.0


def in_belly_band(p: float, belly_band: Tuple[float, float]) -> bool:
    """Inclusive belly-band membership: belly_lo <= p <= belly_hi.
    Single source of truth for spread terms (wing = NOT in belly,
    belly term = in belly) and markout region tagging."""
    return belly_band[0] <= p <= belly_band[1]
