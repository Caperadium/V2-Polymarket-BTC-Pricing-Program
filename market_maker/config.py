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

# Polymarket venue fee parameters (crypto category, docs.polymarket.com,
# verified 2026-07-13). Venue facts, not strategy knobs -- module constants,
# not MMConfig fields. scripts/mm_telegram_bot.py duplicates the product
# 0.20*0.07 = 0.014 in SQL by design (that script is stdlib-only and must not
# import market_maker); keep the two in sync.
# NOTE: contracts.VenueDescriptor already carries dormant maker_fee/
# maker_rebate fields (hardcoded 0.0 in harness.py, consumed nowhere) -- they
# are NOT the source of truth for the maker-rebate accounting layer
# (market_maker/pnl_report.py's rebate_for_fill). Do not wire that knob
# instead of this one.
TAKER_FEE_RATE_CRYPTO = 0.07
MAKER_REBATE_SHARE_CRYPTO = 0.20


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
    # k_arrival history: 1.0 -> 10.0 (2026-07-11 zero-fill recal; the k=1
    # launch placeholder made the Dalen arrival term (1/k)ln(1+gamma/k) ~
    # 0.095 x-units ~ 2.2c/side at ATM -- the single largest term in a
    # 5.7c/side half-spread vs a 0.5c market half-touch, VPS run
    # 20260711_184948 quote journal) -> 18.2 (2026-07-14, FITTED by
    # scripts/mm_calibrate_k.py from 2457 joined trade prints / 289
    # market-hours on the VPS state db: k=18.21, A=3.58/market-hour, implied
    # arrival half-spread ~0.01c ATM). Caveats: only ~3 days of prints
    # (recording began 2026-07-11), lumpy histogram head near the touch;
    # -> 18.3 (2026-07-21 weekly refit, first full volume cycle: 5639 joined
    # prints / 3065 market-hours, k=18.33, A=0.75/market-hour, implied arrival
    # half-spread ~0.007c ATM -- fit stable vs 18.2, term stays negligible);
    # -> 12.8 (2026-08-01 weekly refit #2: 6974 joined prints / 2023
    # market-hours, k=12.75, A=1.05/market-hour, implied arrival half-spread
    # ~0.015c ATM. 30% drop vs 18.3 but stable across sub-windows (5d=11.6,
    # 3d=12.3, both excluding the 2026-07-26 burst weekend) -- genuine flow
    # regime shift, not a burst artifact; term stays negligible).
    k_arrival: float = 12.8  # kappa, arrival decay (fitted 2026-08-01; weekly refit cadence)
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
    # fraction of ladder bankroll notional per quote side, presence floor (plan C3);
    # 0 disables. Caps (depth/inventory/bucket) dominate the floor -- see
    # robustness_sizing.py module docstring for the full pipeline order.
    presence_frac: float = 0.005
    q_max_scale: float = 100.0  # scale on the q_max rule below (launch default)
    # q_max rule (verification decision D1, 2026-07-07): "shrinking" (default,
    # conservative: q_max = q_max_scale * max(S'(x), s_prime_floor), cap
    # shrinks at the wings) or "dalen" (primary's verbatim form: q_max =
    # q_max_scale / max(S'(x), s_prime_floor), cap grows at the wings bounded
    # by 1/s_prime_floor). Dalen mode is dormant, selectable later.
    q_max_mode: str = "shrinking"
    s_prime_floor: float = 1.0e-3  # eps_cap on S'(x) for q_max/beta (launch default)
    beta_max: float = 5.0  # cross-strike hedge clamp |beta| <= beta_max (launch default)
    # risk-based inventory breach cap (package D, 2026-07-15): per-market
    # remaining-loss-notional cap as a fraction of the loop's sizing bankroll
    # (harness._breaches). Replaces the old |q|/q_max ratio, which punished
    # wings hardest exactly where remaining per-share risk is smallest.
    # LAUNCH DEFAULT pending fill data.
    inv_loss_cap_frac: float = 0.10

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
    # near_resolution_pull_hours: 24.0 -> 6.0 (2026-07-11). The 24h plan
    # default pulled quotes for the entire final day of a daily event --
    # combined with ~2d auto-mode lead the bot never quoted 0-1 DTE, the
    # highest-volume regime and the model's 1-2d sweet spot. 6h still clears
    # the settlement-adjacent gap risk window.
    near_resolution_pull_hours: float = 6.0
    settlement_retry_window_hours: float = 6.0  # unsettleable retry window (plan 2.13; 24h -> 6h 2026-07-11, escalation is journal-only)

    # --- confidence-tier day boundaries (plan 2.1; tightened per
    # temp/suitability.md -- no backtest coverage past 7 DTE, 28d tail defect) ---
    tier_full_max_days: float = 7.0  # FULL for tte <= 7d (suitability envelope: no coverage past 7 DTE)
    tier_degraded_max_days: float = 14.0  # DEGRADED for 7-14d
    tier_minimal_max_days: float = 30.0  # MINIMAL for 14-30d; NAIVE_GATED above

    # --- ladder machinery gate (plan 10.7) ---
    min_ladder_width_for_ladder_machinery: int = 3  # min strikes to enable ladder-level machinery (launch default)

    # --- requote tolerances (plan 2.11): no re-quote inside tolerance ---
    # 0.015 = a 1-tick deadband (2026-07-16 boundary-flap fix): quantized
    # prices only ever move in whole ticks (0.01), so the old 0.005 was dead
    # for price -- every 1-tick rounding flap (raw price hovering on an exact
    # tick boundary + sub-tick consensus jitter through spread_builder's
    # _quantize outward floor/ceil) cancelled+reposted the order each tick,
    # losing paper-sim queue position at 15s cadence (observed live 2026-07-16
    # on 64k jul-17: ask 0.80<->0.81 square wave, all spread terms frozen).
    # With 0.015 a 1-tick flap holds the resting order (queue kept); a
    # >=2-tick move still reposts, so the resting order lags the desired
    # price by at most 1 tick, transiently.
    requote_price_tol: float = 0.015  # price units; keep strictly between 1 and 2 venue ticks
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

    # --- ladder mid-velocity pull (Fix 3, risk rule h, 2026-07-26) ---
    # 2026-07-26 VPS diagnosis: all the fill bleed lands INSIDE multi-minute
    # BTC bursts -- fills land ~10c THROUGH the resting mid (stale-quote
    # pick-off), and the vol gate is blind to a live burst because its
    # DATA/btc_intraday_1m.csv only refreshes every 30 min (mm-datafetch.timer).
    # The ladder's OWN mids are visible live, per tick, in the harness; rule
    # (h) (risk_controller) pulls -- or reduce-only when positioned -- when the
    # ladder-wide mid moves more than mid_move_pull_p over the trailing
    # mid_move_window_s. Detection is one tick late by construction (cannot
    # stop the FIRST fill of a burst); it kills the repeated
    # re-quote-into-the-trend fills.
    mid_move_pull_p: float = 0.04  # prob units; ladder-wide mid move over the window that fires rule (h); <= 0 disables
    mid_move_window_s: float = 120.0  # trailing window for the ladder mid-velocity measurement

    # --- markout-based sizing (wave 2 W8) ---
    markout_min_n: int = 20  # min fills in a resolved cell to trust measured markout over the prior
    markout_horizon_s: float = 600.0  # sizing lookup horizon (middle of the 60/600/3600 report horizons)
    markout_prior_var: float = (2 * 0.0085) ** 2  # uninformed sigma2_edge prior, ~2 AS-buffers wide
    depth_cap_floor_shares: float = 1.0  # depth cap never zeroes size below this (venue min order size)
    # Unmeasured-cell size multiplier (Fix 2b, 2026-07-26). A sizing cell with
    # fewer than markout_min_n resolved attempts quotes at this fraction of
    # its full size (reduce side exempt), floored back to depth_cap_floor_
    # shares so it can still fill and become measured. 1.0 disables. Diagnosis:
    # an unmeasured cell's m_prior (~+half-spread) is positive, so both the
    # Kelly path and the presence floor ran full size, paying ~20 cap-sized
    # losses of "tuition" per cell before the m-clamp could turn it off.
    unmeasured_size_mult: float = 0.33

    # --- markout-fed spread widening (package E, spread term 7, 2026-07-15) ---
    # Quoting's counterpart to the markout-based sizing haircut above: widen
    # the posted price on whichever side is measurably getting picked off
    # (spread_builder.markout_widen), instead of only assuming eps_base
    # covers adverse selection. See spread_builder module docstring term 7.
    markout_widen_scale: float = 1.0  # 0 disables the whole term
    markout_widen_cap: float = 0.12  # hard cap per side, prob units (12c)
    # 2026-07-26 VPS measurement: measured 60s side markouts ran -9 to -16c
    # while the prior 5c cap bound everywhere and the bot bled -5c/share over
    # 283 fills; 12c covers most of the toxicity distribution while staying
    # bounded (was 0.05).
    # 60s: the cleanest pick-off signal (measured -4.2c/-3.9c belly/wing at
    # 60s, VPS evidence 2026-07-15) -- deliberately DIFFERENT from sizing's
    # 600s markout_horizon_s above, which measures net edge for Kelly, not
    # pick-off; 600s folds in slower BTC drift variance that dilutes the
    # pick-off signal this term is meant to react to.
    markout_widen_horizon_s: float = 60.0

    # --- wing pricer weight pin (2026-08-08 wing-bleed fix) ---
    # VPS evidence 2026-08-08: the wing region's own Bayes updates re-awarded
    # the pricer ~0.98 weight while every wing YES fill settled worthless (66k
    # NO 6/6 days since 08-01, -3.5 realized). The wing update is a
    # self-confirmation loop (factors score against a consensus built from the
    # pre-update weights), so the wing pricer weight is PINNED and the wing
    # Bayes update is skipped entirely. In [0,1]: pricer weight = pin (clamped
    # into [bankroll_floor, 1-bankroll_floor] at read time so the module's
    # floor invariant holds), remainder to the other models pro rata
    # (all-market in the 2-model case). Negative disables (legacy Bayes).
    # Belly untouched. Runner/harness never read this directly -- only
    # fair_value_anchor.compute_fair_value does.
    wing_pricer_weight_pin: float = 0.5

    # --- slow-horizon sizing haircut (2026-08-08 wing-bleed fix) ---
    # The 600s mid markout is structurally blind to slow theta bleed on
    # low-delta wings (measured -2.7c at 600s vs -10 to -13c realized at
    # settlement, VPS 2026-08-08). A second sizing lookup at this longer mid
    # horizon acts as a strictly ONE-DIRECTIONAL haircut on the Kelly net
    # edge (min(); it can never raise m). 21600 (6h), NOT 86400: quotes pull
    # 6h before settlement (near_resolution_pull_hours), so a 6h markout is
    # resolvable for essentially every fill, while a 24h markout can never
    # resolve for TTE<24h fills (mid_log stops at settlement) -- the
    # highest-volume 0-1d bucket would be permanently unmeasured. <= 0
    # disables (harness skips the lookup AND _leg_edge ignores any supplied
    # slow fields).
    markout_slow_horizon_s: float = 21600.0

    # --- markout epoch (2026-08-08 wing-bleed fix; scope widened
    # 2026-08-13) ---
    # Fills BEFORE this UTC instant are invisible to the EPOCH (sizing)
    # markout report, which feeds (a) BELLY-region sizing (wing sizing
    # keeps the full 28d window -- its measured-toxic verdicts are
    # protective) and, since 2026-08-13, (b) spread term 7's side-widening
    # for ALL markets (the original "term 7 keeps the full window" choice
    # assumed old fills are genuine pick-off evidence; the 2026-08-10
    # skew-incident's own fire-sale fills -- predominantly self-inflicted
    # -- cap-bound the term at 0.12/side and stalled the book for 2 days).
    # Set to the 2026-08-11 skew-fix deploy (start of the current quoting
    # regime; the missed application of the operator rule at that deploy
    # is what caused the stall). OPERATOR RULE: bump at any deploy that
    # materially changes quoting behavior -- but SPARINGLY: every bump
    # also resets the belly slow-channel (21600s) backstop, which needs 6h
    # maturity + 20 fills per cell to re-arm; habitual bumps would keep it
    # permanently unarmed. Empty string disables (sizing view == full
    # view). Runner-only: paper_runner parses it (CLI --markout-epoch
    # overrides); harness/quoting never read it.
    markout_epoch_utc: str = "2026-08-13T23:45:00+00:00"

    # --- sizing-region basis + hysteresis (2026-08-08 wing-bleed fix) ---
    # Basis for the per-market SIZING-region classification.
    # "mid" (default) = live book mid via _market_mid (matches the markout
    # report's mid_at_fill tagging, closing the region-basis mismatch that
    # sustained the exploration-floor wing bleed); "consensus" = legacy
    # behavior (the leaking basis -- kill switch only).
    sizing_region_basis: str = "mid"
    # Hysteresis for the per-market SIZING-region classification.
    # The raw region flips only when the classifying probabilities clear the
    # belly-band edge by this margin (prob units); prevents a boundary market
    # (consensus/mid jitter around 0.20) from flapping its sizing cell between
    # two views with opposite verdicts -- which would alternate resting
    # quotes <-> full cancel each tick (0 <-> N size crosses requote_size_tol)
    # and burn paper-sim queue position. 0 disables (raw region every tick).
    # BOUND: must stay well below (belly_hi - belly_lo)/2 (= 0.3 at the
    # default band); at or above that the wing->belly flip window is empty
    # and every market latches wing forever.
    sizing_region_hysteresis_p: float = 0.02

    # --- inventory-skew displacement cap (2026-08-10 skew-explosion fix) ---
    # The AS/GLFT reservation shift skew_x = -q*gamma*sigma_b^2*tte is
    # UNBOUNDED in q. VPS incident 2026-08-10: a 13.4-share belly fill at
    # sigma_b 2.46 produced skew_x = -8.8 log-odds -- reservation pinned at
    # the p_clamp floor (r_x = logit(0.001)) and the bot liquidated a winning
    # position ~55c under fair (-7.4 realized). Cap the RESERVATION SHIFT at
    # this many x-units: |skew_x| <= skew_x_cap. 1.0 x-unit shifts p at most
    # ~0.5 -> 0.73 (less near the extremes) -- still a strong unload lean,
    # never a fire-sale. <= 0 disables (legacy unbounded).
    skew_x_cap: float = 1.0

    # --- skew q-normalization (2026-08-13 bleed-2 fix, item 1) ---
    # The AS/GLFT skew term skew_x = -q*gamma*sigma_b^2*tte takes q in RAW
    # SHARES; quote_engine.py's module docstring always specified "q is a
    # float (caller normalizes shares by a config unit)" but no caller ever
    # did -- the 2026-08-10 incident (13.4-share fill, sigma_b 2.46 ->
    # skew_x -8.8, fire-sale) is that gap armed. The harness now divides both
    # the quote-engine q and the Stage 6b unit_skew_x by this many shares
    # before the skew term is computed -- a deliberate 20x cut of the skew
    # GAIN (algebraically a skew-only gamma/20; gamma itself is shared with
    # the half-spread and is NOT cut, see temp/mm_bleed2_fix_plan.md).
    # (a) Value rationale: 20 ~ observed live q_max scale (19-22 shares
    #     ATM) -- a full-inventory ATM position lands near 1 skew-unit.
    # (b) Magnitudes at defaults post-normalization: per-share shade =
    #     gamma*sigma_b^2*tte/20 ~ 0.007-0.022 x ~ 0.2-0.5c at belly
    #     (sigma_b 1.2-1.7, tte 1-3d); 5 shares ~ 1-2.5c; MAX lean at full
    #     q_max (25 ATM, calm sigma_b 0.9, tte 1d) is ~0.10 x ~ 2.5c.
    # (c) FIXED norm, not dynamic: q_max ranges ~25 ATM to ~5 deep wing
    #     (q_max_scale*S'), so a full-inventory wing position leans 4-5x
    #     less than a full-inventory ATM one under this fixed unit. The
    #     dimensionless alternative q/q_max (dynamic norm) is the known
    #     follow-up, deferred -- dynamic q_max has prior stranding history
    #     and is out of scope for this wave.
    # (d) Inventory control therefore shifts from the pricing channel to the
    #     caps/hedge channel: skew_x_cap now binds only in extreme-sigma
    #     states (e.g. sigma_b_cap=5.0, tte 4d) -- a catastrophe backstop
    #     again, not the everyday operating regime it was pre-normalization.
    # Semantics: shares per skew-unit. 1.0 = exact legacy raw-share behavior
    # (kill switch). <= 0 (incl. NaN via the `> 0` test) is invalid ->
    # resolved to 1.0 with ONE warning at harness __init__ (precedent:
    # sizing_region_basis validation, harness.py ~265-271).
    skew_q_norm: float = 20.0

    # --- post-only book clamp (2026-08-13 bleed-2 fix, item 2) ---
    # VPS diagnosis (temp/mm_bleed2_fix_plan.md, faucet 1's sibling): a
    # resting bid ABOVE the venue's best ask (or a resting ask BELOW the
    # venue's best bid) is modelled by paper_fill_sim as filling at OUR OWN
    # crossed price with queue_ahead=0 -- a real POST-ONLY maker order would
    # instead be rejected/repriced by the venue. LIVE INTENT IS post-only
    # maker orders, so this clamp is that emulation, not a new strategy
    # layer: spread_builder.post_only_clamp bounds each side of the desired
    # ladder to stay inside the opposite venue touch by at least this many
    # ticks, applied in harness.tick BEFORE the QuoteSet is journaled/sent
    # to the lifecycle (and before the size-skew stage).
    # Sentinel convention (house style, cf. skew_x_cap <= 0): >= 1 is
    # ACTIVE -- ticks of clearance kept inside the opposite touch; <= 0
    # DISABLES the clamp (legacy, byte-identical QuoteSets). 0 is
    # deliberately in the disabled range, NOT "quote exactly at the touch":
    # bid == best_ask would match/take immediately, so the minimum
    # maker-safe margin is 1 tick.
    # Variant decision: the clamp target is the AGGRESSIVE bound
    # (best_ask - margin*tick / best_bid + margin*tick) -- price
    # improvement up to margin ticks inside the opposite touch is still
    # allowed -- NOT join-the-touch (min(bid, best_bid)), which would
    # additionally forbid ALL book improvement everywhere and silently turn
    # the strategy join-only (a strategy decision, out of scope for this
    # safety fix).
    # Known hole (recorded, not fixed here): a ONE-SIDED book (the opposite
    # touch absent/NaN/non-finite/outside (0,1)) leaves that side UNCLAMPED
    # by design -- nothing to cross against -- so unbounded-vs-mid exposure
    # persists exactly on thin, one-sided wing books. Follow-up knob idea:
    # `post_only_join` -- a join-the-touch variant covering that hole, if
    # the belly-consensus-divergence faucet (explicitly out of scope this
    # wave) keeps arming it.
    post_only_margin_ticks: int = 1

    # --- skew-aware entry cap (2026-08-10 skew-fix wave item 2) ---
    # With skew_x_cap binding, additional inventory no longer moves the
    # reservation -- the bot's main inventory-control channel saturates.
    # robustness_sizing.size_ladder (Agent C follow-up) caps the ADD side so
    # the position cannot outrun the skew channel's authority: position may
    # exceed the skew_x_cap clamp-bind quantity by this fraction (some
    # saturation tolerated; fills are the calibration source and the
    # venue-min floor must stay reachable). This cap is a RISK cap and is
    # NOT floored back up to depth_cap_floor_shares (caps dominate floors);
    # a side capped below venue min is a no-quote via the existing
    # order_lifecycle min-size rule. Field only here; unwired until Agent C
    # threads it (0 shares of effect until then).
    skew_q_headroom_mult: float = 1.5

    # --- bankroll update tempering (2026-08-10 skew-fix wave item 3) ---
    # Per-tick Bayes factors at 15s cadence flip a region's weights full
    # range (0.02 <-> 0.98) within hours -- far more weight movement than one
    # tick of mid movement can justify, and the pricer-rich phases create
    # the rich bids + phantom Kelly edges behind the 2026-08-10 incident.
    # Factors are tempered: factor**bankroll_update_temper before the
    # weight update (1.0 = legacy untampered; 0 < t < 1 slows learning; the
    # floor/normalization pipeline is unchanged; factors are non-negative by
    # construction -- ladder_to_buckets clips at 0 -- so factor**t is always
    # real and a zero factor stays zero). 0.1 makes a full-range flip take
    # ~10x as many ticks (~5-7.5h of consistent evidence instead of
    # ~30-45min), clearing the 6h acceptance bar. Tempering changes the
    # RATE, not the attractor -- the 0.98/0.02 corner is still where the
    # self-confirmation dynamic points; this bounds the damage rate. Field
    # only here; unwired until Agent B threads it into fair_value_anchor
    # (no effect until then).
    bankroll_update_temper: float = 0.1


def in_belly_band(p: float, belly_band: Tuple[float, float]) -> bool:
    """Inclusive belly-band membership: belly_lo <= p <= belly_hi.
    Single source of truth for spread terms (wing = NOT in belly,
    belly term = in belly) and markout region tagging."""
    return belly_band[0] <= p <= belly_band[1]
