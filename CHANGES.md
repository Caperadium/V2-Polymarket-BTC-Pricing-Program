# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## Package C: era-conditioned jump calibration (windowed eta_up) + calibration-cache hardening

Adds trailing-window era conditioning to the Kou jump calibration and hardens
the jump-calibration CSV cache. Plan and full audit trail:
temp/mm_package_c_plan.md (6 review rounds), temp/package_c_verification.md,
measurement artifacts temp/package_c_measurement*.json.

- `core/pricing/jump_calibration.py`: `calibrate_jumps` gains
  `window_hours` (default `JUMP_CAL_WINDOW_HOURS = 8760`, i.e. 12 months;
  `None` = byte-identical legacy behavior, golden-pinned). ONLY `eta_up` is
  windowed: the up-jump mean size is re-estimated on the trailing window via
  a MASK-SLICE of the single full-slice Lee-Mykland detection (both legs
  share one critical value C_n -- a fresh short-slice detection uses a
  systematically lower threshold and inflates eta_up), then
  credibility-blended in mean space with `w = min(1, n_window_up_jumps / 6)`
  (`JUMP_CAL_WINDOW_TARGET_UP_JUMPS`, evidence-set). `lam`, `p_crash`,
  `eta_down` and all SVCJ params stay full-slice PINNED. `window_hours <= 0`
  raises ValueError. `JumpCalibrationResult` gains additive fields
  `calibration_window_hours` / `window_weight` / `n_window_jumps` (up-side
  semantics). Flows automatically to backrunner (per-snapshot, leak-free),
  live pipeline and MM (via load_calibrated_jumps) -- deliberate default
  flip for all calibrate_jumps callers, recorded in the plan.
- WHY ONLY eta_up: measured on 10 leak-free historical snapshots
  (production MM engine config), the model's upper tail was rich vs
  recent-era realized exceedance (+0.7..+2.3c at x>=5%, h>=2d) while the
  lower tail and x=2-3% belly were already fair-to-cheap. Windowing all Kou
  params (first design) delivered the tail fix but cheapened the lower
  tail 1-2c and broke the belly (verification FAIL); windowing up-jump
  INTENSITY (lam_up, second design) breached the belly guard both directly
  (shape-blind) and via the jump-drift compensator (whole-curve upward
  drift shift). Final shipped design is the zero-collateral subset:
  measured effect ~0.1-0.2c upper-tail reduction at 1-7d with lower
  tail/belly byte-unchanged. STRUCTURAL LIMIT recorded: the up-side
  mispricing changes sign across strikes (rich tail, cheap belly), so it
  needs a shape change; eta_up is the right shape but its honest era
  signal is small, and the large era signal (jump frequency) is
  shape-blind -- the jump layer alone cannot close the remaining OTM
  richness within collateral guards. Residual stays owned by the E/B2
  quote/consensus mitigations; the MM-only outcome-validated overlay
  (brief approach B) is the documented next option.
- `core/pricing/btc_pricing_engine.py` (`load_calibrated_jumps`): cache
  schema versioning (`JUMP_CAL_SCHEMA_VERSION = 2`, EXACT match -- older,
  missing, or future versions recalibrate; first post-deploy call
  recalibrates automatically, no manual cache delete needed) + stale when
  the cached `calibration_window_hours` differs from the current constant
  (window retunes cannot serve a stale cache). Hardening from the
  post-implementation code review: cache read wrapped (torn/empty/corrupt
  file = stale + self-healing rewrite, was a 30-day crash loop),
  NaN-safe parsing of all optional columns (a NaN cell previously raised
  and silently degraded MM quoting to hardcoded default jumps), atomic
  cache write (tmp + os.replace, data_fetcher idiom), schema_version
  present in the returned dict on both hit and miss paths.
- Tests: +28 (tests/test_jump_calibration_window.py,
  tests/test_load_calibrated_jumps_cache.py incl. an end-to-end real-fit
  integration test); suite 811 -> 839 green; engine self-test ALL TESTS
  PASSED; backrunner 10-snapshot smoke into temp dirs clean (monotone
  ladders, finite probabilities).
- OPERATIONAL NOTES: (1) backtest artifacts (backtested_probabilities/,
  DATA/calibration_shift.csv) predate this change; re-derivation must be a
  CLEAN run into FRESH dirs (idempotent skip would mix pre/post-fix params
  in one series). (2) core/validation/basel_backtest.py pre-calibrates
  jumps once on the full file; with windowing on, its params are now
  end-anchored (final 12 months) -- a sharper look-ahead for historical
  VaR windows; flagged, not fixed here. (3) Live post-deploy expectation
  is pre-registered in the plan: post-consensus OTM skew move of only
  ~0.1c-scale; do not read a small move as failure, and do not escalate to
  the overlay on less than 5 trading days of quotes.
