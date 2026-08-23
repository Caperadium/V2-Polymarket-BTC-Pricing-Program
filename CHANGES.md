# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## fix(mm): C1 drift-scoring bucket-smoothing epsilon (skip-rate fix)

First 7d of C1 shadow measured a 31.3% scoring-event skip rate, all
`non_finite`: flat adjacent ladder values (deep wings pinned at p_clamp,
quantized plateaus) produce zero buckets, and a zero `c_lag` divisor killed
the whole event. New `MMConfig.belly_drift_bucket_eps` (1e-6; <= 0 disables
exactly -- legacy skip-on-zero): every bucket vector entering the
drift/control ratios and s_tail_frac (`fair_value_anchor._smooth_buckets`)
is additively smoothed, (v + eps)/(1 + len(v)*eps). Full support restored
(finite Bayes ratios), sum stays 1, and the dMass == 0 martingale
cancellation is preserved (all vectors smoothed and renormalized). Applied
ONLY in the C1 drift/control block; the legacy applied Bayes loop's
step-3.3 divisor skip discipline is untouched. Distortion (n+1)*eps ~ 1e-5,
below the per-event noise floor; two exact-golden tests pin eps=0 via the
sentinel.

Also records acceptance review 1 (2026-08-21) in
temp/mm_c1_belly_drift_plan.md addendum: HOLD shadow (C1a 199/199 PASS, C3
FAIL in a one-regime week, skip-rate FAIL fixed by this change), criterion
1b amended to trajectory-level (per-event sign gate is structurally
noise-dominated per the plan's own noise analysis).

Files: market_maker/config.py, market_maker/fair_value_anchor.py;
tests/test_mm_belly_drift_scoring.py (+5 tests, 2 goldens pinned to the
eps=0 sentinel). Full suite 1044 passed (baseline 1039).
