# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## feat(mm): C1 mid-drift-anchored belly Bayes scoring (SHADOW mode)

Long-term fix for the belly consensus-richness faucet (user decision after
the 2026-08-13 divergence experiment: belly RICH divergence loses to the mid
at settlement in every bucket). Re-anchors the belly credibility update from
the self-confirmation target (consensus built from pre-update weights) to
external evidence: model forecasts from 1h ago scored against the market's
own sanitized bucket distribution now, over the FULL n+1 bucket vector --
full support makes level divergences visible (interior buckets are ladder
differences) and cancels the static mass bias identically (martingale data
-> market model always wins; the v1 subset form failed both, caught in plan
review). Law: factor_market - factor_pricer = (w_p - alpha)*S + noise;
interior fixed point at the S-weighted average close -- the attractor moves
off the 0.98/0.02 self-confirmation corner for the first time (tempering
only ever changed the rate). Plan-reviewed twice
(temp/mm_c1_belly_drift_plan.md v3).

SHADOW-FIRST (quoting-neutral this deploy): belly_score_mode
"legacy"|"shadow" (DEFAULT)|"live". Shadow keeps the applied belly update
legacy; scoring events (belly_drift_interval_s 900s, lag horizon 3600s,
harness lag deque appended only on BEUOY results) journal drift factors +
a rate-matched control (legacy target on the SAME lag-h pair) into the new
bayes_score_log table (28d retention, skip rows with reasons, s_tail_frac
tail-dominance diagnostic, sanitized belly_snapshot for settlement-Brier
reconstruction) and advance two persisted hypothetical trajectories
(bankrolls region keys belly_drift_shadow / belly_legacy_control, held
outside self.bankroll_states). Six quantified acceptance criteria gate the
one-line flip to "live"; acceptance check ~2026-08-21. Wing pin untouched.

Files: market_maker/config.py (5 fields), state_store.py (bayes_score_log
+ append/get/prune + BAYES_SCORE_RETENTION_S), fair_value_anchor.py
(advance_weights shared helper refactor, belly_lag_* kwargs, drift/control
block, 4 additive AnchorResult fields, C1 docstring section), harness.py
(mode validation, lag buffer + scoring gate + trajectories + journaling +
both restart paths), paper_runner.py (prune wiring);
tests/test_mm_belly_drift_scoring.py (34) +
tests/test_mm_belly_drift_harness.py (18); CLAUDE.md, DOCS
concepts/market-making.md (new 4.8), guides/market-maker-deployment.md
(knob table). Full suite 1036 passed (baseline 984).

## feat(mm): C1 shadow weights in telegram /status and /bankroll

scripts/mm_telegram_bot.py: new _c1_belly_weights helper (latest bankrolls
row per (expiry, region) for belly / belly_drift_shadow /
belly_legacy_control; reports only when at least one shadow row exists, so
legacy/live modes and pre-C1 dbs render unchanged). /status gains a compact
"C1 shadow belly w: MM-DD w(nN)" line; /bankroll gains a per-expiry
"applied / shadow / control (events N)" block. Read-only SQL, stdlib-only
discipline preserved. 3 new tests (tests/test_mm_telegram_bot.py, 27 total).
