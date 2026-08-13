# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## fix(mm): bleed-2 wave -- skew q-normalization + post-only book clamp

Diagnoses and fixes the post-un-stall bleed (-$3.19 over 15 maker fills on
2026-08-13, all lost at fill time; quotes-journal forensics traced ~$2.4 to a
skew-cap-pinned buy-high/sell-low oscillation and the rest to belly consensus
richness executing through crossed resting quotes). Plan-reviewed in two rounds
(temp/mm_bleed2_fix_plan.md, v3).

Item 1 -- skew q-normalization (root fix, deferred since the 2026-08-10 skew
wave): new `MMConfig.skew_q_norm` (20.0; 1.0 = exact legacy raw-share kill
switch, <= 0 -> 1.0 with one init warning). The harness now passes
`q / skew_q_norm` to `make_quote_from_config` and divides Stage 6b's
`unit_skew_x` by the same factor, implementing the caller-side unit
normalization `quote_engine`'s contract always specified. Deliberate 20x cut of
the skew gain: per-share reservation shade drops from ~7-15c (3-5 shares pinned
the +-1.0 skew_x_cap) to ~0.2-0.5c at the belly. Accepted consequence
(documented with loss accounting in the plan): Stage 6b's q_skew_max scales up
20x and binds only in extreme-sigma states; the operating per-strike bound is
q_max plus the per-tick flow caps. gamma left untouched (it is shared with the
vol-adaptive half-spread term; cutting it would strip protective widening).

Item 2 -- post-only book clamp (structural backstop, both faucets): new pure
idempotent `spread_builder.post_only_clamp`, wired in `harness.tick` after PAV
repair and before the size-skew stage, gated on new
`MMConfig.post_only_margin_ticks` (1; <= 0 disables exactly). Bounds each
DESIRED side to margin ticks inside the opposite venue touch (bid <= best_ask -
tick, ask >= best_bid + tick; outward-only, NaN-guarded, band-clamped,
degenerate sides zeroed, suppressed sides never resurrected). Live intent is
post-only maker orders; the paper fill sim otherwise fills a resting crossed
bid at OUR price with queue_ahead=0 -- the execution path of every bleed-2
fill. Clamp displacement journaled additively as `post_only_bid`/
`post_only_ask` terms keys; `QuoteSet.noarb_checked` semantic documented as
"PAV repair ran on the desired ladder" (the clamp can break ask-ladder
monotonicity harmlessly, never the exploitable ask_K < bid_{K+1}). Sizing runs
pre-clamp and is conservative under it on both Kelly legs.

Files: market_maker/config.py, harness.py, spread_builder.py, quote_engine.py
(docstring), robustness_sizing.py (comment), contracts.py (noarb_checked
comment), paper_fill_sim.py (docstring); tests/test_mm_skew_q_norm.py (new, 6),
tests/test_mm_post_only_clamp.py (new, ~26), tests/test_mm_harness_ws1.py (2
threading assertions updated); CLAUDE.md, DOCS/concepts/market-making.md,
DOCS/guides/market-maker-deployment.md. Full suite 984 passed (baseline 948).

Notes: Stage-A shadow quotes change under both items -- historical shadow
comparisons are not apples-to-apples across this deploy. Epoch bumped at
deploy per plan recommendation: `markout_epoch_utc` -> 2026-08-13T23:45Z, so
the 15 crossed-price artifact fills (the sim defect item 2 removes) never
seed the first sizing/term-7 cells to reach n=20; all measured markout
channels were inert at n=15 anyway, so no armed protection is reset.
