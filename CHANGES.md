# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## fix(mm): un-stall quoting -- epoch bump + term 7 reads the epoch view -- 2026-08-13

0 maker fills in the 2 days after the 08-11 skew deploy: the 08-10
incident's own fire-sale fills (predominantly self-inflicted -- the first
rich bid was Bayes-driven) cap-bound term-7 widening at 0.12/side (belly
60s BUY_NO mk=-0.232, n=28) -> 12-15c half-spreads vs ~1c touch, AND
re-clamped belly sizing (the 07-27 epoch predates the incident, so the
epoch view contained the fills). Plan temp/mm_epoch_term7_fix_plan.md,
reviewed ALL CLEAR in one round.

1. markout_epoch_utc -> 2026-08-11T03:21 (the skew-fix deploy; applies
   the documented OPERATOR RULE missed at that deploy). Belly relearns at
   0.33x exploration; wing sizing keeps the full window. Comment now also
   warns: bump SPARINGLY -- every bump resets the belly slow-channel
   backstop (6h + 20 fills to re-arm).
2. Term 7 (side-asymmetric widening) reads the EPOCH (sizing_report) view
   instead of the full window -- one source for all markets, region basis
   unchanged, guard and arguments both flipped. Reverses the wing-wave
   B5 decision on the merits: its premise (old fills = genuine pick-off
   evidence) fails for self-inflicted incident fills; intra-epoch 28d
   decay preserved; no gate -> no deadlock. Fallback: unwired sizing
   provider degrades to the full view; wired-but-cold yields 0 widening
   (safe: the runner assigns both holder entries together).

Dated follow-up (reviewer): ~Sep 5 the pre-epoch wing fills age out of
the 28d sizing window -> the wing exploration carve-out re-arms with
ZERO term-7 widening (epoch view has no wing fills) -- re-assess then:
accept ~20 fills relearn tuition or restore full-window term 7 for the
wing region only. Post-deploy watch: markout_bid/ask ~0 in the terms
journal within one cadence, ATM half-spreads ~3-4c, first fills within
hours, belly BID fill quality (the one legitimately-toxic signal in the
erased window was a Bayes-driven rich bid).

Files: market_maker/config.py, market_maker/harness.py,
tests/test_mm_harness_ws1.py (routing test flipped + un-stall
regression), CLAUDE.md, DOCS.
