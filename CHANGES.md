# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## fix(mm): inventory-skew explosion fix wave (3 items) -- 2026-08-10

Fixes the -7.4 incident of 2026-08-10 13:53-14:22 UTC (64k Aug-11 ladder):
the AS/GLFT inventory skew `-q*gamma*sigma_b^2*tte` is UNBOUNDED in q; a
13.4-share belly fill (the first big fill the 08-08 epoch's belly reopen
allowed) at a genuine sigma_b 2.46 vol spike produced skew_x = -8.83
log-odds, pinned the reservation at logit(0.001), and the bot liquidated a
winning position at ~0.17 into a 0.71 market. Journal-verified forensics +
cross-ladder control (08-12 stable through the window) in
temp/mm_skew_fix_plan.md section 0; sigma_b measured HONEST (realized
x_fair vol ~1.0/sqrt-day vs journaled avg 0.9) -- no estimator recal.
Latent since launch; pre-wave markout clamps kept belly sizes at 0-2
shares so the term was never armed. Plan approved through 3 adversarial
review rounds; final diff ALL CLEAR; 948 tests pass.

1. Skew displacement cap (quote_engine.py, `skew_x_cap` = 1.0 x-units,
   <= 0 disables = exact legacy revert): make_quote clamps x_fair into the
   logit band (step 0, hoisted logit_bounds), clamps skew_x to +-cap and
   rebuilds r_x (step 1), runs the legacy band clamp (step 2), then
   re-derives skew_x = r_x - x_fair (step 3, gated on cap>0) so the
   diagnostic identity x_fair == r_x - skew_x is EXACT whenever the cap is
   on. Clamp lives in make_quote only; new pure helper
   `per_share_skew_x(variant, ...)` (exported) returns the variant-correct
   per-share shift with make_quote-mirrored ValueError guards. Incident
   regression test: q=13.42/sigma_b=2.46 -> ask ~0.77 into a 0.71 market,
   not 0.14.
2. Skew-aware entry cap (robustness_sizing.py Stage 6b, harness.py,
   contracts.py SizingCap.SKEW registered in _CAP_ORDER): add-side shares
   capped at q_skew_max = `skew_q_headroom_mult` (1.5) * skew_x_cap /
   unit_skew_x (threaded from harness via per_share_skew_x, same sigma_b/
   variant as the quote engine -> sizing and quoting agree on the bind
   point); reduce side unconstrained by construction; no floor-back;
   max_add_yes/no journal fields min'd only over active caps. Incident
   replay: 13-17-share bids become ~2.3 shares at sigma_b 2.46; ~17 at
   calm sigma_b 0.9. Structurally inert near expiry (documented).
3. Bankroll update tempering (fair_value_anchor.py,
   `bankroll_update_temper` = 0.1, 1.0 = legacy): Bayes factors tempered
   factor**t before each unpinned region update -- belly weights were
   flipping full-range (0.02 <-> 0.98 pricer) within hours (pre-existing
   instability, pre-dates the pin wave), and the pricer-rich phases
   produced the rich oversized entries that armed the skew; tempering
   slows a full flip ~10x (~5-7.5h). Rate bound, not an attractor fix
   (documented); pinned wing untouched.

Deferred follow-up recorded in the plan: harness passes RAW SHARES as the
AS q (docstring says caller normalizes) -- root cause of the term's scale
and why gamma is uncalibratable; normalizing q changes every
gamma-dependent calibration at once, deliberately not this wave.

Files: market_maker/{config,quote_engine,fair_value_anchor,
robustness_sizing,harness,contracts,__init__}.py; tests: quote_engine
(+14), fair_value_anchor (+11, LEGACY_CFG now disables pin AND temper),
robustness_sizing (+11), harness_ws1 (+3), contracts (enum set);
CLAUDE.md, DOCS/concepts/market-making.md (4.7, 6.3, 8.2 Stage 6b),
DOCS/guides/market-maker-deployment.md (knobs + post-deploy watch). Kill
switches: skew_x_cap<=0, skew_q_headroom_mult, bankroll_update_temper=1.0.
Post-deploy watch: journaled |skew_x| <= 1.0 on every quotes row; no fill
>2 half-spreads through mid; belly full flip takes >6h; |q| <= ~1.5x
clamp-bind quantity.
