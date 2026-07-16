# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## MM negative-PnL fix: markout widening + per-region bankrolls + risk-based breach (2026-07-15)

Investigated the VPS paper run's negative PnL (marked -13.05 on ~1000 bankroll,
144 fills): every markout cell negative (~-4c/fill at 60s vs ~3c half-spread),
systematic anchor FV skew vs market (OTM +0.8c rich / ITM -2.9c cheap over
221k-387k quote snapshots), pricer Beuoy weight 0.979 on the far ladder, and
inventory stranded over the S'(x)-shrinking q_max (70k jul-20 at 3.1x cap with
$0.72 remaining loss). Plan (temp/mm_pnl_fix_plan.md) reviewed to GO in two
rounds; three packages implemented:

- Package D (risk-based inventory breach): harness._breaches now emits
  InvBreach with ratio = remaining-loss notional L / cap, L = q*p (long) or
  |q|*(1-p) (short), cap = MMConfig.inv_loss_cap_frac (0.10) * ladder bankroll;
  cap <= 0 emits none. is_long from raw q (phase 1: no hedge adjustment, so
  rules (c)/(f) keep agreeing and can never escalate to PULLED). Sizing
  headroom caps still use the S'-based q_max (mode path only). Resolves the
  deferred 2026-07-14 "option C" gate. Files: market_maker/harness.py,
  market_maker/config.py, market_maker/risk_controller.py (docstrings).

- Package E (markout-fed spread term 7): markout_report cells/by_region gain a
  "sides" key (BUY_YES/BUY_NO sub-stats, lockstep with aggregates); new
  pnl_report.markout_stats_side resolver (cell -> region-rollup fallback);
  spread_builder.markout_widen (clamp(-mk_avg,0,cap)*scale) feeds new
  side-asymmetric compute_posted_prices args markout_widen_bid/ask (terms
  audit keys markout_bid/markout_ask); harness wires both sides per market at
  MMConfig.markout_widen_horizon_s (60s), gated on markout_min_n per side.
  Rolling 7d report window is the decay. PAV interaction characterized:
  mid-ladder no-arb preserved; residual bid-ladder non-monotonicity is not
  exploitable. Files: market_maker/pnl_report.py, spread_builder.py,
  harness.py, config.py.

- Package B2 (per-region Beuoy bankrolls): fair_value_anchor.compute_fair_value
  now takes/returns Dict[belly|wing -> BankrollState]; region per strike from
  the sanitized market ladder, tail buckets always wing; region-restricted
  Bayes factors with per-region skip on empty/degenerate regions (never
  fallback/freeze; skips logged); ladder-space per-strike-weight consensus
  (cummin repair then band clamp -- sanity band passes by construction);
  FairValue.credibility_by_region (additive), legacy credibility = weighted
  average; harness holds bankroll_states dict, legacy bankroll_state is a
  read-only OR/mean property (paper_runner heartbeat unchanged); bankrolls
  table gains a region column ('' = legacy) via a guarded ALTER TABLE
  migration in _init_schema; resume seed: belly inherits the legacy row, wing
  resets to 0.5/0.5 parity. mm_monitor gains a per-region bankroll panel.
  Files: market_maker/fair_value_anchor.py, contracts.py, harness.py,
  state_store.py, app/pages/mm_monitor.py.

Full-diff review (8 finder angles + verification) produced 10 findings: 2
fixed pre-push (unlogged per-region update skip now logged; harness region
literals replaced with BELLY_REGION/WING_REGION constants + .get() on
credibility_by_region), 3 refuted, remainder are logged follow-ups (extract
shared cummin-repair helper, shared sanity-band envelope, markout_stats_side
branch dedup, weight array/dict round-trip, cells-scan index, property
allocation in unfreeze gate). Tests: 765 baseline -> 809 passing.
