# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## MM sizing wave 2: posted-spread / measured-markout Kelly (plan-reviewed, temp/mm_sizing_wave2_plan.md)

Supersedes wave 1's mid-edge with the literature-correct maker frame
(research: temp/mm_sizing_research.md -- Chen-Pennock utility maker,
realized-spread decomposition): Kelly at our own POSTED quote, belief
haircut by the bot's own measured markout.

- spread_builder.py: compute_posted_prices() extracted from build_quote_set
  (terms 1-6 + floor/quantize + terms dict, bit-identical); build_quote_set
  gains posted= param so the harness computes prices once, BEFORE sizing.
- robustness_sizing.py: ContractSizingInput drops mkt_mid (wave-1-internal,
  no consumers), gains mk_avg/mk_var/mk_n/mk_n_attempted. Per-leg edge:
  m = measured mk_avg (cell n >= markout_min_n) else m_prior =
  (belief - posted_price) - eps_base; m clamped at 0 (Glosten-Milgrom:
  negative net edge = no size); Kelly via kelly_buy(price + m, price).
  Baker-McHale sigma2 = mk_var/mk_n when measured else markout_prior_var --
  per-strike MC-SE no longer shrinks legs (it double-charged the spread bet
  and was the post-recal size killer; param-posterior sigma2 now affects
  spread widening and phi only). Presence floor gated on net edge >= 0 OR
  exploration (cell n_attempted < min_n). Reduce-side exemption: the
  inventory-unload leg is floored at min(|q|, s_presence) UNgated (fixes the
  live defect where skew > half-spread zeroed the unload side). Depth cap
  floored at depth_cap_floor_shares (1.0) so a dead book can be restored.
- pnl_report.py: _summarize gains mk_var; markout_stats() lookup helper
  (cell -> by_region rollup -> null, never raises); tte_bucket_label export.
- harness.py: _compose_quote_sets computes posted prices before sizing,
  feeds them + resolved markout stats into ContractSizingInput;
  PaperTradingLoop gains markout_provider (called once per tick).
- paper_runner.py / multi_runner.py: shared markout_provider threaded
  through the orchestrator into every slot loop; seeded at startup from the
  persistent store's fills; refreshed at the existing periodic
  markout_report write.
- MMConfig: markout_min_n (20), markout_horizon_s (600), markout_prior_var
  ((2*eps_base)^2), depth_cap_floor_shares (1.0).
- Tests: robustness_sizing suite rewritten around posted-edge/markout
  semantics (34 tests); spread-split bit-identity, markout_stats, provider
  threading, gate/exemption/depth-floor tests added. Suite 732 passed.
- DOCS/concepts/market-making.md section 8 updated (incl. the
  param-posterior note).

Known scoped risks (plan Section 7): 7-day markout lag after regime turns;
region-key approximation near the belly boundary (degrades to prior path);
cold-start wings lose param-posterior extra shrinkage (bounded by q_max +
wing widen + bucket cap).

## VPS deferred-deploy script (deploy/deploy_after_rollover.sh)

One-shot script armed via systemd-run on the VPS: waits for the current
nearest expiry's in-process rollover (baseline expiry no longer active +
replacement ladder live in heartbeat), then stop mm-paper -> git reset
--hard + pull --ff-only -> import smoke check -> start -> heartbeat
freshness verify, webhook-notified at every step, fail-safe restart on the
old code if any step fails. Repo is not pulled until trigger time so a
crash-restart before the rollover still boots the old code.
