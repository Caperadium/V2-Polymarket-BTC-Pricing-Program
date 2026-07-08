# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

- Stage-A shadow runs from the dev machine caught and fixed TWO real quoting
  defects (this is exactly what shadow mode is for): (1) sigma_b estimation
  annualized 30s REST-mid microstructure jitter into belief vol (~2.5-3.5 per
  sqrt-day, near the cap) -- new `MMConfig.sigma_b_sample_s` (300s) and the
  harness now estimates on a newest-anchored subsample of the consensus-x
  history; (2) ARRIVAL MARKUP DOUBLE-COUNT: spread_builder added a raw
  1/kappa markup (~23c at the belly with launch k=1) on top of the quote
  engine's delta_x which already carries Dalen's arrival term -- markup is
  now AUDIT-ONLY in the terms dict (same treatment the plan mandates for the
  skew term); decomposition test updated. Two live runs on the
  bitcoin-above-on-july-10-2026 ladder (11 strikes, 30s ticks, 5-min
  re-price cache over the real SVCJ+skewed-t+FIGARCH engine, real vol gate):
  58 + 7 ticks, zero self-inflicted no-arb violations, zero book-fetch
  failures, correct DEGENERATE pull on the one-sided 52k book, no quote
  churn, model p_hat tracked market mid within 1-3pp across the ladder,
  re-price 8.5s warm / 18s cold. Full MM suite: 249 passed.

- Add `market_maker/shadow_runner.py` -- Stage-A SHADOW runner executable from
  any machine with network (no VPS needed): resolves a bitcoin-above event's
  ladder via the Gamma API, REST-polls live CLOB books into the existing
  PaperTradingLoop (BookMirror snapshot messages; no prints -> fill-free by
  construction, strictly read-only against the venue), wraps the REAL pricing
  engine in a re-price cache (`CachedEngine`, full live feature set
  SVCJ+skewed-t+FIGARCH + garch_cache, re-prices every --reprice-s), binds the
  real vol gate over fresh intraday data, and journals per-tick market touch
  vs our quotes, modes, credibility, and no-arb status to
  temp/shadow_run/<ts>/{quotes,ticks}.csv + summary.md. Smoke-tested live
  (re-price 18.8s warm; empty-book wing correctly PULLED via the DEGENERATE
  liquidity trigger).

- Implement the three verification-pass decisions (D1/D2/D3): (D1) q_max rule
  becomes `MMConfig.q_max_mode` -- "shrinking" (default, conservative,
  unchanged behavior) or "dalen" (primary's verbatim 1/max(S',eps), dormant
  for later use) in `inventory_manager`. (D2) pricer adapter now fills WING
  strikes' sigma2 (consensus p outside belly_band) from the PARAM_POSTERIOR
  channel (`core/pricing/bayesian_estimation.posterior_probability_bands`),
  per Baker-McHale's total-estimator-error semantics: injectable
  `posterior_fn`, cached per `MMConfig.posterior_refresh_s` (1h), sigma from
  the 90% band width /3.29, never reduced below the MC floor, hard fallback
  to MC on any failure; `sigma2_source` flips to PARAM_POSTERIOR and
  `engine_meta['param_posterior_strikes']` records coverage; controlled by
  `MMConfig.use_param_posterior_wings` (default True). Existing adapter tests
  now stub the posterior channel (they were silently invoking the real MCMC
  -- 4-minute test run caught it). (D3) `quote_engine` gains
  `arrival_denominator` ("k" = Dalen verbatim default, "gamma" = classical
  AS), threaded through `make_quote`/`make_quote_from_config`/`params_id`;
  side-by-side comparison run against the live Polymarket BTC ladder
  (belly touch spread 1c; Dalen ~2.4-4c half-spread at launch params, AS
  ~22c -- uncompetitive) written to
  `Market Maker/verification/spread_settings_comparison.md`; "k" stays
  default. New tests for all three switches. Full MM suite: 249 passed.

- Verify all MM formulas against their primary sources (gates V1/V2/V3 +
  Baker-McHale/GLFT): read Dalen via arXiv HTML (local PDF glyph-ciphered),
  Beuoy/Baker-McHale/Fodra-Labadie/GLFT via extracted PDF text. 10 of 12
  checklist items VERIFIED; results written into
  `Market Maker/verification/formula_verification_checklist.md` (verbatim
  quotes kept). ONE REAL BUG FOUND AND FIXED: the synthesis mis-transcribed
  Dalen's cross-strike hedge ratio as S'_i*S'_j*rho/(S'_j^2*sigma_b^2)
  (drops sigma_bi, inflates beta ~1/sigma_b^2); primary says beta(i<-j) =
  Cov/Var ~= (S'_i/S'_j)*rho. `ladder_hedger.beta_ratio` corrected to
  (S'_i/S'_j)*(sigma_bi/sigma_bj)*rho with unchanged clamps; plan 2.7(c)
  updated; `market_making_synthesis.md` deliberately left untouched but its
  Section 2.6 error is flagged in the checklist. Other outcomes: Dalen Eq 9
  really is (2/k)ln(1+gamma/k) (code correct; divergence from classical AS
  noted as optional Stage-A sensitivity check); Beuoy formula (2) verified
  with w_i = win shares (code's convex combination = exact zero-win-share
  case, m = p*c identity confirmed); Baker-McHale shrinkage verified
  verbatim and its sigma^2 confirmed as TOTAL estimator variance (decision
  D2: keep inert MC-SE wiring vs enable PARAM_POSTERIOR channel);
  Fodra-Labadie local PDF verified genuine (gate V3 satisfied). Remaining
  user decisions: D1 q_max wing direction (Dalen verbatim = 1/max{S',eps},
  cap grows at wings; conservative shrinking cap kept), D2 shrinkage
  variance channel, D3 optional 2/k-vs-2/gamma sensitivity. Full MM suite:
  242 passed.

- Align MM settlement with the venue's confirmed resolution rule
  (`market_maker/settlement_handler.py`): Polymarket resolves YES only if
  price is STRICTLY above the strike, so the outcome comparison changes from
  `spot >= strike` to `spot > strike` (now consistent with the backtester's
  `resolve_outcome_yes`; the pricing engine's `P(S_T >= K)` differs only on
  the measure-zero boundary and is unchanged). Adds an exact-tie boundary
  test (tie -> NO). Also adds `Market Maker/mm_open_items_answers.md`
  (reference doc: primaries-review gates V1-V3, queue-behind vs
  trade-through fill models, q_max wing-cap question, fee-wedge fix,
  settlement rule) and a User-Agent header on the feed probe's REST call
  (Gamma/CLOB return 403 without one). Full MM suite: 242 passed.

- Resolve plan task P0b (market-data boundary): ran the feed probe against
  three live Polymarket BTC markets from the dev machine. Verdict: FULL_L2
  (full price-level depth both sides where two-sided; 24 bid / 21 ask levels
  on the ATM up-or-down market), trade prints with price+size present,
  message types map 1:1 onto BookMirror's snapshot/delta/trade inputs.
  DECISION: queue-behind fill model (fillmodel-v1-queuebehind); the
  trade-through fallback is not needed. Boundary note with evidence and one
  config consequence (quiet books go silent for 80s+, so feed health must
  key off WebSocket ping/pong, not message arrival -- revisit BookMirror
  staleness before Stage A) at
  `Market Maker/verification/p0b_feed_boundary_note.md`.

- Fix the arb-half-life estimator for Polymarket's structural fee wedge
  (`market_maker/liquidity_monitor.py`): YES+NO does not settle at exactly 1,
  so the previous AR(1)-through-origin on |deviation| measured decay toward
  the wrong level (zero) and overstated the half-life. Now stores the SIGNED
  deviation and fits AR(1) WITH intercept (slope is exact for decay toward
  any baseline; the intercept absorbs the wedge). Adds a fee-wedge regression
  test; plan Section 2.9 updated to match. Also adds
  `scripts/utilities/polymarket_feed_probe.py` (plan task P0b tooling):
  read-only probe of the CLOB REST snapshot + WebSocket market channel that
  reports depth-level counts, trade-event availability, and update cadence,
  and prints the FULL_L2 vs TOP_OF_BOOK fill-model decision input. Full MM
  suite: 241 passed.

- Author the market-making implementation plan
  (`Market Maker/mm_implementation_plan.md`) per `prompt.md`: resolves the
  MUST-RESOLVE items against the pricer codebase (sigma2_mc(P_hat) =
  P_hat*(1-P_hat)/n_sims is derivable consumer-side with zero engine change;
  no vol-smile output exists and none is needed -- the MC engine prices
  digitals directly off the simulated CDF, so the BS skew correction is
  N/A and replaced by dense-grid CDF sampling), decomposes 14 components +
  state store with exhaustive interface contracts (Section 4), build DAG,
  conservative paper-trading fill model, gate thresholds, risk register, and
  model-tier tags. Iterated through 3 plan-reviewer rounds to GO (round 1:
  settlement ownership + market-data client blockers; round 2: settlement
  pseudo-fill channel preserving the fills-fold invariant; round 3: GO with
  PnL-authority and UNSETTLEABLE-retry clarifications applied).

- Implement market-maker foundation + quoting/risk components not logged in
  earlier entries: `market_maker/contracts.py` (all Section-4 interface
  dataclasses/enums + VenueAdapter ABC), `market_maker/config.py` (MMConfig
  launch defaults), `market_maker/logodds.py` (clamped logit/sigmoid,
  Jacobians, exact two-point spread conversion, spread floor),
  `market_maker/inventory_manager.py` (I1 -- per-contract/per-ladder q,
  SETTLEMENT fills through the normal channel, R3 metric; NOTE: implements
  the CONSERVATIVE shrinking wing cap `q_max = q_max_scale * max(S'(x),
  floor)` -- deliberate deviation from Dalen's divisive form, recorded as
  plan Open Question 11 pending the V1 source review),
  `market_maker/spread_builder.py` (S1 -- four-term additive spread + the
  launch-default wing/tail widening term), `market_maker/ladder_hedger.py`
  (L1/L2 -- PAV isotonic no-arb repair, vertical-spread offsets, clamped
  cross-strike beta behind a default-off flag), `market_maker/
  robustness_sizing.py` (Z1 -- Kelly -> Baker-McHale on sigma2_ladder ->
  joint-ladder cap -> ruin/bankroll caps -> fractional-Kelly c<=0.5 always
  last), `market_maker/market_data_client.py` (D1 -- BookMirror with seq-gap
  health + FULL_L2/TOP_OF_BOOK capability; live PolymarketFeedAdapter is a
  documented stub pending P0b on the VPS), `market_maker/
  liquidity_monitor.py` (M1 -- realized depth, unsigned Kyle-lambda impact
  proxy, YES+NO arb half-life, regime tags; no direction signals per
  Finding 10), `market_maker/paper_fill_sim.py` (X1 -- conservative fill
  model, all eight plan-6.3 assumptions, trade-through fallback variant),
  `market_maker/risk_controller.py` (R1 -- reuses core/strategy/vol_gate,
  trigger matrix (a)-(f), hysteresis latching, journaled transitions).
  Tests: test_mm_contracts/logodds/inventory_manager/spread_builder/
  ladder_hedger/robustness_sizing/market_data_client/liquidity_monitor/
  paper_fill_sim/risk_controller. Consolidated `market_maker/__init__.py`
  exports (85 symbols). Loosened the flaky 1e-9 tolerance in
  test_invariant5_monotone_credibility_gain to 1e-6 (20-step float
  accumulation). Full suite `tests/test_mm_*.py`: 240 passed.

- Build the market-maker integration test harness (plan task G1, scripted-feed
  subset -- no live data, no backtest replay). Adds `market_maker/harness.py`
  (`PaperTradingLoop`: a THIN one-expiry-ladder orchestrator wiring one tick of
  the full loop -- pricer snapshot with reuse-on-failure -> book mirrors ->
  Beuoy fair value threading BankrollState -> risk directives -> quotes ->
  joint sizing -> spread builder -> MANDATORY ladder-hedger no-arb check/repair
  -> order lifecycle over PaperVenueAdapter -> fill sim -> atomic
  record_fill_and_update_inventory -> filled-order reconciliation; plus
  `settle()`/`restart()` with catch-up and inventory rebuild) and
  `tests/test_mm_integration.py` (10 scripted end-to-end scenarios: happy path,
  closed-loop stability, feed-gap PULL/recover, pricer-failure widen-then-pull,
  forced no-arb repair/reject, inventory-cap one-sided-then-pull, settlement
  close-out + PnL + idempotency, kill/restart reconcile, Beuoy credibility
  motion). The harness carries a `_PaperFillSimBridge` because PaperVenueAdapter
  probes its fill_sim for `place_order/cancel_order` but PaperFillSimulator
  exposes `place/cancel` with a decision_ts and geometric book side -- the
  bridge translates the Side enum + sell-YES-via-buy-NO price convention and
  supplies the SimClock time. No existing files modified; full suite
  `tests/test_mm_*.py` 240 passed (230 + 10).

- Implement market-maker quoting-core components: `market_maker/quote_engine.py`
  (task Q1 -- Dalen AS + GLFT stationary variant in log-odds, separable EWMA
  `estimate_sigma_b`, `QuoteProposal` output with params_id fingerprint) and
  `market_maker/fair_value_anchor.py` (task F1 -- Beuoy bankroll-credibility
  consensus via bucket decomposition + convex-combination fixed point, Bayes
  mark-to-market bankroll update with hard-clip floor, degeneracy fixed-blend
  fallback per risk 8.8). Adds `tests/test_mm_quote_engine.py` and
  `tests/test_mm_fair_value_anchor.py` (25 tests; the five normative invariants
  each covered). No existing files modified.

- Implement market-maker foundation components: `market_maker/pricer_adapter.py`
  (task P1 -- sole boundary to `core/pricing/btc_pricing_engine.py`; one
  engine call over a densified quoted+midpoint grid per refresh, per-strike
  `sigma2_mc = p*(1-p)/n_sims` derived from the engine's own `_meta['n_sims']`
  (never hardcoded), `sigma2_ladder` = max over quoted strikes, confidence-tier
  derivation from `tte_days` against `MMConfig` day boundaries,
  `horizon_gate_active` passthrough, a `stale` flag combining an engine-meta
  passthrough with a snapshot-age check against `MMConfig.pricer_max_age_s`,
  and a monotonicity guard that logs a warning on a non-increasing-in-strike
  violation) and `market_maker/state_store.py` (task T2 -- SQLite/WAL store,
  one file per plan Section 5 schema: `inventory`, `ladder_state`, `orders`,
  `fills` (append-only), `quotes`, `pnl`, `settlements`, `bankrolls`,
  `risk_journal`, `liquidity_windows`; `mark_all_live_orders_unknown()` restart
  helper; `fold_fills_to_inventory()` invariant helper (SETTLEMENT fills
  included, no special-casing); `record_fill_and_update_inventory()` atomic
  transactional write for crash-consistency; `settlements` idempotency guard
  trips only on an existing TERMINAL YES/NO row, UNSETTLEABLE rows are
  overwritten by a later successful settlement). Adds
  `tests/test_mm_pricer_adapter.py` and `tests/test_mm_state_store.py`
  (40 tests: stub-engine sigma2/grid/tier/staleness/monotonicity/kwargs-
  passthrough coverage; full per-table round trips, kill/restart round trip,
  settlement idempotency, and fills-fold-to-inventory invariant coverage).
  `market_maker/__init__.py` updated to export the two new public entry
  points (`build_snapshot`, `MMStateStore` + its small local record types).
  No other existing files modified. Note: observed one flaky, apparently
  pre-existing, numerical failure in
  `tests/test_mm_fair_value_anchor.py::test_invariant5_monotone_credibility_gain`
  on a single run out of ~10 full-suite runs (razor-thin 1e-9 tolerance on a
  hash-order-sensitive float sum); not reproducible afterward and out of
  scope for this task (F1, not P1/T2) -- flagging for a separate look.

- Implement market-maker execution components: `market_maker/order_lifecycle.py`
  (task O1 -- converts `(QuoteSet, RiskDirective)` per market into
  `VenueAdapter` actions with minimal churn against `MMConfig.requote_price_tol`
  / `requote_size_tol`; deterministic sha256 client order IDs over
  `(market_id, side, price, size, source_seq)` mirroring
  `polymarket/intent_builder.py`'s `compute_intent_id` pattern (idempotent
  replays never duplicate orders); cancel-all on `RiskDirective.cancel_all`
  or either mode being PULLED; restart reconciliation (mark LIVE->UNKNOWN,
  reconcile against venue open orders/positions, cancel unknowns/orphans,
  report position discrepancies for a MANUAL trigger); injectable `SimClock`;
  thin `PaperVenueAdapter` wrapping a fill-sim-like object) and
  `market_maker/settlement_handler.py` (task E1 -- 12:00 ET settlement
  instant, settleability + spot resolution replicated from
  `core/backtesting/backtest_engine.py`'s `_expiry_is_settleable` /
  `_settlement_price` / `_spot_as_of` conventions (that file read-only, not
  modified or imported -- its constructor has no injection point for a
  fixture daily-close frame); SYNTHETIC CLOSING FILL emitted through the
  existing `SETTLEMENT`-tagged fill channel so `fold(fills) == inventory`
  holds through resolution with no special case; terminal-only idempotency
  via `MMStateStore.upsert_settlement`; UNSETTLEABLE retry via
  `MMConfig.settlement_retry_window_hours` with escalation flagging;
  `catch_up()` restart-protocol scan). Adds `tests/test_mm_order_lifecycle.py`
  and `tests/test_mm_settlement_handler.py` (23 tests). No existing files
  modified. Two documented deviations: (1) settlement outcome uses `spot >=
  strike` (matches the pricing engine's `P(S_T >= K)` convention per the task
  spec), not the backtester's strict `>`; (2) `catch_up()` takes an explicit
  `registry: {market_id: (expiry_key, strike)}` argument because the
  `inventory` table has no expiry_key/strike columns to scan from directly.

- Wire the live Polymarket WebSocket feed adapter and add the Stage-B paper
  runner. `market_maker/market_data_client.py`: replace the
  `PolymarketFeedAdapter` stub with a real CLOB WebSocket client (one
  connection per ladder via `token_by_market`; background thread + asyncio
  receive loop; translates observed venue payloads `book` -> snapshot,
  `price_change` -> per-token deltas (BUY=bid/SELL=ask, absolute sizes, entries
  filtered to subscribed tokens), `last_trade_price` -> trade;
  `tick_size_change`/unknown ignored; no venue seq so messages carry no "seq"
  and the harness assigns its own; reconnect with exponential backoff, venue
  re-sends the full book on resubscribe). Feed health per the P0b boundary
  note keys off WS ping/pong connection liveness (`healthy()`), NOT message
  arrival -- quiet books go silent 80s+; runners pass `healthy()` as the
  tick's `feed_healthy` override so BookMirror staleness never false-alarms.
  New `market_maker/paper_runner.py` (Stage B): same skeleton as the Stage-A
  shadow runner (reuses its `resolve_event`/`CachedEngine`) but fed from the
  WS adapter, so trade prints reach the queue-behind `PaperFillSimulator` and
  simulated fills flow through inventory + state store; journals quotes.csv /
  fills.csv / ticks.csv / summary.md (fills, ending inventory,
  fold==inventory check). Tests: stub test replaced with 6 translation/drain/
  health tests driven by recorded P0b payloads (no network);
  `tests/test_mm_market_data_client.py` now 15 tests, full MM suite 255
  passing. Verified against the live feed: 30s read-only smoke (4 tokens,
  July-10 ladder) connected, subscribed, and mirrored full L2 books.

- Add the MM monitor dashboard page and engine start/stop control for VPS
  deployment. New `app/pages/mm_monitor.py`: read-only Streamlit page (status
  row with RUNNING/STARTING/STALLED/CRASHED/STOPPED badge + heartbeat age,
  START/STOP/FORCE KILL buttons, PnL cards + equity curve, positions table
  with q/q_max utilization and marks, fills tail, risk/liquidity/settlement
  panels, our-spread-vs-market and tick-latency charts, historical run
  selector, sleep+rerun auto-refresh); reads paper_state.db strictly read-only
  (sqlite mode=ro URI + query_only, never MMStateStore) and CSVs via
  mtime-busted cache with on_bad_lines=skip. New `market_maker/run_control.py`
  (stdlib-only launcher/status side of the control-file protocol:
  pid_alive posix/win32, engine_status with STARTING grace + reprice-aware
  STALLED threshold, start_engine with O_EXCL start lock + parent-side PID
  write + detached spawn + runner.log redirect, request_stop with PID-stamped
  stop file, stop_engine, kill_engine). New `market_maker/pnl_report.py`
  (settlement-aware fill_cash: SETTLEMENT pseudo-fills carry price=payoff_yes,
  regular BUY_NO carries the NO-price; cash_by_market folded from the durable
  fills table each snapshot so restarts cannot corrupt PnL; compute_pnl_rows
  writes per-market + market_id-NULL TOTAL rows via the previously-unused pnl
  table; realized = cash + q*avg_cost identity, unrealized vs mid/consensus,
  worst-case bankroll_utilization; equity = bankroll + realized +
  unrealized_mid with settlement payoffs inside realized). `paper_runner.py`
  gains: --config JSON (new committed template
  `market_maker/paper_run_config.json`; CLI flags override), --control-dir,
  --minutes 0 = indefinite, --btc-refresh-s mtime-based re-read of the BTC
  intraday csv (vol gate + settlement stay fresh on long runs), control-file
  protocol (PID file written before heavy startup work, PID-stamped stop-file
  polling, SIGTERM/SIGINT graceful stop, current_run.json with exit_reason,
  per-run run_meta.json, atomic per-tick heartbeat.json carrying
  tick_s/reprice_s), per-tick settlement gate (loop.settle once past the
  12:00-ET expiry instant) and settle-then-snapshot PnL journaling (TOTAL
  every tick, per-market every 20th). Plan iterated to GO with a plan-reviewer
  agent (3 blockers fixed pre-build: settlement cash sign, settlement never
  invoked, restart cash corruption). Tests: tests/test_mm_pnl_report.py (14),
  tests/test_mm_run_control.py (10), tests/test_mm_paper_runner_control.py
  (5); MM suite now 284 passing. Also fixes a ~1-in-8 suite flake (dummy
  runner in the run_control tests wrote a single heartbeat, leaving RUNNING
  observable for only 3x tick_s; it now heartbeats every loop iteration like
  the real runner) and a production STALLED false-alarm (reprice ticks block
  the loop for minutes; heartbeat now carries reprice_s and the threshold is
  max(3*tick_s, reprice_s + 60)).
