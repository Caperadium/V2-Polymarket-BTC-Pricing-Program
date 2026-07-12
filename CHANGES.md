# Changes

<!-- Append one entry per logical task. Cleared after each push. -->

## MM sizing overhaul: decouple quote sizes from spread calibration (plan-reviewed, temp/mm_sizing_fix_plan.md)

Fixes the sizing layer's self-referential Kelly edge (belief vs OUR OWN raw
proposal quote), which mechanically collapsed flat-inventory quote sizes ~10x
when the spread recal (429e833) cut delta_x -- the design-doc synthesis
(Market Maker/market_making_synthesis.md sec 2.7) specifies edge vs MARKET
price; the implementation plan (sec 2.8) silently substituted "vs our quote".

- robustness_sizing.py: Kelly edge now prices against the market mid
  (ContractSizingInput.mkt_mid, fallback to our quote when the book is
  one-sided/crossed/empty -- old behavior exactly); Baker-McHale uses
  per-strike sigma2 (fallback sigma2_ladder); pipeline restructured --
  fraction space (kelly -> baker-mchale -> bankroll-util -> fractional-c)
  then share space (presence floor -> inventory headroom cap -> depth cap ->
  bucket worst-case cap). The old joint-ladder stand-in (sum f <= max single
  f, over-charged internally-hedged books) and fraction-space ruin scale are
  replaced by a bucket-decomposition worst-case cap: strikes partition
  outcomes into n+1 intervals; cap the MAX single-bucket loss at
  per_expiry_cap_frac (records RUIN; LADDER_JOINT retained in the enum but
  no longer emitted). Presence floor (MMConfig.presence_frac, default 0.005
  of ladder bankroll notional per side, 0 disables) keeps the maker present
  at zero directional edge, tapered by inventory toward each side; caps
  dominate floors. Fractional-c invariant restated: last FRACTION-SPACE
  ceiling; the bucket/ruin cap is a final share-space override.
- contracts.py: SizingCap.INVENTORY added (recorded when headroom binds);
  SizingDecision.max_add_yes/max_add_no added (contract 4.8 "max position
  add"; 0.0 = not computed when inventory absent).
- harness.py: _compose_quote_sets now receives market_states, computes a
  both-sides-present-and-uncrossed sizing mid per market (deliberately NOT
  _market_mid(), which has a one-sided fallback), hoists one
  inv.snapshot(now) per tick and passes it to size_ladder(inventory=...).
- Tests: test_mm_robustness_sizing.py rewritten around the new pipeline
  (12 -> 30 tests, incl. mid-edge decoupling regression, hedged-book-not-
  scaled, floor-dominated-by-caps); test_mm_integration.py inventory-cap
  test mechanism updated (old mechanism required the inventory-blind-sizing
  bug this change fixes); harness wiring test added. Suite 686 passed.
- DOCS/concepts/market-making.md section 8 updated.

Paper-only change; not deployed to the VPS in this task. Follow-up research
(temp/mm_sizing_research.md) recommends a wave 2: size on POSTED spread minus
measured markout (round-trip economics) once markout data accumulates,
reduce-side exemption from the f>=0 floor, and a net-edge-gated floor.
