## 2026-05-29 — Remove 4 dead features from pricing engine

**Intent**: Remove momentum drift, RV blending, momentum gating, and strict_above from `core/pricing/btc_pricing_engine.py`. None of the 4 features were engaged by any caller — all 3 call sites pass default `None`/`False`.

**Scope**:
- `core/pricing/btc_pricing_engine.py`: Removed `MOMENTUM_GATE_MULT` constant, `drift_window` from `fit_garch_model()`, `initial_variance`/`use_momentum_gating` from `simulate_paths()`, `strict_above` from `get_contract_probability()`, and `drift_window`/`rv_intraday`/`rv_blend_weight`/`strict_above`/`use_momentum_gating` from `calculate_probabilities()`. Dropped validation Test 4 (Global Momentum Gating). Net removal: ~45 lines of dead code.
- `CLAUDE.md`: Replaced momentum injection / variance blending bullet points with structural mean drift and jump drift correction summary.
- `DOCS/concepts/pricing-engine.md`: Replaced momentum injection + global gating sections with structural mean drift. Removed RV blending section. Updated API signature, parameter table, and validation test count (5→4).
- `DOCS/guides/troubleshooting.md`: Removed stale `drift_window` troubleshooting tip.

**Verification**: All 4 built-in validation tests pass. All 5 downstream imports confirmed working.
