# BTC Pricing Engine: Feature Summary (Phases 0-2)

Based on 17-paper meta-analysis. All changes backward-compatible via feature flags.

## Phase 0: Basel Baseline + Jump Calibration

| # | Feature | File | Flag |
|---|---|---|---|
| 0.1 | Training start date parameter | `btc_pricing_engine.py` | `training_start_date` param |
| 0.2 | Historical jump calibration | `core/pricing/jump_calibration.py` (new) | N/A (utility) |
| 0.3 | Teng Basel backtest framework | `core/validation/basel_backtest.py` (new) | N/A (utility) |

## Phase 1: Structural Foundations

| # | Feature | File | Flag |
|---|---|---|---|
| 1.1 | Naive prior (μ=0) | `btc_pricing_engine.py` | `use_naive_prior: bool = True` |
| 1.2 | 3-state HMM regime detection | `core/pricing/regime_detector.py` (new) | `use_regime_switching: bool = False` |
| 1.3 | SVCJ volatility jumps (bivariate) | `btc_pricing_engine.py` | `use_svcj: bool = False` |
| 1.4 | Skewed-t innovations (Hansen 1994) | `btc_pricing_engine.py` | `use_skewed_t: bool = False` |
| 1.5 | Horizon-gating | `btc_pricing_engine.py` | Automatic based on T |

## Phase 2: Features & Regime-Conditioning

| # | Feature | File | Flag |
|---|---|---|---|
| 2.1 | Macro data feed | `core/data/macro_fetcher.py` (new) | N/A (utility) |
| 2.2 | Macro-augmented HMM | `core/pricing/regime_detector.py` | `use_macro_features` param |
| 2.3 | Directional XGBoost classifier | `core/pricing/directional_xgb.py` (new) | `use_xgb_direction: bool = False` |
| 2.4 | Regime-conditional jump params | `btc_pricing_engine.py` | Via `regime_params` dict |
| 2.5 | FIGARCH long memory volatility | `btc_pricing_engine.py` | `use_figarch: bool = False` |
| 2.6 | Regime-vol gate interaction protocol | `btc_pricing_engine.py` | Automatic (vol gate overrides) |

## Extended API

```python
def calculate_probabilities(
    strikes, hours_to_expiry,
    # Existing params preserved
    hourly_df=None, intraday_df=None,
    hourly_csv="DATA/btc_hourly.csv",
    intraday_csv="DATA/btc_intraday_1m.csv",
    n_sims=15000, jump_params=None, seed=None,
    # New params (all default-off for backward compat)
    use_naive_prior=True,           # Phase 1.1 — on by default (per synthesis evidence)
    use_regime_switching=False,     # Phase 1.2
    use_svcj=False,                 # Phase 1.3
    use_skewed_t=False,             # Phase 1.4
    use_figarch=False,              # Phase 2.5
    use_xgb_direction=False,        # Phase 2.3
    training_start_date="2019-10-01",  # Phase 0.1
    regime_params=None,             # Phase 2.4
    macro_df=None,                  # Phase 2.2
) -> dict:
```

## Synthesis Evidence Mapping

| Feature | Synthesis Finding | Evidence Strength |
|---|---|---|
| Naive prior (μ=0) | #5 Naive baseline dominance | HIGH (3/0) |
| 3-state HMM | #4 Regime changes / structural breaks | CONCLUSIVE (6/0) |
| SVCJ vol jumps | #2 Jumps essential | CONCLUSIVE (5/0) |
| Skewed-t innovations | #1 Extreme non-normality | CONCLUSIVE (7/0) |
| Horizon-gating | #11 Frequency-dependent predictability | HIGH (3/0) |
| Macro features | #7 Growing macro entanglement | HIGH (4/0) |
| FIGARCH | #8 Long memory in volatility | MODERATE (1/0) |
| Directional XGBoost | #6 Directional vs magnitude trade-off | HIGH (3/0) |
