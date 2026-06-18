# BTC Polymarket Pricing

Welcome to the documentation for the **V2-BTC-Contract-Pricing** system.

This system prices Bitcoin binary options on Polymarket using a GARCH(1,1) + SVCJ (Kou Double Exponential Jump Diffusion with correlated volatility jumps) Monte Carlo engine, with optional Hansen skewed-t innovations and FIGARCH long-memory volatility. It incorporates 3-state HMM regime detection, MAD-based jump calibration, Basel VaR backtesting, and XGBoost directional prediction.

## Key Features

- **SVCJ Pricing Engine v2**: Monte Carlo simulation with GARCH(1,1) volatility, Kou double-exponential jumps, correlated volatility jumps (SVCJ), Hansen skewed-t innovations, and FIGARCH long memory — all on hourly steps
- **Regime-Conditional Pricing**: 3-state HMM (bear/sideways/bull) with post-hoc regime weighting of independent MC simulations
- **Historical Jump Calibration**: MAD-based jump detection + MLE estimation of Kou parameters + SVCJ vol jump params from BTC hourly returns
- **Basel VaR Validation**: Kupiec POF traffic light test + expected shortfall (Acerbi-Szekely) at multiple horizons and confidence levels
- **Directional XGBoost**: P(up) classifier on BTC momentum + macro features (Gold, DXY, VIX, SPX) with 30% blend weight
- **Probability Calibration**: Logistic curve fitting with logit-shift calibration per expiry
- **3-Stage Strategy Pipeline**:
    - **Stage 1 — Build Targets**: Identify +EV positions with directional consistency constraints
    - **Stage 2 — Compute Deltas**: Calculate position changes using fractional Kelly sizing
    - **Stage 3 — Determine Actions**: Apply risk controls, caps, and generate final BUY/SELL signals
- **Risk Controls**: Volatility gating, per-expiry caps, total exposure limits, staleness controls, exit hysteresis, correlation penalties
- **Polymarket Integration**: Full CLOB API integration for order book enrichment and trade execution
- **Interactive Dashboard**: 8-tab Streamlit monitoring dashboard with backtesting and parameter sweep capabilities
- **Comprehensive Backtesting**: Chronological time-travel simulation with decile-conditioned Monte Carlo shuffle tests

## Quick Links

- [Quickstart](getting-started/quickstart.md) — Set up and run your first pipeline
- [Architecture](concepts/architecture.md) — System overview and data flow
- [Pricing Engine](concepts/pricing-engine.md) — GARCH+SVCJ+FIGARCH model specification
- [Jump Calibration](concepts/jump-calibration.md) — MAD-based Kou parameter estimation
- [Regime Detection](concepts/regime-detection.md) — 3-state HMM regime classification
- [Directional XGBoost](concepts/directional-xgb.md) — P(up) classifier with 30% blend weight
- [Calibration Metrics](concepts/calibration-metrics.md) — Brier score, ECE, reliability diagrams
- [Macro Data](concepts/macro-data.md) — Gold, DXY, VIX, SPX feature pipeline
- [Basel Validation](concepts/basel-validation.md) — Kupiec POF VaR backtesting
- [Strategy Pipeline](concepts/strategy-pipeline.md) — 3-stage Target → Delta → Action
- [Configuration](getting-started/configuration.md) — All 24 strategy parameters
- [API Reference](api-reference/core/auto-reco.md) — Core module documentation
