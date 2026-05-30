# BTC Polymarket Pricing

Welcome to the documentation for the **V2-BTC-Contract-Pricing** system.

This system prices Bitcoin binary options on Polymarket using a GARCH(1,1) + Student-t + Jump Diffusion Monte Carlo engine, calibrates probability curves to market data, and generates trade recommendations through a 3-stage strategy pipeline with Kelly sizing and multi-layered risk controls.

## Key Features

- **GARCH + Jump Diffusion Pricing Engine**: Monte Carlo simulation with GARCH(1,1) volatility, Student-t errors, and Kou double-exponential jumps
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
- [Strategy Pipeline](concepts/strategy-pipeline.md) — 3-stage Target → Delta → Action
- [Configuration](getting-started/configuration.md) — All 24 strategy parameters
- [API Reference](api-reference/core/auto-reco.md) — Core module documentation

