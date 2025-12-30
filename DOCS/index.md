# BTC Polymarket Pricing

Welcome to the documentation for the **V2-Polymarket-BTC-Pricing-Program**. 

This system automates the pricing, strategy generation, and execution for Bitcoin binary options on Polymarket. It leverages a custom Black-Scholes pricing engine, calibrates probability curves to live market data, and executes trades using a 3-stage recommendation pipeline with Kelly sizing and risk controls.

## Key Features

- **Custom-built Pricing Engine**: Prices BTC binary options using implied volatility and spot prices.
- **Probability Calibration**: Fits probability curves to observed market prices.
- **Automated Strategy**: 
    - **Target Generation**: Identifies +EV trades based on model vs. market probabilities.
    - **Delta Sizing**: Calculates optimal trade sizes using Kelly criterion adjustments.
    - **Risk Management**: Enforces exposure caps, staleness checks, and volatility gates.
- **Polymarket Integration**: Full integration with Polymarket's CLOB API for market data and trade execution.
- **Interactive Dashboard**: Streamlit-based dashboard for monitoring and manual intervention.

## Quick Links

- [Getting Started](getting-started/quickstart.md): Set up the environment and run your first pipeline.
- [Architecture](concepts/architecture.md): Understand the 3-stage strategy pipeline.
- [API Reference](api-reference/core/auto-reco.md): Explore the core strategy modules.
