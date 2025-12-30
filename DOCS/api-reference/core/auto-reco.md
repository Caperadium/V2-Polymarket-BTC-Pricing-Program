# Auto-Reco Strategy Module

`core.strategy.auto_reco`

The **Auto-Reco** module is the core decision-making engine of the trading system. It implements a **3-stage pipeline** to transform market data and model probabilities into executable trade recommendations (Delta Intents).

## Key Features

- **3-Stage Pipeline**:
    1.  **Build Targets**: Identifies ideal positions (`TargetPosition`) based on EV/Kelly.
    2.  **Compute Deltas**: Calculates difference between target and current exposure.
    3.  **Determine Actions**: Applies caps, filters, and generates final BUY/SELL/HOLD actions (`DeltaIntent`).
- **Kelly Sizing**: Uses fractional Kelly criterion ($f^* = p - q / 1-q$) scaled by volatility gate and stability metrics.
- **Risk Controls**:
    - **Exposure Caps**: Limits per-expiry and total portfolio exposure.
    - **Staleness Checks**: Penalizes or blocks stale batch data.
    - **Churn Prevention**: Hysteresis and minimum trade thresholds.
- **Target Roles**: explicitly classifies targets as `ENTRY` (new), `EXIT` (reduce), `HOLD_SAFETY` (keep), or `NEUTRAL`.

## Usage

```python
from core.strategy.auto_reco import recommend_trades, RebalanceConfig
from core.strategy.vol_gate import compute_vol_gate

# 1. Load data
batch_df = ... # your fitted probability data
positions_df = ... # current wallet positions

# 2. Compute Vol Gate (market regime)
vol_res = compute_vol_gate(btc_price_df)

# 3. Configure Strategy
config = RebalanceConfig(
    bankroll=1000.0,
    min_edge=0.06,
    kelly_fraction=0.15,
    max_capital_total_frac=0.35
)

# 4. Generate Recommendations
intents = recommend_trades(
    batch_df,
    positions_df,
    vol_gate_result=vol_res,
    config=config
)

# 5. Execute
for intent in intents:
    print(f"{intent.action} {intent.key}: ${intent.amount_usd}")
```

## API Reference

::: core.strategy.auto_reco
    options:
      show_root_heading: true
      show_source: true
