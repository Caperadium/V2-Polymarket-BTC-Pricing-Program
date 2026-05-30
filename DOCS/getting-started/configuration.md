# Configuration

All strategy parameters live in one place: `sweep_config.py`'s `SweepConfig` dataclass. Both the dashboard sidebar and the parameter sweep CLI tool read from this single source of truth.

## View All Parameters

```bash
python scripts/utilities/parameter_sweep.py --list-params
```

## Parameter Reference

### Strategy Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_edge` | float | 0.06 | Minimum edge (model_prob − market_price) to enter a trade |
| `max_bets_per_expiry` | int | 3 | Max simultaneous contracts per expiry date |
| `max_capital_per_expiry_frac` | float | 0.15 | Max bankroll fraction in one expiry |
| `max_capital_total_frac` | float | 0.40 | Max total bankroll deployed |
| `max_net_delta_frac` | float | 0.20 | Max net directional exposure (Long − Short) |
| `min_price` | float | 0.03 | Minimum contract price to consider |
| `max_price` | float | 0.95 | Maximum contract price to consider |
| `min_model_prob` | float | 0.0 | Minimum model probability filter |
| `max_model_prob` | float | 1.0 | Maximum model probability filter |
| `use_stability_penalty` | bool | True | Penalize Kelly sizing for low-quality curve fits |
| `correlation_penalty` | float | 0.25 | Penalty for correlated positions in same direction |

### Kelly & Sizing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kelly_fraction` | float | 0.15 | Fractional Kelly multiplier (0.15 = 15% Kelly) |
| `use_fixed_stake` | bool | False | Use fixed dollar amounts instead of Kelly |
| `fixed_stake_amount` | float | 10.0 | Fixed stake in USD when `use_fixed_stake=True` |
| `bankroll` | float | 500.0 | Starting bankroll |

### Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_max_dte` | bool | True | Filter by days-to-expiry |
| `max_dte` | float | 2.0 | Maximum allowed days to expiry |
| `use_prob_threshold` | bool | False | Use probability thresholds instead of edge |
| `prob_threshold_yes` | float | 0.70 | Trade YES when model prob ≥ this |
| `prob_threshold_no` | float | 0.30 | Trade NO when model prob ≤ this |
| `use_max_moneyness` | bool | False | Filter by moneyness |
| `min_moneyness` | float | 0.0 | Minimum absolute moneyness |
| `max_moneyness` | float | 0.05 | Maximum absolute moneyness |

## Dashboard Configuration

The Streamlit dashboard (`app/dashboard.py`) loads defaults from `SweepConfig` and provides sidebar sliders for all parameters. Changes in the sidebar apply immediately to trade recommendations.

## CLI Overrides

Both `auto_reco.py` and `parameter_sweep.py` accept overrides via CLI:

```bash
# Override min_edge
python core/strategy/auto_reco.py --bankroll 1000 --min-edge 0.08

# Sweep multiple params
python scripts/utilities/parameter_sweep.py \
    --sweep min_edge=0.04,0.06,0.08 \
    --sweep kelly_fraction=0.10,0.15 \
    --fixed bankroll=1000
```

## Programmatic Configuration

```python
from sweep_config import SweepConfig

# Start with defaults
config = SweepConfig()

# Override specific values
config = config.update({
    "min_edge": 0.08,
    "kelly_fraction": 0.12,
    "max_dte": 3.0,
})

# Convert to strategy params dict
params = config.to_strategy_params()
```
