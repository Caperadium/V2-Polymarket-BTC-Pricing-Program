# Risk Controls

The strategy pipeline enforces multiple layers of risk control, applied across the 3-stage pipeline (Target → Delta → Action).

## Portfolio-Level Caps

### Per-Expiry Cap

Limits exposure to any single expiry date:

```python
expiry_cap_usd = bankroll * max_capital_per_expiry_frac  # default: 0.15
```

No single expiry can consume more than 15% of bankroll.

### Total Cap

Limits total deployed capital:

```python
cap_usd = bankroll * max_capital_total_frac  # default: 0.35
```

At most 35% of bankroll is deployed across all positions. Remaining capital stays in reserve.

### Net Delta Limit

Limits net directional exposure:

```python
net_delta = sum(signed_deltas)  # + for BUY, - for SELL
max_net_delta = bankroll * max_net_delta_frac  # default: 0.20
```

Prevents the portfolio from being overly long or short.

### Cap Breach Deleveraging

When hold budget exceeds the total cap, positions are ranked by `exit_score` and reduced (worst first) until compliant:

```python
if hold_budget > cap_usd and cap_breach_delever:
    # Reduce lowest exit_score positions to target=0
```

## Churn Prevention

### Exit Hysteresis

Prevents flipping in and out of positions on small edge changes:

$$edge_{exit} < edge_{entry} - hysteresis$$

Default hysteresis = 0.02 (2 cents). A position entered at edge=0.08 won't exit until edge drops below 0.06.

### Minimum Trade Size

Trades below `min_trade_usd` (default: $5) are filtered out. Prevents micro-orders that cost more in fees than they're worth.

```python
rebalance_min_add_usd = 5.0   # Minimum BUY
rebalance_min_reduce_usd = 10.0  # Minimum SELL (2x to prevent churn)
```

## Staleness Controls

### Soft Stale (4 hours)

After 4 hours, Kelly sizing decays linearly toward zero. Positions can still be exited.

### Hard Stale (12 hours)

After 12 hours, **all new entries blocked**. Existing positions can still be reduced/exited.

```python
if batch_age > STALE_HARD_HOURS:  # 12 hours
    allow_new_entries = False
```

Both thresholds are configurable via `STALE_SOFT_HOURS` and `STALE_HARD_HOURS` in `common.py`.

## Correlation Penalty

Positions in the same expiry with the same direction (YES or NO) are penalized:

$$multiplier = \frac{1}{1 + penalty \times (n - 1)}$$

With `correlation_penalty = 0.25`, having 3 YES positions in one expiry reduces each to 1/(1 + 0.25×2) = 0.67×.

This prevents concentration risk from correlated binary outcomes on the same underlying.

## Directional Consistency

Within each expiry, at most **one sign change** is allowed, and only the pattern **YES-then-NO** (higher strikes are NO). This enforces:
- No contradictory positions (YES and NO on same strike)
- Coherent directional thesis per expiry
- At most `max_bets_per_expiry` (default: 3) contracts per expiry

## Probability Threshold Mode

When `use_prob_threshold=True`, trades are filtered by absolute probability rather than edge:

| Trade | Condition |
|-------|-----------|
| YES | `model_prob ≥ prob_threshold_yes` (default: 0.70) |
| NO | `model_prob ≤ prob_threshold_no` (default: 0.30) |

Positions already held are exempt from threshold filtering to allow exits.

## Risk-Off Override

When the Vol Gate blocks new entries AND `risk_off_targets_to_zero=True`, all position targets are set to zero, triggering full portfolio exit:

```python
if not vol_gate_result.allow_new_entries and config.risk_off_targets_to_zero:
    tgt_usd = 0.0  # Force exit
```

## Price Filters

- **Min price** (0.03): Skip contracts below 3¢ (low liquidity, wide spreads)
- **Max price** (0.95): Skip contracts above 95¢ (tiny edge potential)
- **Model prob range** (0.0–1.0): Filter out degenerate model outputs
