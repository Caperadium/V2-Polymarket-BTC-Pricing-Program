# Kelly Sizing

The strategy uses **fractional Kelly criterion** to size trades. Full Kelly maximizes expected log-wealth; fractional Kelly (typically 10–25%) reduces volatility and protects against estimation error.

## Kelly Formula

For binary contracts with payout normalized to 1:

### YES Side

$$f^* = \frac{p - q}{1 - q}$$

Where $p$ = model probability (chance YES wins), $q$ = market price (cost of YES share).

```python
def kelly_fraction_yes(p: float, q: float) -> float:
    return max((p - q) / (1.0 - q), 0.0)
```

### NO Side

$$f^* = \frac{q - p}{q}$$

Where $p$ = model probability, $q$ = market price (cost of YES). The NO side wins when outcome is NO, which has probability $1-p$, and the NO share costs $1-q$.

```python
def kelly_fraction_no(p: float, q: float) -> float:
    return max((q - p) / q, 0.0)
```

## Fractional Multiplier

Raw Kelly fraction is multiplied down for safety:

```python
f_target = min(kelly_fraction * f_star, 0.30)
```

With `kelly_fraction=0.15`, the maximum allocation to any single contract is 30% of bankroll.

## Multiplier Stack

The final Kelly fraction passes through a stack of multipliers:

$$f_{final} = f^* \times \text{kelly\_fraction} \times \text{kelly\_mult} \times \text{stability} \times \text{stale\_mult}$$

| Multiplier | Source | Range | Effect |
|------------|--------|-------|--------|
| `kelly_fraction` | Config (0.15) | 0–1.0 | Base fractional scaling |
| `kelly_mult` | Vol Gate | 1.0 (normal), 0.5 (high), 0.0 (extreme) | Reduce in volatile markets |
| `stability` | Curve fit quality | 0.2–1.0 | Penalize poor logistic fits |
| `stale_mult` | Data freshness | 0.0–1.0 | Decay for old batch data |

## Stability Penalty

The stability multiplier penalizes contracts with noisy curve fits:

```python
penalty = np.clip(1.0 - abs(fit_residual) / 0.3, 0.2, 1.0)   # Fit residual
penalty *= 0.6 if monotonicity_violated else 1.0               # Violation flag
penalty *= np.clip(1.0 - abs(edge_zscore) / 4.0, 0.2, 1.0)    # Edge z-score
```

## Staleness Multiplier

Data freshness decays linearly between soft and hard limits:

- **≤ 4 hours**: multiplier = 1.0 (fresh)
- **4–12 hours**: linear decay from 1.0 → 0.0
- **≥ 12 hours**: hard block on new entries

## Per-Contract Cap

Individual position capped at 30% of bankroll:

```python
f_target = min(kelly_fraction * f_star_scaled, 0.30)
```

## Fixed Stake Mode

Set `use_fixed_stake=True` and `fixed_stake_amount=10.0` to bypass Kelly entirely. Every trade uses the same dollar amount regardless of edge magnitude.
