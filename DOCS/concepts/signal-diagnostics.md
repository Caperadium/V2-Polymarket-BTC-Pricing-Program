# Signal Diagnostics

`core/strategy/signal_diagnostics.py`

Signal diagnostics measure whether the model's edge (model probability − market price) actually predicts outcomes. If the edge is a real signal, higher edges should correlate with higher win rates.

## Metrics

### Spearman Rank Correlation

$$\rho = \text{Spearman}(edge, outcome)$$

Measures monotonic relationship between edge magnitude and binary outcome. Non-parametric — doesn't assume linearity.

```python
rho, p_value = spearmanr(edge_array, outcome_array)
```

- **ρ > 0**: Higher edge → higher win probability (good)
- **ρ ≈ 0**: No relationship (random)
- **ρ < 0**: Higher edge → lower win probability (anti-signal)

### AUC (Area Under ROC Curve)

Measures discriminative power: how well does the edge separate winners from losers?

```python
auc = roc_auc_score(outcomes, edges)
```

- **> 0.55**: Positive signal — model edge predicts wins
- **0.45–0.55**: No discrimination — roughly random
- **< 0.45**: Anti-signal — inverted relationship

### Edge Difference

Mean edge for winners vs losers:

```python
edge_diff = mean(edge | outcome=1) - mean(edge | outcome=0)
```

Positive difference means winning trades had higher edges.

## Usage

### CLI

```bash
python core/strategy/signal_diagnostics.py path/to/all_priced.csv
```

### Programmatic

```python
from core.backtesting.diagnostics import run_diagnostics

run_diagnostics("path/to/all_priced.csv")
```

## Output

```
OVERALL METRICS
  Spearman rho: 0.1234  (p-value: 0.045678)
  AUC:          0.5678  (positive signal - model edge predicts wins)

  Mean edge (outcome=1): 0.0567
  Mean edge (outcome=0): 0.0345
  Edge difference:       0.0222
```

## Subset Analysis

The diagnostics break down signal quality by:

### By DTE (Days to Expiry)

| Bucket | What It Tests |
|--------|--------------|
| DTE 1–2 | Signal quality for near-expiry contracts |
| DTE 3–4 | Medium-term signal decay |
| DTE 5–6 | Longer-term predictions |
| DTE 7+ | Far-expiry edge reliability |

### By Moneyness

| Bucket | What It Tests |
|--------|--------------|
| ATM (｜m｜ ≤ 2%) | Signal at-the-money (most liquid) |
| Near-ATM (｜m｜ ≤ 5%) | Broader near-money range |
| OTM (m > 5%) | Out-of-the-money edge quality |
| ITM (m < −5%) | In-the-money edge quality |

## Data Requirements

Expected CSV columns (detected via precedence chains):

| Role | Candidate Columns |
|------|------------------|
| Outcome | `outcome_yes`, `outcome` |
| Model Probability | `model_prob_used` |
| Market Price | `market_yes_price` |
| DTE (optional) | `dte_days`, `t_days`, `T_days` |
| Moneyness (optional) | `moneyness` |

Outcome values are coerced from multiple formats: `{0,1}`, `True/False`, `"YES"/"NO"`.

## Interpretation Guide

| Scenario | Spearman ρ | AUC | Meaning |
|----------|-----------|-----|---------|
| Strong signal | > 0.15 | > 0.60 | Model reliably identifies value |
| Weak signal | 0.05–0.15 | 0.52–0.60 | Some predictive power, needs improvement |
| No signal | −0.05 to 0.05 | 0.48–0.52 | Model is noise — strategy likely unprofitable |
| Anti-signal | < −0.05 | < 0.48 | Model systematically wrong — invert it |

**Important**: A significant Spearman ρ or AUC > 0.50 does NOT guarantee profitability. It means the edge contains information about outcomes. Whether that information can be monetized depends on trade sizing, costs, and execution.
