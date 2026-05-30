# Strategy Pipeline

`core/strategy/auto_reco.py`

The strategy converts model probabilities into executable trade recommendations through a **3-stage pipeline**: Target → Delta → Action.

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Stage 1       │     │   Stage 2        │     │   Stage 3       │
│   Build Targets │────▶│   Compute Deltas  │────▶│   Enforce       │
│   TargetPosition│     │   DeltaIntent     │     │   BUY/SELL/HOLD │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

Each stage is independently testable with clear inputs and outputs.

## Stage 1: Build Targets (`build_targets`)

**Input**: Batch DataFrame, current positions, VolGate result, config
**Output**: `Dict[key, TargetPosition]`

### Candidate Generation

For each contract in the batch, both YES and NO sides are evaluated:

```python
# YES side
yes_edge = (model_prob - yes_ask) - spread_cost

# NO side
no_edge = ((1.0 - model_prob) - no_ask) - spread_cost
```

### Scoring

Candidates are scored for capital allocation:

```python
score = effective_edge × kelly_mult × stability_penalty × stale_mult
```

### Per-Expiry Selection

Within each expiry, candidates obey two constraints:

1. **≤ `max_bets_per_expiry`** contracts (default: 3)
2. **≤ 1 sign change**, and only the pattern YES→NO (higher strikes are NO)

Exit signals and safety holds bypass consistency filtering — only new entries are constrained.

### Target Roles

Each target is classified with a `TargetRole`:

| Role | Meaning | Filtered by consistency? |
|------|---------|-------------------------|
| `ENTRY` | New or increased position | Yes |
| `EXIT` | Reduce to zero | No |
| `HOLD_SAFETY` | Keep at current exposure | No |
| `NEUTRAL` | No action | No |

### Portfolio Allocation (Rank-and-Fill)

1. Compute hold budget (existing positions to keep)
2. Check cap breach — if exceeded, deleverage worst positions
3. Rank entry candidates by allocation score
4. Fill sequentially, respecting per-expiry and total caps
5. Partial fills allowed when budget is tight

## Stage 2: Compute Deltas (`compute_deltas`)

**Input**: Targets dict, current exposure, VolGate result, config
**Output**: `List[DeltaIntent]`

### Delta Calculation

```python
raw_delta = target_usd - current_usd

if abs(raw_delta) < 0.01:    action = "HOLD"
elif raw_delta > 0:           action = "BUY"
else:                         action = "SELL"
```

### Entry Blocking

Vol Gate overrides in Stage 2:
- Risk-off: all targets → 0
- Entry blocked: positive deltas zeroed
- Minimum trade size enforced

### Directional Sorting

- **BUYs** ranked by allocation_score (highest first)
- **SELLs** ranked by exit_score (worst first — reduce worst positions first)
- Cycle caps limit total add/reduce per batch

## Stage 3: Action Generation

The `recommend_trades()` function wraps stages 1+2 and returns actionable `DeltaIntent` objects:

```python
from core.strategy.auto_reco import recommend_trades

intents = recommend_trades(
    df=batch_df,
    bankroll=1000.0,
    positions_df=current_positions,
    kelly_fraction=0.15,
    min_edge=0.06,
)
```

Each intent contains: `action` (BUY/SELL/HOLD), `amount_usd`, `limit_price_hint`, `model_prob`, `effective_edge`, `reason`.

## Key Data Types (`common.py`)

### TargetPosition

```python
@dataclass
class TargetPosition:
    key: str              # Unique position identifier
    side: str             # YES or NO
    target_fraction: float  # Kelly fraction
    target_usd: float     # Dollar target
    model_prob: float     # Model probability
    effective_edge: float  # Risk-adjusted edge
    role: TargetRole      # ENTRY/EXIT/HOLD_SAFETY/NEUTRAL
```

### DeltaIntent

```python
@dataclass
class DeltaIntent:
    action: Literal["BUY", "SELL", "HOLD"]
    amount_usd: float
    signed_delta_usd: float  # +BUY, -SELL
    price_mode: str         # TAKER_ASK or TAKER_BID
    limit_price_hint: float
    effective_edge: float
    reason: str
```

### RebalanceConfig

```python
@dataclass
class RebalanceConfig:
    bankroll: float
    min_edge_entry: float = 0.02
    kelly_fraction: float = 0.15
    max_capital_per_expiry_frac: float = 0.15
    max_capital_total_frac: float = 0.35
    max_bets_per_expiry: int = 3
    # ... 20+ parameters total
```

## CLI Usage

```bash
python core/strategy/auto_reco.py --bankroll 1000 --min-edge 0.06 --kelly-fraction 0.15
```

Outputs a table of recommended trades with action, amount, edge, and model probability.
