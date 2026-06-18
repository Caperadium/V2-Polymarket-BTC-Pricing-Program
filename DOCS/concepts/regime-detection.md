# Regime Detection

`core/pricing/regime_detector.py`

3-state Hidden Markov Model (HMM) regime classifier for BTC markets. Labels market phases as **bear**, **sideways**, or **bull** and provides forward-looking weights for regime-conditional pricing.

Based on: Paskaleva & Vasenska (2025), Köse et al. (2025), Kim et al. (2025).

## Why It Matters

BTC exhibits persistent regime behavior — bear markets have more frequent and larger downward jumps, bull markets have upward momentum. A single-model approach (one set of parameters for all conditions) underperforms regime-aware models in directional accuracy and tail risk estimation.

## Model

### 3-State GaussianHMM

Uses `hmmlearn.GaussianHMM` on daily BTC returns:

- **2-year training window** (≈730 days) — balances recency with stability
- **Weekly re-fit** (configurable via `reestimate_days=7`) — avoids per-day noise while staying current
- **Full covariance** — captures state-dependent return distributions

### State Labeling

States ordered by annualized mean return:

```python
annualized_means = {state: mean_daily × 365}
# Label: lowest → "bear", middle → "sideways", highest → "bull"
```

This ensures consistent interpretation — HMM state index is arbitrary, but labeling by return magnitude is deterministic.

### Forward Prediction

N-day-ahead regime weights via transition matrix powering:

```python
T = model.transmat_           # 3×3 transition matrix
T_forward = T^n_days          # n-step transition probabilities
weights = posterior @ T_forward  # current × forward transition
```

This captures regime persistence — if current weight is 0.8 sideways, and bear→sideways transitions are rare, the forward weight stays mostly sideways.

## RegimeResult

```python
@dataclass
class RegimeResult:
    labels: Dict[str, str]       # {"0": "bear", "1": "sideways", "2": "bull"}
    weights: Dict[str, float]    # {"bear": 0.05, "sideways": 0.73, "bull": 0.22}
    means: Dict[str, float]      # Annualized means per regime
    transition_matrix: np.ndarray  # 3×3 transition probabilities
    converged: bool               # HMM fit convergence status
    n_days: int                   # Training window days
```

## Integration with Pricing Engine

Regime weights flow into `calculate_probabilities()`:

1. Three independent MC simulations run — one per regime with scaled jump parameters
2. `build_regime_jump_params()` applies regime-specific scaling:
    - **Bear**: λ × 1.5, p_crash × 1.3
    - **Bull**: λ × 0.7, p_crash × 0.7
    - **Sideways**: Default parameters unchanged
3. Terminal price distributions weighted by HMM posterior:

```python
P(S_T ≥ K) = Σ_{r} w_r · P_r(S_T ≥ K)
```

**Post-hoc weighting** not intra-path switching — avoids path-continuity issues.

## Usage

### CLI

```bash
python core/pricing/regime_detector.py --input DATA/btc_hourly.csv
```

### Programmatic

```python
from core.pricing.regime_detector import RegimeDetector

detector = RegimeDetector(training_years=2, reestimate_days=7)
result = detector.fit("DATA/btc_hourly.csv")

print(result.weights)
# {"bear": 0.006, "sideways": 0.767, "bull": 0.227}

# 30-day forward weights
forward = detector.predict_weights(n_days_ahead=30)
```

### Hourly to Daily

Helper for resampling hourly close prices:

```python
from core.pricing.regime_detector import hourly_to_daily_returns

daily_returns = hourly_to_daily_returns("DATA/btc_hourly.csv")
```
