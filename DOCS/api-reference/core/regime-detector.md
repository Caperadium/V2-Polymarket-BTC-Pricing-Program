# Regime Detector

`core/pricing/regime_detector.py`

3-state Hidden Markov Model regime classifier using hmmlearn.GaussianHMM on daily BTC returns.

## Classes

### `RegimeDetector`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_years` | 2 | Training window in years |
| `reestimate_days` | 7 | Days between HMM re-fits |

#### Methods

**`fit(hourly_csv)`** — Fit HMM and label states by annualized mean return. Returns `RegimeResult`.

**`predict_weights(n_days_ahead)`** — Forward prediction via transition matrix powering. Returns `{regime: weight}` dict.

### `RegimeResult`

| Field | Type | Description |
|-------|------|-------------|
| `labels` | Dict[str,str] | State index → regime name mapping |
| `weights` | Dict[str,float] | Current posterior weights |
| `means` | Dict[str,float] | Annualized means per regime |
| `transition_matrix` | np.ndarray | 3×3 transition probabilities |
| `converged` | bool | HMM fit convergence |
| `n_days` | int | Training window days |

## Functions

### `hourly_to_daily_returns(hourly_csv)`

Resample hourly close to daily last price, compute log returns.

See [Regime Detection Concept](../../concepts/regime-detection.md) for full methodology.
