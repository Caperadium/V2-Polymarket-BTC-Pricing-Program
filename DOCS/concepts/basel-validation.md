# Basel Validation

`core/validation/basel_backtest.py`

Regulatory-standard VaR backtesting using Kupiec POF (Proportion of Failures) traffic light test and Acerbi-Szekely expected shortfall statistics. Validates multi-horizon model adequacy at 95%, 97.5%, and 99% confidence levels.

Based on: Basel Committee (1996, 2016), Teng et al. (2025), Acerbi & Szekely (2014).

## Why It Matters

Predictive models should demonstrate calibration across confidence levels, not just point accuracy. A model that prices P=0.70 correctly but fails to capture 1% tail events (P=0.99) will produce catastrophic losses in extreme markets.

SVCJ passes Basel backtests where SVJ fails — Teng et al. (2025) show SVJ exceedance rates at h=14, 99% VaR are 4× the expected rate (Red), while SVCJ falls to 1.1× (Green).

## Tests

### Kupiec POF (Proportion of Failures)

Tests whether the observed violation rate matches the expected rate for a given VaR confidence level:

- **H₀**: Observed violation rate = expected violation rate (1 - confidence)
- **Test statistic**: Likelihood ratio, asymptotically χ²(1) distributed
- **Traffic light classification**:

| POF p-value | Classification | Interpretation |
|-------------|----------------|----------------|
| p > 0.05 | 🟢 **Green** | Model adequately captures risk |
| 0.01 < p ≤ 0.05 | 🟡 **Yellow** | Borderline — monitor closely |
| p ≤ 0.01 | 🔴 **Red** | Model rejected — recalibration needed |

### Expected Shortfall (Acerbi-Szekely)

Tests whether the model captures tail severity, not just frequency:

- **Z1 statistic**: Average tail loss vs expected tail loss (robust to small samples)
- **Z2 statistic**: Severity-weighted exceedance count
- **Z3 statistic**: Combined frequency + severity test

## Multi-Horizon Rolling Windows

```python
run_basel_backtest(hourly_csv="DATA/btc_hourly.csv")
```

Runs validation across:

- **Horizons**: h ∈ {1, 7, 14, 30} days
- **Confidence levels**: 95%, 97.5%, 99%
- **Method**: Rolling windows — at each point t, compute VaR from t-h to t, compare to realized h-day return

## BaselBacktestResult

```python
@dataclass
class BaselBacktestResult:
    horizon_days: int
    confidence: float
    n_windows: int
    n_violations: int
    violation_rate: float
    expected_rate: float
    pof_pvalue: float
    traffic_light: str          # "Green", "Yellow", "Red"
    z1: Optional[float]         # ES Z1 statistic
    z2: Optional[float]         # ES Z2 statistic
    z3: Optional[float]         # ES Z3 statistic
```

## Usage

### CLI

```bash
python core/validation/basel_backtest.py --input DATA/btc_hourly.csv
```

Outputs traffic light table for all horizon-confidence combinations plus ES Z-statistics.

### Programmatic

```python
from core.validation.basel_backtest import run_basel_backtest

results = run_basel_backtest("DATA/btc_hourly.csv")
for r in results:
    print(f"h={r.horizon_days}d {r.confidence*100:.0f}%: "
          f"violations={r.n_violations}/{r.n_windows} "
          f"({r.violation_rate*100:.2f}%) — {r.traffic_light}")
```

## Interpretation

- **All Green**: Model well-calibrated across horizons and confidence levels
- **Yellow at 99%**: Tail risk slightly underestimated; consider increasing jump intensity
- **Red at any level**: Model inadequately captures risk at that horizon/confidence; recalibration required
- **Red at h≥14 without SVCJ**: Expected — SVJ models understate volatility persistence; enable SVCJ
