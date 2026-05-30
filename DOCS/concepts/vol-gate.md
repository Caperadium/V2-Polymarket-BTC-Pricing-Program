# Vol Gate

`core/strategy/vol_gate.py`

The Vol Gate is a standalone risk module that gates trading based on BTC realized volatility. It classifies the current market regime and adjusts edge requirements and Kelly sizing accordingly.

## How It Works

1. Compute 1-minute log returns from BTC intraday data
2. Calculate rolling realized volatility over **15-minute** and **60-minute** windows
3. Rank current volatility against a **14-day trailing baseline** (percentile)
4. Classify regime: `normal`, `high`, or `extreme`
5. Optionally detect **shock** events (sudden 5-minute moves)

## Regime Classification

| Regime | Percentile | New Entries | Edge Add | Kelly Mult |
|--------|-----------|-------------|----------|------------|
| `normal` | < 80th | Allowed | 0¢ | 1.0 |
| `high` | 80th–95th | Allowed | +2¢ | 0.5 |
| `extreme` | > 95th | **Blocked** | ∞ | 0.0 |

### Shock Detection

A separate gate detects sudden price moves over a 5-minute window. If the absolute 5-minute return ranks ≥ 90th percentile vs baseline, a shock is flagged. Shocks immediately trigger `extreme` regime regardless of vol15 percentile.

## VolGateResult

```python
@dataclass(frozen=True)
class VolGateResult:
    now_utc: str
    regime: str                   # "normal" | "high" | "extreme" | "unknown"
    vol15: Optional[float]        # 15-minute realized vol
    vol60: Optional[float]        # 60-minute realized vol
    vol15_pct: Optional[float]    # Percentile vs baseline
    shock: bool                   # Sudden-move flag
    allow_new_entries: bool       # False in extreme
    edge_add_cents: float         # Additional edge required
    kelly_mult: float             # Kelly multiplier
    reason: str                   # Human-readable explanation
```

## Usage

### CLI

```bash
python core/strategy/vol_gate.py --now 2025-12-29T12:34:00Z --file DATA/btc_intraday_1m.csv
```

Outputs JSON with regime, vol metrics, and gating decisions.

### Programmatic

```python
from core.strategy.vol_gate import compute_vol_gate
import pandas as pd

btc_df = pd.read_csv("DATA/btc_intraday_1m.csv")
result = compute_vol_gate(btc_df, now_utc="2025-12-29T12:00:00Z")

print(result.regime)            # "normal"
print(result.allow_new_entries)  # True
print(result.kelly_mult)         # 1.0
```

## Integration in Strategy Pipeline

The Vol Gate is computed once at the start of `recommend_trades()` and flows through all three stages:

1. **Stage 1 (Targets)**: Edge thresholds adjusted by `edge_add_cents`; new entries blocked in extreme regime
2. **Stage 2 (Deltas)**: Kelly multiplier applied to all sizing; adds blocked in extreme
3. **Stage 3 (Actions)**: Final risk enforcement uses the allow/block decision

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `baseline_days` | 14 | Lookback for percentile ranking |
| `vol15_window_min` | 15 | Primary volatility window |
| `vol60_window_min` | 60 | Secondary volatility window |
| `high_pct` | 80.0 | Percentile threshold for HIGH regime |
| `extreme_pct` | 95.0 | Percentile threshold for EXTREME regime |
| `high_edge_add_cents` | 2.0 | Additional edge in HIGH regime |
| `high_kelly_mult` | 0.5 | Kelly multiplier in HIGH regime |
| `extreme_kelly_mult` | 0.0 | Kelly multiplier in EXTREME regime |
| `shock_window_min` | 5 | Shock detection window |
| `shock_pct` | 90.0 | Shock percentile threshold |

## Robustness

- **Stale data protection**: Requires data within 5 minutes of `now_utc`
- **Insufficient baseline**: Defaults to conservative (edge_add=2¢, kelly=0.5) when baseline < ~1 day
- **Missing data**: Returns `unknown` regime with conservative fallback
- **Self-testing**: Includes built-in unit tests (`--test` flag); validates constant price, missing data, volatility spike, and shock detection
