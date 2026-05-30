# Signal Diagnostics Guide

`core/strategy/signal_diagnostics.py`

Run signal quality analysis to measure whether model edge predicts actual outcomes.

## Quick Start

```bash
python core/strategy/signal_diagnostics.py path/to/all_priced.csv
```

## What It Measures

- **Spearman rank correlation** between edge and binary outcomes
- **AUC** (Area Under ROC Curve) — discriminative power of the edge
- **Edge difference** — mean edge for winners vs losers

## Subset Analysis

Optionally breaks down by:

- **DTE** (Days to Expiry) — does signal decay with time?
- **Moneyness** — is signal stronger ATM or OTM?

## Interpretation

See [Signal Diagnostics Concept](../concepts/signal-diagnostics.md) for detailed interpretation guide.
