# Quickstart

Run the full pipeline end-to-end: fetch data → price markets → fit curves → generate trade recommendations.

## 1. Fetch BTC Data

```bash
python core/data/data_fetcher.py
```

Downloads daily and intraday BTC price data to `DATA/`.

## 2. Run Full Pipeline

Price BTC binary contracts for a date range:

```bash
python scripts/pipelines/run_full_pipeline.py \
    --slug-pattern "bitcoin-above-on-december-{}" \
    --day-range 1 31
```

This runs three stages internally:

1. **Batch pricing** — fetches Polymarket contracts, runs Monte Carlo simulations per expiry, outputs `batch_results/<timestamp>/`
2. **Curve fitting** — fits logistic probability curves, applies calibration shift, outputs `fitted_batch_results/<timestamp>/`
3. **Order book enrichment** — fetches live ask/bid prices from CLOB

## 3. Generate Trade Recommendations

```bash
python core/strategy/auto_reco.py --bankroll 1000 --min-edge 0.06 --kelly-fraction 0.15
```

Outputs a list of BUY/SELL actions with Kelly-sized stakes.

## 4. Launch Dashboard

```bash
streamlit run app/dashboard.py
```

Opens the 8-tab monitoring dashboard in your browser. Load fitted batch data from the sidebar.

## Next Steps

- [Configuration](configuration.md) — tune all strategy parameters
- [Full Pipeline Guide](../guides/full-pipeline.md) — detailed pipeline walkthrough
- [Architecture](../concepts/architecture.md) — understand the system design
