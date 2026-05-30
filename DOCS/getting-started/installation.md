# Installation

## Prerequisites

- **Python 3.9+** (tested on 3.9–3.12)
- **pip** (or conda)
- Git (optional — for cloning)

## Clone & Setup

```bash
git clone <repo-url>
cd "V2 BTC Contract Pricing"
```

## Virtual Environment

Create and activate a venv:

```bash
# Create
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

If no `requirements.txt` exists, install the core dependencies manually:

```bash
pip install numpy pandas scipy arch streamlit plotly pyarrow requests
```

### Additional Dependencies

| Package | Used By |
|---------|---------|
| `scikit-learn` | signal_diagnostics (AUC) |
| `mkdocs` / `mkdocs-material` / `mkdocstrings` | Documentation build |
| `pytest` | Test suite |
| `python-dotenv` | Polymarket API credentials |

## Initial Data Setup

The pricing engine requires BTC historical data. Fetch it once:

```bash
python core/data/data_fetcher.py
```

This downloads:

- **5 years of daily closes** from CoinGecko → `DATA/btc_daily.csv`
- **~3 months of 1-minute candles** from Binance → `DATA/btc_intraday_1m.csv`

## Verify Installation

Run the pricing engine validation tests:

```bash
python core/pricing/btc_pricing_engine.py
```

All 5 tests should pass. Successful output ends with `ALL TESTS PASSED`.

## Polymarket API Credentials (Optional)

For live trading features, set environment variables (or create a `.env` file):

```bash
POLYMARKET_USER_ADDRESS=0x...
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_PASSPHRASE=...
POLYMARKET_PRIVATE_KEY=...    # Only needed for trading mode
```

These are **not required** for backtesting, dashboard, or pipeline operation.
