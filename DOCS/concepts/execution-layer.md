# Execution Layer

`polymarket/`

The execution layer converts trade recommendations into live orders on Polymarket's CLOB (Central Limit Order Book).

## Component Stack

```
auto_reco.py                    Strategy output (DeltaIntent list)
     │
     ▼
intent_builder.py               DeltaIntent → OrderIntent
     │
     ▼
execution_gateway.py            Validate → Submit → Track
     │
     ▼
provider_polymarket.py          CLOB API (place/cancel/fill)
     │
     ▼
accounting.py                   Collateral + balance tracking
```

## Intent Lifecycle

Each trade intent progresses through statuses:

```
DRAFT → APPROVED → SUBMITTED → FILLED
  │         │          │
  └─────────┴──────────┴──→ CANCELLED / FAILED / SKIPPED
```

### 1. Generate (DRAFT)

`intent_builder.py` converts `DeltaIntent` objects from `auto_reco` into `OrderIntent`:

```python
from polymarket.intent_builder import build_intents_from_reco, create_run

run = create_run(strategy="auto_reco", params=params_dict)
intents = build_intents_from_reco(delta_intents, run)
```

Key transformations:

- **Share computation**: `size_shares = floor(stake_usd / limit_price * 100) / 100` — rounds DOWN to 2 decimals
- **Deterministic ID**: `intent_id = sha256(run_id | intent_key)` — same logical trade gets same ID on retry
- **Notional cap**: `notional_usd = size_shares × limit_price ≤ stake_usd`

### 2. Approve (APPROVED)

Operator reviews DRAFT intents in the Polymarket Console and approves. Approval captures an account state snapshot for audit.

### 3. Submit (SUBMITTED)

`execution_gateway.py` validates and submits:

**Validation checks**:

- Status must be APPROVED
- `limit_price` in (0, 1)
- `size_shares > 0`
- `notional_usd ≤ available_collateral`
- `notional_usd ≤ collateral_allowance`

**Batch submission** with cumulative collateral tracking:

```python
from polymarket.execution_gateway import submit_approved_batch

results = submit_approved_batch(intents, account_state, provider)
```

Collateral is tracked cumulatively: each submitted order reduces the available balance for subsequent orders in the batch.

### 4. Monitor (SUBMITTED → FILLED)

`reconcile.py` syncs submission statuses with the provider. MVP implementation keeps submissions OPEN.

## Core Data Types (`models.py`)

### OrderIntent

```python
@dataclass
class OrderIntent:
    intent_id: str          # Deterministic SHA256 hash
    run_id: str
    contract: str           # e.g. "bitcoin-above-90k-on-dec-31"
    outcome: str            # "YES" or "NO"
    action: str             # "BUY" or "SELL"
    limit_price: float      # Max price for BUY, min for SELL
    stake_usd: float        # Max USD to spend
    size_shares: float      # Rounded DOWN to 2 decimals
    notional_usd: float     # size_shares × limit_price ≤ stake_usd
    status: str             # IntentStatus
```

### Submission

```python
@dataclass
class Submission:
    submission_id: str
    intent_id: str
    order_id: str           # CLOB order ID from API
    submitted_price: float
    submitted_size: float
    status: str             # SubmissionStatus
```

### AccountState

```python
@dataclass
class AccountState:
    collateral_balance: float
    collateral_allowance: float
    reserved_open_buys: float
    available_collateral: float  # balance − reserved
```

## SQLite Database (`db.py`)

Uses WAL mode for Streamlit concurrency safety. Connection-per-operation pattern:

| Table | Purpose |
|-------|---------|
| `runs` | Generation batch tracking |
| `intents` | OrderIntent lifecycle |
| `submissions` | CLOB submission records |
| `account_state` | Collateral snapshots |
| `pm_trades` | Ingested CLOB trades |
| `pm_closed_positions` | Realized PnL source |
| `pm_sync_metadata` | Idempotent sync cursors |

## Provider Layer (`provider_polymarket.py`)

Abstract `PolymarketProvider` interface with two implementations:

- `RealPolymarketProvider` — Full CLOB + Data-API integration
- `FakePolymarketProvider` — MVP testing without real API calls

API endpoints used:

| Endpoint | Purpose |
|----------|---------|
| Gamma API `/events` | Market discovery (slug → condition_id, token_ids) |
| CLOB `/book` | Order book (best bid/ask) |
| CLOB `/order` | Place/cancel orders |
| Data-API `/closed-positions` | Realized PnL source |
| Data-API `/balance-allowance` | Collateral state |

## Ingestion (`ingest.py`)

Pulls historical trade and position data from Polymarket APIs into local SQLite for PnL metrics:

- **Trades**: Incremental sync from CLOB `/data/trades`
- **Closed positions**: From Data-API `/closed-positions`
- **Idempotent**: Uses timestamp cursors to avoid re-processing

## Metrics (`metrics.py`)

Computes from ingested data:

- **Daily realized PnL**: From closed positions
- **Daily loss limit**: Flags days where PnL ≤ −4% of bankroll
- **Rolling 7-day max drawdown**: Peak-to-trough over 7-day window
- **Staleness warning**: Data older than 60 minutes

## Console Workflow (`app/pages/polymarket_console.py`)

Operator steps:

1. **Generate** — Run `auto_reco` → build intents → store as DRAFT
2. **Audit** — Review intents, check prices, duplicate detection
3. **Approve** — Select intents → snapshot account state → mark APPROVED
4. **Submit** — Validate against collateral → submit batch → mark SUBMITTED
5. **Monitor** — Track submission statuses, view PnL metrics
