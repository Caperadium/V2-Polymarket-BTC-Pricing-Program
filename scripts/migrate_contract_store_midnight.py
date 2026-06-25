#!/usr/bin/env python3
"""migrate_contract_store_midnight.py

One-shot migration: floor every contract-price observation in
``DATA/historical_contract_prices.csv`` to midnight UTC and deduplicate on
``(clobTokenId, date)``.

Why: CLOB candle timestamps carried second-level jitter (00:00:1X-3X), so the
backrunner — which groups contracts on exact-timestamp equality — scattered each
expiry's ~11 strikes across as many per-second "snapshots" (~1.4 strikes each),
starving the logistic curve fit and producing all-NaN ``p_model_fit``. Ingest now
floors to midnight (polymarket_fetcher._normalize_to_midnight); this script
repairs the already-stored rows.

Dedup policy: when several jittered rows collapse to one (clobTokenId, day),
keep the row whose ORIGINAL timestamp was CLOSEST to midnight (the intended
candle) — not an arbitrary load-order "first".

Safety: writes ``historical_contract_prices.csv.bak`` before overwriting.
Idempotent: running twice is a no-op (already floored → dist 0, dedup stable).

Usage:
    python scripts/migrate_contract_store_midnight.py
    python scripts/migrate_contract_store_midnight.py --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.backtesting.contract_store import ContractPriceStore


def migrate(dry_run: bool = False) -> int:
    store = ContractPriceStore()
    csv_path = store.csv_path

    if not csv_path.exists():
        print(f"No store at {csv_path} — nothing to migrate.")
        return 0

    df = store.load()  # typed: date/expiry_date as UTC datetime
    if df.empty:
        print("Store is empty — nothing to migrate.")
        return 0

    before = len(df)

    # distance (seconds) of each original timestamp from the nearest midnight
    secs = (
        df["date"].dt.hour * 3600
        + df["date"].dt.minute * 60
        + df["date"].dt.second
    )
    df = df.assign(
        _floored=df["date"].dt.floor("D"),
        _dist=pd.concat([secs, 86400 - secs], axis=1).min(axis=1),
    )

    # keep closest-to-midnight row per (clobTokenId, floored-day)
    df = (
        df.sort_values("_dist", kind="stable")
        .drop_duplicates(subset=["clobTokenId", "_floored"], keep="first")
        .copy()
    )
    df["date"] = df["_floored"]
    df = df.drop(columns=["_floored", "_dist"])
    df = df.sort_values("date").reset_index(drop=True)

    after = len(df)
    n_offgrid = int((secs != 0).sum())
    print(
        f"Rows: {before} -> {after} (removed {before - after} duplicate/"
        f"collapsed; {n_offgrid} originally off-midnight)"
    )

    if dry_run:
        print("--dry-run: no files written.")
        return 0

    bak = csv_path.with_suffix(csv_path.suffix + ".bak")
    shutil.copy2(csv_path, bak)
    print(f"Backup written: {bak}")

    store.save(df)
    print(f"Migrated store written: {csv_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Report changes; write nothing.")
    args = ap.parse_args()
    return migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
