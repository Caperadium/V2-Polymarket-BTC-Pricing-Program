#!/usr/bin/env python3
"""
contract_store.py

Manages historical_contract_prices.csv: read, write, deduplicate merge,
and conversion to the market_df format expected by the backrunner engine.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Default CSV path relative to repo root
DATA_DIR = Path("DATA")
DEFAULT_CSV_PATH = DATA_DIR / "historical_contract_prices.csv"

# Schema: every row is one contract at one midnight-UTC date
COLUMNS = ["slug", "clobTokenId", "date", "price", "resolution", "strike", "expiry_date"]

# Columns produced by to_market_df() for backrunner consumption
MARKET_COLUMNS = ["slug", "strike", "market_price", "date", "expiry_date"]


class ContractPriceStore:
    """CSV-backed store for historical Polymarket contract prices.

    Schema (7 columns):
        slug          — contract slug string (e.g. "bitcoin-above-94k-on-november-15")
        clobTokenId   — Polymarket YES token ID (string)
        date          — midnight UTC datetime of this price observation
        price         — float, market YES price at midnight UTC
        resolution    — float (1.0=YES, 0.0=NO, NaN=pending/unresolved)
        strike        — float, BTC strike price
        expiry_date   — datetime (UTC), contract expiry at 12:00 ET → UTC
    """

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Read CSV. Returns empty DataFrame with correct columns on any error."""
        try:
            if not self.csv_path.exists():
                logger.info("No existing store at %s — cold start", self.csv_path)
                return self._empty_df()

            df = pd.read_csv(self.csv_path)

            # Validate required columns
            missing = [c for c in COLUMNS if c not in df.columns]
            if missing:
                logger.warning(
                    "Store CSV missing columns %s — treating as empty", missing
                )
                return self._empty_df()

            # Ensure dtypes
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df["expiry_date"] = pd.to_datetime(
                df["expiry_date"], utc=True, errors="coerce"
            )
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            df["resolution"] = pd.to_numeric(df["resolution"], errors="coerce")
            df["slug"] = df["slug"].astype(str)
            df["clobTokenId"] = df["clobTokenId"].astype(str)

            # Drop rows with unparseable dates
            df = df.dropna(subset=["date", "clobTokenId"]).reset_index(drop=True)

            logger.info("Loaded %d records from %s", len(df), self.csv_path)
            return df

        except pd.errors.EmptyDataError:
            logger.warning("Store CSV is empty — cold start")
            return self._empty_df()
        except Exception:
            logger.exception("Failed to load store CSV — treating as empty")
            return self._empty_df()

    def save(self, df: pd.DataFrame) -> None:
        """Overwrite CSV with the given DataFrame."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, index=False)
        logger.info("Saved %d records to %s", len(df), self.csv_path)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_latest_date(self) -> Optional[datetime]:
        """Most recent date in the store, or None if empty."""
        df = self.load()
        if df.empty or df["date"].isna().all():
            return None
        return df["date"].max().to_pydatetime()

    def get_known_clob_token_ids(self) -> Set[str]:
        """Set of all clobTokenIds already stored."""
        df = self.load()
        if df.empty:
            return set()
        return set(df["clobTokenId"].dropna().unique())

    def get_clob_token_max_date(self, token_id: str) -> Optional[datetime]:
        """Most recent stored date for a specific clobTokenId, or None."""
        df = self.load()
        if df.empty:
            return None
        mask = df["clobTokenId"] == str(token_id)
        if not mask.any():
            return None
        return df.loc[mask, "date"].max().to_pydatetime()

    # ------------------------------------------------------------------
    # Merge & Append
    # ------------------------------------------------------------------

    @staticmethod
    def merge(new_records: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate by (clobTokenId, date), keeping first occurrence.

        Returns combined DataFrame sorted by date.
        """
        if existing.empty:
            out = new_records.copy()
        elif new_records.empty:
            out = existing.copy()
        else:
            combined = pd.concat([existing, new_records], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(
                subset=["clobTokenId", "date"], keep="first"
            )
            after = len(combined)
            if before != after:
                logger.debug("Dedup removed %d duplicate rows", before - after)
            out = combined

        out = out.sort_values("date").reset_index(drop=True)
        return out

    def append_incremental(self, new_records: pd.DataFrame) -> int:
        """Load existing store, merge new records, save. Returns count of NEW rows."""
        if new_records.empty:
            logger.debug("No new records to append")
            return 0

        existing = self.load()
        before_count = len(existing)
        merged = self.merge(new_records, existing)
        after_count = len(merged)
        added = after_count - before_count

        self.save(merged)
        logger.info("Appended %d new records (%d duplicates skipped)", added, after_count - before_count - added)
        return max(added, 0)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_market_df(self) -> pd.DataFrame:
        """Convert to the format expected by BackrunnerEngine.

        Returns DataFrame with exactly columns:
            slug, strike, market_price, date, expiry_date
        """
        df = self.load()
        if df.empty:
            return pd.DataFrame(columns=MARKET_COLUMNS)

        out = df.copy()
        out = out.rename(columns={"price": "market_price"})
        # Drop columns not needed by the backrunner
        drop_cols = [c for c in ["clobTokenId", "resolution"] if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

        # Ensure all required columns exist
        for col in MARKET_COLUMNS:
            if col not in out.columns:
                out[col] = None

        return out[MARKET_COLUMNS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Return an empty DataFrame with the correct schema and dtypes."""
        return pd.DataFrame(
            {
                "slug": pd.Series(dtype="str"),
                "clobTokenId": pd.Series(dtype="str"),
                "date": pd.Series(dtype="datetime64[ns, UTC]"),
                "price": pd.Series(dtype="float64"),
                "resolution": pd.Series(dtype="float64"),
                "strike": pd.Series(dtype="float64"),
                "expiry_date": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        )
