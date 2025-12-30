#!/usr/bin/env python3
"""Position tracking helpers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo

    ET_ZONE = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback for older Python
    ET_ZONE = timezone(timedelta(hours=-5))

POSITIONS_DEFAULT_PATH = Path("positions.csv")

POSITION_COLUMNS = [
    "id",
    "slug",
    "question",
    "expiry_key",
    "expiry_date",
    "strike",
    "side",
    "status",
    "entry_date",
    "entry_price",
    "size_shares",
    "notional",
    "current_price",
    "mtm_value",
    "realized_pnl",
    "unrealized_pnl",
    "model_prob_at_entry",
    "model_prob_latest",
    "notes",
]


@dataclass
class Position:
    id: Optional[str]
    slug: str
    question: str
    expiry_key: str
    strike: float
    side: str  # "YES" or "NO"
    status: str  # "OPEN" or "CLOSED"
    entry_date: datetime
    entry_price: float
    size_shares: float
    notional: float
    current_price: Optional[float] = None
    mtm_value: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: Optional[float] = None
    model_prob_at_entry: Optional[float] = None
    model_prob_latest: Optional[float] = None
    notes: str = ""


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in POSITION_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[POSITION_COLUMNS].copy()


def _parse_expiry_cutoff(value: object) -> Optional[datetime]:
    """Return expiry datetime (UTC) corresponding to 12PM ET on expiry date."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    # Handle timestamps with time component first
    if "T" in text or " " in text:
        try:
            cleaned = text.replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    # Fallback to date-only parsing
    try:
        date_part = text[:10]
        date_obj = datetime.strptime(date_part, "%Y-%m-%d")
    except ValueError:
        return None
    cutoff_local = datetime(
        date_obj.year,
        date_obj.month,
        date_obj.day,
        12,
        0,
        tzinfo=ET_ZONE,
    )
    return cutoff_local.astimezone(timezone.utc)


def _auto_close_expired(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Mark positions as closed if expiry (12PM ET) has passed."""
    if df.empty:
        return df
    now_utc = datetime.now(timezone.utc)
    changed = False
    for idx, row in df.iterrows():
        status = str(row.get("status", "")).upper()
        if status != "OPEN":
            continue
        expiry_val = row.get("expiry_date")
        if not expiry_val:
            expiry_val = row.get("expiry_key")
        cutoff = _parse_expiry_cutoff(expiry_val)
        if cutoff is None:
            continue
        if now_utc >= cutoff:
            df.at[idx, "status"] = "CLOSED"
            # Preserve realized PnL if already set, else copy unrealized if available
            realized = row.get("realized_pnl")
            unrealized = row.get("unrealized_pnl")
            if (pd.isna(realized) or realized is None) and pd.notna(unrealized):
                df.at[idx, "realized_pnl"] = unrealized
            note = str(row.get("notes") or "")
            timestamp = now_utc.isoformat()
            suffix = f" Auto-closed at expiry ({timestamp})."
            df.at[idx, "notes"] = (note + " " + suffix).strip()
            changed = True
    if changed:
        save_positions(df, path)
    return df


def load_positions(path: Path = POSITIONS_DEFAULT_PATH) -> pd.DataFrame:
    """Load positions from CSV. Return empty DataFrame if file missing."""
    if not path.exists():
        return pd.DataFrame(columns=POSITION_COLUMNS)
    df = pd.read_csv(path)
    df = _ensure_columns(df)
    df = _auto_close_expired(df, path)
    return df


def save_positions(df: pd.DataFrame, path: Path = POSITIONS_DEFAULT_PATH) -> None:
    """Persist positions DataFrame."""
    df = _ensure_columns(df)
    df.to_csv(path, index=False)


def get_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to OPEN positions."""
    if df.empty:
        return df.copy()
    mask = df["status"].astype(str).str.upper() == "OPEN"
    return df[mask].copy()


def append_position(new_pos: Position, path: Path = POSITIONS_DEFAULT_PATH) -> None:
    """Append new position row."""
    df = load_positions(path)
    record = {
        "id": new_pos.id or str(uuid.uuid4()),
        "slug": new_pos.slug,
        "question": new_pos.question,
        "expiry_key": new_pos.expiry_key,
        "expiry_date": new_pos.expiry_key,
        "strike": new_pos.strike,
        "side": new_pos.side,
        "status": new_pos.status,
        "entry_date": new_pos.entry_date.isoformat(),
        "entry_price": new_pos.entry_price,
        "size_shares": new_pos.size_shares,
        "notional": new_pos.notional,
        "current_price": new_pos.current_price,
        "mtm_value": new_pos.mtm_value,
        "realized_pnl": new_pos.realized_pnl,
        "unrealized_pnl": new_pos.unrealized_pnl,
        "model_prob_at_entry": new_pos.model_prob_at_entry,
        "model_prob_latest": new_pos.model_prob_latest,
        "notes": new_pos.notes or "",
    }
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save_positions(df, path)


def mark_position_closed(
    path: Path,
    position_id: str,
    realized_pnl: float,
    close_price: float,
    close_date: Optional[datetime] = None,
) -> None:
    """Mark a position as closed and update realized PnL."""
    df = load_positions(path)
    if df.empty:
        raise ValueError("No positions available.")
    mask = df["id"].astype(str) == str(position_id)
    if not mask.any():
        raise ValueError(f"Position id {position_id} not found.")
    df.loc[mask, "status"] = "CLOSED"
    df.loc[mask, "realized_pnl"] = realized_pnl
    df.loc[mask, "current_price"] = close_price
    df.loc[mask, "mtm_value"] = close_price * df.loc[mask, "size_shares"].astype(float)
    if close_date:
        df.loc[mask, "notes"] = df.loc[mask, "notes"].fillna("").astype(str) + f" Closed {close_date.isoformat()}"
    save_positions(df, path)


def enrich_positions_with_batch(
    positions_df: pd.DataFrame,
    batch_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join open positions with latest batch info."""
    if positions_df.empty or batch_df is None or batch_df.empty:
        return positions_df.copy()

    df = positions_df.copy()
    batch = batch_df.copy()
    batch["slug"] = batch.get("slug", "").astype(str)
    batch["expiry_key"] = batch.get("expiry_key", batch.get("expiry_date", ""))
    price_col = batch.columns.intersection(["market_price", "market_pr"]).tolist()
    if price_col:
        price_col = price_col[0]
    else:
        batch["market_price"] = np.nan
        price_col = "market_price"
    model_col = batch.columns.intersection(["p_model_fit", "p_real_mc"]).tolist()
    if model_col:
        model_col = model_col[0]
    else:
        batch["p_model_fit"] = np.nan
        model_col = "p_model_fit"

    merge_cols = ["slug"]
    if "slug" not in df.columns:
        merge_cols = ["expiry_key", "strike"]

    df = df.merge(
        batch[
            merge_cols
            + [
                price_col,
                model_col,
            ]
        ],
        on=merge_cols,
        how="left",
        suffixes=("", "_batch"),
    )

    df["current_price"] = np.where(
        df["side"].str.upper() == "NO",
        1.0 - df[price_col],
        df[price_col],
    )
    df["mtm_value"] = df["current_price"] * df["size_shares"].astype(float)
    entry_cost = df["entry_price"].astype(float) * df["size_shares"].astype(float)
    df["unrealized_pnl"] = df["mtm_value"] - entry_cost
    df["model_prob_latest"] = df[model_col]
    return df


def ensure_position_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a stable position_key exists for joining across files."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    def _key(row: pd.Series) -> str:
        slug = str(row.get("slug") or "").strip()
        side = str(row.get("side") or "").strip().upper()
        expiry = str(row.get("expiry_key") or row.get("expiry_date") or "").strip()
        strike = row.get("strike")
        strike_part = ""
        if strike is not None and not (isinstance(strike, float) and np.isnan(strike)):
            try:
                strike_part = f"{float(strike):.8g}"
            except Exception:
                strike_part = str(strike).strip()
        parts = [slug, side, expiry, strike_part]
        return "|".join(parts)

    df = df.copy()
    df["position_key"] = df.apply(_key, axis=1)
    return df


# Snapshot for open positions on disk
OPEN_POSITIONS_DEFAULT_PATH = Path("open_positions.csv")


def load_open_positions(path: Path = OPEN_POSITIONS_DEFAULT_PATH) -> pd.DataFrame:
    """Load open positions snapshot; return empty DataFrame if missing or invalid."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_open_positions(df: pd.DataFrame, path: Path = OPEN_POSITIONS_DEFAULT_PATH) -> None:
    """Persist open positions snapshot."""
    df.to_csv(path, index=False)


def sync_open_positions_with_batch(
    positions_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    path: Path = OPEN_POSITIONS_DEFAULT_PATH,
) -> pd.DataFrame:
    """
    Maintain open_positions.csv as a live snapshot:
    - Start from OPEN rows in positions_df.
    - Enrich them with the current batch.
    - Drop any rows that are no longer OPEN.
    - Save snapshot to disk and return the enriched DataFrame.
    """
    positions_df = ensure_position_keys(positions_df.copy())
    open_df = get_open_positions(positions_df)

    # Keep only currently OPEN positions
    open_df = open_df.copy()
    open_df = _ensure_columns(open_df)

    # Enrich with latest batch if provided
    enriched = enrich_positions_with_batch(open_df, batch_df) if batch_df is not None else open_df

    # Save snapshot
    save_open_positions(enriched, path)
    return enriched
