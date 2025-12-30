"""
polymarket/ingest.py

Ingestion pipeline for Polymarket API data.

Syncs trades and closed positions from the Polymarket API to local SQLite ledger.
Supports idempotent incremental updates using cursor/timestamp tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from polymarket.db import get_connection
from polymarket.provider_polymarket import RealPolymarketProvider, get_provider

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Statistics from a sync operation."""
    closed_positions_fetched: int = 0
    closed_positions_inserted: int = 0
    closed_positions_updated: int = 0
    last_sync_ts: Optional[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "closed_positions_fetched": self.closed_positions_fetched,
            "closed_positions_inserted": self.closed_positions_inserted,
            "closed_positions_updated": self.closed_positions_updated,
            "last_sync_ts": self.last_sync_ts,
            "errors": self.errors,
        }


def get_sync_metadata(key: str) -> Optional[str]:
    """
    Get a sync metadata value by key.
    
    Args:
        key: Metadata key (e.g., "last_closed_positions_sync")
        
    Returns:
        Value if exists, None otherwise
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM pm_sync_metadata WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def set_sync_metadata(key: str, value: str) -> None:
    """
    Set a sync metadata value.
    
    Uses INSERT OR REPLACE for idempotency.
    
    Args:
        key: Metadata key
        value: Value to store
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO pm_sync_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, now)
        )
        conn.commit()
    finally:
        conn.close()


def get_last_sync_time() -> Optional[datetime]:
    """
    Get the last sync timestamp.
    
    Returns:
        datetime if synced before, None otherwise
    """
    value = get_sync_metadata("last_closed_positions_sync")
    if value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _generate_position_id(position: Dict[str, Any], user_address: str) -> str:
    """
    Generate a stable position ID for deduplication.
    
    Uses conditionId + outcomeIndex + user to create unique key.
    """
    condition_id = position.get("conditionId", "")
    outcome_index = position.get("outcomeIndex", 0)
    return f"{user_address}:{condition_id}:{outcome_index}"


def upsert_closed_position(position: Dict[str, Any], user_address: str) -> Tuple[bool, bool]:
    """
    Insert or update a closed position.
    
    Args:
        position: Position dict from API
        user_address: User's wallet address
        
    Returns:
        Tuple of (inserted, updated) booleans
    """
    position_id = _generate_position_id(position, user_address)
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute(
            "SELECT position_id FROM pm_closed_positions WHERE position_id = ?",
            (position_id,)
        )
        exists = cursor.fetchone() is not None
        
        # Extract fields with safe defaults
        condition_id = position.get("conditionId", "")
        market_slug = position.get("slug", position.get("marketSlug", ""))
        title = position.get("title", "")
        outcome = position.get("outcome", "")
        outcome_index = position.get("outcomeIndex", 0)
        avg_price = float(position.get("avgPrice", 0) or 0)
        size = float(position.get("size", position.get("totalBought", 0)) or 0)
        total_bought = float(position.get("totalBought", 0) or 0)
        realized_pnl = float(position.get("realizedPnl", 0) or 0)
        cur_price = float(position.get("curPrice", 0) or 0)
        timestamp = position.get("timestamp", "")
        end_date = position.get("endDate", "")
        raw_json = json.dumps(position)
        
        if exists:
            # Update existing
            cursor.execute(
                """
                UPDATE pm_closed_positions SET
                    condition_id = ?,
                    market_slug = ?,
                    title = ?,
                    outcome = ?,
                    outcome_index = ?,
                    avg_price = ?,
                    size = ?,
                    total_bought = ?,
                    realized_pnl = ?,
                    cur_price = ?,
                    resolved_at = ?,
                    end_date = ?,
                    raw_json = ?
                WHERE position_id = ?
                """,
                (
                    condition_id, market_slug, title, outcome, outcome_index,
                    avg_price, size, total_bought, realized_pnl, cur_price,
                    timestamp, end_date, raw_json, position_id
                )
            )
            conn.commit()
            return False, True
        else:
            # Insert new
            cursor.execute(
                """
                INSERT INTO pm_closed_positions (
                    position_id, user_address, condition_id, market_slug, title,
                    outcome, outcome_index, avg_price, size, total_bought,
                    realized_pnl, cur_price, resolved_at, end_date, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position_id, user_address, condition_id, market_slug, title,
                    outcome, outcome_index, avg_price, size, total_bought,
                    realized_pnl, cur_price, timestamp, end_date, raw_json
                )
            )
            conn.commit()
            return True, False
    finally:
        conn.close()


def sync_closed_positions(
    provider: Optional[RealPolymarketProvider] = None,
    max_positions: int = 1000,
) -> SyncStats:
    """
    Sync closed positions from Polymarket API to local database.
    
    Args:
        provider: RealPolymarketProvider instance (creates one if None)
        max_positions: Maximum positions to fetch
        
    Returns:
        SyncStats with counts and any errors
    """
    stats = SyncStats()
    
    if provider is None:
        provider = get_provider(mode="read_only")
    
    try:
        # Fetch all closed positions
        positions = provider.fetch_all_closed_positions(max_positions=max_positions)
        stats.closed_positions_fetched = len(positions)
        
        # Upsert each position
        for pos in positions:
            try:
                inserted, updated = upsert_closed_position(pos, provider.user_address)
                if inserted:
                    stats.closed_positions_inserted += 1
                elif updated:
                    stats.closed_positions_updated += 1
            except Exception as e:
                logger.error(f"Failed to upsert position: {e}")
                stats.errors.append(str(e))
        
        # Update sync timestamp
        now = datetime.now(timezone.utc).isoformat()
        set_sync_metadata("last_closed_positions_sync", now)
        stats.last_sync_ts = now
        
        logger.info(
            f"Sync complete: {stats.closed_positions_inserted} inserted, "
            f"{stats.closed_positions_updated} updated"
        )
        
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        stats.errors.append(str(e))
    
    return stats


def sync_polymarket_ledger(
    days_back: int = 30,
    max_positions: int = 1000,
) -> Dict[str, Any]:
    """
    Main entry point for syncing Polymarket data.
    
    This is the function to call from Streamlit.
    
    Args:
        days_back: How many days of history to fetch (for trades)
        max_positions: Maximum closed positions to fetch
        
    Returns:
        Dict with sync statistics
    """
    logger.info(f"Starting Polymarket ledger sync (days_back={days_back})")
    
    provider = get_provider(mode="read_only")
    
    # Sync closed positions (primary source for realized PnL)
    closed_stats = sync_closed_positions(provider, max_positions=max_positions)
    
    # TODO: Add trades sync when CLOB API integration is complete
    
    return {
        "success": len(closed_stats.errors) == 0,
        "closed_positions": closed_stats.to_dict(),
        "last_sync": closed_stats.last_sync_ts,
    }


def get_closed_positions_count() -> int:
    """
    Get count of closed positions in local database.
    
    Useful for verification.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM pm_closed_positions")
        row = cursor.fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def get_total_realized_pnl() -> float:
    """
    Get sum of realized PnL from all closed positions.
    
    Useful for verification and reconciliation.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(realized_pnl) as total FROM pm_closed_positions")
        row = cursor.fetchone()
        return float(row["total"]) if row and row["total"] else 0.0
    finally:
        conn.close()
