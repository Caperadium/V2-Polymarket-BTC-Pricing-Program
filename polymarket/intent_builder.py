"""
polymarket/intent_builder.py

Converts auto_reco DataFrame output to OrderIntent objects.

Handles:
- Column mapping from auto_reco output
- Deterministic intent_id generation (includes run_id)
- Safe size/notional calculation (rounds DOWN, never exceeds stake)
- Row validation with warning collection
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
import math
import sqlite3
import uuid
from typing import List, Tuple, Optional, Dict, Any, Union

import pandas as pd
import numpy as np

from polymarket.db import get_connection
from polymarket.models import Run, OrderIntent, IntentStatus, utc_now_iso

logger = logging.getLogger(__name__)


def compute_shares_and_notional(stake_usd: float, limit_price: float) -> Tuple[float, float]:
    """
    Compute shares rounded DOWN to 2 decimals, ensuring notional never exceeds stake.
    
    Args:
        stake_usd: Maximum USD to spend
        limit_price: Price per share (0 < price < 1)
        
    Returns:
        (size_shares, notional_usd) where notional_usd <= stake_usd
    """
    if limit_price <= 0 or limit_price >= 1:
        raise ValueError(f"limit_price must be in (0, 1), got {limit_price}")
    if stake_usd <= 0:
        raise ValueError(f"stake_usd must be > 0, got {stake_usd}")
    
    # Compute raw shares
    raw_shares = stake_usd / limit_price
    # Round DOWN to 2 decimal places (conservative)
    size_shares = math.floor(raw_shares * 100) / 100
    # Recompute actual notional
    notional_usd = size_shares * limit_price
    
    # Safety check: notional should never exceed stake
    if notional_usd > stake_usd + 1e-9:
        logger.warning(f"Notional {notional_usd:.4f} > stake {stake_usd:.4f}, clamping")
        size_shares = math.floor((stake_usd / limit_price) * 100) / 100
        notional_usd = size_shares * limit_price
    
    return size_shares, notional_usd


def compute_params_hash(params: Dict[str, Any]) -> str:
    """Compute deterministic hash of params dict."""
    # Sort keys for determinism
    sorted_json = json.dumps(params, sort_keys=True)
    return hashlib.sha256(sorted_json.encode()).hexdigest()[:16]


def compute_intent_id(
    run_id: str,
    intent_key: str,
) -> str:
    """
    Compute deterministic intent_id from logical key (run_id + logical_intent_key).
    
    This ensures that retrying a recommendation (with slightly different price/size)
    maps to the SAME intent_id, allowing safe upserts.
    """
    payload = f"{run_id}|{intent_key}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def validate_reco_row(row: pd.Series) -> List[str]:
    """
    Validate a recommendation row and return list of warnings.
    
    Args:
        row: Single row from auto_reco DataFrame
        
    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []
    
    # Get entry price
    entry_price = row.get("entry_price")
    if entry_price is None:
        entry_price = row.get("limit_price")
    
    if entry_price is not None:
        if pd.isna(entry_price):
            warnings.append("entry_price is NaN")
        elif not (0 < entry_price < 1):
            warnings.append(f"Price {entry_price:.4f} outside (0,1)")
    else:
        warnings.append("Missing entry_price")
    
    # Get stake
    stake = row.get("suggested_stake")
    if stake is None:
        stake = row.get("stake_dollars")
    
    if stake is not None:
        if pd.isna(stake):
            warnings.append("stake is NaN")
        elif stake <= 0:
            warnings.append(f"Stake {stake:.2f} <= 0")
    else:
        warnings.append("Missing stake")
    
    # Check for zero shares after rounding
    if entry_price and stake and not pd.isna(entry_price) and not pd.isna(stake):
        if entry_price > 0:
            raw_shares = stake / entry_price
            rounded_shares = math.floor(raw_shares * 100) / 100
            if rounded_shares <= 0:
                warnings.append(f"Shares round to 0 ({raw_shares:.4f} -> {rounded_shares})")
    
    # Check side/outcome
    side = row.get("side", "")
    if str(side).upper() not in ("YES", "NO"):
        warnings.append(f"Invalid side: {side}")
    
    return warnings


def create_run(
    strategy: str = "auto_reco", 
    params: Optional[Dict] = None,
    run_id: Optional[str] = None
) -> Run:
    """
    Create a new run record in the database.
    
    Args:
        strategy: Strategy name
        params: Parameters used for generation
        run_id: Optional explicit run_id (generates new UUID if None)
        
    Returns:
        The created Run object
    """
    if run_id is None:
        run_id = str(uuid.uuid4())
    created_at = utc_now_iso()
    params_json = json.dumps(params or {}, sort_keys=True)
    
    run = Run(
        run_id=run_id,
        created_at=created_at,
        strategy=strategy,
        params_json=params_json,
        notes="",
    )
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO runs (run_id, created_at, strategy, params_json, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run.run_id, run.created_at, run.strategy, run.params_json, run.notes),
        )
        conn.commit()
        logger.info(f"Created run {run_id}")
    except sqlite3.IntegrityError:
        # Idempotent: if run exists, that's fine (e.g. retry)
        logger.info(f"Run {run_id} already exists.")
    finally:
        conn.close()
    
    return run


def build_intents_from_reco(
    data: Union[pd.DataFrame, List[Any]],
    run: Union[Run, str],
) -> List[OrderIntent]:
    """
    Convert recommendations (DataFrame or List[DeltaIntent]) to OrderIntent objects.
    
    Args:
        data: DataFrame or List of DeltaIntent objects
        run: Run object or run_id string
        
    Returns:
        List of valid OrderIntent objects
    """
    intents: List[OrderIntent] = []
    
    # Resolve run attributes
    if isinstance(run, str):
        run_id = run
        strategy = "auto_reco"
        params_json = "{}"
    else:
        run_id = run.run_id
        strategy = run.strategy
        params_json = run.params_json
        
    # Validating Input
    if not isinstance(data, list):
         raise TypeError(f"build_intents_from_reco expects List[DeltaIntent], got {type(data)}")

    for item in data:
        # We assume item is DeltaIntent or compatible object
        try:
            intent = _build_intent_from_delta(item, run_id, strategy, params_json)
            if intent:
                intents.append(intent)
        except Exception as e:
            logger.error(f"Failed to build intent for item: {e}")
            
    return intents


def _create_dummy_run(run_id: str) -> Run:
    """Create dummy Run object for compatibility."""
    return Run(
        run_id=run_id,
        created_at=utc_now_iso(),
        strategy="auto_reco",
        params_json="{}",
        notes="dummy"
    )


def _build_intent_from_delta(
    delta: Any, # DeltaIntent
    run_id: str,
    strategy: str,
    params_json: str
) -> Optional[OrderIntent]:
    """Build OrderIntent from DeltaIntent."""
    # Validate
    if delta.action not in ["BUY", "SELL"]:
        return None
        
    # Invariants
    if delta.amount_usd < 0:
         logger.warning(f"Skipping intent with negative amount: {delta.amount_usd}")
         return None
         
    if delta.action in ["BUY", "SELL"] and getattr(delta, "price_mode", None) not in ["TAKER_ASK", "TAKER_BID"]:
        # We allow it, but log a warning as this is unusual for an actionable intent
        logger.warning(f"Intent {delta.slug} has actionable side {delta.action} but price_mode={getattr(delta, 'price_mode', 'None')}")

    contract = f"{delta.slug}_{delta.expiry_key}_{delta.strike}"
    expiry = delta.expiry_key
    outcome = "YES" if delta.side.upper() == "YES" else "NO" 
    
    intent_key = getattr(delta, "intent_key", None) or getattr(delta, "key", contract) or contract

    # Price and Size
    # For execution, limit_price_hint is our provisional limit
    limit_price = delta.limit_price_hint
    if not limit_price or limit_price <= 0:
        limit_price = delta.market_price # Fallback
        
    if not limit_price or limit_price <= 0:
        return None
        
    stake_usd = delta.amount_usd
    
    size_shares, notional_usd = compute_shares_and_notional(stake_usd, limit_price)
    
    if size_shares <= 0:
        return None
        
    # Deterministic ID (from Logical Key)
    intent_id = compute_intent_id(run_id, intent_key)
    
    return OrderIntent(
        intent_id=intent_id,
        run_id=run_id,
        intent_key=intent_key,
        created_at=utc_now_iso(),
        contract=contract,
        expiry=expiry,
        outcome=outcome,
        action=delta.action,
        limit_price=limit_price,
        stake_usd=stake_usd,
        size_shares=size_shares,
        notional_usd=notional_usd,
        strategy=strategy,
        params_json=params_json,
        model_prob=delta.model_prob,
        market_prob=delta.market_price,
        edge=delta.effective_edge,
        ev=delta.expected_value_dollars,
        status=IntentStatus.DRAFT,
        raw_reco_json=json.dumps(asdict(delta)) if hasattr(delta, "__dataclass_fields__") else "{}"
    )

# Legacy helpers _extract_contract and _build_single_intent have been removed.


def _safe_float(value) -> Optional[float]:
    """Safely convert value to float, returning None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def save_intents(intents: List[OrderIntent]) -> int:
    """
    Save intents to database with upsert logic.
    
    Uses ON CONFLICT(intent_id) to update existing drafts if parameters change.
    Only updates DRAFT intents; if an intent is already approved/submitted,
    this creates a potential conflict if the intent_id is the same.
    
    However, since intent_id includes run_id, and run_id is unique per run,
    we are only upserting within the CURRENT run context.
    
    Args:
        intents: List of OrderIntent objects
        
    Returns:
        Number of intents upserted
    """
    if not intents:
        return 0
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Guard: Check for attempts to overwrite non-DRAFT intents
        intent_ids = [i.intent_id for i in intents]
        placeholders = ",".join("?" * len(intent_ids))
        cursor.execute(f"SELECT intent_id, status FROM intents WHERE intent_id IN ({placeholders})", intent_ids)
        existing = {row[0]: row[1] for row in cursor.fetchall()}
        
        filtered_intents = []
        for intent in intents:
            current_status = existing.get(intent.intent_id)
            if current_status and current_status != IntentStatus.DRAFT:
                logger.warning(f"Skipping upsert for {intent.intent_id} (Status: {current_status}), preserves economics.")
                continue
            filtered_intents.append(intent)
            
        inserted = 0
        
        for intent in filtered_intents:
            # We use upsert logic. 
            # Note: We only update if status is DRAFT to avoid overwriting approved/submitted orders
            # (though normally we shouldn't be re-generating for the same run if already approved)
            cursor.execute(
                """
                INSERT INTO intents (
                    intent_id, run_id, intent_key, created_at, strategy, params_json,
                    contract, expiry, outcome, action, limit_price,
                    stake_usd, size_shares, notional_usd, model_prob, market_prob,
                    edge, ev, status, approved_at, approved_snapshot_json,
                    submitted_at, notes, raw_reco_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(intent_id) DO UPDATE SET
                    intent_key=excluded.intent_key,
                    limit_price=excluded.limit_price,
                    stake_usd=excluded.stake_usd,
                    size_shares=excluded.size_shares,
                    notional_usd=excluded.notional_usd,
                    model_prob=excluded.model_prob,
                    market_prob=excluded.market_prob,
                    edge=excluded.edge,
                    ev=excluded.ev,
                    raw_reco_json=excluded.raw_reco_json,
                    notes=excluded.notes
                WHERE status = 'DRAFT'
                """,
                intent.to_insert_tuple(),
            )
            if cursor.rowcount > 0:
                inserted += 1
        
        conn.commit()
        logger.info(f"Saved/Upserted {inserted} intents (of {len(intents)} total)")
        return inserted
    finally:
        conn.close()


def get_intents_by_status(status: str) -> List[OrderIntent]:
    """Fetch all intents with a given status."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM intents WHERE status = ? ORDER BY created_at DESC",
            (status,),
        )
        rows = cursor.fetchall()
        return [OrderIntent.from_row(row) for row in rows]
    finally:
        conn.close()


def get_intents_by_run(run_id: str) -> List[OrderIntent]:
    """Fetch all intents for a given run."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM intents WHERE run_id = ? ORDER BY created_at DESC",
            (run_id,),
        )
        rows = cursor.fetchall()
        return [OrderIntent.from_row(row) for row in rows]
    finally:
        conn.close()


def update_intent_status(
    intent_id: str,
    new_status: str,
    notes: Optional[str] = None,
    approved_at: Optional[str] = None,
    approved_snapshot_json: Optional[str] = None,
    submitted_at: Optional[str] = None,
) -> bool:
    """
    Update the status of an intent.
    
    Args:
        intent_id: The intent to update
        new_status: New status value
        notes: Optional notes to append
        approved_at: Optional approval timestamp
        approved_snapshot_json: Optional approval snapshot
        submitted_at: Optional submission timestamp
        
    Returns:
        True if update was successful
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Build dynamic UPDATE
        updates = ["status = ?"]
        params = [new_status]
        
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if approved_at is not None:
            updates.append("approved_at = ?")
            params.append(approved_at)
        if approved_snapshot_json is not None:
            updates.append("approved_snapshot_json = ?")
            params.append(approved_snapshot_json)
        if submitted_at is not None:
            updates.append("submitted_at = ?")
            params.append(submitted_at)
        
        params.append(intent_id)
        
        cursor.execute(
            f"UPDATE intents SET {', '.join(updates)} WHERE intent_id = ?",
            tuple(params),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def check_duplicate_intents(intents: List[OrderIntent]) -> List[str]:
    """
    Check for duplicate contract/expiry/outcome in a batch and against existing orders.
    
    Only considers existing orders that are APPROVED, SUBMITTED, or FILLED.
    Cancelled orders are excluded from duplicate check.
    
    Returns list of duplicate descriptions.
    """
    seen = set()
    duplicates = []
    
    # First, load existing non-cancelled intents from DB
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Query existing intents that are NOT cancelled (DRAFT, APPROVED, SUBMITTED, FILLED)
        # Only SUBMITTED and FILLED are truly "active"
        cursor.execute("""
            SELECT contract, expiry, outcome, status
            FROM intents
            WHERE status IN ('APPROVED', 'SUBMITTED', 'FILLED')
        """)
        
        for row in cursor.fetchall():
            key = (row[0], row[1], row[2])  # contract, expiry, outcome
            seen.add(key)
            
    finally:
        conn.close()
    
    # Now check new intents against existing + each other
    for intent in intents:
        key = (intent.contract, intent.expiry, intent.outcome)
        if key in seen:
            duplicates.append(f"{intent.contract} {intent.outcome}")
        seen.add(key)
    
    return duplicates


def clear_draft_intents() -> int:
    """
    Delete all DRAFT intents from the database.
    
    This should be called before generating new orders to clear the slate.
    
    Returns:
        Number of intents deleted
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM intents WHERE status = ?",
            (IntentStatus.DRAFT,),
        )
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"Cleared {deleted} draft intents")
        return deleted
    finally:
        conn.close()

