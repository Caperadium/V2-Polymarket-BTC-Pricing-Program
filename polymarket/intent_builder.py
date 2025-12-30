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

import hashlib
import json
import logging
import math
import uuid
from typing import List, Tuple, Optional, Dict, Any

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
    contract: str,
    expiry: str,
    outcome: str,
    action: str,
    limit_price: float,
    size_shares: float,
) -> str:
    """
    Compute deterministic intent_id from key fields.
    
    Including run_id ensures each generation batch has unique intents,
    even if the same trade is recommended multiple times.
    """
    payload = f"{run_id}|{contract}|{expiry}|{outcome}|{action}|{limit_price:.6f}|{size_shares:.6f}"
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


def create_run(strategy: str = "auto_reco", params: Optional[Dict] = None) -> Run:
    """
    Create a new run record in the database.
    
    Args:
        strategy: Strategy name
        params: Parameters used for generation
        
    Returns:
        The created Run object
    """
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
    finally:
        conn.close()
    
    return run


def build_intents_from_reco(
    df: pd.DataFrame,
    run: Run,
) -> Tuple[List[OrderIntent], List[Dict[str, Any]]]:
    """
    Convert auto_reco DataFrame to OrderIntent objects.
    
    Args:
        df: DataFrame from recommendations_to_dataframe()
        run: The Run object for this batch
        
    Returns:
        Tuple of (valid intents, list of warning dicts for invalid rows)
    """
    if df is None or df.empty:
        return [], []
    
    intents: List[OrderIntent] = []
    warnings_list: List[Dict[str, Any]] = []
    
    for idx, row in df.iterrows():
        # Validate row first
        row_warnings = validate_reco_row(row)
        
        if row_warnings:
            warnings_list.append({
                "index": idx,
                "contract": _extract_contract(row),
                "warnings": row_warnings,
            })
            continue
        
        try:
            intent = _build_single_intent(row, run)
            if intent:
                intents.append(intent)
        except Exception as e:
            warnings_list.append({
                "index": idx,
                "contract": _extract_contract(row),
                "warnings": [f"Build error: {str(e)}"],
            })
            logger.warning(f"Failed to build intent for row {idx}: {e}")
    
    return intents, warnings_list


def _extract_contract(row: pd.Series) -> str:
    """Extract contract identifier from row."""
    slug = str(row.get("slug", "") or "")
    expiry_key = str(row.get("expiry_key", "") or "")
    strike = row.get("strike", "")
    
    if slug:
        return f"{slug}_{expiry_key}_{strike}"
    return f"{expiry_key}_{strike}"


def _build_single_intent(row: pd.Series, run: Run) -> Optional[OrderIntent]:
    """Build a single OrderIntent from a DataFrame row."""
    # Extract fields with fallbacks for column name variations
    entry_price = row.get("entry_price")
    if entry_price is None or pd.isna(entry_price):
        entry_price = row.get("limit_price")
    
    stake = row.get("suggested_stake")
    if stake is None or pd.isna(stake):
        stake = row.get("stake_dollars")
    
    # Compute contract identifier
    contract = _extract_contract(row)
    expiry = str(row.get("expiry_key", "") or "")
    
    # Side -> outcome
    side = str(row.get("side", "YES")).upper()
    outcome = side if side in ("YES", "NO") else "YES"
    
    # Action is BUY for MVP
    action = "BUY"
    
    # Compute shares and notional (rounds DOWN)
    limit_price = float(entry_price)
    stake_usd = float(stake)
    
    size_shares, notional_usd = compute_shares_and_notional(stake_usd, limit_price)
    
    # Skip if shares round to 0
    if size_shares <= 0:
        logger.warning(f"Skipping {contract}: shares round to 0")
        return None
    
    # Generate deterministic intent_id
    intent_id = compute_intent_id(
        run.run_id, contract, expiry, outcome, action, limit_price, size_shares
    )
    
    # Extract optional numeric fields
    model_prob = _safe_float(row.get("model_prob"))
    market_prob = _safe_float(row.get("market_price"))
    edge = _safe_float(row.get("edge"))
    ev = _safe_float(row.get("expected_value_dollars")) or _safe_float(row.get("ev_dollars"))
    
    # Store original row for debugging
    raw_reco_json = row.to_json()
    
    return OrderIntent(
        intent_id=intent_id,
        run_id=run.run_id,
        created_at=utc_now_iso(),
        contract=contract,
        expiry=expiry,
        outcome=outcome,
        action=action,
        limit_price=limit_price,
        stake_usd=stake_usd,
        size_shares=size_shares,
        notional_usd=notional_usd,
        strategy=run.strategy,
        params_json=run.params_json,
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        ev=ev,
        status=IntentStatus.DRAFT,
        raw_reco_json=raw_reco_json,
    )


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
    
    Uses INSERT OR IGNORE to handle duplicates (idempotent).
    
    Args:
        intents: List of OrderIntent objects
        
    Returns:
        Number of intents actually inserted (not duplicates)
    """
    if not intents:
        return 0
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        inserted = 0
        
        for intent in intents:
            cursor.execute(
                """
                INSERT OR IGNORE INTO intents (
                    intent_id, run_id, created_at, strategy, params_json,
                    contract, expiry, outcome, action, limit_price,
                    stake_usd, size_shares, notional_usd, model_prob, market_prob,
                    edge, ev, status, approved_at, approved_snapshot_json,
                    submitted_at, notes, raw_reco_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                intent.to_insert_tuple(),
            )
            if cursor.rowcount > 0:
                inserted += 1
        
        conn.commit()
        logger.info(f"Saved {inserted} intents (of {len(intents)} total)")
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

