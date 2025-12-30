"""
polymarket/execution_gateway.py

Order validation and submission for the Polymarket Trade Console.

Provides:
- Single intent validation
- Batch submission with cumulative collateral tracking
- Submission record creation
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Tuple, Optional, Dict, Any

from polymarket.db import get_connection
from polymarket.models import (
    OrderIntent,
    Submission,
    AccountState,
    IntentStatus,
    SubmissionStatus,
    utc_now_iso,
)
from polymarket.accounting import PolymarketProvider
from polymarket.intent_builder import update_intent_status

logger = logging.getLogger(__name__)


def validate_intent(
    intent: OrderIntent,
    available_collateral: float,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a single intent against available collateral.
    
    Checks:
    - status == APPROVED
    - available_collateral >= notional_usd
    - size_shares > 0
    - limit_price in (0, 1)
    
    Args:
        intent: The OrderIntent to validate
        available_collateral: Available collateral after prior orders
        
    Returns:
        (is_valid, error_message or None)
    """
    # Status check
    if intent.status != IntentStatus.APPROVED:
        return False, f"Intent status is {intent.status}, expected APPROVED"
    
    # Price range check
    if intent.limit_price <= 0 or intent.limit_price >= 1:
        return False, f"Price {intent.limit_price:.4f} outside valid range (0, 1)"
    
    # Size check
    if intent.size_shares <= 0:
        return False, f"Size {intent.size_shares:.4f} must be > 0"
    
    # Collateral check
    if intent.notional_usd > available_collateral:
        return False, (
            f"Insufficient collateral: need {intent.notional_usd:.2f}, "
            f"have {available_collateral:.2f}"
        )
    
    return True, None


def submit_intent(
    intent: OrderIntent,
    provider: PolymarketProvider,
) -> Tuple[Optional[Submission], Optional[str]]:
    """
    Submit a single intent to the provider.
    
    Args:
        intent: The approved intent to submit
        provider: The PolymarketProvider to use
        
    Returns:
        (Submission, None) on success or (None, error_message) on failure
    """
    try:
        # Resolve the actual CLOB token_id from the contract slug and outcome
        # The intent.contract is a slug, but CLOB needs the numeric token_id
        # Use intent.outcome (YES/NO) not intent.action (BUY/SELL)
        token_id = provider.fetch_clob_token_id(intent.contract, intent.outcome)
        
        if not token_id:
            error_msg = f"Could not resolve token_id for {intent.contract} ({intent.outcome})"
            logger.error(error_msg)
            return None, error_msg
        
        # Call provider to place order
        response = provider.place_order(
            token_id=token_id,
            side=intent.action,
            price=intent.limit_price,
            size=intent.size_shares,
        )
        
        # Check if order placement succeeded
        if not response.get("success"):
            error_msg = response.get("error", "Unknown error from place_order")
            logger.error(f"Order placement failed for {intent.intent_id[:8]}...: {error_msg}")
            return None, f"Order placement failed: {error_msg}"
        
        # Create submission record only on success
        submission_id = str(uuid.uuid4())
        submitted_at = utc_now_iso()
        
        submission = Submission(
            submission_id=submission_id,
            intent_id=intent.intent_id,
            submitted_at=submitted_at,
            order_id=response.get("order_id"),
            submitted_price=intent.limit_price,
            submitted_size=intent.size_shares,
            status=SubmissionStatus.OPEN,  # OPEN, not PENDING
            raw_response_json=json.dumps(response),
        )
        
        # Save submission to database
        _save_submission(submission)
        
        logger.info(
            f"Submitted intent {intent.intent_id[:8]}... -> order {submission.order_id}"
        )
        
        return submission, None
        
    except Exception as e:
        error_msg = f"Submission failed: {str(e)}"
        logger.error(f"Failed to submit intent {intent.intent_id}: {e}")
        return None, error_msg


def submit_approved_batch(
    intents: List[OrderIntent],
    account_state: AccountState,
    provider: PolymarketProvider,
) -> List[Dict[str, Any]]:
    """
    Submit multiple approved intents with cumulative collateral tracking.
    
    Algorithm:
    1. Sort intents by creation order
    2. Track available collateral cumulatively
    3. For each intent:
       - Validate against remaining available collateral
       - Submit if valid, mark FAILED if not
    
    Args:
        intents: List of APPROVED intents to submit
        account_state: Current account state
        provider: The PolymarketProvider to use
        
    Returns:
        List of result dicts with keys: intent_id, success, submission, error
    """
    results = []
    
    # Start with current available collateral
    available = account_state.available_collateral
    
    # Also check allowance
    allowance = account_state.collateral_allowance
    
    # Sort by creation time for deterministic ordering
    sorted_intents = sorted(intents, key=lambda i: i.created_at)
    
    for intent in sorted_intents:
        result = {
            "intent_id": intent.intent_id,
            "contract": intent.contract,
            "success": False,
            "submission": None,
            "error": None,
        }
        
        # Validate against remaining available collateral
        is_valid, error_msg = validate_intent(intent, available)
        
        if not is_valid:
            # Mark intent as FAILED
            update_intent_status(
                intent.intent_id,
                IntentStatus.FAILED,
                notes=error_msg,
            )
            result["error"] = error_msg
            results.append(result)
            logger.warning(f"Intent {intent.intent_id[:8]}... failed validation: {error_msg}")
            continue
        
        # Check allowance too
        if intent.notional_usd > allowance:
            error_msg = f"Insufficient allowance: need {intent.notional_usd:.2f}, have {allowance:.2f}"
            update_intent_status(
                intent.intent_id,
                IntentStatus.FAILED,
                notes=error_msg,
            )
            result["error"] = error_msg
            results.append(result)
            logger.warning(f"Intent {intent.intent_id[:8]}... failed allowance check: {error_msg}")
            continue
        
        # Submit the order
        submission, submit_error = submit_intent(intent, provider)
        
        if submission:
            # Success - update collateral tracking
            available -= intent.notional_usd
            allowance -= intent.notional_usd  # Conservative: reduce allowance too
            
            # Update intent status to SUBMITTED
            update_intent_status(
                intent.intent_id,
                IntentStatus.SUBMITTED,
                submitted_at=submission.submitted_at,
            )
            
            result["success"] = True
            result["submission"] = submission
            logger.info(f"Intent {intent.intent_id[:8]}... submitted successfully")
        else:
            # Submission failed
            update_intent_status(
                intent.intent_id,
                IntentStatus.FAILED,
                notes=submit_error,
            )
            result["error"] = submit_error
            logger.error(f"Intent {intent.intent_id[:8]}... submission failed: {submit_error}")
        
        results.append(result)
    
    # Summary log
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Batch submission complete: {success_count}/{len(results)} successful")
    
    return results


def _save_submission(submission: Submission) -> None:
    """Save a submission record to the database."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO submissions (
                submission_id, intent_id, submitted_at, order_id,
                submitted_price, submitted_size, status, raw_response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            submission.to_insert_tuple(),
        )
        conn.commit()
    finally:
        conn.close()


def get_submissions_by_intent(intent_id: str) -> List[Submission]:
    """Get all submissions for an intent."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM submissions WHERE intent_id = ? ORDER BY submitted_at DESC",
            (intent_id,),
        )
        rows = cursor.fetchall()
        return [Submission.from_row(row) for row in rows]
    finally:
        conn.close()


def get_open_submissions() -> List[Submission]:
    """Get all submissions with status=OPEN."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM submissions WHERE status = ? ORDER BY submitted_at DESC",
            (SubmissionStatus.OPEN,),
        )
        rows = cursor.fetchall()
        return [Submission.from_row(row) for row in rows]
    finally:
        conn.close()
