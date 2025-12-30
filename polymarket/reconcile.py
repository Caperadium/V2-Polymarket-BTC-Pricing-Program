"""
polymarket/reconcile.py

Status reconciliation for the Polymarket Trade Console.

Provides:
- Submission status synchronization with provider
- Intent status updates based on submission states

MVP Implementation:
- All submissions remain OPEN (no real API calls)
- Intent status stays SUBMITTED while submission is OPEN
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from polymarket.db import get_connection
from polymarket.models import (
    Submission,
    IntentStatus,
    SubmissionStatus,
)
from polymarket.accounting import PolymarketProvider
from polymarket.execution_gateway import get_open_submissions

logger = logging.getLogger(__name__)


def reconcile_submissions(provider: PolymarketProvider) -> Dict[str, Any]:
    """
    Reconcile submission statuses with the Polymarket CLOB API.
    
    For each open submission:
    - Queries the CLOB API for the order's current status
    - Updates local submission/intent status based on real status
    
    CLOB statuses: LIVE, MATCHED, CANCELLED
    - LIVE = still open on the book
    - MATCHED = fully filled
    - CANCELLED = cancelled
    
    Args:
        provider: The PolymarketProvider to query
        
    Returns:
        Summary dict with counts of updated/unchanged submissions
    """
    open_submissions = get_open_submissions()
    
    if not open_submissions:
        logger.info("No open submissions to reconcile")
        return {"open": 0, "filled": 0, "cancelled": 0, "still_open": 0, "errors": 0}
    
    logger.info(f"Reconciling {len(open_submissions)} open submissions")
    
    filled = 0
    cancelled = 0
    still_open = 0
    errors = 0
    
    for submission in open_submissions:
        if not submission.order_id:
            # No order_id means the order was never actually placed
            # Mark as cancelled so it doesn't stay in "submitted" state forever
            logger.info(f"Submission {submission.submission_id[:8]}... has no order_id - marking as CANCELLED")
            update_submission_status(submission.submission_id, SubmissionStatus.CANCELLED)
            cancelled += 1
            continue
        
        try:
            # Query CLOB API for order status
            order_status = provider.fetch_order_status(submission.order_id)
            
            if order_status is None:
                # Order not found - may have been fully matched and removed
                # Check if it could have been filled
                logger.info(f"Order {submission.order_id[:12]}... not found - marking as FILLED")
                update_submission_status(submission.submission_id, SubmissionStatus.FILLED)
                filled += 1
                continue
            
            status = order_status.get("status", "").upper()
            
            if status == "MATCHED":
                # Fully filled
                logger.info(f"Order {submission.order_id[:12]}... is MATCHED (filled)")
                update_submission_status(submission.submission_id, SubmissionStatus.FILLED)
                filled += 1
            elif status in ("CANCELLED", "CANCELED"):  # Handle both spellings
                logger.info(f"Order {submission.order_id[:12]}... is CANCELLED")
                update_submission_status(submission.submission_id, SubmissionStatus.CANCELLED)
                cancelled += 1
            elif status == "LIVE":
                # Still open
                logger.debug(f"Order {submission.order_id[:12]}... still LIVE")
                still_open += 1
            else:
                # Unknown status
                logger.warning(f"Order {submission.order_id[:12]}... has unknown status: {status}")
                still_open += 1
                
        except Exception as e:
            logger.error(f"Error reconciling submission {submission.submission_id[:8]}...: {e}")
            errors += 1
    
    result = {
        "open": len(open_submissions),
        "filled": filled,
        "cancelled": cancelled,
        "still_open": still_open,
        "errors": errors,
    }
    
    logger.info(f"Reconciliation complete: {result}")
    return result


def update_submission_status(
    submission_id: str,
    new_status: str,
) -> bool:
    """
    Update the status of a submission.
    
    Also updates the linked intent status if appropriate:
    - FILLED submission -> FILLED intent
    - CANCELLED submission -> CANCELLED intent
    - PARTIAL submission -> PARTIAL intent
    
    Args:
        submission_id: The submission to update
        new_status: New status value
        
    Returns:
        True if update was successful
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Get current submission
        cursor.execute(
            "SELECT * FROM submissions WHERE submission_id = ?",
            (submission_id,),
        )
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Submission {submission_id} not found")
            return False
        
        submission = Submission.from_row(row)
        
        # Update submission status
        cursor.execute(
            "UPDATE submissions SET status = ? WHERE submission_id = ?",
            (new_status, submission_id),
        )
        
        # Map submission status to intent status
        intent_status_map = {
            SubmissionStatus.FILLED: IntentStatus.FILLED,
            SubmissionStatus.PARTIAL: IntentStatus.PARTIAL,
            SubmissionStatus.CANCELLED: IntentStatus.CANCELLED,
            SubmissionStatus.FAILED: IntentStatus.FAILED,
        }
        
        if new_status in intent_status_map:
            new_intent_status = intent_status_map[new_status]
            cursor.execute(
                "UPDATE intents SET status = ? WHERE intent_id = ?",
                (new_intent_status, submission.intent_id),
            )
            logger.info(
                f"Updated submission {submission_id[:8]}... to {new_status}, "
                f"intent to {new_intent_status}"
            )
        else:
            logger.info(f"Updated submission {submission_id[:8]}... to {new_status}")
        
        conn.commit()
        return True
    finally:
        conn.close()


def get_reconciliation_summary() -> Dict[str, int]:
    """
    Get a summary of submission statuses.
    
    Returns:
        Dict mapping status -> count
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT status, COUNT(*) as count
            FROM submissions
            GROUP BY status
            """
        )
        rows = cursor.fetchall()
        return {row["status"]: row["count"] for row in rows}
    finally:
        conn.close()
