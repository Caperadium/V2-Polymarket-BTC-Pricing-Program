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
    Reconcile submission statuses with the provider.
    
    MVP Implementation:
    - Queries open submissions from database
    - Queries open orders from provider
    - For submissions not found in open orders, marks as FILLED
    - For submissions found, keeps status as OPEN
    
    In production, this would check actual order status from the API.
    
    Args:
        provider: The PolymarketProvider to query
        
    Returns:
        Summary dict with counts of updated/unchanged submissions
    """
    open_submissions = get_open_submissions()
    
    if not open_submissions:
        logger.info("No open submissions to reconcile")
        return {"open": 0, "updated": 0, "unchanged": 0}
    
    # Get open orders from provider
    open_orders = provider.get_open_orders()
    open_order_ids = {order.get("order_id") for order in open_orders if order.get("order_id")}
    
    updated = 0
    unchanged = 0
    
    for submission in open_submissions:
        if submission.order_id in open_order_ids:
            # Order still open - no change
            unchanged += 1
            logger.debug(f"Submission {submission.submission_id[:8]}... still open")
        else:
            # Order not found in open orders
            # In MVP, this likely means it was never really placed (fake provider)
            # In production, we'd check if it was filled/cancelled
            # For now, keep it OPEN
            unchanged += 1
            logger.debug(f"Submission {submission.submission_id[:8]}... not in open orders (keeping OPEN for MVP)")
    
    result = {
        "open": len(open_submissions),
        "updated": updated,
        "unchanged": unchanged,
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
