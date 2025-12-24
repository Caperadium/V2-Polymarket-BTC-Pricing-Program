"""
polymarket/accounting.py

Account state management and provider abstraction for the Polymarket Trade Console.

Provides:
- Abstract PolymarketProvider interface
- FakePolymarketProvider for MVP testing
- Account state fetching and caching
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

from polymarket.db import get_connection
from polymarket.models import AccountState, Submission, SubmissionStatus, utc_now_iso

logger = logging.getLogger(__name__)


class PolymarketProvider(ABC):
    """
    Abstract interface for Polymarket API operations.
    
    Implement this interface to swap between fake and real providers.
    """
    
    @abstractmethod
    def get_balance_allowance(self) -> Tuple[float, float]:
        """
        Get current balance and allowance.
        
        Returns:
            (collateral_balance, collateral_allowance)
        """
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get list of open orders.
        
        Returns:
            List of order dicts with at least: order_id, price, size, side
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            token_id: The token/contract to trade
            side: BUY or SELL
            price: Limit price
            size: Number of shares
            
        Returns:
            Response dict with at least: order_id, status
        """
        pass


class FakePolymarketProvider(PolymarketProvider):
    """
    Fake provider for MVP testing.
    
    Returns deterministic fake values without making real API calls.
    """
    
    def __init__(self, balance: float = 100.0, allowance: float = 100.0):
        self.balance = balance
        self.allowance = allowance
        self._order_counter = 0
    
    def get_balance_allowance(self) -> Tuple[float, float]:
        """Return configured fake balance and allowance."""
        logger.debug(f"FakeProvider: balance={self.balance}, allowance={self.allowance}")
        return self.balance, self.allowance
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Return open orders from submissions table.
        
        This queries the local database for submissions with status=OPEN.
        """
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.*, i.contract, i.outcome, i.action
                FROM submissions s
                JOIN intents i ON s.intent_id = i.intent_id
                WHERE s.status = ?
                """,
                (SubmissionStatus.OPEN,),
            )
            rows = cursor.fetchall()
            
            orders = []
            for row in rows:
                orders.append({
                    "order_id": row["order_id"],
                    "price": row["submitted_price"],
                    "size": row["submitted_size"],
                    "side": row["action"],
                    "contract": row["contract"],
                    "outcome": row["outcome"],
                })
            
            logger.debug(f"FakeProvider: found {len(orders)} open orders")
            return orders
        finally:
            conn.close()
    
    def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Dict[str, Any]:
        """
        Simulate placing an order.
        
        Returns a fake order_id and success status.
        """
        self._order_counter += 1
        order_id = f"FAKE-ORDER-{self._order_counter:06d}"
        
        response = {
            "order_id": order_id,
            "status": "OPEN",
            "token_id": token_id,
            "side": side,
            "price": price,
            "size": size,
            "timestamp": utc_now_iso(),
        }
        
        logger.info(f"FakeProvider: placed order {order_id} for {size} @ {price}")
        return response


def compute_reserved_collateral(provider: PolymarketProvider) -> float:
    """
    Compute total collateral reserved by open BUY orders.
    
    For BUY orders, reserved = sum(price * size) for all open orders.
    """
    open_orders = provider.get_open_orders()
    
    reserved = 0.0
    for order in open_orders:
        if order.get("side", "").upper() == "BUY":
            price = float(order.get("price", 0) or 0)
            size = float(order.get("size", 0) or 0)
            reserved += price * size
    
    logger.debug(f"Reserved collateral from open orders: {reserved:.2f}")
    return reserved


def fetch_account_state(provider: PolymarketProvider) -> AccountState:
    """
    Fetch current account state from provider.
    
    Args:
        provider: The PolymarketProvider to query
        
    Returns:
        AccountState with current balance, allowance, reserved, and available
    """
    balance, allowance = provider.get_balance_allowance()
    reserved = compute_reserved_collateral(provider)
    available = balance - reserved
    
    # Ensure available doesn't go negative
    available = max(0.0, available)
    
    state = AccountState(
        timestamp=utc_now_iso(),
        collateral_balance=balance,
        collateral_allowance=allowance,
        reserved_open_buys=reserved,
        available_collateral=available,
        raw_json=json.dumps({
            "balance": balance,
            "allowance": allowance,
            "reserved": reserved,
            "available": available,
        }),
    )
    
    logger.info(
        f"Account state: balance={balance:.2f}, allowance={allowance:.2f}, "
        f"reserved={reserved:.2f}, available={available:.2f}"
    )
    
    return state


def save_account_state(state: AccountState) -> None:
    """
    Save account state to database.
    
    Uses INSERT OR REPLACE to update if timestamp exists.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO account_state (
                timestamp, collateral_balance, collateral_allowance,
                reserved_open_buys, available_collateral, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                state.timestamp,
                state.collateral_balance,
                state.collateral_allowance,
                state.reserved_open_buys,
                state.available_collateral,
                state.raw_json,
            ),
        )
        conn.commit()
        logger.debug(f"Saved account state at {state.timestamp}")
    finally:
        conn.close()


def get_latest_account_state() -> Optional[AccountState]:
    """
    Get the most recent account state from database.
    
    Returns:
        AccountState or None if no records exist
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM account_state ORDER BY timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            return AccountState.from_row(row)
        return None
    finally:
        conn.close()


def create_approval_snapshot(state: AccountState) -> str:
    """
    Create a JSON snapshot for approval records.
    
    Args:
        state: Current account state
        
    Returns:
        JSON string with snapshot data
    """
    return state.to_snapshot_json()
