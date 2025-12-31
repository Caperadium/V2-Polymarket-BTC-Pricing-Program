"""
polymarket/models.py

Dataclasses for the Polymarket Trade Console.

Provides type-safe representations for:
- Run: A generation batch
- OrderIntent: A trade intent
- Submission: An order submission record
- AccountState: Account balance snapshot
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


# Status constants for type safety
class IntentStatus:
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    
    ALL = [DRAFT, APPROVED, SUBMITTED, FILLED, PARTIAL, CANCELLED, FAILED, SKIPPED]


class SubmissionStatus:
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    
    ALL = [OPEN, FILLED, PARTIAL, CANCELLED, FAILED]


@dataclass
class Run:
    """Represents a generation run/batch of intents."""
    run_id: str
    created_at: str  # ISO format
    strategy: str = ""
    params_json: str = "{}"
    notes: str = ""
    
    @classmethod
    def from_row(cls, row) -> "Run":
        """Create Run from a database row."""
        return cls(
            run_id=row["run_id"],
            created_at=row["created_at"],
            strategy=row["strategy"] or "",
            params_json=row["params_json"] or "{}",
            notes=row["notes"] or "",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderIntent:
    """Represents a trade intent."""
    intent_id: str
    run_id: str
    intent_key: str  # Logical key for stability
    created_at: str  # ISO format
    contract: str
    outcome: str  # YES or NO
    action: str  # BUY or SELL
    limit_price: float
    stake_usd: float
    size_shares: float
    notional_usd: float
    
    # Optional fields
    expiry: str = ""
    strategy: str = ""
    params_json: str = "{}"
    model_prob: Optional[float] = None
    market_prob: Optional[float] = None
    edge: Optional[float] = None
    ev: Optional[float] = None
    status: str = IntentStatus.DRAFT
    approved_at: Optional[str] = None
    approved_snapshot_json: Optional[str] = None
    submitted_at: Optional[str] = None
    notes: str = ""
    raw_reco_json: str = "{}"
    
    @classmethod
    def from_row(cls, row) -> "OrderIntent":
        """
        Create OrderIntent from a database row (dict or sqlite3.Row).
        """
        # Helper to safely get value whether row is dict or sqlite3.Row
        def get_val(key, default=None):
            if hasattr(row, "keys"):
                return row[key] if key in row.keys() else default
            return default

        intent_key = get_val("intent_key", "") or "" 
        
        return cls(
            intent_id=row["intent_id"],
            run_id=row["run_id"],
            intent_key=intent_key,
            created_at=row["created_at"],
            contract=row["contract"],
            outcome=row["outcome"],
            action=row["action"],
            limit_price=row["limit_price"],
            stake_usd=row["stake_usd"],
            size_shares=row["size_shares"],
            notional_usd=row["notional_usd"],
            expiry=row["expiry"] or "",
            strategy=row["strategy"] or "",
            params_json=row["params_json"] or "{}",
            model_prob=get_val("model_prob"),
            market_prob=get_val("market_prob"),
            edge=get_val("edge"),
            ev=get_val("ev"),
            status=row["status"],
            approved_at=get_val("approved_at"),
            approved_snapshot_json=get_val("approved_snapshot_json"),
            submitted_at=get_val("submitted_at"),
            notes=row["notes"] or "",
            raw_reco_json=row["raw_reco_json"] or "{}",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_insert_tuple(self) -> tuple:
        """Return tuple for INSERT statement."""
        return (
            self.intent_id,
            self.run_id,
            self.intent_key,
            self.created_at,
            self.strategy,
            self.params_json,
            self.contract,
            self.expiry,
            self.outcome,
            self.action,
            self.limit_price,
            self.stake_usd,
            self.size_shares,
            self.notional_usd,
            self.model_prob,
            self.market_prob,
            self.edge,
            self.ev,
            self.status,
            self.approved_at,
            self.approved_snapshot_json,
            self.submitted_at,
            self.notes,
            self.raw_reco_json,
        )


@dataclass
class Submission:
    """Records an order submission attempt."""
    submission_id: str
    intent_id: str
    submitted_at: str  # ISO format
    status: str = SubmissionStatus.OPEN
    order_id: Optional[str] = None
    submitted_price: Optional[float] = None
    submitted_size: Optional[float] = None
    raw_response_json: str = "{}"
    
    @classmethod
    def from_row(cls, row) -> "Submission":
        """Create Submission from a database row."""
        return cls(
            submission_id=row["submission_id"],
            intent_id=row["intent_id"],
            submitted_at=row["submitted_at"],
            status=row["status"],
            order_id=row["order_id"],
            submitted_price=row["submitted_price"],
            submitted_size=row["submitted_size"],
            raw_response_json=row["raw_response_json"] or "{}",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_insert_tuple(self) -> tuple:
        """Return tuple for INSERT statement."""
        return (
            self.submission_id,
            self.intent_id,
            self.submitted_at,
            self.order_id,
            self.submitted_price,
            self.submitted_size,
            self.status,
            self.raw_response_json,
        )


@dataclass
class AccountState:
    """Snapshot of account balance and collateral."""
    timestamp: str  # ISO format
    collateral_balance: float
    collateral_allowance: float
    reserved_open_buys: float
    available_collateral: float
    raw_json: str = "{}"
    
    @classmethod
    def from_row(cls, row) -> "AccountState":
        """Create AccountState from a database row."""
        return cls(
            timestamp=row["timestamp"],
            collateral_balance=row["collateral_balance"],
            collateral_allowance=row["collateral_allowance"],
            reserved_open_buys=row["reserved_open_buys"],
            available_collateral=row["available_collateral"],
            raw_json=row["raw_json"] or "{}",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_snapshot_json(self) -> str:
        """Return JSON string for approval snapshots."""
        return json.dumps({
            "balance": self.collateral_balance,
            "allowance": self.collateral_allowance,
            "reserved": self.reserved_open_buys,
            "available": self.available_collateral,
            "timestamp": self.timestamp,
        })


def utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()
