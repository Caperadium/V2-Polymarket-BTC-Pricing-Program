"""
polymarket/db.py

SQLite database connection and schema management for the Polymarket Trade Console.

Design decisions:
- Connection-per-operation pattern for Streamlit concurrency safety
- WAL journal mode for reduced lock contention
- Indexes on frequently queried columns (status, run_id, intent_id)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Database file location - in the project root
DB_DIR = Path(__file__).parent.parent
DB_PATH = DB_DIR / "polymarket_console.db"


def get_connection() -> sqlite3.Connection:
    """
    Create a new SQLite connection per operation.
    
    Connections are cheap; this pattern avoids concurrency issues
    with Streamlit's re-run behavior and shared state.
    
    Returns:
        sqlite3.Connection with WAL mode enabled and Row factory set.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the database schema.
    
    Creates all tables and indexes if they don't exist.
    Safe to call multiple times (idempotent).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Runs table for batch separation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                strategy TEXT,
                params_json TEXT,
                notes TEXT
            )
        """)
        
        # Intents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS intents (
                intent_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                strategy TEXT,
                params_json TEXT,
                contract TEXT NOT NULL,
                expiry TEXT,
                outcome TEXT NOT NULL CHECK (outcome IN ('YES', 'NO')),
                action TEXT NOT NULL CHECK (action IN ('BUY', 'SELL')),
                limit_price REAL NOT NULL CHECK (limit_price > 0 AND limit_price < 1),
                stake_usd REAL NOT NULL CHECK (stake_usd > 0),
                size_shares REAL NOT NULL CHECK (size_shares > 0),
                notional_usd REAL NOT NULL,
                model_prob REAL,
                market_prob REAL,
                edge REAL,
                ev REAL,
                status TEXT NOT NULL DEFAULT 'DRAFT' 
                    CHECK (status IN ('DRAFT', 'APPROVED', 'SUBMITTED', 'FILLED', 'PARTIAL', 'CANCELLED', 'FAILED', 'SKIPPED')),
                approved_at TEXT,
                approved_snapshot_json TEXT,
                submitted_at TEXT,
                notes TEXT,
                raw_reco_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        
        # Submissions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                submission_id TEXT PRIMARY KEY,
                intent_id TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                order_id TEXT,
                submitted_price REAL,
                submitted_size REAL,
                status TEXT NOT NULL DEFAULT 'OPEN'
                    CHECK (status IN ('OPEN', 'FILLED', 'PARTIAL', 'CANCELLED', 'FAILED')),
                raw_response_json TEXT,
                FOREIGN KEY (intent_id) REFERENCES intents(intent_id)
            )
        """)
        
        # Account state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account_state (
                timestamp TEXT PRIMARY KEY,
                collateral_balance REAL,
                collateral_allowance REAL,
                reserved_open_buys REAL,
                available_collateral REAL,
                raw_json TEXT
            )
        """)
        
        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intents_status ON intents(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intents_created_at ON intents(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intents_run_id ON intents(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submissions_intent ON submissions(intent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_submissions_status ON submissions(status)")
        
        conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        conn.close()


def execute_query(query: str, params: tuple = (), fetch: bool = False) -> Optional[list]:
    """
    Execute a query with a fresh connection.
    
    Args:
        query: SQL query string
        params: Query parameters
        fetch: If True, return all results
        
    Returns:
        List of rows if fetch=True, else None
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        if fetch:
            return cursor.fetchall()
        conn.commit()
        return None
    finally:
        conn.close()


def execute_many(query: str, params_list: list) -> None:
    """
    Execute a query with multiple parameter sets.
    
    Args:
        query: SQL query string with placeholders
        params_list: List of parameter tuples
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        conn.commit()
    finally:
        conn.close()
