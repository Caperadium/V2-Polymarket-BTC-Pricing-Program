
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import uuid
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

# Mocking modules that might rely on env vars or heavy imports
import sys
# sys.modules['polymarket.provider_polymarket'] = MagicMock()

from polymarket.models import OrderIntent, Run, IntentStatus
from polymarket.intent_builder import build_intents_from_reco, save_intents, get_intents_by_run, clear_draft_intents, update_intent_status, create_run
from core.strategy.auto_reco import DeltaIntent

@dataclass
class MockDeltaIntent:
    # Mimics DeltaIntent for testing
    key: str
    slug: str
    side: str
    expiry_key: str
    strike: float
    action: str
    amount_usd: float
    limit_price_hint: float
    model_prob: float
    effective_edge: float
    market_price: float
    expected_value_dollars: float = 0.0
    
    # Required for asdict
    def as_dict(self):
        return asdict(self)

class TestDashboardRefactor(unittest.TestCase):
    
    def setUp(self):
        self.run_id = str(uuid.uuid4())
        self.contract = "Bitcoin-Above-100k-2024-12-31"
        
    def test_workflow_logic(self):
        print("\nTesting Dashboard Workflow Logic...")
        
        # 1. Simulate Pipeline Output (DeltaIntent List)
        print("1. Creating Mock DeltaIntents...")
        deltas = [
            DeltaIntent(
                key="key1", slug="slug1", side="YES", expiry_key="2024-12-31", strike=100000.0, condition_id="c1",
                action="BUY", amount_usd=100.0, signed_delta_usd=100.0, current_usd=0.0, target_usd=100.0,
                price_mode="TAKER_ASK", limit_price_hint=0.50, model_prob=0.60, effective_edge=0.10,
                reason="Test", market_price=0.50, question="Test Question",
                kelly_fraction_full=0.1, kelly_fraction_target=0.1
            ),
            DeltaIntent(
                key="key2", slug="slug2", side="NO", expiry_key="2024-12-31", strike=90000.0, condition_id="c2",
                action="SELL", amount_usd=50.0, signed_delta_usd=-50.0, current_usd=100.0, target_usd=50.0,
                price_mode="TAKER_BID", limit_price_hint=0.40, model_prob=0.30, effective_edge=0.05,
                reason="Test Sell", market_price=0.40, question="Test Question 2"
            ),
            DeltaIntent(
                key="key3", slug="slug3", side="YES", expiry_key="2024-12-31", strike=110000.0, condition_id="c3",
                action="HOLD", amount_usd=0.0, signed_delta_usd=0.0, current_usd=0.0, target_usd=0.0,
                price_mode=None, limit_price_hint=0.10, model_prob=0.10, effective_edge=0.0,
                reason="Hold", market_price=0.10, question="Test Question 3"
            )
        ]
        
        # Filter actionable (Dashboard logic)
        actionable_deltas = [d for d in deltas if d.action in ["BUY", "SELL"]]
        self.assertEqual(len(actionable_deltas), 2)
        
        # 2. Build Intents
        print("2. Building Orders from Deltas...")
        orders = build_intents_from_reco(actionable_deltas, self.run_id)
        
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0].action, "BUY")
        self.assertEqual(orders[0].contract, "slug1_2024-12-31_100000.0")
        self.assertEqual(orders[0].run_id, self.run_id)
        self.assertEqual(orders[0].status, IntentStatus.DRAFT)
        
        # 2b. Create Run Record (Required for FK)
        create_run(run_id=self.run_id, strategy="test_console", params={})

        # 3. Save to DB
        print("3. Saving to DB...")
        # Note: We rely on the real DB here (sqlite in git ignored file usually). 
        # Ideally we use a test DB. Assuming dev environment is fine with test data in main DB 
        # as long as we clean up or use unique run_id.
        
        saved_count = save_intents(orders)
        # Note: save_intents uses INSERT OR IGNORE. 
        # If run_id is unique, it should save.
        print(f"Saved {saved_count} intents.")
        
        # 4. Retrieve by Run ID
        print("4. Retrieving by Run ID...")
        fetched = get_intents_by_run(self.run_id)
        self.assertEqual(len(fetched), 2)
        
        # 5. Verify Isolation (Random run_id shouldn't see these)
        other_run = str(uuid.uuid4())
        fetched_other = get_intents_by_run(other_run)
        self.assertEqual(len(fetched_other), 0)
        
        # 6. Update Status (Simulate Approval)
        print("6. Simulating Approval...")
        intent_to_approve = fetched[0]
        update_intent_status(intent_to_approve.intent_id, IntentStatus.APPROVED)
        
        # Verify status update
        # Re-fetch
        fetched_updated = get_intents_by_run(self.run_id)
        approved = [i for i in fetched_updated if i.status == IntentStatus.APPROVED]
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0].intent_id, intent_to_approve.intent_id)
        
        print("[SUCCESS] Workflow Verification Successful!")

if __name__ == "__main__":
    unittest.main()
