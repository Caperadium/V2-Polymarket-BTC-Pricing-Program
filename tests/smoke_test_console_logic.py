
import unittest
import uuid
import sys
import shutil
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polymarket.intent_builder import create_run, build_intents_from_reco, save_intents, get_intents_by_run, update_intent_status, get_intents_by_status
from polymarket.models import IntentStatus, OrderIntent
from core.strategy.auto_reco import DeltaIntent
from polymarket.db import init_db, get_connection

class TestConsoleLogicSmoke(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We assume the main DB exists and key columns are present (verified by previous step)
        # We will use unique run_ids to avoid colliding with existing data.
        pass

    def test_01_sqlite3_import_smoke(self):
        """Confirm create_run works (sqlite3 imported)."""
        print("\n[SMOKE] Testing create_run...")
        run = create_run(strategy="smoke_test", params={"test": True})
        self.assertIsNotNone(run.run_id)
        print(f"Created run: {run.run_id}")

    def test_02_dry_run_and_isolation(self):
        """Test isolation between two runs."""
        print("\n[SMOKE] Testing Isolation...")
        
        # Run A
        run_a_id = str(uuid.uuid4())
        create_run(run_id=run_a_id, strategy="test_A")
        
        delta_a = DeltaIntent(
            key="A1", slug="slugA", side="YES", expiry_key="2025-01-01", strike=50000,
            action="BUY", amount_usd=10, signed_delta_usd=10, current_usd=0, target_usd=10,
            price_mode="TAKER_ASK", limit_price_hint=0.5, model_prob=0.6, effective_edge=0.1, reason="Test",
            market_price=0.5, condition_id="condA"
        )
        intents_a = build_intents_from_reco([delta_a], run_a_id)
        save_intents(intents_a)
        
        # Run B
        run_b_id = str(uuid.uuid4())
        create_run(run_id=run_b_id, strategy="test_B")
        
        delta_b = DeltaIntent(
            key="B1", slug="slugB", side="NO", expiry_key="2025-01-01", strike=60000,
            action="SELL", amount_usd=10, signed_delta_usd=-10, current_usd=10, target_usd=0,
            price_mode="TAKER_BID", limit_price_hint=0.4, model_prob=0.3, effective_edge=0.1, reason="Test",
            market_price=0.4, condition_id="condB"
        )
        intents_b = build_intents_from_reco([delta_b], run_b_id)
        save_intents(intents_b)
        
        # Verify A sees only A
        fetched_a = get_intents_by_run(run_a_id)
        self.assertEqual(len(fetched_a), 1)
        self.assertEqual(fetched_a[0].contract, "slugA_2025-01-01_50000")
        
        # Verify B sees only B
        fetched_b = get_intents_by_run(run_b_id)
        self.assertEqual(len(fetched_b), 1)
        self.assertEqual(fetched_b[0].contract, "slugB_2025-01-01_60000")
        
        print("Isolation confirmed.")
        
        # Save run_b_id for approval test
        self.__class__.approval_run_id = run_b_id
        self.__class__.approval_intent_id = fetched_b[0].intent_id

    def test_03_approval_persistence(self):
        """Test approval persistence."""
        print("\n[SMOKE] Testing Approval Persistence...")
        
        run_id = self.__class__.approval_run_id
        intent_id = self.__class__.approval_intent_id
        
        # Approve
        update_intent_status(intent_id, IntentStatus.APPROVED)
        
        # Re-fetch
        fetched = get_intents_by_run(run_id)
        target = next(i for i in fetched if i.intent_id == intent_id)
        self.assertEqual(target.status, IntentStatus.APPROVED)
        
        print("Approval persisted.")

    def test_04_idempotent_upsert(self):
        """Test that re-generating the exact same intent updates/upserts safely."""
        print("\n[SMOKE] Testing Idempotent Upsert...")
        
        run_id = str(uuid.uuid4())
        create_run(run_id=run_id)
        
        # First save
        # First save
        delta = DeltaIntent(
            key="U1", slug="slugU", side="YES", expiry_key="2025-01-01", strike=100,
            action="BUY", amount_usd=100, signed_delta_usd=100, current_usd=0, target_usd=100,
            price_mode="TAKER_ASK", limit_price_hint=0.5, model_prob=0.6, effective_edge=0.1, reason="Test",
            market_price=0.5, condition_id="condU"
        )
        i1 = build_intents_from_reco([delta], run_id)[0]
        save_intents([i1])
        
        # Verify 1
        f1 = get_intents_by_run(run_id)
        self.assertEqual(len(f1), 1)
        self.assertEqual(f1[0].stake_usd, 100.0)
        
        # Second save (Modified amount, same key logic)
        delta.amount_usd = 200.0
        i2 = build_intents_from_reco([delta], run_id)[0]
        
        # The ID should match if our logic is deterministic based on key
        # Wait - compute_intent_id includes run_id + intent_key. 
        # Intent key is stable. So yes, ID matches.
        self.assertEqual(i1.intent_id, i2.intent_id)
        
        save_intents([i2])
        
        # Verify update (count 1, amount 200)
        f2 = get_intents_by_run(run_id)
        self.assertEqual(len(f2), 1)
        self.assertEqual(f2[0].stake_usd, 200.0)
        
        print("Upsert successful.")

    def test_05_collision_check(self):
        """Test that BUY/SELL on same contract do not collide if keys differ."""
        print("\n[SMOKE] Testing Collision (BUY vs SELL)...")
        run_id = str(uuid.uuid4())
        create_run(run_id=run_id)
        
        # BUY
        d1 = DeltaIntent(
            key="C1", slug="slugC", side="YES", expiry_key="2025-01-01", strike=50000,
            action="BUY", amount_usd=100, signed_delta_usd=100, current_usd=0, target_usd=100,
            price_mode="TAKER_ASK", limit_price_hint=0.5, model_prob=0.6, effective_edge=0.1, reason="Test",
            market_price=0.5, condition_id="condC", intent_key="C1|BUY"
        )
            
        # SELL (Same C1, Same slug)
        d2 = DeltaIntent(
            key="C1", slug="slugC", side="YES", expiry_key="2025-01-01", strike=50000,
            action="SELL", amount_usd=50, signed_delta_usd=-50, current_usd=100, target_usd=50,
            price_mode="TAKER_BID", limit_price_hint=0.4, model_prob=0.3, effective_edge=0.1, reason="Test",
            market_price=0.4, condition_id="condC", intent_key="C1|SELL"
        )
            
        intents = build_intents_from_reco([d1, d2], run_id)
        save_intents(intents)
        
        saved = get_intents_by_run(run_id)
        self.assertEqual(len(saved), 2, "Should have saved 2 distinct intents")
        
        # Verify IDs different
        ids = {i.intent_id for i in saved}
        self.assertEqual(len(ids), 2)
        print("Collision avoidance confirmed.")

if __name__ == "__main__":
    unittest.main()
