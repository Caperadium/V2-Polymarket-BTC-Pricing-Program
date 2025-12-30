#!/usr/bin/env python3
"""
test_auto_reco_refactor.py

Unit tests for the refactored auto_reco 3-stage pipeline.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import TestCase, main

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.strategy.auto_reco import (
    DeltaIntent,
    RebalanceConfig,
    TargetPosition,
    TargetRole,
    build_targets,
    compute_current_exposure_usd,
    compute_deltas,
    generate_key,
    recommend_trades,
)
from core.strategy.vol_gate import VolGateResult


class TestKeyGeneration(TestCase):
    """Test stable key generation."""
    
    def test_key_with_condition_id(self):
        """Prefer condition_id when available, with explicit side."""
        row = pd.Series({
            "condition_id": "0x123abc",
            "outcome_index": 0,
            "slug": "btc-above-100k",
            "expiry_key": "2025-01-15",
            "strike": 100000,
            "side": "YES"
        })
        # When called without explicit side, uses row["side"]
        key = generate_key(row)
        self.assertEqual(key, "0x123abc|YES")
        
        # When called with explicit side, uses that
        key_no = generate_key(row, side="NO")
        self.assertEqual(key_no, "0x123abc|NO")
    
    def test_key_fallback_composite(self):
        """Use composite key when no condition_id."""
        row = pd.Series({
            "slug": "btc-above-100k",
            "expiry_key": "2025-01-15",
            "strike": 100000,
            "side": "YES"
        })
        key = generate_key(row)
        self.assertEqual(key, "btc-above-100k|2025-01-15|100000.00|YES")
    
    def test_key_collision_prevention(self):
        """Different markets should have different keys."""
        row1 = pd.Series({
            "slug": "btc-above-100k",
            "expiry_key": "2025-01-15",
            "strike": 100000,
            "side": "YES"
        })
        row2 = pd.Series({
            "slug": "btc-above-100k",
            "expiry_key": "2025-01-15",
            "strike": 105000,  # Different strike
            "side": "YES"
        })
        self.assertNotEqual(generate_key(row1), generate_key(row2))


class TestCurrentExposure(TestCase):
    """Test current exposure calculation."""
    
    def test_empty_positions(self):
        """Empty positions returns empty dict."""
        result = compute_current_exposure_usd(None, 1000)
        self.assertEqual(result, {})
        
        result = compute_current_exposure_usd(pd.DataFrame(), 1000)
        self.assertEqual(result, {})
    
    def test_single_position(self):
        """Single position returns correct cost basis."""
        positions = pd.DataFrame([{
            "slug": "test-market",
            "expiry_key": "2025-01-15",
            "strike": 100000,
            "side": "YES",
            "entry_price": 0.50,
            "size_shares": 100,
        }])
        result = compute_current_exposure_usd(positions, 1000)
        key = "test-market|2025-01-15|100000.00|YES"
        self.assertAlmostEqual(result[key], 50.0)  # 0.50 * 100
    
    def test_clamped_at_zero(self):
        """Cost basis should be clamped at 0."""
        # For now sells aren't tracked, but structure is in place
        positions = pd.DataFrame([{
            "slug": "test-market",
            "expiry_key": "2025-01-15",
            "strike": 100000,
            "side": "YES",
            "entry_price": 0.50,
            "size_shares": 100,
        }])
        result = compute_current_exposure_usd(positions, 1000)
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)


class TestDeltaSignHandling(TestCase):
    """Test delta computation with various current/target combinations."""
    
    def get_normal_vol_gate(self):
        return VolGateResult(
            now_utc=datetime.now(timezone.utc).isoformat(),
            regime="normal",
            vol15=0.001,
            vol60=0.001,
            vol15_pct=50.0,
            shock=False,
            allow_new_entries=True,
            edge_add_cents=0.0,
            kelly_mult=1.0,
            reason="normal"
        )
    
    def get_config(self, bankroll=1000):
        return RebalanceConfig(
            bankroll=bankroll,
            min_edge_entry=0.02,
            min_edge_exit=0.00,
            rebalance_min_add_usd=5.0,
            rebalance_min_reduce_usd=10.0,
        )
    
    def test_current_zero_target_positive_is_buy(self):
        """current=0, target>0 -> BUY"""
        targets = {
            "test-key": TargetPosition(
                key="test-key",
                slug="test",
                side="YES",
                expiry_key="2025-01-15",
                strike=100000,
                condition_id=None,
                target_fraction=0.05,
                target_usd=50.0,
                model_prob=0.60,
                market_price=0.50,
                entry_price=0.50,
                effective_edge=0.10,
                allocation_score=0.10,
                exit_score=0.10,
                role=TargetRole.ENTRY,
            )
        }
        current_exposure = {}  # No current position
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_normal_vol_gate(),
            config=self.get_config(),
        )
        
        buy_intents = [i for i in intents if i.action == "BUY"]
        self.assertEqual(len(buy_intents), 1)
        self.assertAlmostEqual(buy_intents[0].amount_usd, 50.0)
        self.assertEqual(buy_intents[0].price_mode, "TAKER_ASK")
    
    def test_current_positive_target_zero_is_sell(self):
        """current>0, target=0 -> SELL"""
        targets = {
            "test-key": TargetPosition(
                key="test-key",
                slug="test",
                side="YES",
                expiry_key="2025-01-15",
                strike=100000,
                condition_id=None,
                target_fraction=0.0,
                target_usd=0.0,
                model_prob=0.40,
                market_price=0.50,
                entry_price=0.50,
                effective_edge=-0.10,
                allocation_score=0.0,
                exit_score=-0.10,
                role=TargetRole.ENTRY,
            )
        }
        current_exposure = {"test-key": 50.0}
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_normal_vol_gate(),
            config=self.get_config(),
        )
        
        sell_intents = [i for i in intents if i.action == "SELL"]
        self.assertEqual(len(sell_intents), 1)
        self.assertAlmostEqual(sell_intents[0].amount_usd, 50.0)
        self.assertEqual(sell_intents[0].price_mode, "TAKER_BID")
    
    def test_current_approx_target_is_hold(self):
        """current ≈ target -> HOLD"""
        targets = {
            "test-key": TargetPosition(
                key="test-key",
                slug="test",
                side="YES",
                expiry_key="2025-01-15",
                strike=100000,
                condition_id=None,
                target_fraction=0.05,
                target_usd=50.0,
                model_prob=0.55,
                market_price=0.50,
                entry_price=0.50,
                effective_edge=0.05,
                allocation_score=0.05,
                exit_score=0.05,
                role=TargetRole.ENTRY,
            )
        }
        current_exposure = {"test-key": 52.0}  # Close to target
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_normal_vol_gate(),
            config=self.get_config(),
        )
        
        # Delta is only 2 USD, below min thresholds
        hold_intents = [i for i in intents if i.action == "HOLD"]
        self.assertEqual(len(hold_intents), 1)
    
    def test_hold_price_mode_is_none(self):
        """HOLD action should have price_mode=None (not TAKER_BID)"""
        targets = {
            "test-key": TargetPosition(
                key="test-key",
                slug="test",
                side="YES",
                expiry_key="2025-01-15",
                strike=100000,
                condition_id=None,
                target_fraction=0.05,
                target_usd=50.0,
                model_prob=0.55,
                market_price=0.50,
                entry_price=0.50,
                effective_edge=0.05,
                allocation_score=0.05,
                exit_score=0.05,
                role=TargetRole.ENTRY,
            )
        }
        # Current exactly equals target -> HOLD
        current_exposure = {"test-key": 50.0}
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_normal_vol_gate(),
            config=self.get_config(),
        )
        
        hold_intents = [i for i in intents if i.action == "HOLD"]
        self.assertEqual(len(hold_intents), 1)
        # Critical: price_mode must be None for HOLD
        self.assertIsNone(hold_intents[0].price_mode)


class TestVolGateEntryBlock(TestCase):
    """Test vol gate blocks new entries but allows reductions."""
    
    def get_blocked_vol_gate(self, risk_off=True):
        return VolGateResult(
            now_utc=datetime.now(timezone.utc).isoformat(),
            regime="extreme",
            vol15=0.01,
            vol60=0.01,
            vol15_pct=99.0,
            shock=True,
            allow_new_entries=False,
            edge_add_cents=100.0,  # Effectively blocks entries
            kelly_mult=0.0,
            reason="extreme volatility"
        )
    
    def get_config(self, risk_off_targets_to_zero=True):
        return RebalanceConfig(
            bankroll=1000,
            min_edge_entry=0.02,
            min_edge_exit=0.00,
            rebalance_min_add_usd=5.0,
            rebalance_min_reduce_usd=10.0,
            risk_off_targets_to_zero=risk_off_targets_to_zero,
        )
    
    def test_entry_blocked_produces_reductions_when_risk_off(self):
        """allow_new_entries=False + risk_off -> reductions"""
        targets = {
            "test-key": TargetPosition(
                key="test-key",
                slug="test",
                side="YES",
                expiry_key="2025-01-15",
                strike=100000,
                condition_id=None,
                target_fraction=0.05,
                target_usd=50.0,
                model_prob=0.60,
                market_price=0.50,
                entry_price=0.50,
                effective_edge=0.10,
                allocation_score=0.10,
                exit_score=0.10,
                role=TargetRole.ENTRY,
            )
        }
        current_exposure = {"test-key": 50.0}
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_blocked_vol_gate(),
            config=self.get_config(risk_off_targets_to_zero=True),
        )
        
        # Risk-off should set target to 0, causing a sell
        sell_intents = [i for i in intents if i.action == "SELL"]
        self.assertEqual(len(sell_intents), 1)
    
    def test_entry_blocked_prevents_adds_when_risk_off_false(self):
        """allow_new_entries=False + risk_off=False -> no adds, but holds kept"""
        targets = {
            "test-key": TargetPosition(
                key="test-key",
                slug="test",
                side="YES",
                expiry_key="2025-01-15",
                strike=100000,
                condition_id=None,
                target_fraction=0.10,
                target_usd=100.0,  # Want to increase
                model_prob=0.60,
                market_price=0.50,
                entry_price=0.50,
                effective_edge=0.10,
                allocation_score=0.10,
                exit_score=0.10,
                role=TargetRole.ENTRY,
            )
        }
        current_exposure = {"test-key": 50.0}  # Current < target
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_blocked_vol_gate(),
            config=self.get_config(risk_off_targets_to_zero=False),
        )
        
        # Positive delta should be blocked
        buy_intents = [i for i in intents if i.action == "BUY"]
        self.assertEqual(len(buy_intents), 0)
        
        # Should hold
        hold_intents = [i for i in intents if i.action == "HOLD"]
        self.assertEqual(len(hold_intents), 1)


class TestMissingDataSafetyPolicy(TestCase):
    """Test that missing data doesn't cause accidental sells."""
    
    def get_normal_vol_gate(self):
        return VolGateResult(
            now_utc=datetime.now(timezone.utc).isoformat(),
            regime="normal",
            vol15=0.001,
            vol60=0.001,
            vol15_pct=50.0,
            shock=False,
            allow_new_entries=True,
            edge_add_cents=0.0,
            kelly_mult=1.0,
            reason="normal"
        )
    
    def test_missing_target_keep_policy_holds(self):
        """Position missing from batch with KEEP policy -> HOLD"""
        targets = {}  # No target for existing position
        current_exposure = {"missing-key": 50.0}
        
        config = RebalanceConfig(
            bankroll=1000,
            missing_target_policy="KEEP",
            rebalance_min_add_usd=5.0,
            rebalance_min_reduce_usd=10.0,
        )
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_normal_vol_gate(),
            config=config,
        )
        
        # Missing target with KEEP policy means target=current -> HOLD
        hold_intents = [i for i in intents if i.action == "HOLD"]
        self.assertEqual(len(hold_intents), 1)
    
    def test_risk_off_overrides_missing_keep(self):
        """Risk-off overrides KEEP policy -> SELL"""
        targets = {}  # No target
        current_exposure = {"missing-key": 50.0}
        
        vol_gate = VolGateResult(
            now_utc=datetime.now(timezone.utc).isoformat(),
            regime="extreme",
            vol15=0.01,
            vol60=0.01,
            vol15_pct=99.0,
            shock=True,
            allow_new_entries=False,
            edge_add_cents=100.0,
            kelly_mult=0.0,
            reason="extreme"
        )
        
        config = RebalanceConfig(
            bankroll=1000,
            missing_target_policy="KEEP",
            risk_off_targets_to_zero=True,
            rebalance_min_add_usd=5.0,
            rebalance_min_reduce_usd=10.0,
        )
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=vol_gate,
            config=config,
        )
        
        # Risk-off should force a sell even with KEEP policy
        sell_intents = [i for i in intents if i.action == "SELL"]
        self.assertEqual(len(sell_intents), 1)


class TestChurnThresholds(TestCase):
    """Test that small deltas are filtered to prevent churn."""
    
    def get_normal_vol_gate(self):
        return VolGateResult(
            now_utc=datetime.now(timezone.utc).isoformat(),
            regime="normal",
            vol15=0.001,
            vol60=0.001,
            vol15_pct=50.0,
            shock=False,
            allow_new_entries=True,
            edge_add_cents=0.0,
            kelly_mult=1.0,
            reason="normal"
        )
    
    def test_small_buy_delta_becomes_hold(self):
        """Small positive delta below threshold -> HOLD"""
        targets = {
            "test-key": TargetPosition(
                key="test-key",
                slug="test",
                side="YES",
                expiry_key="2025-01-15",
                strike=100000,
                condition_id=None,
                target_fraction=0.053,
                target_usd=53.0,  # Only $3 more than current
                model_prob=0.55,
                market_price=0.50,
                entry_price=0.50,
                effective_edge=0.05,
                allocation_score=0.05,
                exit_score=0.05,
                role=TargetRole.ENTRY,
            )
        }
        current_exposure = {"test-key": 50.0}
        
        config = RebalanceConfig(
            bankroll=1000,
            rebalance_min_add_usd=5.0,  # $3 delta is below threshold
            rebalance_min_reduce_usd=10.0,
        )
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_normal_vol_gate(),
            config=config,
        )
        
        buy_intents = [i for i in intents if i.action == "BUY"]
        self.assertEqual(len(buy_intents), 0)
    
    def test_small_sell_delta_becomes_hold(self):
        """Small negative delta below threshold -> HOLD"""
        targets = {
            "test-key": TargetPosition(
                key="test-key",
                slug="test",
                side="YES",
                expiry_key="2025-01-15",
                strike=100000,
                condition_id=None,
                target_fraction=0.045,
                target_usd=45.0,  # Only $5 less than current
                model_prob=0.50,
                market_price=0.50,
                entry_price=0.50,
                effective_edge=0.00,
                allocation_score=0.00,
                exit_score=0.00,
                role=TargetRole.ENTRY,
            )
        }
        current_exposure = {"test-key": 50.0}
        
        config = RebalanceConfig(
            bankroll=1000,
            rebalance_min_add_usd=5.0,
            rebalance_min_reduce_usd=10.0,  # $5 delta is below threshold
        )
        
        intents = compute_deltas(
            targets=targets,
            current_exposure=current_exposure,
            vol_gate_result=self.get_normal_vol_gate(),
            config=config,
        )
        
        sell_intents = [i for i in intents if i.action == "SELL"]
        self.assertEqual(len(sell_intents), 0)


if __name__ == "__main__":
    main()
