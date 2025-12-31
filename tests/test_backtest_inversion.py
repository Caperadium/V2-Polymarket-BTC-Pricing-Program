
import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from unittest.mock import MagicMock, patch
from scripts.backtesting.backtest_engine import BacktestEngine

# Mock data
@pytest.fixture
def mock_batch_df():
    return pd.DataFrame({
        "slug": ["btc-up", "btc-down"],
        "expiry_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
        "strike": [100000, 100000],
        "market_price": [0.6, 0.4], # Q (YES Price)
        "model_prob": [0.7, 0.3],   # P
        # ... other required cols if needed by get_column checks
    })

def test_execute_trades_no_side_inversion():
    """
    Verify that when auto_reco recommends a NO trade, the backtest engine
    uses the PROVIDED price as the entry price, and does NOT invert it again.
    
    Scenario:
    - Market YES Price (q) = 0.60
    - Market NO Price (1-q) = 0.40
    - Model Probability (p) = 0.50
    - Edge YES: p - q = 0.5 - 0.6 = -0.1 (No trade)
    - Edge NO: q - p = 0.6 - 0.5 = +0.1 (Trade!)
    
    auto_reco will return:
    - side: "NO"
    - market_price: 0.40 (The entry price for NO)
    
    Current Bug: BacktestEngine takes 0.40, sees "NO", and calculates 1.0 - 0.40 = 0.60.
    Desired Behavior: BacktestEngine takes 0.40, sees "NO", uses 0.40 as entry price.
    """
    
    # Setup Engine
    engine = BacktestEngine(
        market_data_batches=[],
        initial_bankroll=1000.0,
        strategy_params={
            "min_edge": 0.0,
            "max_add_per_cycle_usd": 1000.0,
        },
    )
    
    # Mock auto_reco result
    # We simulate exactly what auto_reco returns for a NO trade
    mock_reco = pd.DataFrame([{
        "slug": "test-slug",
        "side": "NO",
        "market_price": 0.40, # This is 1-q, the price we pay
        "model_prob": 0.50,
        "suggested_stake": 100.0,
        "expiry_key": "2025-01-01",
        "strike": 100000.0,
        "kelly_fraction_applied": 0.1,
        "effective_edge": 0.10, # Mock edge
    }])
    
    # We need to mock recommend_trades to return our pre-cooked frame
    # But BacktestEngine calls recommend_trades then recommendations_to_dataframe
    # Simpler to mock `recommend_trades` to return a list that `recommendations_to_dataframe` would process,
    # OR just patch internal `_execute_trades` logic? No, we want to test `_execute_trades`.
    # `_execute_trades` calls `recommend_trades`.
    
    with patch("scripts.backtesting.backtest_engine.recommend_trades") as mock_reco_func, \
         patch("scripts.backtesting.backtest_engine.recommendations_to_dataframe") as mock_to_df:
             
        mock_to_df.return_value = mock_reco
        
        # Create a dummy batch_df (content doesn't matter much as we mock the reco output)
        batch_df = pd.DataFrame({"dummy": [1]})
        
        # Act
        current_time = pd.Timestamp("2024-12-30", tz="UTC")
        engine._execute_trades(batch_df, current_time)
        
        # Assert
        assert len(engine._open_positions) == 1
        pos = engine._open_positions[0]
        
        # CRITICAL CHECKS
        
        # 1. Entry Price should be 0.40 (what we paid), NOT 0.60
        assert pos.entry_price == 0.40, f"Entry Price incorrect! Expected 0.40, got {pos.entry_price}"
        
        # 2. Market Price (stored for analytics) should be normalized to YES price (0.60)
        # implied_yes = 1.0 - 0.40 = 0.60
        # This ensures edge calculation later (q - p) works: 0.60 - 0.50 = 0.10
        assert pos.market_price == 0.60, f"Stored Market Price (YES-basis) incorrect! Expected 0.60, got {pos.market_price}"
        
        # 3. Check bankroll deduction
        expected_bankroll = 1000.0 - 100.0
        assert engine._running_bankroll == expected_bankroll
        
        # 4. Check edge capture
        assert pos.edge == 0.10, f"Edge incorrect! Expected 0.10, got {pos.edge}"

def test_execute_trades_yes_side_normal():
    """Verify YES trades work as expected (no inversion checks needed, but sanity check)."""
    engine = BacktestEngine(
        market_data_batches=[],
        initial_bankroll=1000.0,
        strategy_params={"min_edge": 0.0},
    )
    
    mock_reco = pd.DataFrame([{
        "slug": "test-yes",
        "side": "YES",
        "market_price": 0.70, # Price we pay
        "model_prob": 0.80,
        "suggested_stake": 100.0,
        "expiry_key": "2025-01-01",
        "strike": 100000.0,
    }])
    
    with patch("scripts.backtesting.backtest_engine.recommend_trades"), \
         patch("scripts.backtesting.backtest_engine.recommendations_to_dataframe") as mock_to_df:
        
        mock_to_df.return_value = mock_reco
        engine._execute_trades(pd.DataFrame(), pd.Timestamp("2024-12-30", tz="UTC"))
        
        assert len(engine._open_positions) == 1
        pos = engine._open_positions[0]
        
        assert pos.entry_price == 0.70
        assert pos.market_price == 0.70

if __name__ == "__main__":
    # If run directly, minimal manual check
    try:
        test_execute_trades_no_side_inversion()
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
