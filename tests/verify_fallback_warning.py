
import pytest
import pandas as pd
from core.strategy.auto_reco import recommend_trades
from core.strategy.common import RebalanceConfig, TargetRole
from dataclasses import asdict

def test_fallback_price_warning_flag():
    # Setup mock data with ONE missing ask price
    data = pd.DataFrame([
        {
            "market_price": 0.50,
            "model_prob": 0.60,
            "yes_ask_price": 0.52, # Valid ask
            "no_ask_price": 0.52,
            "slug": "contract-1",
            "expiry_key": "2025-01-01",
            "condition_id": "c1",
            "strike": 100000,
            "side": "YES"
        },
        {
            "market_price": 0.40,
            "model_prob": 0.60,
            "yes_ask_price": None, # MISSING ask -> fallback
            "no_ask_price": 0.62,
            "slug": "contract-2",
            "expiry_key": "2025-01-01",
            "condition_id": "c2",
            "strike": 100000,
            "side": "YES"
        }
    ])
    
    # Run reco
    recommendations = recommend_trades(
        df=data,
        bankroll=1000,
        min_edge=0.01,
        return_all=True
    )
    
    # Check results
    # Contract 1: explicit ask, is_fallback_price should be False
    rec1 = next(r for r in recommendations if r.slug == "contract-1" and r.side == "YES")
    assert rec1.is_fallback_price == False, "Explicit ask price should NOT be fallback"
    assert rec1.entry_price == 0.52
    
    # Contract 2: missing ask, is_fallback_price should be True
    rec2 = next(r for r in recommendations if r.slug == "contract-2" and r.side == "YES")
    assert rec2.is_fallback_price == True, "Missing ask price SHOULD be fallback"
    assert rec2.entry_price == 0.40 # Falls back to market_price (q)
    
    print("Verification passed!")

if __name__ == "__main__":
    test_fallback_price_warning_flag()
