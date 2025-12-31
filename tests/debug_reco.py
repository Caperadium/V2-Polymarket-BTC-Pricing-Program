
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from core.strategy.auto_reco import recommend_trades

def debug_reco():
    # Setup Logger
    logging.basicConfig(level=logging.INFO)
    
    # Create mock dataframe matching T96 scenario
    # Side: NO
    # Entry Price: 0.92
    # Model Prob: 0.0821
    # Market Price (YES): 0.08
    # Edge: -0.0021
    
    df = pd.DataFrame([{
        "slug": "test-slug",
        "market_price": 0.08,  # q
        "p_model_cal": 0.0821,  # p
        "expiry_date": "2025-01-01",
        "strike": 100000.0,
        "side": "NO",
        "t_days": 1.0, 
        "expiry_key": "2025-01-01"
    }])
    
    print("--- Input Data ---")
    print(df)
    
    # Run auto_reco
    # min_edge = 0.06
    bankroll = 1000.0
    
    print("\n--- Running recommend_trades ---")
    intents = recommend_trades(
        df=df,
        bankroll=bankroll,
        min_edge=0.06,
        kelly_fraction=0.15,
        allow_no=True,
        # Ensure other filters don't block it
        min_price=0.01,
        max_price=0.99,
        disable_staleness=True,
    )
    
    print(f"\nIntents Generated: {len(intents)}")
    for i in intents:
        print(f"Action: {i.action}, Side: {i.side}, Amount: {i.amount_usd}, Edge: {i.effective_edge}, Kelly: {i.kelly_fraction_full}")

if __name__ == "__main__":
    debug_reco()
