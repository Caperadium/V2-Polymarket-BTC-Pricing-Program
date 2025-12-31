"""
pages/polymarket_console.py

Streamlit page for the Polymarket Trade Console.

Provides an operator workflow for:
1. Generating trade recommendations from auto_reco
2. Reviewing and approving draft intents
3. Submitting approved orders to market
4. Monitoring account state and order status
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
import math
import concurrent.futures
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.strategy.auto_reco import recommend_trades, recommendations_to_dataframe, load_latest_fitted_batch, RebalanceConfig, DeltaIntent
from polymarket.db import init_db, get_connection
from polymarket.models import (
    OrderIntent,
    AccountState,
    IntentStatus,
    SubmissionStatus,
    utc_now_iso,
)
from polymarket.intent_builder import (
    create_run,
    build_intents_from_reco,
    save_intents,
    get_intents_by_status,
    update_intent_status,
    check_duplicate_intents,
    clear_draft_intents,
)
from polymarket.accounting import (
    FakePolymarketProvider,
    fetch_account_state,
    save_account_state,
    get_latest_account_state,
    create_approval_snapshot,
)
from polymarket.execution_gateway import (
    submit_approved_batch,
    get_open_submissions,
)
from polymarket.reconcile import reconcile_submissions, get_reconciliation_summary
from polymarket.ingest import sync_polymarket_ledger, get_last_sync_time, get_closed_positions_count
from polymarket.metrics import get_drawdown_warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database on module load
init_db()

# Page config
st.set_page_config(
    page_title="Polymarket Trade Console",
    page_icon="ðŸ“Š",
    layout="wide",
)

st.title("ðŸ“Š Polymarket Trade Console")

# Initialize provider - use real provider in read_only mode for metrics
from polymarket.provider_polymarket import (
    RealPolymarketProvider,
    check_provider_config,
    snap_to_tick,
    get_executable_price,
)

@st.cache_resource
def get_provider():
    """Get the Polymarket provider (cached)."""
    config = check_provider_config()
    if config["user_address_set"]:
        return RealPolymarketProvider(mode="read_only")
    else:
        # Fall back to fake provider if no address configured
        return FakePolymarketProvider(balance=100.0, allowance=100.0)

provider = get_provider()

# -----------------------------------------------------------------------------
# Account State Panel
# -----------------------------------------------------------------------------

st.header("Account State")

col1, col2, col3, col4, col5 = st.columns(5)

# Get or refresh account state
if "account_state" not in st.session_state:
    st.session_state.account_state = get_latest_account_state()

if col5.button("ðŸ”„ Refresh Balance"):
    with st.spinner("Fetching account state..."):
        state = fetch_account_state(provider)
        save_account_state(state)
        st.session_state.account_state = state
        st.success("Account state refreshed!")

state = st.session_state.account_state

if state:
    col1.metric("Balance", f"${state.collateral_balance:.2f}")
    col2.metric("Allowance", f"${state.collateral_allowance:.2f}")
    col3.metric("Reserved", f"${state.reserved_open_buys:.2f}")
    col4.metric("Available", f"${state.available_collateral:.2f}")
    st.caption(f"Last updated: {state.timestamp}")
else:
    st.info("Click 'Refresh Balance' to fetch account state")

st.divider()

# -----------------------------------------------------------------------------
# Polymarket Sync Section
# -----------------------------------------------------------------------------

st.header("ðŸ“¡ Polymarket Sync")

sync_col1, sync_col2 = st.columns([1, 3])

with sync_col1:
    sync_clicked = st.button("ðŸ”„ Sync Last 30 Days", type="primary", width='stretch')

with sync_col2:
    last_sync = get_last_sync_time()
    if last_sync:
        age_minutes = (datetime.now(timezone.utc) - last_sync).total_seconds() / 60
        sync_status = f"Last sync: {last_sync.strftime('%Y-%m-%d %H:%M UTC')} ({int(age_minutes)} min ago)"
        if age_minutes > 60:
            st.warning(f"âš ï¸ {sync_status} - Data may be stale")
        else:
            st.success(f"âœ“ {sync_status}")
    else:
        st.info("No sync data. Click 'Sync' to fetch from Polymarket API.")

if sync_clicked:
    with st.spinner("Syncing from Polymarket API..."):
        try:
            result = sync_polymarket_ledger(days_back=30)
            if result["success"]:
                cp = result["closed_positions"]
                st.success(
                    f"âœ… Sync complete! "
                    f"Fetched: {cp['closed_positions_fetched']}, "
                    f"Inserted: {cp['closed_positions_inserted']}, "
                    f"Updated: {cp['closed_positions_updated']}"
                )
                # Rerun to show updated sync time immediately
                st.rerun()
            else:
                st.error(f"Sync failed: {result.get('closed_positions', {}).get('errors', [])}")
        except Exception as e:
            st.error(f"Sync error: {e}")

st.divider()

# -----------------------------------------------------------------------------
# Metrics Section
# -----------------------------------------------------------------------------

st.header("ðŸ“Š Metrics")

# Get bankroll from account state or use default
metrics_bankroll = state.collateral_balance if state else 500.0

# Get drawdown metrics
metrics = get_drawdown_warnings(bankroll=metrics_bankroll, days=30)

if metrics.has_data:
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    # Yesterday Realized PnL
    yesterday_delta = f"{metrics.yesterday_pnl:+.2f}" if metrics.yesterday_pnl != 0 else "0.00"
    mcol1.metric(
        "Yesterday PnL",
        f"${metrics.yesterday_pnl:.2f}",
        delta=yesterday_delta,
        delta_color="normal",
    )
    
    # Daily Loss %
    daily_loss_display = f"{metrics.yesterday_loss_pct:.1%}" if metrics.yesterday_pnl < 0 else "0.0%"
    mcol2.metric(
        "Yesterday Loss %",
        daily_loss_display,
        delta="âš ï¸ Limit exceeded" if metrics.daily_loss_warn else None,
        delta_color="inverse" if metrics.daily_loss_warn else "off",
    )
    
    # Rolling 7D Max Drawdown
    mcol3.metric(
        "7D Max Drawdown",
        f"{metrics.rolling_mdd:.1%}",
        delta="âš ï¸ Threshold exceeded" if metrics.rolling_mdd_warn else None,
        delta_color="inverse" if metrics.rolling_mdd_warn else "off",
    )
    
    # Total Realized PnL
    mcol4.metric(
        "Total Realized PnL",
        f"${metrics.total_realized_pnl:.2f}",
    )
    
    # Warnings
    if metrics.daily_loss_warn:
        st.warning(f"âš ï¸ **Daily Loss Limit Alert**: Yesterday's loss ({metrics.yesterday_loss_pct:.1%}) exceeded the 4% limit.")
    
    if metrics.rolling_mdd_warn:
        st.warning(f"âš ï¸ **Rolling Drawdown Alert**: 7-day max drawdown ({metrics.rolling_mdd:.1%}) exceeded the 10% threshold.")
    
    if metrics.data_stale:
        st.warning("âš ï¸ **Data Freshness**: Last sync was over 60 minutes ago. Consider syncing for latest data.")
    
    # Optional: Show equity curve
    with st.expander("ðŸ“ˆ Equity Curve & Daily PnL"):
        if metrics.equity_curve is not None and not metrics.equity_curve.empty:
            st.line_chart(metrics.equity_curve.set_index("date")["equity"])
        else:
            st.info("No equity data available.")
        
        if metrics.daily_pnl_df is not None and not metrics.daily_pnl_df.empty:
            st.bar_chart(metrics.daily_pnl_df.set_index("date")["realized_pnl"])

else:
    st.info(metrics.notes or "No closed positions data. Sync from Polymarket to see metrics.")
    st.caption(f"Closed positions in DB: {get_closed_positions_count()}")

st.divider()

# -----------------------------------------------------------------------------
# Parameters Section
# -----------------------------------------------------------------------------

st.header("Generation Parameters")

with st.expander("Auto-Reco Parameters", expanded=True):
    # Row 1: Core settings
    st.subheader("Core Settings")
    param_row1_col1, param_row1_col2, param_row1_col3, param_row1_col4 = st.columns(4)
    
    with param_row1_col1:
        bankroll = st.number_input(
            "Bankroll ($)",
            min_value=100.0,
            max_value=100000.0,
            value=500.0,
            step=50.0,
            help="Total bankroll for position sizing",
        )
    
    with param_row1_col2:
        kelly_fraction = st.slider(
            "Kelly Fraction",
            min_value=0.05,
            max_value=0.30,
            value=0.15,
            step=0.01,
            help="Fractional Kelly multiplier",
        )
    
    with param_row1_col3:
        min_edge = st.number_input(
            "Min Edge",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.005,
            help="Minimum edge threshold",
        )
    
    with param_row1_col4:
        max_bets_per_expiry = st.number_input(
            "Max Bets per Expiry",
            min_value=1,
            max_value=10,
            value=6,
            step=1,
        )
    
    # Row 2: Capital allocation
    st.subheader("Capital Allocation")
    param_row2_col1, param_row2_col2, param_row2_col3 = st.columns(3)
    
    with param_row2_col1:
        max_capital_per_expiry_frac = st.slider(
            "Max Capital per Expiry (frac)",
            min_value=0.05,
            max_value=1.0,
            value=0.15,
            step=0.05,
            help="Maximum fraction of bankroll per expiry",
        )
    
    with param_row2_col2:
        max_capital_total_frac = st.slider(
            "Max Capital Total (frac)",
            min_value=0.05,
            max_value=1.0,
            value=0.40,
            help="Maximum total fraction of bankroll to deploy",
        )
    
    # Row 3: Price filters
    st.subheader("Price & Probability Filters")
    param_row3_col1, param_row3_col2, param_row3_col3, param_row3_col4 = st.columns(4)
    
    with param_row3_col1:
        min_price = st.number_input(
            "Min Price",
            min_value=0.01,
            max_value=0.99,
            value=0.03,
            step=0.01,
        )
    
    with param_row3_col2:
        max_price = st.number_input(
            "Max Price",
            min_value=0.01,
            max_value=0.99,
            value=0.95,
            step=0.01,
        )
    
    with param_row3_col3:
        min_model_prob = st.number_input(
            "Min Model Prob",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )
    
    with param_row3_col4:
        max_model_prob = st.number_input(
            "Max Model Prob",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
        )
    
    # Row 4: Advanced settings
    st.subheader("Advanced Settings")
    param_row4_col1, param_row4_col2, param_row4_col3 = st.columns(3)
    
    with param_row4_col1:
        use_stability_penalty = st.checkbox("Use Stability Penalty", value=False)
    
    with param_row4_col2:
        min_trade_pct = st.slider(
            "Min Stake (% of bankroll)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )
        min_trade_usd = (min_trade_pct / 100.0) * bankroll
        st.caption(f"= ${min_trade_usd:.2f}")
    
    with param_row4_col3:
        use_fixed_stake = st.checkbox("Use Fixed Stake (not Kelly)", value=True)
        fixed_stake_amount = st.number_input(
            "Fixed Stake Amount ($)",
            min_value=1.0,
            value=10.0,
            step=1.0,
            disabled=not use_fixed_stake,
            help="When enabled, all trades use this fixed dollar amount instead of Kelly sizing.",
        )
    
    # Row 5: DTE and Probability Threshold filters
    st.subheader("Optional Filters")
    param_row5_col1, param_row5_col2, param_row5_col3, param_row5_col4 = st.columns(4)
    
    with param_row5_col1:
        use_max_dte = st.checkbox("Limit Max Days to Expiry", value=False)
        max_dte_value = st.number_input(
            "Max DTE (days)",
            min_value=1.0,
            value=2.0,
            step=1.0,
            disabled=not use_max_dte,
            help="Only recommend trades on contracts expiring within this many days.",
        )
    
    with param_row5_col2:
        use_prob_threshold = st.checkbox("Use Probability Thresholds", value=True)
        prob_threshold_yes = st.number_input(
            "Trade YES Above or Equal To",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            disabled=not use_prob_threshold,
            help="Trade YES when model probability >= this value.",
        )
        prob_threshold_no = st.number_input(
            "Trade NO Below or Equal To",
# Strategy Settings (Sidebar)
# -----------------------------------------------------------------------------

st.sidebar.header("Strategy Settings")

# Core Settings
st.sidebar.subheader("Position Sizing")
bankroll = st.sidebar.number_input("Bankroll ($)", 100.0, 100000.0, 500.0, 50.0)
kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.05, 0.50, 0.15, 0.01)
min_edge = st.sidebar.number_input("Min Edge", 0.0, 0.5, 0.0, 0.005)
max_bets_per_expiry = st.sidebar.number_input("Max Bets/Expiry", 1, 10, 6)

# Capital Limits (New)
st.sidebar.subheader("Capital Limits")
max_capital_per_expiry_frac = st.sidebar.slider("Max Capital/Expiry (frac)", 0.05, 1.0, 0.15, 0.05)
max_capital_total_frac = st.sidebar.slider("Max Capital Total (frac)", 0.05, 1.0, 0.40, 0.05)

max_add_per_cycle_usd = st.sidebar.number_input("Max Add per Cycle ($)", 0.0, 10000.0, 200.0, step=50.0)
max_reduce_per_cycle_usd = st.sidebar.number_input("Max Reduce per Cycle ($)", 0.0, 10000.0, 200.0, step=50.0)

# Price Filters
st.sidebar.subheader("Filters")
min_price = st.sidebar.number_input("Min Price", 0.01, 0.99, 0.03, 0.01)
max_price = st.sidebar.number_input("Max Price", 0.01, 0.99, 0.95, 0.01)
min_model_prob = st.sidebar.number_input("Min Model Prob", 0.0, 1.0, 0.0, 0.01)
max_model_prob = st.sidebar.number_input("Max Model Prob", 0.0, 1.0, 1.0, 0.01)

# Advanced Toggles (New)
st.sidebar.subheader("Risk Controls")
use_prob_threshold = st.sidebar.checkbox("Use Prob Thresholds", value=False)
prob_threshold_yes = st.sidebar.number_input("Trade YES >=", 0.0, 1.0, 0.8, disabled=not use_prob_threshold)
prob_threshold_no = st.sidebar.number_input("Trade NO <=", 0.0, 1.0, 0.3, disabled=not use_prob_threshold)

disable_staleness = st.sidebar.checkbox("Disable Staleness Checks", value=False)
cap_breach_delever = st.sidebar.checkbox("De-lever on Cap Breach", value=False)
risk_off_targets_to_zero = st.sidebar.checkbox("Risk Off (Targets -> 0)", value=False)
if risk_off_targets_to_zero:
    st.sidebar.warning("âš ï¸ RISK OFF: All targets forced to zero. Exits only.")

use_stability_penalty = st.sidebar.checkbox("Use Stability Penalty", value=False)
use_spread_gate = st.sidebar.checkbox("Spread-Aware Gate", value=True)
spread_gate_buffer = st.sidebar.number_input("Gate Buffer", -0.5, 0.5, 0.01, disabled=not use_spread_gate)

# Legacy/Misc
with st.sidebar.expander("Legacy / Misc"):
    min_trade_pct = st.slider("Min Stake (% Bankroll)", 1, 10, 1)
    min_trade_usd = (min_trade_pct / 100.0) * bankroll
    st.caption(f"Min Trade: ${min_trade_usd:.2f}")
    
    use_fixed_stake = st.checkbox("Use Fixed Stake", value=True)
    fixed_stake_amount = st.number_input("Fixed Stake ($)", 1.0, 1000.0, 10.0, disabled=not use_fixed_stake)
    
    use_max_dte = st.checkbox("Limit Max DTE", value=False)
    max_dte_value = st.number_input("Max DTE", 1.0, 30.0, 2.0, disabled=not use_max_dte)
    
    use_moneyness_filter = st.checkbox("Limit Moneyness", value=False)
    min_moneyness = st.number_input("Min Moneyness", 0.0, 0.5, 0.0, disabled=not use_moneyness_filter)
    max_moneyness = st.number_input("Max Moneyness", 0.0, 0.5, 0.05, disabled=not use_moneyness_filter)


# Construct Configuration Object
# Note: This is re-created on every rerun based on sidebar state
reco_config = RebalanceConfig(
    # Core
    bankroll=bankroll,
    kelly_fraction=kelly_fraction,
    min_edge=min_edge,
    
    # Capital Limits
    max_capital_total_frac=max_capital_total_frac,
    max_capital_per_expiry_frac=max_capital_per_expiry_frac,
    max_add_per_cycle_usd=max_add_per_cycle_usd,
    max_reduce_per_cycle_usd=max_reduce_per_cycle_usd,
    
    # Filters
    min_price=min_price,
    max_price=max_price,
    min_model_prob=min_model_prob,
    max_model_prob=max_model_prob,
    
    # Features
    use_stability_penalty=use_stability_penalty,
    disable_staleness=disable_staleness,
    cap_breach_delever=cap_breach_delever,
    risk_off_targets_to_zero=risk_off_targets_to_zero,
    
    # Advanced / Legacy
    min_trade_usd=min_trade_usd,
    use_fixed_stake=use_fixed_stake,
    fixed_stake_amount=fixed_stake_amount,
    
    # Optional logic handled via filtering params dict if not in Config object yet
    # We pass these as extra_params to recommend_trades if needed
)

# Extra params for recommend_trades that might not be in RebalanceConfig dataclass yet
# (Assuming RebalanceConfig was updated or these passed separately)
reco_extra_params = {
    "max_bets_per_expiry": max_bets_per_expiry,
    "use_max_dte": use_max_dte,
    "max_dte_value": max_dte_value if use_max_dte else None,
    "use_prob_threshold": use_prob_threshold,
    "prob_threshold_yes": prob_threshold_yes,
    "prob_threshold_no": prob_threshold_no,
    "use_moneyness_filter": use_moneyness_filter,
    "min_moneyness": min_moneyness if use_moneyness_filter else None,
    "max_moneyness": max_moneyness if use_moneyness_filter else None,
    "use_spread_gate": use_spread_gate,
    "spread_gate_buffer": spread_gate_buffer,
}


st.divider()

# -----------------------------------------------------------------------------
# Execution Workflow (State Machine)
# -----------------------------------------------------------------------------

# Imports for execution
from dataclasses import asdict
from polymarket.date_utils import compute_expiry_dates, group_dates_by_month, format_date_range_summary
from scripts.pipelines.run_full_pipeline import run_pipeline_programmatic, verify_pipeline_output

st.header("ðŸš€ Execution Control")

# 1. Initialize State
if "program_phase" not in st.session_state:
    st.session_state.program_phase = "IDLE"
if "program_run_id" not in st.session_state:
    st.session_state.program_run_id = None
if "program_error" not in st.session_state:
    st.session_state.program_error = None
if "delta_df" not in st.session_state:
    st.session_state.delta_df = None
if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = None

# 2. Control Buttons
col_run, col_reset, col_status = st.columns([1, 1, 3])

with col_run:
    # Disable if not IDLE
    disable_run = st.session_state.program_phase != "IDLE"
    if st.button("â–¶ï¸  Run Program", type="primary", disabled=disable_run, width="stretch"):
        # State Clear
        st.session_state.program_phase = "PIPELINE"
        st.session_state.program_error = None
        st.session_state.delta_df = None
        st.session_state.program_run_id = str(uuid.uuid4())
        # Note: We do NOT clear pipeline_output to allow introspection, but it will be overwritten on success
        st.rerun()

with col_reset:
    if st.button("ðŸ”„ Reset State", type="secondary", width="stretch"):
        st.session_state.program_phase = "IDLE"
        st.session_state.program_error = None
        st.rerun()

with col_status:
    # Phase Display
    phase_map = {
        "IDLE": "âšª IDLE",
        "PIPELINE": "â ³ RUNNING PIPELINE...",
        "RECO": "ðŸ§  GENERATING RECOS...",
        "DONE": "âœ… DONE (Ready to Review)",
        "ERROR": "â Œ ERROR"
    }
    current_phase = st.session_state.program_phase
    st.markdown(f"**Status:** {phase_map.get(current_phase, current_phase)}")
    if st.session_state.program_run_id:
        st.caption(f"Run ID: `{st.session_state.program_run_id}`")

st.divider()

# 3. State Machine Handlers

# --- PHASE: PIPELINE ---
if st.session_state.program_phase == "PIPELINE":
    try:
        with st.spinner("Step 1/2: Validating inputs & Pricing contracts..."):
            expiry_dates = compute_expiry_dates()
            result = run_pipeline_programmatic(expiry_dates, num_sims=10000)
            
            if result.get("ok"):
                st.session_state.pipeline_output = result
                st.session_state.program_phase = "RECO"
                st.rerun()
            else:
                raise Exception(f"Pipeline Failed: {result.get('error')}")
                
    except Exception as e:
        import traceback
        st.session_state.program_error = f"Pipeline Error: {str(e)}\n{traceback.format_exc()}"
        st.session_state.program_phase = "ERROR"
        st.rerun()

# --- PHASE: RECO ---
elif st.session_state.program_phase == "RECO":
    try:
        with st.spinner("Step 2/2: Generating Recommendations..."):
            # 1. Load Data
            if not st.session_state.pipeline_output:
                 raise Exception("No pipeline output found.")
            
            output_dir = st.session_state.pipeline_output.get("output_dir")
            batch_path = Path(output_dir) / "batch_with_fits.csv"
            if not batch_path.exists():
                raise Exception(f"Batch file not found at {batch_path}")
            
            batch_df = pd.read_csv(batch_path)
            
            # 2. Generate Deltas (All, including HOLD)
            deltas: List[DeltaIntent] = recommend_trades(
                df=batch_df,
                **asdict(reco_config),
                **reco_extra_params,
                return_all=True 
            )
            
            # 3. Process Deltas & Stable Keys
            enhanced_deltas = []
            valid_actionable_deltas = []
            
            for d in deltas:
                # Generate Stable Key: {run_id}|{key}|{action}|{price_mode}|{amount:.2f}
                pm = d.price_mode or "NONE"
                amt_str = f"{d.amount_usd:.2f}"
                stable_key = f"{st.session_state.program_run_id}|{d.key}|{d.action}|{pm}|{amt_str}"
                
                # Assign to dict for DF
                d_dict = asdict(d)
                d_dict["intent_key"] = stable_key
                d_dict["run_id"] = st.session_state.program_run_id
                enhanced_deltas.append(d_dict)
                
                # 4. Filter for Execution (BUY/SELL only)
                if d.action in ["BUY", "SELL"]:
                    valid_actionable_deltas.append(d)

            # 5. Convert to DF for Display
            st.session_state.delta_df = pd.DataFrame(enhanced_deltas)
            
            # 6. Build & Save Executable Intents
            if valid_actionable_deltas:
                # build_intents_from_reco returns List[OrderIntent]
                exec_intents = build_intents_from_reco(valid_actionable_deltas) # Note: Adapter sets TAKER mode
                
                # Upsert / Save
                # We save group_id = run_id to filter later
                for ei in exec_intents:
                    ei.group_id = st.session_state.program_run_id
                
                save_intents(exec_intents)

            st.session_state.program_phase = "DONE"
            st.rerun()

    except Exception as e:
        import traceback
        st.session_state.program_error = f"Reco Error: {str(e)}\n{traceback.format_exc()}"
        st.session_state.program_phase = "ERROR"
        st.rerun()

# --- PHASE: DONE ---
elif st.session_state.program_phase == "DONE":
    # 1. Display Informational Deltas
    st.subheader("ðŸ“Š Recommendation Results")
    
    if st.session_state.delta_df is not None:
        ddf = st.session_state.delta_df.copy()
        
        # Display Tabs
        tab_buy, tab_sell, tab_hold = st.tabs(["BUY Actions", "SELL Actions", "HOLD / Skips"])
        
        # Columns to hide from display
        hide_cols = [
            "key", "condition_id", "rn_prob", "pricing_date", "stability_penalty", 
            "stale_mult", "question", "price_mode", "batch_age_hours", 
            "expiry_group_risk", "expiry_shape_label", "direction", 
            "display_key", "run_id", "intent_key",
            "kelly_fraction_full", "kelly_fraction_effective"
        ]
        
        with tab_buy:
            buys = ddf[ddf["action"] == "BUY"].drop(columns=[c for c in hide_cols if c in ddf.columns], errors="ignore")
            st.dataframe(buys, width="stretch")
            st.caption(f"Count: {len(buys)}")
            
        with tab_sell:
            sells = ddf[ddf["action"] == "SELL"].drop(columns=[c for c in hide_cols if c in ddf.columns], errors="ignore")
            st.dataframe(sells, width="stretch")
            st.caption(f"Count: {len(sells)}")
            
        with tab_hold:
            # Hide additional columns for HOLD tab
            hide_cols_hold = hide_cols + ["amount_usd", "signed_delta_usd", "target_usd"]
            holds = ddf[ddf["action"] == "HOLD"].drop(columns=[c for c in hide_cols_hold if c in ddf.columns], errors="ignore")
            st.dataframe(holds, width="stretch")
            st.caption(f"Count: {len(holds)}")
    
    st.divider()
    
    # 2. Draft Orders Management (DB Backed)
    st.subheader("ðŸ“  Draft Orders (Execution)")
    
    all_drafts = get_intents_by_status(IntentStatus.DRAFT)
    # Filter by run_id (stored in group_id)
    run_drafts = [i for i in all_drafts if i.group_id == st.session_state.program_run_id]
    
    if not run_drafts:
        st.info("No executable orders generated for this run.")
    else:
        st.write(f"Found {len(run_drafts)} draft orders.")
        
        # Interactive Table for Approval
        draft_data = []
        for i in run_drafts:
            draft_data.append({
                "Select": False, 
                "ID": i.intent_id,
                "Action": i.action,
                "Contract": i.contract,
                "Price": i.limit_price,
                "Shares": i.size_shares,
                "Value ($)": i.notional_usd,
                "Prob": i.model_prob,
            })
        
        draft_df = pd.DataFrame(draft_data)
        draft_df["Select"] = True # Default select all
        
        edited_df = st.data_editor(
            draft_df,
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=["ID", "Action", "Contract", "Price", "Shares", "Value ($)", "Prob"],
            key=f"editor_{st.session_state.program_run_id}"
        )
        
        # Process actions
        selected_ids = edited_df[edited_df["Select"]]["ID"].tolist()
        
        if st.button(f"Mark Selected as APPROVED ({len(selected_ids)})"):
            for iid in selected_ids:
                update_intent_status(iid, IntentStatus.APPROVED)
            st.success("Updated status to APPROVED")
            st.rerun()

    # Submit Section (Loads APPROVED)
    approved_intents = get_intents_by_status(IntentStatus.APPROVED)
    run_approved = [i for i in approved_intents if i.group_id == st.session_state.program_run_id]
    
    if run_approved:
        st.divider()
        st.subheader(f"ðŸš€ Ready to Submit ({len(run_approved)})")
        
        st.dataframe(pd.DataFrame([asdict(i) for i in run_approved]), width="stretch")
        
        if st.button("ðŸš€ SUBMIT TO MARKET", type="primary"):
             with st.spinner("Resolving Live Prices & Submitting..."):
                 from polymarket.provider_polymarket import RealPolymarketProvider
                 import os
                 pkey = os.getenv("POLYMARKET_PRIVATE_KEY")
                 if not pkey:
                     st.error("No Private Key found!")
                 else:
                    trade_provider = RealPolymarketProvider(mode="trading")
                    
                    # --- Price Resolution Logic ---
                    unique_keys = set((i.contract, i.outcome) for i in run_approved)
                    c_map = {}
                    for c, o in unique_keys:
                         # Resolving token might be slow, consider caching or parallel if needed
                         # But for submission, we do it live
                        c_map[(c,o)] = trade_provider.fetch_clob_token_id(c, o)
                        
                    tids = list(val for val in c_map.values() if val)
                    live_books = trade_provider.fetch_live_prices_with_depth(tids, order_dollars=10)
                    
                    final_batch = []
                    blocked = []
                    
                    for intent in run_approved:
                        tid = c_map.get((intent.contract, intent.outcome))
                        book = live_books.get(tid)
                        
                        if not book:
                            blocked.append(intent.contract)
                            continue
                        
                        # TAKER Logic
                        if intent.action == "BUY":
                            ask = float(book.get("ask") or 0)
                            # Slippage check: price must be reasonable relative to provisional limit
                            # If provisional was 0.50 and ask is 0.52, maybe OK. 
                            # If provisional was 0.50 and ask is 0.80, maybe NO?
                            # For now, we trust the live price if it meets min_price checks?
                            # Rebalance config had min/max price.
                            # But we want to ensure we don't buy way worse than we thought.
                            # Let's verify ask < intent.limit_price + 0.05 (5 cents slippage?)
                            # intent.limit_price from RECO was likely a batch update.
                            # Let's update limit to ask.
                            if ask > 0 and ask < (intent.limit_price or 0.99) + 0.05:
                                intent.limit_price = ask
                                final_batch.append(intent)
                            else:
                                blocked.append(f"{intent.contract} (Ask {ask} > Limit {intent.limit_price} + 5c)")
                        elif intent.action == "SELL":
                            bid = float(book.get("bid") or 0)
                            if bid > 0 and bid > (intent.limit_price or 0.01) - 0.05:
                                intent.limit_price = bid
                                final_batch.append(intent)
                            else:
                                blocked.append(f"{intent.contract} (Bid {bid} < Limit {intent.limit_price} - 5c)")

                    if blocked:
                         st.warning(f"Skipped {len(blocked)} orders due to price/liquidity: {blocked}")
                    
                    if final_batch:
                         # Need current account state for collateral check
                         acct = fetch_account_state(provider) 
                         results = submit_approved_batch(final_batch, acct, trade_provider)
                         success_count = sum(1 for r in results if r["success"])
                         st.success(f"Submitted {success_count}/{len(final_batch)} orders!")
                         st.rerun()

# --- PHASE: ERROR ---
elif st.session_state.program_phase == "ERROR":
    st.error("âŒ An error occurred")
    st.error(st.session_state.program_error)
    
    if st.session_state.pipeline_output:
        if st.button("ðŸ”„ Retry Recommendation (Reuse Pipeline Data)"):
            st.session_state.program_phase = "RECO"
            st.session_state.program_error = None
            st.rerun()



st.divider()

# -----------------------------------------------------------------------------
# Order History
# -----------------------------------------------------------------------------

st.header("Order History")

# Status filter
history_status = st.selectbox(
    "Filter by status",
    options=["All", "SUBMITTED", "FILLED", "PARTIAL", "FAILED", "CANCELLED"],
    index=0,
)

# Fetch intents
conn = get_connection()
try:
    cursor = conn.cursor()
    if history_status == "All":
        cursor.execute(
            """
            SELECT * FROM intents 
            WHERE status NOT IN ('DRAFT', 'APPROVED')
            ORDER BY created_at DESC
            LIMIT 100
            """
        )
    else:
        cursor.execute(
            """
            SELECT * FROM intents 
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT 100
            """,
            (history_status,),
        )
    rows = cursor.fetchall()
finally:
    conn.close()

if rows:
    history_data = []
    for row in rows:
        intent = OrderIntent.from_row(row)
        history_data.append({
            "contract": intent.contract,
            "outcome": intent.outcome,
            "price": f"${intent.limit_price:.4f}",
            "shares": f"{intent.size_shares:.2f}",
            "notional": f"${intent.notional_usd:.2f}",
            "status": intent.status,
            "submitted_at": intent.submitted_at[:19] if intent.submitted_at else "N/A",
            "notes": intent.notes[:50] + "..." if intent.notes and len(intent.notes) > 50 else (intent.notes or ""),
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, width='stretch', hide_index=True)
else:
    st.info("No order history yet.")

# Reconciliation summary
with st.expander("Submission Status Summary"):
    summary = get_reconciliation_summary()
    if summary:
        for status, count in summary.items():
            st.write(f"â€¢ {status}: {count}")
    else:
        st.write("No submissions yet.")
    
    if st.button("Run Reconciliation"):
        with st.spinner("Querying Polymarket CLOB API for order statuses..."):
            result = reconcile_submissions(provider)
        st.success(
            f"âœ… Reconciled {result.get('open', 0)} orders: "
            f"**{result.get('filled', 0)} filled**, "
            f"{result.get('cancelled', 0)} cancelled, "
            f"{result.get('still_open', 0)} still open"
        )
        if result.get('errors', 0) > 0:
            st.warning(f"âš ï¸ {result.get('errors')} errors encountered")
        st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.caption("Polymarket Trade Console MVP - Using fake provider for testing")
