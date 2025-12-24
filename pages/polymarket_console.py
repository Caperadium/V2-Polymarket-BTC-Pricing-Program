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
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_reco import recommend_trades, recommendations_to_dataframe, load_latest_fitted_batch
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database on module load
init_db()

# Page config
st.set_page_config(
    page_title="Polymarket Trade Console",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Polymarket Trade Console")

# Initialize provider (fake for MVP)
@st.cache_resource
def get_provider():
    """Get the Polymarket provider (cached)."""
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

if col5.button("🔄 Refresh Balance"):
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
            value=0.06,
            step=0.005,
            help="Minimum edge threshold",
        )
    
    with param_row1_col4:
        max_bets_per_expiry = st.number_input(
            "Max Bets per Expiry",
            min_value=1,
            max_value=10,
            value=8,
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
            step=0.05,
            help="Maximum total fraction of bankroll to deploy",
        )
    
    with param_row2_col3:
        max_net_delta_frac = st.slider(
            "Net Delta Limit (+/-)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="Cap net directional exposure (Long - Short). 0.2 = max 20% net long or short. 1.0 = no cap.",
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
        correlation_penalty = st.slider(
            "Correlation Penalty",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="0.0 = no shrink; 1.0 = strong shrink when many trades share same expiry/direction",
        )
    
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
            min_value=5.0,
            value=10.0,
            step=5.0,
            disabled=not use_fixed_stake,
            help="When enabled, all trades use this fixed dollar amount instead of Kelly sizing.",
        )
    
    # Row 5: DTE and Probability Threshold filters
    st.subheader("Optional Filters")
    param_row5_col1, param_row5_col2, param_row5_col3 = st.columns(3)
    
    with param_row5_col1:
        use_max_dte = st.checkbox("Limit Max Days to Expiry", value=True)
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
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            disabled=not use_prob_threshold,
            help="Trade NO when model probability <= this value.",
        )
    
    with param_row5_col3:
        use_moneyness_filter = st.checkbox("Limit Moneyness", value=False)
        min_moneyness = st.number_input(
            "Min |Moneyness|",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.01,
            format="%.2f",
            disabled=not use_moneyness_filter,
            help="Only trade contracts where |moneyness| >= this value. Use to exclude ATM.",
        )
        max_moneyness = st.number_input(
            "Max |Moneyness|",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.2f",
            disabled=not use_moneyness_filter,
            help="Only trade contracts where |moneyness| <= this value. 0.05 = ±5% from spot.",
        )

# Store params for later use
reco_params = {
    "bankroll": bankroll,
    "kelly_fraction": kelly_fraction,
    "min_edge": min_edge,
    "max_bets_per_expiry": max_bets_per_expiry,
    "max_capital_per_expiry_frac": max_capital_per_expiry_frac,
    "max_capital_total_frac": max_capital_total_frac,
    "max_net_delta_frac": max_net_delta_frac,
    "min_price": min_price,
    "max_price": max_price,
    "min_model_prob": min_model_prob,
    "max_model_prob": max_model_prob,
    "use_stability_penalty": use_stability_penalty,
    "correlation_penalty": correlation_penalty,
    "min_trade_usd": min_trade_usd,
    "use_fixed_stake": use_fixed_stake,
    "fixed_stake_amount": fixed_stake_amount,
    "use_max_dte": use_max_dte,
    "max_dte_value": max_dte_value if use_max_dte else None,
    "use_prob_threshold": use_prob_threshold,
    "prob_threshold_yes": prob_threshold_yes,
    "prob_threshold_no": prob_threshold_no,
    "use_moneyness_filter": use_moneyness_filter,
    "min_moneyness": min_moneyness if use_moneyness_filter else None,
    "max_moneyness": max_moneyness if use_moneyness_filter else None,
}

st.divider()

# -----------------------------------------------------------------------------
# Generate Orders Section
# -----------------------------------------------------------------------------

st.header("Order Generation")

gen_col1, gen_col2 = st.columns([1, 3])

with gen_col1:
    generate_clicked = st.button("🎯 Generate Orders", type="primary", use_container_width=True)

with gen_col2:
    st.caption("Loads latest batch data, runs auto_reco, and creates DRAFT intents")

if generate_clicked:
    with st.spinner("Generating trade recommendations..."):
        try:
            # Load latest batch data
            batch_df = load_latest_fitted_batch()
            
            if batch_df is None or batch_df.empty:
                st.error("No batch data found. Please run the pricing pipeline first.")
            else:
                # Build recommend_trades kwargs
                reco_kwargs = {
                    "df": batch_df,
                    "bankroll": bankroll,
                    "kelly_fraction": kelly_fraction,
                    "min_edge": min_edge,
                    "max_bets_per_expiry": max_bets_per_expiry,
                    "max_capital_per_expiry_frac": max_capital_per_expiry_frac,
                    "max_capital_total_frac": max_capital_total_frac,
                    "max_net_delta_frac": max_net_delta_frac,
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_model_prob": min_model_prob,
                    "max_model_prob": max_model_prob,
                    "use_stability_penalty": use_stability_penalty,
                    "correlation_penalty": correlation_penalty,
                    "min_trade_usd": min_trade_usd,
                    "use_fixed_stake": use_fixed_stake,
                    "fixed_stake_amount": fixed_stake_amount,
                    "use_prob_threshold": use_prob_threshold,
                    "prob_threshold_yes": prob_threshold_yes,
                    "prob_threshold_no": prob_threshold_no,
                }
                
                # Add optional filters
                if use_max_dte:
                    reco_kwargs["max_dte"] = max_dte_value
                if use_moneyness_filter:
                    reco_kwargs["min_moneyness"] = min_moneyness
                    reco_kwargs["max_moneyness"] = max_moneyness
                
                # Run auto_reco
                recommendations = recommend_trades(**reco_kwargs)
                
                if not recommendations:
                    st.warning("No trade recommendations generated with current parameters.")
                else:
                    # Convert to DataFrame
                    reco_df = recommendations_to_dataframe(recommendations)
                    
                    # Create run and build intents
                    run = create_run(strategy="auto_reco", params=reco_params)
                    intents, warnings = build_intents_from_reco(reco_df, run)
                    
                    # Check for duplicates
                    duplicates = check_duplicate_intents(intents)
                    if duplicates:
                        st.warning(f"Duplicate contracts detected: {', '.join(duplicates[:5])}")
                    
                    # Display warnings
                    if warnings:
                        with st.expander(f"⚠️ {len(warnings)} validation warnings", expanded=False):
                            for w in warnings[:10]:
                                st.write(f"• {w['contract']}: {', '.join(w['warnings'])}")
                    
                    # Save intents
                    if intents:
                        saved_count = save_intents(intents)
                        st.success(f"✅ Generated {len(intents)} intents ({saved_count} new)")
                        # Force refresh of draft intents
                        if "draft_intents" in st.session_state:
                            del st.session_state["draft_intents"]
                    else:
                        st.warning("No valid intents could be created from recommendations.")
                        
        except Exception as e:
            st.error(f"Error generating orders: {str(e)}")
            logger.exception("Failed to generate orders")

st.divider()

# -----------------------------------------------------------------------------
# Draft Intents Table
# -----------------------------------------------------------------------------

st.header("Draft Intents")

# Fetch draft intents
draft_intents = get_intents_by_status(IntentStatus.DRAFT)

if not draft_intents:
    st.info("No draft intents. Click 'Generate Orders' to create some.")
else:
    # Convert to DataFrame for display
    intent_data = []
    for intent in draft_intents:
        intent_data.append({
            "intent_id": intent.intent_id[:12] + "...",
            "contract": intent.contract,
            "outcome": intent.outcome,
            "limit_price": f"${intent.limit_price:.4f}",
            "size_shares": f"{intent.size_shares:.2f}",
            "notional": f"${intent.notional_usd:.2f}",
            "edge": f"{intent.edge:.2%}" if intent.edge else "N/A",
            "ev": f"${intent.ev:.2f}" if intent.ev else "N/A",
            "expiry": intent.expiry,
            "status": intent.status,
            "_intent_id_full": intent.intent_id,
        })
    
    display_df = pd.DataFrame(intent_data)
    
    # Show table with selection
    st.write(f"**{len(draft_intents)} draft intents**")
    
    # Use data_editor for selection
    selection_df = display_df.copy()
    selection_df.insert(0, "Select", False)
    
    edited_df = st.data_editor(
        selection_df.drop(columns=["_intent_id_full"]),
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=False),
            "intent_id": st.column_config.TextColumn("Intent ID", width="small"),
            "contract": st.column_config.TextColumn("Contract", width="medium"),
            "outcome": st.column_config.TextColumn("Side", width="small"),
            "limit_price": st.column_config.TextColumn("Price", width="small"),
            "size_shares": st.column_config.TextColumn("Shares", width="small"),
            "notional": st.column_config.TextColumn("Notional", width="small"),
            "edge": st.column_config.TextColumn("Edge", width="small"),
            "ev": st.column_config.TextColumn("EV", width="small"),
            "expiry": st.column_config.TextColumn("Expiry", width="small"),
            "status": st.column_config.TextColumn("Status", width="small"),
        },
        disabled=["intent_id", "contract", "outcome", "limit_price", "size_shares", 
                  "notional", "edge", "ev", "expiry", "status"],
        use_container_width=True,
    )
    
    # Get selected intent IDs
    selected_mask = edited_df["Select"].tolist()
    selected_intent_ids = [
        display_df.iloc[i]["_intent_id_full"] 
        for i, selected in enumerate(selected_mask) 
        if selected
    ]
    
    # Approval button
    approval_col1, approval_col2 = st.columns([1, 3])
    
    with approval_col1:
        approve_clicked = st.button(
            f"✅ Approve Selected ({len(selected_intent_ids)})",
            disabled=len(selected_intent_ids) == 0,
            use_container_width=True,
        )
    
    if approve_clicked and selected_intent_ids:
        # Get current account state for snapshot
        current_state = st.session_state.get("account_state")
        if not current_state:
            current_state = fetch_account_state(provider)
        
        snapshot_json = create_approval_snapshot(current_state)
        approved_at = utc_now_iso()
        
        approved_count = 0
        for intent_id in selected_intent_ids:
            success = update_intent_status(
                intent_id,
                IntentStatus.APPROVED,
                approved_at=approved_at,
                approved_snapshot_json=snapshot_json,
            )
            if success:
                approved_count += 1
        
        st.success(f"✅ Approved {approved_count} intents")
        st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Approved Intents & Submission
# -----------------------------------------------------------------------------

st.header("Approved Orders")

approved_intents = get_intents_by_status(IntentStatus.APPROVED)

if not approved_intents:
    st.info("No approved intents. Select draft intents above and click 'Approve Selected'.")
else:
    # Display approved intents
    approved_data = []
    for intent in approved_intents:
        approved_data.append({
            "contract": intent.contract,
            "outcome": intent.outcome,
            "limit_price": f"${intent.limit_price:.4f}",
            "size_shares": f"{intent.size_shares:.2f}",
            "notional": f"${intent.notional_usd:.2f}",
            "approved_at": intent.approved_at[:19] if intent.approved_at else "N/A",
        })
    
    approved_df = pd.DataFrame(approved_data)
    st.dataframe(approved_df, use_container_width=True, hide_index=True)
    
    # Total notional
    total_notional = sum(i.notional_usd for i in approved_intents)
    st.write(f"**Total notional: ${total_notional:.2f}**")
    
    # Check against available collateral
    if state:
        if total_notional > state.available_collateral:
            st.warning(
                f"⚠️ Total notional (${total_notional:.2f}) exceeds available collateral "
                f"(${state.available_collateral:.2f}). Some orders may fail."
            )
    
    # Submit button
    submit_col1, submit_col2 = st.columns([1, 3])
    
    with submit_col1:
        submit_clicked = st.button(
            "🚀 Place Approved Orders",
            type="primary",
            use_container_width=True,
        )
    
    if submit_clicked:
        with st.spinner("Submitting orders..."):
            # Refresh account state before submission
            current_state = fetch_account_state(provider)
            save_account_state(current_state)
            st.session_state.account_state = current_state
            
            # Submit batch with cumulative validation
            results = submit_approved_batch(approved_intents, current_state, provider)
            
            # Display results
            success_count = sum(1 for r in results if r["success"])
            failed_count = len(results) - success_count
            
            if success_count > 0:
                st.success(f"✅ Successfully submitted {success_count} orders")
            if failed_count > 0:
                st.error(f"❌ Failed to submit {failed_count} orders")
                with st.expander("Failed order details"):
                    for r in results:
                        if not r["success"]:
                            st.write(f"• {r['contract']}: {r['error']}")
            
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
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("No order history yet.")

# Reconciliation summary
with st.expander("Submission Status Summary"):
    summary = get_reconciliation_summary()
    if summary:
        for status, count in summary.items():
            st.write(f"• {status}: {count}")
    else:
        st.write("No submissions yet.")
    
    if st.button("Run Reconciliation"):
        result = reconcile_submissions(provider)
        st.write(f"Reconciled: {result}")

st.divider()

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.caption("Polymarket Trade Console MVP - Using fake provider for testing")
