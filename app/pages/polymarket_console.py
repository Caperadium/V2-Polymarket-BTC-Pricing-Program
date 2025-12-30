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
import concurrent.futures
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.strategy.auto_reco import recommend_trades, recommendations_to_dataframe, load_latest_fitted_batch
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
    page_icon="📊",
    layout="wide",
)

st.title("📊 Polymarket Trade Console")

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
# Polymarket Sync Section
# -----------------------------------------------------------------------------

st.header("📡 Polymarket Sync")

sync_col1, sync_col2 = st.columns([1, 3])

with sync_col1:
    sync_clicked = st.button("🔄 Sync Last 30 Days", type="primary", width='stretch')

with sync_col2:
    last_sync = get_last_sync_time()
    if last_sync:
        age_minutes = (datetime.now(timezone.utc) - last_sync).total_seconds() / 60
        sync_status = f"Last sync: {last_sync.strftime('%Y-%m-%d %H:%M UTC')} ({int(age_minutes)} min ago)"
        if age_minutes > 60:
            st.warning(f"⚠️ {sync_status} - Data may be stale")
        else:
            st.success(f"✓ {sync_status}")
    else:
        st.info("No sync data. Click 'Sync' to fetch from Polymarket API.")

if sync_clicked:
    with st.spinner("Syncing from Polymarket API..."):
        try:
            result = sync_polymarket_ledger(days_back=30)
            if result["success"]:
                cp = result["closed_positions"]
                st.success(
                    f"✅ Sync complete! "
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

st.header("📊 Metrics")

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
        delta="⚠️ Limit exceeded" if metrics.daily_loss_warn else None,
        delta_color="inverse" if metrics.daily_loss_warn else "off",
    )
    
    # Rolling 7D Max Drawdown
    mcol3.metric(
        "7D Max Drawdown",
        f"{metrics.rolling_mdd:.1%}",
        delta="⚠️ Threshold exceeded" if metrics.rolling_mdd_warn else None,
        delta_color="inverse" if metrics.rolling_mdd_warn else "off",
    )
    
    # Total Realized PnL
    mcol4.metric(
        "Total Realized PnL",
        f"${metrics.total_realized_pnl:.2f}",
    )
    
    # Warnings
    if metrics.daily_loss_warn:
        st.warning(f"⚠️ **Daily Loss Limit Alert**: Yesterday's loss ({metrics.yesterday_loss_pct:.1%}) exceeded the 4% limit.")
    
    if metrics.rolling_mdd_warn:
        st.warning(f"⚠️ **Rolling Drawdown Alert**: 7-day max drawdown ({metrics.rolling_mdd:.1%}) exceeded the 10% threshold.")
    
    if metrics.data_stale:
        st.warning("⚠️ **Data Freshness**: Last sync was over 60 minutes ago. Consider syncing for latest data.")
    
    # Optional: Show equity curve
    with st.expander("📈 Equity Curve & Daily PnL"):
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
            min_value=0.0,
            max_value=1.0,
            value=0.3,
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

    with param_row5_col4:
        use_spread_gate = st.checkbox("Spread-Aware Edge Gate", value=True)
        spread_gate_buffer = st.number_input(
            "Gate Buffer",
            min_value=-0.5,
            max_value=0.5,
            value=0.01,
            step=0.005,
            format="%.3f",
            disabled=not use_spread_gate,
            help="Require edge >= (spread/2) + buffer. Filters out trades where edge doesn't justify crossing the spread."
        )

# Store params for later use
reco_params = {
    "bankroll": bankroll,
    "kelly_fraction": kelly_fraction,
    "min_edge": min_edge,
    "max_bets_per_expiry": max_bets_per_expiry,
    "max_capital_per_expiry_frac": max_capital_per_expiry_frac,
    "max_capital_total_frac": max_capital_total_frac,
    "min_price": min_price,
    "max_price": max_price,
    "min_model_prob": min_model_prob,
    "max_model_prob": max_model_prob,
    "use_stability_penalty": use_stability_penalty,
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
# Run Engine Section
# -----------------------------------------------------------------------------

st.header("🚀 Run Pricing Engine")

# Import date utilities
from polymarket.date_utils import compute_expiry_dates, group_dates_by_month, format_date_range_summary
from scripts.pipelines.run_full_pipeline import run_pipeline_programmatic, verify_pipeline_output

# Initialize session state for engine
if "engine_running" not in st.session_state:
    st.session_state.engine_running = False
if "engine_log" not in st.session_state:
    st.session_state.engine_log = []
if "engine_result" not in st.session_state:
    st.session_state.engine_result = None

# Compute dates for display
expiry_dates = compute_expiry_dates()
date_groups = group_dates_by_month(expiry_dates)
date_summary = format_date_range_summary(date_groups)

engine_col1, engine_col2 = st.columns([1, 3])

with engine_col1:
    run_engine_clicked = st.button(
        "🔄 Run Full Pipeline",
        type="primary",
        width='stretch',
        disabled=st.session_state.engine_running,
    )

with engine_col2:
    st.caption(f"Will price contracts for: **{date_summary}** ({len(expiry_dates)} days)")
    if st.session_state.engine_running:
        st.warning("⏳ Pipeline is running...")

# Handle button click
if run_engine_clicked and not st.session_state.engine_running:
    st.session_state.engine_running = True
    st.session_state.engine_log = []
    st.session_state.engine_result = None
    st.rerun()

# Execute pipeline if running (runs after rerun)
if st.session_state.engine_running:
    try:
        with st.spinner("Running pricing pipeline (this may take a few minutes)..."):
            import traceback
            expiry_dates = compute_expiry_dates()
            result = run_pipeline_programmatic(expiry_dates, num_sims=10000)
            
            st.session_state.engine_result = result
            st.session_state.engine_log = result.get("logs", [])
            
            # Verify output
            if result["ok"] and result["output_dir"]:
                verification = verify_pipeline_output(expiry_dates, result["output_dir"])
                if verification["missing"]:
                    st.session_state.engine_log.append(
                        f"⚠️ Missing dates: {', '.join(str(d) for d in verification['missing'])}"
                    )
                else:
                    st.session_state.engine_log.append(
                        f"✅ Verified: All {len(verification['verified'])} dates processed"
                    )
                    
    except Exception as e:
        st.session_state.engine_log.append(f"❌ Error: {traceback.format_exc()}")
        st.session_state.engine_result = {"ok": False, "error": str(e)}
    finally:
        st.session_state.engine_running = False
        st.rerun()

# Display results
if st.session_state.engine_result is not None:
    result = st.session_state.engine_result
    
    if result.get("ok"):
        st.success(f"✅ Pipeline complete! Processed {len(result.get('processed', []))} dates")
        if result.get("output_dir"):
            st.info(f"📁 Output: `{result['output_dir']}`")
    else:
        failed = result.get("failed", [])
        if failed:
            st.error(f"❌ Failed to process {len(failed)} date(s)")
        else:
            st.error("❌ Pipeline failed")

# Display log
if st.session_state.engine_log:
    with st.expander("Pipeline Log", expanded=False):
        log_text = "\n".join(st.session_state.engine_log)
        st.code(log_text, language="text")

st.divider()

# -----------------------------------------------------------------------------
# Generate Orders Section
# -----------------------------------------------------------------------------

st.header("Order Generation")

gen_col1, gen_col2 = st.columns([1, 3])

with gen_col1:
    generate_clicked = st.button("🎯 Generate Orders", type="primary", width='stretch')

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
                # =================================================================
                # LIVE PRICE REFRESH - Fetch fresh prices BEFORE recommendations
                # OPTIMIZED: Batch fetch by expiry date to minimize API calls
                # =================================================================
                from decimal import Decimal
                import re
                import json
                import requests
                
                st.info("🔄 Fetching live prices from CLOB...")
                
                # Determine the price column name in batch_df
                price_col = None
                for col in ["market_price", "market_pr", "Polymarket_Price"]:
                    if col in batch_df.columns:
                        price_col = col
                        break
                
                if price_col is None:
                    st.error("No price column found in batch data")
                else:
                    # OPTIMIZATION: Group contracts by expiry date and fetch tokens in batch
                    # This reduces API calls from ~200 (one per row) to ~5 (one per expiry date)
                    
                    # Step 1: Group slugs by expiry date (month-day)
                    slug_to_idx = {}  # slug -> list of row indices
                    date_to_slugs = {}  # (month, day) -> list of slugs
                    
                    for idx, row in batch_df.iterrows():
                        slug = row.get("slug", "")
                        if not slug:
                            continue
                        
                        # Track which rows have this slug
                        if slug not in slug_to_idx:
                            slug_to_idx[slug] = []
                        slug_to_idx[slug].append(idx)
                        
                        # Parse month and day from slug
                        month_match = re.search(
                            r'(january|february|march|april|may|june|july|august|september|october|november|december)-(\d+)',
                            slug, re.IGNORECASE
                        )
                        if month_match:
                            month = month_match.group(1).lower()
                            day = int(month_match.group(2))
                            date_key = (month, day)
                            if date_key not in date_to_slugs:
                                date_to_slugs[date_key] = set()
                            date_to_slugs[date_key].add(slug)
                    
                    # Debug: Show what was parsed
                    st.text(f"Found {len(slug_to_idx)} unique slugs, grouped into {len(date_to_slugs)} dates")
                    logger.info(f"date_to_slugs: {dict((f'{m}-{d}', len(s)) for (m,d), s in date_to_slugs.items())}")
                    
                    # Step 2: For each expiry date, make ONE API call to get all tokens
                    slug_to_tokens = {}  # slug -> {"YES": token_id, "NO": token_id}
                    GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
                    
                    progress_text = st.empty()
                    total_dates = len(date_to_slugs)
                    
                    # Debug: Test API connectivity with first date
                    if date_to_slugs:
                        test_month, test_day = list(date_to_slugs.keys())[0]
                        test_slug = f"bitcoin-above-on-{test_month}-{test_day}"
                        try:
                            test_resp = requests.get(GAMMA_API_URL, params={"slug": test_slug}, timeout=15)
                            test_events = test_resp.json() if test_resp.status_code == 200 else []
                            test_markets = sum(len(e.get("markets", [])) for e in test_events) if test_events else 0
                            st.text(f"API test: {test_slug} returned {len(test_events)} events, {test_markets} markets")
                        except Exception as e:
                            st.error(f"API test failed: {e}")
                    
                    
                    # Define helper for parallel execution
                    def fetch_tokens_for_date(m, d):
                        api_slug_local = f"bitcoin-above-on-{m}-{d}"
                        try:
                            # Use a new request for each thread (or pass session if available, but simple get is fine here)
                            r = requests.get(
                                GAMMA_API_URL, 
                                params={"slug": api_slug_local},
                                timeout=15
                            )
                            if r.status_code == 200:
                                return r.json(), None
                            else:
                                return None, f"Status {r.status_code}"
                        except Exception as ex:
                            return None, str(ex)

                    progress_text.text(f"Fetching tokens for {total_dates} dates in parallel...")
                    
                    # Parallel execution
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        # Map futures to args
                        future_to_args = {
                            executor.submit(fetch_tokens_for_date, m, d): (m, d, s) 
                            for (m, d), s in date_to_slugs.items()
                        }
                        
                        completed_count = 0
                        for future in concurrent.futures.as_completed(future_to_args):
                            completed_count += 1
                            month_arg, day_arg, slugs_arg = future_to_args[future]
                            progress_text.text(f"Processed {month_arg}-{day_arg} ({completed_count}/{total_dates})...")
                            
                            try:
                                events, error = future.result()
                                
                                if error:
                                    logger.warning(f"Failed to fetch {month_arg}-{day_arg}: {error}")
                                    continue
                                    
                                if not events:
                                    logger.warning(f"No events for {month_arg}-{day_arg}")
                                    continue
                                
                                total_markets = sum(len(e.get("markets", [])) for e in events)
                                logger.info(f"Found {len(events)} events, {total_markets} markets for {month_arg}-{day_arg}")
                                
                                # Process events (Match tokens)
                                for event in events:
                                    markets = event.get("markets", [])
                                    for market in markets:
                                        question = market.get("question", "")
                                        
                                        # Parse strike from question
                                        strike_match = re.search(r'\$?(\d+(?:,\d+)?(?:\.\d+)?)(k)?', question, re.IGNORECASE)
                                        if not strike_match:
                                            continue
                                        val_str = strike_match.group(1).replace(',', '')
                                        market_strike = float(val_str)
                                        if strike_match.group(2) and strike_match.group(2).lower() == 'k':
                                            market_strike *= 1000
                                        
                                        # Get token IDs
                                        clob_token_ids = market.get("clobTokenIds", "[]")
                                        outcomes = market.get("outcomes", "[]")
                                        
                                        try:
                                            token_ids = json.loads(clob_token_ids) if isinstance(clob_token_ids, str) else clob_token_ids
                                            outcome_list = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                                        except:
                                            continue
                                        
                                        if not token_ids or not outcome_list:
                                            continue
                                        
                                        yes_token = None
                                        no_token = None
                                        for j, outcome in enumerate(outcome_list):
                                            if outcome.upper() == "YES" and j < len(token_ids):
                                                yes_token = token_ids[j]
                                            elif outcome.upper() == "NO" and j < len(token_ids):
                                                no_token = token_ids[j]
                                        
                                        if not yes_token or not no_token:
                                            continue
                                        
                                        # Match to slugs
                                        for slug in slugs_arg:
                                            # Check if strike matches
                                            slug_strike_match = re.search(r'above-(\d+)-on', slug)
                                            if slug_strike_match:
                                                slug_strike = float(slug_strike_match.group(1))
                                                if abs(market_strike - slug_strike) < 100:
                                                    slug_to_tokens[slug] = {"YES": yes_token, "NO": no_token}
                                            # Also check strike in slug suffix
                                            parts = slug.split("_")
                                            if len(parts) >= 3 and parts[2].replace('.', '').isdigit():
                                                suffix_strike = float(parts[2])
                                                if abs(market_strike - suffix_strike) < 100:
                                                    slug_to_tokens[slug] = {"YES": yes_token, "NO": no_token}

                            except Exception as e:
                                logger.error(f"Error processing future result for {month_arg}-{day_arg}: {e}")
                    
                    progress_text.empty()
                    
                    # Debug: Show how many tokens matched
                    st.text(f"Matched {len(slug_to_tokens)} slugs to CLOB tokens (YES + NO)")
                    
                    # If no matches, show detailed debug info
                    if len(slug_to_tokens) == 0:
                        with st.expander("🔍 Debug: Token Matching Details", expanded=True):
                            st.write("**Dates searched:**")
                            for (m, d), s in date_to_slugs.items():
                                st.write(f"• {m}-{d}: {len(s)} slugs")
                            st.write("**Sample slug format:**")
                            sample_slugs = list(slug_to_idx.keys())[:3]
                            for s in sample_slugs:
                                st.code(s)
                            st.write("Try running the debug script for more details:") 
                            st.code("python debug_token_match.py", language="bash")
                    
                    logger.info(f"slug_to_tokens count: {len(slug_to_tokens)}")
                    
                    # Step 3: Batch fetch prices for BOTH YES and NO tokens
                    if slug_to_tokens:
                        # Create trading provider for live price fetching
                        trading_provider = RealPolymarketProvider(mode="trading")
                        
                        # Get unique token IDs (both YES and NO)
                        all_tokens = set()
                        for tokens in slug_to_tokens.values():
                            all_tokens.add(tokens["YES"])
                            all_tokens.add(tokens["NO"])
                        unique_tokens = list(all_tokens)
                        
                        st.text(f"Fetching order books for {len(unique_tokens)} tokens (YES + NO)...")
                        
                        # Calculate expected order size in dollars
                        if use_fixed_stake:
                            expected_dollars = fixed_stake_amount
                        else:
                            # For Kelly sizing, use min_trade_usd as the floor
                            expected_dollars = min_trade_usd
                        
                        st.text(f"Checking depth for ~${expected_dollars:.0f} orders (using actual prices)")
                        
                        # Fetch prices with depth check for all tokens
                        token_prices = trading_provider.fetch_live_prices_with_depth(
                            unique_tokens, 
                            order_dollars=expected_dollars,
                            timeout=5.0
                        )
                        
                        # Step 4: Update batch_df with BOTH YES and NO prices
                        # Add new columns for YES and NO ask prices and depths
                        batch_df["yes_ask_price"] = None
                        batch_df["no_ask_price"] = None
                        batch_df["yes_depth"] = None
                        batch_df["no_depth"] = None
                        batch_df["yes_has_depth"] = True
                        batch_df["no_has_depth"] = True
                        batch_df["spread"] = None  # To track spread for filtering
                        
                        updated_count = 0
                        price_updates = []
                        fallback_count = 0
                        fallback_slugs = []
                        low_depth_yes_count = 0
                        low_depth_no_count = 0
                        slug_to_spread = {}  # Map slug -> spread for gate logic
                        
                        for slug, tokens in slug_to_tokens.items():
                            yes_token = tokens["YES"]
                            no_token = tokens["NO"]
                            
                            yes_price_info = token_prices.get(yes_token)
                            no_price_info = token_prices.get(no_token)
                            
                            yes_ask = float(yes_price_info["ask"]) if yes_price_info and yes_price_info.get("ask") else None
                            no_ask = float(no_price_info["ask"]) if no_price_info and no_price_info.get("ask") else None
                            yes_depth = float(yes_price_info.get("ask_depth", 0)) if yes_price_info else 0
                            no_depth = float(no_price_info.get("ask_depth", 0)) if no_price_info else 0
                            yes_has_depth = yes_price_info.get("has_sufficient_depth", True) if yes_price_info else True
                            no_has_depth = no_price_info.get("has_sufficient_depth", True) if no_price_info else True
                            
                            if not yes_has_depth:
                                low_depth_yes_count += 1
                            if not no_has_depth:
                                low_depth_no_count += 1
                            
                            # Update all rows with this slug
                            for idx in slug_to_idx.get(slug, []):
                                if yes_ask is not None:
                                    # Update the main price column with YES ask (for compatibility)
                                    original_price = float(batch_df.at[idx, price_col])
                                    batch_df.at[idx, price_col] = yes_ask
                                    batch_df.at[idx, "yes_ask_price"] = yes_ask
                                    batch_df.at[idx, "yes_depth"] = yes_depth
                                    batch_df.at[idx, "yes_has_depth"] = yes_has_depth
                                    updated_count += 1
                                    
                                    price_diff = yes_ask - original_price
                                    if abs(price_diff) > 0.001 and slug not in [p["slug"] for p in price_updates]:
                                        price_updates.append({
                                            "slug": slug[:25],
                                            "old": original_price,
                                            "yes_ask": yes_ask,
                                            "no_ask": no_ask,
                                            "spread": (yes_ask + no_ask - 1) if (yes_ask and no_ask) else None,
                                            "yes_depth": yes_depth,
                                            "no_depth": no_depth,
                                        })
                                else:
                                    fallback_count += 1
                                    fallback_slugs.append(slug)
                                
                                if no_ask is not None:
                                    batch_df.at[idx, "no_ask_price"] = no_ask
                                    batch_df.at[idx, "no_depth"] = no_depth
                                    batch_df.at[idx, "no_has_depth"] = no_has_depth

                                # Capture spread for this slug
                                if yes_ask and no_ask:
                                    current_spread = yes_ask + no_ask - 1.0
                                    slug_to_spread[slug] = current_spread
                                    batch_df.at[idx, "spread"] = current_spread

                        
                        # Show price update summary with depth info
                        if price_updates:
                            with st.expander(f"📊 Live Price Updates ({len(price_updates)} markets)", expanded=False):
                                for pu in price_updates[:10]:
                                    spread_str = f"spread={pu['spread']:.2%}" if pu.get('spread') else ""
                                    no_ask_str = f"{pu['no_ask']:.4f}" if pu.get('no_ask') else "N/A"
                                    st.write(f"• {pu['slug']}: YES={pu['yes_ask']:.4f}, NO={no_ask_str} {spread_str}")
                                if len(price_updates) > 10:
                                    st.write(f"... and {len(price_updates) - 10} more")
                        
                        if fallback_count > 0:
                            msg = f"⚠️ {fallback_count} contracts using batch prices (CLOB unavailable)"
                            with st.expander(msg, expanded=False):
                                st.write("Live buy price (YES Ask) not available for:")
                                for s in fallback_slugs:
                                    st.caption(s)
                        
                        st.success(f"✅ Live prices fetched for {updated_count}/{len(batch_df)} contracts ({len(date_to_slugs)} API calls)")
                    else:
                        st.warning("⚠️ Could not match any contracts to CLOB tokens. Using batch prices.")
                # =================================================================
                
                # Build recommend_trades kwargs with LIVE PRICE DATA
                reco_kwargs = {
                    "df": batch_df,
                    "bankroll": bankroll,
                    "kelly_fraction": kelly_fraction,
                    "min_edge": min_edge,
                    "max_bets_per_expiry": max_bets_per_expiry,
                    "max_capital_per_expiry_frac": max_capital_per_expiry_frac,
                    "max_capital_total_frac": max_capital_total_frac,
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_model_prob": min_model_prob,
                    "max_model_prob": max_model_prob,
                    "use_stability_penalty": use_stability_penalty,
                    # When using fixed stake, set min to that amount to avoid filtering
                    "min_trade_usd": fixed_stake_amount if use_fixed_stake else min_trade_usd,
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
                
                # Run auto_reco with LIVE PRICES
                recommendations = recommend_trades(**reco_kwargs)
                
                if not recommendations:
                    st.warning("No trade recommendations generated with current parameters.")
                else:
                    # Convert to DataFrame
                    reco_df = recommendations_to_dataframe(recommendations)
                    
                    st.success(f"✅ Generated {len(reco_df)} recommendations with live prices")
                    
                    # -----------------------------------------------------------------------------
                    # Spread-Aware Edge Gate Filtering
                    # -----------------------------------------------------------------------------
                    if use_spread_gate:
                        filtered_recos = []
                        accepted_recos = []
                        
                        for rec in recommendations:
                            rec_slug = rec.slug
                            rec_edge = rec.edge
                            
                            # Get spread
                            spread = slug_to_spread.get(rec_slug)
                            
                            # If spread is unknown (fallback), default to blocking or passing? 
                            # Since we have "fallback_count" warning, maybe we skip gate or block.
                            # "Spread-Aware" implies we need spread.
                            if spread is None:
                                # Block if spread unknown
                                filtered_recos.append((rec, "Unknown Spread"))
                                continue
                            
                            required_edge = (spread / 2.0) + spread_gate_buffer
                            
                            if rec_edge >= required_edge:
                                accepted_recos.append(rec)
                            else:
                                filtered_recos.append((rec, f"Edge {rec_edge:.1%} < Req {required_edge:.1%} (S={spread:.1%})"))
                        
                        if filtered_recos:
                            st.warning(f"🛡️ Spread-Aware Gate filtered out {len(filtered_recos)} trades")
                            with st.expander("Filtered Trades (Insufficient Edge vs Spread)", expanded=False):
                                for frec, reason in filtered_recos:
                                    st.write(f"• **{frec.side} {frec.slug[:30]}...**: {reason}")
                        
                        # Update recommendations to only accepted
                        recommendations = accepted_recos
                        # Re-create DataFrame from filtered list
                        reco_df = recommendations_to_dataframe(recommendations)
                        
                        if not recommendations:
                             st.warning("All recommendations were filtered by Spread Gate.")

                    # -----------------------------------------------------------------------------

                    # Check for low depth ONLY on recommended trades
                    low_depth_trades = []
                    if "slug" in reco_df.columns and "side" in reco_df.columns:
                        for _, row in reco_df.iterrows():
                            slug = row.get("slug", "")
                            side = row.get("side", "YES")
                            
                            # Find the batch_df row for this slug
                            batch_row = batch_df[batch_df["slug"] == slug]
                            if not batch_row.empty:
                                if side == "YES" and "yes_has_depth" in batch_df.columns:
                                    has_depth = batch_row.iloc[0].get("yes_has_depth", True)
                                    if not has_depth:
                                        low_depth_trades.append(f"{slug[:30]}... (YES)")
                                elif side == "NO" and "no_has_depth" in batch_df.columns:
                                    has_depth = batch_row.iloc[0].get("no_has_depth", True)
                                    if not has_depth:
                                        low_depth_trades.append(f"{slug[:30]}... (NO)")
                    
                    if low_depth_trades:
                        st.warning(f"⚠️ {len(low_depth_trades)} recommended trades have low depth:")
                        for trade in low_depth_trades[:5]:
                            st.write(f"  • {trade}")
                        if len(low_depth_trades) > 5:
                            st.write(f"  ... and {len(low_depth_trades) - 5} more")
                    # =================================================================
                    
                    # Clear existing draft intents before generating new ones
                    clear_draft_intents()
                    
                    # Create run and build intents
                    run = create_run(strategy="auto_reco", params=reco_params)
                    intents, warnings = build_intents_from_reco(reco_df, run)
                    
                    # Check for duplicates
                    duplicates = check_duplicate_intents(intents)
                    if duplicates:
                        with st.expander(f"⚠️ {len(duplicates)} duplicate contracts detected", expanded=True):
                            for dup in duplicates:
                                st.write(f"• {dup}")
                    
                    # Display warnings
                    if warnings:
                        with st.expander(f"⚠️ {len(warnings)} validation warnings", expanded=False):
                            for w in warnings[:10]:
                                st.write(f"• {w['contract']}: {', '.join(w['warnings'])}")
                    
                    # Save intents
                    if intents:
                        saved_count = save_intents(intents)
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

# Duplicate filtering removed as requested
pass

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
            "p_model": f"{intent.model_prob:.2%}" if intent.model_prob else "N/A",
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
    
    # Check for orders below minimum (5 shares for CLOB)
    MIN_CLOB_SHARES = 5
    below_minimum = [i for i in draft_intents if i.size_shares < MIN_CLOB_SHARES]
    
    # Show table with selection
    st.write(f"**{len(draft_intents)} draft intents**")
    
    if below_minimum:
        st.warning(
            f"⚠️ **{len(below_minimum)} order(s) below 5-share minimum** - "
            f"Polymarket CLOB requires at least 5 shares per order. "
            f"These orders will fail if submitted."
        )
    
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
        width='stretch',
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
            width='stretch',
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
    # Display approved intents with selection checkboxes
    approved_data = []
    for intent in approved_intents:
        approved_data.append({
            "contract": intent.contract,
            "outcome": intent.outcome,
            "limit_price": f"${intent.limit_price:.4f}",
            "size_shares": f"{intent.size_shares:.2f}",
            "notional": f"${intent.notional_usd:.2f}",
            "approved_at": intent.approved_at[:19] if intent.approved_at else "N/A",
            "_intent_id_full": intent.intent_id,
        })
    
    approved_display_df = pd.DataFrame(approved_data)
    
    # Add selection column
    approved_selection_df = approved_display_df.copy()
    approved_selection_df.insert(0, "Select", False)
    
    approved_edited_df = st.data_editor(
        approved_selection_df.drop(columns=["_intent_id_full"]),
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=False),
            "contract": st.column_config.TextColumn("Contract", width="medium"),
            "outcome": st.column_config.TextColumn("Side", width="small"),
            "limit_price": st.column_config.TextColumn("Price", width="small"),
            "size_shares": st.column_config.TextColumn("Shares", width="small"),
            "notional": st.column_config.TextColumn("Notional", width="small"),
            "approved_at": st.column_config.TextColumn("Approved At", width="medium"),
        },
        width='stretch',
        key="approved_intents_editor",
    )
    
    # Get selected approved intent IDs
    approved_selected_mask = approved_edited_df["Select"].tolist()
    approved_selected_ids = [
        approved_display_df.iloc[i]["_intent_id_full"]
        for i, selected in enumerate(approved_selected_mask)
        if selected
    ]
    
    # Total notional
    total_notional = sum(i.notional_usd for i in approved_intents)
    st.write(f"**Total notional: ${total_notional:.2f}** | {len(approved_intents)} orders")
    
    # Check against available collateral
    if state:
        if total_notional > state.available_collateral:
            st.warning(
                f"⚠️ Total notional (${total_notional:.2f}) exceeds available collateral "
                f"(${state.available_collateral:.2f}). Some orders may fail."
            )
    
    # Action buttons row
    submit_col1, remove_col, submit_col2 = st.columns([1, 1, 2])
    
    with submit_col1:
        submit_clicked = st.button(
            "🚀 Place Approved Orders",
            type="primary",
            width='stretch',
        )
    
    with remove_col:
        remove_clicked = st.button(
            f"🗑️ Remove Selected ({len(approved_selected_ids)})",
            type="secondary",
            width='stretch',
            disabled=len(approved_selected_ids) == 0,
        )
    
    # Handle remove selected
    if remove_clicked and approved_selected_ids:
        removed_count = 0
        for intent_id in approved_selected_ids:
            # Move back to DRAFT status
            success = update_intent_status(
                intent_id,
                IntentStatus.DRAFT,
                notes="Removed from approved list",
            )
            if success:
                removed_count += 1
        
        st.success(f"✅ Removed {removed_count} orders from approved list (moved back to drafts)")
        st.rerun()
    
    if submit_clicked:
        # Check for orders below minimum FIRST
        MIN_CLOB_SHARES = 5
        below_min_approved = [i for i in approved_intents if i.size_shares < MIN_CLOB_SHARES]
        
        if below_min_approved:
            st.error(
                f"❌ **Cannot submit: {len(below_min_approved)} order(s) below 5-share minimum**\n\n"
                f"Polymarket CLOB requires at least 5 shares per order. "
                f"Please remove these orders before submitting:\n"
                + "\n".join([f"• {i.contract} ({i.outcome}): {i.size_shares:.2f} shares" for i in below_min_approved[:5]])
            )
        else:
            with st.spinner("Submitting orders..."):
                # Check for private key
                import os
                private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
                if not private_key:
                    st.error("❌ POLYMARKET_PRIVATE_KEY not set in .env file. Required for trading.")
                else:
                    # Create trading mode provider for submissions
                    trading_provider = RealPolymarketProvider(mode="trading")
                    
                    # Refresh account state before submission
                    current_state = fetch_account_state(provider)
                    save_account_state(current_state)
                    st.session_state.account_state = current_state
                    
                    # -------------------------------------------------------------------------
                    # Pre-Submission Price Validation & Auto-Improvement
                    # -------------------------------------------------------------------------
                    price_validation_passed = True
                    valid_intents_to_submit = []
                    
                    with st.status("🔍 Validating live prices...", expanded=True) as status:
                        # 1. Resolve Token IDs (Parallel)
                        status.write("Resolving token IDs...")
                        unique_keys = set((i.contract, i.outcome) for i in approved_intents)
                        contract_outcome_map = {} 
                        
                        def resolve_token(c, o):
                            return trading_provider.fetch_clob_token_id(c, o)
                            
                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                            future_to_key = {executor.submit(resolve_token, c, o): (c, o) for c, o in unique_keys}
                            for future in concurrent.futures.as_completed(future_to_key):
                                c, o = future_to_key[future]
                                try:
                                    tid = future.result()
                                    if tid:
                                        contract_outcome_map[(c, o)] = tid
                                except Exception as e:
                                    logger.error(f"Failed to resolve token for {c} {o}: {e}")
                        
                        # 2. Fetch Live Prices
                        status.write("Fetching live order books...")
                        token_ids_to_fetch = list(set(contract_outcome_map.values()))
                        if not token_ids_to_fetch:
                            st.error("❌ Critical: Could not resolve any token IDs.")
                            live_prices = {}
                        else:
                            # Use minimal dollar depth just to get the best price
                            live_prices = trading_provider.fetch_live_prices_with_depth(token_ids_to_fetch, order_dollars=10)
                        
                        # 3. Validate & Improve
                        status.write("Comparing prices...")
                        blocked_intents = [] 
                        improved_intents = [] 
                        
                        for intent in approved_intents:
                            tid = contract_outcome_map.get((intent.contract, intent.outcome))
                            price_info = live_prices.get(tid) if tid else None
                            
                            conn_key = f"{intent.contract[:20]}... ({intent.outcome})"
                            
                            if not price_info:
                                blocked_intents.append((intent, f"Could not fetch live price (Token ID: {tid})"))
                                continue
                                
                            action = intent.action.upper()
                            
                            if action == "BUY":
                                # We want to buy. Comparison against Best Ask.
                                market_price = float(price_info.get("ask")) if price_info.get("ask") else None
                                if market_price is None:
                                    blocked_intents.append((intent, "No live Sell orders (Ask) on market"))
                                    continue
                                
                                # Logic:
                                # If Market > Limit: Bad. (Price went up). BLOCK.
                                # If Market < Limit: Good. (Price went down). IMPROVE.
                                
                                if market_price > intent.limit_price + 0.0001:
                                    blocked_intents.append((intent, f"Price mismatch: Market Ask {market_price:.4f} > Limit {intent.limit_price:.4f}"))
                                elif market_price < intent.limit_price - 0.0001:
                                    old_p = intent.limit_price
                                    intent.limit_price = market_price
                                    improved_intents.append((intent, old_p, market_price))
                                    valid_intents_to_submit.append(intent)
                                else:
                                    valid_intents_to_submit.append(intent)

                            elif action == "SELL":
                                # We want to sell. Comparison against Best Bid.
                                market_price = float(price_info.get("bid")) if price_info.get("bid") else None
                                if market_price is None:
                                    blocked_intents.append((intent, "No live Buy orders (Bid) on market"))
                                    continue
                                
                                # Logic:
                                # If Market < Limit: Bad. (Price went down). BLOCK.
                                # If Market > Limit: Good. (Price went up). IMPROVE.
                                
                                if market_price < intent.limit_price - 0.0001:
                                    blocked_intents.append((intent, f"Price mismatch: Market Bid {market_price:.4f} < Limit {intent.limit_price:.4f}"))
                                elif market_price > intent.limit_price + 0.0001:
                                    old_p = intent.limit_price
                                    intent.limit_price = market_price
                                    improved_intents.append((intent, old_p, market_price))
                                    valid_intents_to_submit.append(intent)
                                else:
                                    valid_intents_to_submit.append(intent)
                        
                        # Feedback
                        if blocked_intents:
                            status.update(label="⚠️ Price Validation found issues!", state="error", expanded=True)
                            st.error(f"🛑 Blocking {len(blocked_intents)} orders due to unfavorable price movements:")
                            for i, r in blocked_intents:
                                st.write(f"• **{i.outcome} {i.contract[:30]}...**: {r}")
                            price_validation_passed = False # Only stops full success message, but we still allow partial submission of valid ones? 
                            # User said "that trade should not go through". Valid ones okay?
                            # I will submit valid ones.
                        
                        if improved_intents:
                            st.success(f"✨ Auto-Improving {len(improved_intents)} orders to better market prices:")
                            for i, old, new in improved_intents:
                                st.write(f"• **{i.outcome} {i.contract[:30]}...**: Improved {old:.3f} ➔ {new:.3f}")
                        
                        if not valid_intents_to_submit:
                            st.warning("No valid orders remaining to submit.")
                            st.stop()
                        
                        status.update(label=f"✅ Ready to submit {len(valid_intents_to_submit)} orders", state="complete", expanded=False)

                    # Submit batch with cumulative validation
                    results = submit_approved_batch(valid_intents_to_submit, current_state, trading_provider)
                    
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
    st.dataframe(history_df, width='stretch', hide_index=True)
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
        with st.spinner("Querying Polymarket CLOB API for order statuses..."):
            result = reconcile_submissions(provider)
        st.success(
            f"✅ Reconciled {result.get('open', 0)} orders: "
            f"**{result.get('filled', 0)} filled**, "
            f"{result.get('cancelled', 0)} cancelled, "
            f"{result.get('still_open', 0)} still open"
        )
        if result.get('errors', 0) > 0:
            st.warning(f"⚠️ {result.get('errors')} errors encountered")
        st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.caption("Polymarket Trade Console MVP - Using fake provider for testing")
