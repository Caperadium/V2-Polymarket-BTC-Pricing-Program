"""Binary BTC market-making module (Polymarket, paper-traded).

Foundation layer: interface contracts (Section 4), launch config, and the pure
log-odds transform library (Section 2.2). Downstream components consume these.
"""
from __future__ import annotations

from market_maker.config import MMConfig
from market_maker.contracts import (
    AnchorMethod,
    BankrollState,
    ConfidenceTier,
    ContractInv,
    FairValue,
    Fill,
    HedgeReason,
    HedgeRecommendation,
    InventoryState,
    LadderInv,
    LiquidityRegime,
    LiquiditySource,
    LiquidityState,
    MarketState,
    PaperFill,
    PricerSnapshot,
    QuoteMode,
    QuoteProposal,
    QuoteSet,
    RiskDirective,
    RiskTrigger,
    SettlementEvent,
    SettlementOutcome,
    Side,
    Sigma2Source,
    SizingCap,
    SizingDecision,
    SpotSource,
    VenueAdapter,
    VenueDescriptor,
)
from market_maker.logodds import (
    DEFAULT_P_HI,
    DEFAULT_P_LO,
    floor_half_spread,
    half_spread_p_exact,
    half_spread_p_linear,
    logit,
    logit_bounds,
    s_double_prime,
    s_prime,
    sigmoid,
)
from market_maker.pricer_adapter import build_snapshot
from market_maker.state_store import (
    DEFAULT_DB_PATH,
    ORDER_STATUSES,
    MMStateStore,
    OrderRecord,
    PnlSnapshot,
    QuoteRecord,
)
from market_maker.quote_engine import (
    estimate_sigma_b,
    glft_side_deltas,
    make_quote,
    make_quote_from_config,
    per_share_skew_x,
)
from market_maker.fair_value_anchor import (
    AnchorResult,
    buckets_to_ladder,
    compute_fair_value,
    ladder_to_buckets,
)
from market_maker.inventory_manager import InventoryManager
from market_maker.spread_builder import (
    build_quote_set,
    make_stub_directive,
    make_stub_sizing,
)
from market_maker.ladder_hedger import LadderHedger, NoArbVerdict
from market_maker.robustness_sizing import (
    ContractSizingInput,
    baker_mchale,
    kelly_buy,
    size_ladder,
)
from market_maker.liquidity_monitor import LiquidityMonitor
from market_maker.market_data_client import (
    BookMirror,
    FeedCapability,
    PolymarketFeedAdapter,
)
from market_maker.paper_fill_sim import ExposureIncident, PaperFillSimulator
from market_maker.risk_controller import InvBreach, RiskController, default_vol_gate_fn
from market_maker.order_lifecycle import (
    OrderLifecycleManager,
    PaperVenueAdapter,
    ReconciliationResult,
    SimClock,
    client_order_id,
)
from market_maker.settlement_handler import (
    BTCDataProvider,
    MarketPosition,
    SettlementHandler,
    SettlementRunResult,
    settlement_instant_utc,
)

__all__ = [
    # config
    "MMConfig",
    # enums
    "Sigma2Source",
    "ConfidenceTier",
    "AnchorMethod",
    "QuoteMode",
    "RiskTrigger",
    "LiquidityRegime",
    "Side",
    "LiquiditySource",
    "HedgeReason",
    "SizingCap",
    "SettlementOutcome",
    "SpotSource",
    # contracts
    "PricerSnapshot",
    "MarketState",
    "FairValue",
    "QuoteProposal",
    "QuoteSet",
    "InventoryState",
    "ContractInv",
    "LadderInv",
    "HedgeRecommendation",
    "SizingDecision",
    "LiquidityState",
    "RiskDirective",
    "Fill",
    "PaperFill",
    "VenueAdapter",
    "VenueDescriptor",
    "BankrollState",
    "SettlementEvent",
    # logodds
    "logit",
    "sigmoid",
    "logit_bounds",
    "s_prime",
    "s_double_prime",
    "half_spread_p_exact",
    "half_spread_p_linear",
    "floor_half_spread",
    "DEFAULT_P_LO",
    "DEFAULT_P_HI",
    # pricer_adapter
    "build_snapshot",
    # state_store
    "MMStateStore",
    "OrderRecord",
    "QuoteRecord",
    "PnlSnapshot",
    "DEFAULT_DB_PATH",
    "ORDER_STATUSES",
    # quote engine
    "make_quote",
    "make_quote_from_config",
    "glft_side_deltas",
    "estimate_sigma_b",
    "per_share_skew_x",
    # fair-value anchor
    "compute_fair_value",
    "AnchorResult",
    "ladder_to_buckets",
    "buckets_to_ladder",
    # inventory
    "InventoryManager",
    # spread builder
    "build_quote_set",
    "make_stub_directive",
    "make_stub_sizing",
    # ladder hedger
    "LadderHedger",
    "NoArbVerdict",
    # sizing
    "size_ladder",
    "kelly_buy",
    "baker_mchale",
    "ContractSizingInput",
    # liquidity
    "LiquidityMonitor",
    # market data
    "BookMirror",
    "FeedCapability",
    "PolymarketFeedAdapter",
    # fill sim
    "PaperFillSimulator",
    "ExposureIncident",
    # risk controller
    "RiskController",
    "InvBreach",
    "default_vol_gate_fn",
    # order lifecycle
    "OrderLifecycleManager",
    "PaperVenueAdapter",
    "SimClock",
    "ReconciliationResult",
    "client_order_id",
    # settlement
    "SettlementHandler",
    "BTCDataProvider",
    "MarketPosition",
    "SettlementRunResult",
    "settlement_instant_utc",
]
