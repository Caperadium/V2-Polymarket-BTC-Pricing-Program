"""Interface contracts (plan Section 4).

Frozen dataclasses for message/state contracts, plain dataclasses where the
component clearly mutates state (inventory, bankrolls), plus the shared enums
and the VenueAdapter abstraction (4.12).

Conventions (plan Section 4 header): probabilities are floats in [0,1]; log-odds
are floats (dimensionless); prices are USDC per share in [0,1]; sizes are shares
(float); timestamps are timezone-aware datetimes in memory; tte_days is float
days; market_id is str; expiry_key is YYYY-MM-DD str; strike is float USD. Every
message carries ts; source_seq is carried where the plan table names it.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Sigma2Source(Enum):
    MC = "MC"
    PARAM_POSTERIOR = "PARAM_POSTERIOR"


class ConfidenceTier(Enum):
    FULL = "FULL"
    DEGRADED = "DEGRADED"
    MINIMAL = "MINIMAL"
    NAIVE_GATED = "NAIVE_GATED"


class AnchorMethod(Enum):
    BEUOY = "BEUOY"
    FIXED_BLEND_FALLBACK = "FIXED_BLEND_FALLBACK"


class QuoteMode(Enum):
    TWO_SIDED = "TWO_SIDED"
    BID_ONLY = "BID_ONLY"
    ASK_ONLY = "ASK_ONLY"
    PULLED = "PULLED"


class RiskTrigger(Enum):
    SPOT_JUMP = "SPOT_JUMP"
    NEAR_RESOLUTION = "NEAR_RESOLUTION"
    SPOT_GAPPING_STRIKE = "SPOT_GAPPING_STRIKE"
    INV_CAP = "INV_CAP"
    FEED_STALE = "FEED_STALE"
    PRICER_STALE = "PRICER_STALE"
    LIQ_DEGENERATE = "LIQ_DEGENERATE"
    MANUAL = "MANUAL"
    # Wave 1 W1.2: write-only additive member -- the only deserializer is
    # state_store.get_risk_journal -> RiskTrigger(t), which reads values
    # written by the same-or-older code, so adding a member here cannot
    # break reads of an existing db.
    FAIR_VALUE_STALE = "FAIR_VALUE_STALE"


class LiquidityRegime(Enum):
    THICK = "THICK"
    NORMAL = "NORMAL"
    THIN = "THIN"
    DEGENERATE = "DEGENERATE"


class Side(Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"


class LiquiditySource(Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"
    SETTLEMENT = "SETTLEMENT"


class HedgeReason(Enum):
    VERTICAL_OFFSET = "VERTICAL_OFFSET"
    BETA_HEDGE = "BETA_HEDGE"


class SizingCap(Enum):
    LADDER_JOINT = "LADDER_JOINT"  # retained for old-journal compat; no longer emitted (plan C5)
    RUIN = "RUIN"
    BANKROLL = "BANKROLL"
    INVENTORY = "INVENTORY"  # per-side headroom cap (plan C2)
    DEPTH = "DEPTH"
    FRACTIONAL_C = "FRACTIONAL_C"


class SettlementOutcome(Enum):
    YES = "YES"
    NO = "NO"
    UNSETTLEABLE = "UNSETTLEABLE"


class SpotSource(Enum):
    INTRADAY = "INTRADAY"
    DAILY_PRIOR = "DAILY_PRIOR"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Light validation helpers (cheap only, per plan: do not over-engineer)
# ---------------------------------------------------------------------------


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _require_unit(name: str, v: Any) -> None:
    """Probability/fraction in [0,1]; rejects NaN."""
    if v is None:
        return
    if not _is_num(v) or math.isnan(float(v)) or not (0.0 <= float(v) <= 1.0):
        raise ValueError(name + " must be in [0,1], got " + repr(v))


def _require_price(name: str, v: Any) -> None:
    """Price in the venue band [0,1]; None allowed (empty side)."""
    if v is None:
        return
    if _is_num(v) and math.isnan(float(v)):
        return  # NaN price = empty side, permitted by contract
    _require_unit(name, v)


def _require_nonneg(name: str, v: Any) -> None:
    if v is None:
        return
    if not _is_num(v) or math.isnan(float(v)) or float(v) < 0.0:
        raise ValueError(name + " must be >= 0, got " + repr(v))


def _require_prob_dict(name: str, d: Dict[float, float]) -> None:
    for k, v in d.items():
        _require_unit(name + "[" + repr(k) + "]", v)


# ---------------------------------------------------------------------------
# 4.1 PricerSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PricerSnapshot:
    ts: datetime
    expiry_key: str
    tte_days: float
    s0: float
    n_sims: int
    strikes: List[float]
    grid_strikes: List[float]
    p_hat: Dict[float, float]
    p_grid: Dict[float, float]
    sigma2: Dict[float, float]
    sigma2_ladder: float
    sigma2_source: Sigma2Source
    confidence_tier: ConfidenceTier
    horizon_gate_active: bool
    stale: bool
    engine_meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_prob_dict("p_hat", self.p_hat)
        _require_prob_dict("p_grid", self.p_grid)
        for k, v in self.sigma2.items():
            _require_nonneg("sigma2[" + repr(k) + "]", v)
        _require_nonneg("sigma2_ladder", self.sigma2_ladder)


# ---------------------------------------------------------------------------
# 4.2 MarketState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketState:
    ts: datetime
    market_id: str
    expiry_key: str
    strike: float
    best_bid: Optional[float]  # NaN/None if empty side
    best_ask: Optional[float]
    bid_depth: List[Tuple[float, float]]  # (price, size) top-N
    ask_depth: List[Tuple[float, float]]
    last_prints: List[Tuple[datetime, float, float]]  # (ts, price, size)
    feed_healthy: bool

    def __post_init__(self) -> None:
        _require_price("best_bid", self.best_bid)
        _require_price("best_ask", self.best_ask)


# ---------------------------------------------------------------------------
# 4.3 FairValue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FairValue:
    ts: datetime
    expiry_key: str
    consensus_p: Dict[float, float]
    consensus_x: Dict[float, float]
    credibility: float
    anchor_method: AnchorMethod
    inputs_ts: Tuple[datetime, datetime]  # (pricer snapshot ts, market state ts)
    skew_correction: Optional[Dict[float, float]] = None  # None in build one
    # Package B2 (2026-07-15), additive: per-region ("belly"/"wing") pricer
    # credibility, alongside the legacy scalar `credibility` (the strike-
    # count-weighted average of the two). None for the FIXED_BLEND_FALLBACK
    # path, where a region split is not meaningful.
    credibility_by_region: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        _require_prob_dict("consensus_p", self.consensus_p)
        _require_unit("credibility", self.credibility)
        if self.credibility_by_region is not None:
            for k, v in self.credibility_by_region.items():
                _require_unit("credibility_by_region[" + repr(k) + "]", v)


# ---------------------------------------------------------------------------
# 4.4 QuoteProposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuoteProposal:
    ts: datetime
    market_id: str
    r_x: float
    delta_x: float
    skew_x: float
    sigma_b: float
    params_id: str
    x_bid: float
    x_ask: float
    p_bid_raw: float
    p_ask_raw: float

    def __post_init__(self) -> None:
        _require_nonneg("delta_x", self.delta_x)
        _require_nonneg("sigma_b", self.sigma_b)


# ---------------------------------------------------------------------------
# 4.5 QuoteSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuoteSet:
    ts: datetime
    market_id: str
    bid_price: float
    ask_price: float
    bid_size: float  # 0 = do not quote that side
    ask_size: float
    terms: Dict[str, float]  # markup, eps, skew, robust (prob units)
    risk_mode: QuoteMode
    noarb_checked: bool
    source_seq: int

    def __post_init__(self) -> None:
        _require_price("bid_price", self.bid_price)
        _require_price("ask_price", self.ask_price)
        _require_nonneg("bid_size", self.bid_size)
        _require_nonneg("ask_size", self.ask_size)


# ---------------------------------------------------------------------------
# 4.6 InventoryState (+ ContractInv, LadderInv) -- mutable bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class ContractInv:
    q: float  # signed shares, YES-positive
    avg_cost: float  # price
    q_max: float  # current cap from S'(x)
    age_weighted_holding: float  # hours, R3 input

    def __post_init__(self) -> None:
        _require_price("avg_cost", self.avg_cost)
        _require_nonneg("q_max", self.q_max)


@dataclass
class LadderInv:
    net_band_exposure: List[float]  # per inter-strike bucket, shares
    gross: float
    phi: float  # current running penalty
    r3_histogram: Dict[int, float]  # holding-time distribution


@dataclass
class InventoryState:
    ts: datetime
    per_contract: Dict[str, ContractInv]
    per_ladder: Dict[str, LadderInv]


# ---------------------------------------------------------------------------
# 4.7 HedgeRecommendation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HedgeRecommendation:
    ts: datetime
    expiry_key: str
    target_market_id: str
    side: Side
    size: float
    max_price: float  # passive-preferred limit
    reason: HedgeReason
    paired_market_id: str
    beta: Optional[float]  # clamped value used; None for vertical offset
    expires: datetime

    def __post_init__(self) -> None:
        _require_nonneg("size", self.size)
        _require_price("max_price", self.max_price)


# ---------------------------------------------------------------------------
# 4.8 SizingDecision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizingDecision:
    ts: datetime
    market_id: str
    bid_size: float
    ask_size: float
    f_kelly: float  # raw per-contract Kelly
    k_shrink: float  # [0,1] Baker-McHale factor
    ladder_alloc: float
    caps_applied: List[SizingCap]
    sigma2_used: float
    phi_directive: float
    # = headroom when inventory provided at sizing time, else 0.0 (not computed)
    max_add_yes: float = 0.0
    max_add_no: float = 0.0

    def __post_init__(self) -> None:
        _require_nonneg("bid_size", self.bid_size)
        _require_nonneg("ask_size", self.ask_size)
        _require_unit("k_shrink", self.k_shrink)
        _require_nonneg("sigma2_used", self.sigma2_used)


# ---------------------------------------------------------------------------
# 4.9 LiquidityState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquidityState:
    ts: datetime
    market_id: str
    realized_depth_bid: float  # shares within X ticks of touch
    realized_depth_ask: float
    kyle_lambda: Optional[float]  # NaN/None until estimable
    arb_halflife_s: Optional[float]  # seconds, YES+NO deviation decay
    regime: LiquidityRegime
    window: str
    vol_discount: float = 2.5

    def __post_init__(self) -> None:
        _require_nonneg("realized_depth_bid", self.realized_depth_bid)
        _require_nonneg("realized_depth_ask", self.realized_depth_ask)


# ---------------------------------------------------------------------------
# 4.10 RiskDirective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskDirective:
    ts: datetime
    market_id: str
    mode: QuoteMode
    eps_add: float  # extra adverse-selection widening (prob)
    kelly_mult: float  # [0,1] from vol_gate; journaled-only by decision 2026-07-08 --
    # NOT applied to sizing (sizing protection is Baker-McHale + caps + fractional-c;
    # vol gate acts on quotes via eps_add + PULL instead). See risk_controller.py.
    triggers: List[RiskTrigger]
    latched_until: datetime  # hysteresis
    cancel_all: bool

    def __post_init__(self) -> None:
        _require_nonneg("eps_add", self.eps_add)
        _require_unit("kelly_mult", self.kelly_mult)


# ---------------------------------------------------------------------------
# 4.11 Fill / PaperFill
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    ts: datetime
    market_id: str
    order_id: str  # or synthetic settlement id
    side: Side
    price: float
    size: float
    liquidity: LiquiditySource  # MAKER / TAKER / SETTLEMENT
    venue_ts: datetime

    def __post_init__(self) -> None:
        _require_price("price", self.price)
        _require_nonneg("size", self.size)


@dataclass(frozen=True)
class PaperFill(Fill):
    queue_ahead_at_fill: float = 0.0  # shares
    print_size: float = 0.0  # aggressor print that produced the fill
    latency_applied_ms: int = 0
    assumption_set: str = ""  # fill-model version id
    mid_at_fill: Optional[float] = None
    mid_p1m: Optional[float] = None  # NaN/None until elapsed
    mid_p10m: Optional[float] = None
    mid_p1h: Optional[float] = None


# ---------------------------------------------------------------------------
# 4.12 VenueAdapter (+ VenueDescriptor) -- order_lifecycle boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VenueDescriptor:
    tick_size: float
    min_size: float
    price_band: Tuple[float, float]  # (0.001, 0.999) both venues
    maker_fee: float
    maker_rebate: float
    settlement_rule: str  # Polymarket: 12:00 ET on expiry date
    supports_ladder: bool  # False on HIP-4 -> ladder components run N=1


class VenueAdapter(abc.ABC):
    """Venue boundary abstraction (plan 4.12). HIP-4 port supplies a second
    implementation. Paper mode implements this against the fill simulator.
    """

    @abc.abstractmethod
    def submit_order(self, client_order_id: str, market_id: str, side: Side,
                     price: float, size: float) -> Any:
        """Submit a limit order (idempotent on client_order_id)."""

    @abc.abstractmethod
    def replace_order(self, client_order_id: str, price: float, size: float) -> Any:
        """Replace price/size of an existing order (idempotent on client_order_id)."""

    @abc.abstractmethod
    def cancel_order(self, client_order_id: str) -> Any:
        """Cancel an existing order."""

    @abc.abstractmethod
    def fetch_open_orders(self) -> Any:
        """Return currently open orders (for reconciliation)."""

    @abc.abstractmethod
    def fetch_positions(self) -> Any:
        """Return current venue positions (for reconciliation)."""

    @abc.abstractmethod
    def stream_market_data(self) -> Any:
        """Book snapshots/deltas and trade prints stream."""

    @abc.abstractmethod
    def descriptor(self) -> VenueDescriptor:
        """Static venue descriptor (tick, min size, band, fees, settlement rule)."""


# ---------------------------------------------------------------------------
# 4.13 BankrollState (state_store <-> fair_value_anchor) -- mutable
# ---------------------------------------------------------------------------


@dataclass
class BankrollState:
    model_ids: List[str]  # e.g. ["pricer", "market"]
    bankrolls: Dict[str, float]  # nonnegative, sum normalized
    last_update: datetime
    update_count: int
    frozen: bool  # set on degeneracy (Section 8)

    def __post_init__(self) -> None:
        for k, v in self.bankrolls.items():
            _require_nonneg("bankrolls[" + repr(k) + "]", v)


# ---------------------------------------------------------------------------
# 4.14 SettlementEvent (audit record; inventory mutation travels as a
# SETTLEMENT-tagged pseudo-fill, contract 4.11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlementEvent:
    ts: datetime  # processing time
    settlement_ts: datetime  # the 12:00 ET instant, in UTC
    market_id: str
    expiry_key: str
    strike: float
    outcome: SettlementOutcome
    spot_used: Optional[float]  # USD or None
    spot_source: SpotSource
    q_settled: float  # signed shares
    payoff: Optional[float]  # USDC; None if UNSETTLEABLE
    pnl_realized: Optional[float]  # USDC vs avg_cost; None if UNSETTLEABLE
    excluded_from_gate: bool  # True iff UNSETTLEABLE
