"""Inventory manager (plan Section 2.6, task I1, contract 4.6).

Tracks signed per-contract inventory q (YES-positive) and per-ladder aggregates
through ONE fill channel (MAKER/TAKER/SETTLEMENT alike -- a SETTLEMENT fill is
just a closing fill, no special-casing beyond bookkeeping, per plan 2.13).

avg_cost convention: volume-weighted average price while a position grows in
its CURRENT direction; unchanged while a position is being reduced (realized
PnL is explicitly NOT computed here -- that is settlement/PnL's job); reset to
the fill price when a position opens from flat or flips sign through zero.

q_max deviation from the plan's literal text (documented, not silent): plan
Section 2.6 states "q_max proportional to 1/max(S'(x), eps_cap) (cap shrinks as
p -> 0/1)". Those two clauses contradict for the given defaults (dividing by
S'(x) makes q_max GROW in the wings, since S'(x) -> 0 there), and the task's own
DoD requires q_max(p=0.5) >> q_max(p=0.99). Implemented instead as
q_max = q_max_scale * max(S'(x), s_prime_floor) -- multiplicative, so the cap
shrinks toward the wings as the parenthetical and the DoD both require, with
s_prime_floor still acting as a floor (prevents q_max collapsing to exactly
zero at the clamps). Flagged for plan/owner review.

HedgeState (ladder_hedger output, not yet a Section-4 dataclass) is modeled
here as Dict[market_id, float] -- a vertical offset in shares applied only for
net_band_exposure aggregation, never mutating the raw per-contract q.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from market_maker.config import MMConfig
from market_maker.contracts import ContractInv, Fill, InventoryState, LadderInv, Side
from market_maker.logodds import s_prime

HedgeState = Dict[str, float]  # market_id -> vertical offset shares


@dataclass
class _ContractState:
    q: float = 0.0
    avg_cost: float = 0.0
    fair_x: float = 0.0
    q_max: float = 0.0
    age_weighted_holding: float = 0.0
    position_open_ts: Optional[datetime] = None
    expiry_key: Optional[str] = None
    strike: Optional[float] = None


@dataclass
class _LadderState:
    phi: float = 0.0
    hedge_state: HedgeState = field(default_factory=dict)
    r3_last_ts: Optional[datetime] = None
    r3_last_level: int = 0
    r3_histogram: Dict[int, float] = field(default_factory=dict)


class InventoryManager:
    """Pure in-memory bookkeeping. No sqlite/state-store import (plan 2.6)."""

    def __init__(self, config: Optional[MMConfig] = None) -> None:
        self.config = config if config is not None else MMConfig()
        self._contracts: Dict[str, _ContractState] = {}
        self._ladders: Dict[str, _LadderState] = {}
        self._ladder_strikes: Dict[str, List[Tuple[float, str]]] = {}

    # -- registration / config --------------------------------------------

    def _ensure_ladder(self, expiry_key: str) -> _LadderState:
        ladder = self._ladders.get(expiry_key)
        if ladder is None:
            ladder = _LadderState()
            self._ladders[expiry_key] = ladder
            self._ladder_strikes.setdefault(expiry_key, [])
        return ladder

    def _get_or_create_contract(self, market_id: str) -> _ContractState:
        c = self._contracts.get(market_id)
        if c is None:
            c = _ContractState(q_max=self._q_max_from_x(0.0))
            self._contracts[market_id] = c
        return c

    def register_market(self, market_id: str, expiry_key: str, strike: float) -> None:
        c = self._get_or_create_contract(market_id)
        c.expiry_key = expiry_key
        c.strike = strike
        self._ensure_ladder(expiry_key)
        members = [m for m in self._ladder_strikes[expiry_key] if m[1] != market_id]
        members.append((strike, market_id))
        members.sort(key=lambda t: t[0])
        self._ladder_strikes[expiry_key] = members

    def set_phi(self, expiry_key: str, phi: float) -> None:
        self._ensure_ladder(expiry_key).phi = phi

    def set_hedge_state(self, expiry_key: str, hedge_state: Optional[HedgeState] = None) -> None:
        self._ensure_ladder(expiry_key).hedge_state = dict(hedge_state or {})

    def _q_max_from_x(self, x: float) -> float:
        # Decision D1 (verification pass 2026-07-07): "shrinking" is the
        # conservative default (cap shrinks at the wings); "dalen" is the
        # primary's verbatim form 1/max(S', eps) (cap grows at the wings,
        # bounded by 1/s_prime_floor) -- dormant, selectable via config.
        sp = max(float(s_prime(x, *self.config.p_clamp)), self.config.s_prime_floor)
        if getattr(self.config, "q_max_mode", "shrinking") == "dalen":
            return self.config.q_max_scale / sp
        return self.config.q_max_scale * sp

    def update_fair_x(self, market_id: str, x: float) -> None:
        c = self._get_or_create_contract(market_id)
        c.fair_x = x
        c.q_max = self._q_max_from_x(x)

    # -- fill channel -------------------------------------------------------

    def apply_fill(self, fill: Fill, now: Optional[datetime] = None) -> None:
        ts = now if now is not None else fill.ts
        c = self._get_or_create_contract(fill.market_id)
        sign = 1.0 if fill.side == Side.BUY_YES else -1.0
        delta_q = sign * fill.size
        ek = c.expiry_key
        if ek is not None:
            self._accrue_ladder(ek, ts)
        self._apply_contract_fill(c, delta_q, fill.price, ts)
        if ek is not None:
            self._update_ladder_level(ek)

    def _apply_contract_fill(self, c: _ContractState, delta_q: float, price: float, ts: datetime) -> None:
        old_q = c.q
        new_q = old_q + delta_q

        if old_q == 0.0:
            c.avg_cost = price
        elif new_q == 0.0:
            c.avg_cost = 0.0
        elif (old_q > 0) != (new_q > 0):
            c.avg_cost = price  # flip through zero: reset basis (documented convention)
        elif (delta_q > 0) == (old_q > 0):
            c.avg_cost = (abs(old_q) * c.avg_cost + abs(delta_q) * price) / (abs(old_q) + abs(delta_q))
        # else: reducing toward (not through) zero -> avg_cost unchanged
        c.q = new_q

        if old_q == 0.0 and new_q != 0.0:
            c.position_open_ts = ts
            c.age_weighted_holding = 0.0
        elif new_q == 0.0:
            if c.position_open_ts is not None:
                c.age_weighted_holding = (ts - c.position_open_ts).total_seconds() / 3600.0
            c.position_open_ts = None
        elif (old_q > 0) != (new_q > 0):
            c.position_open_ts = ts
            c.age_weighted_holding = 0.0
        else:
            if c.position_open_ts is not None:
                c.age_weighted_holding = (ts - c.position_open_ts).total_seconds() / 3600.0

    def mark(self, now: datetime) -> None:
        """Heartbeat: attribute elapsed time to R3/age without any fill."""
        for ek in list(self._ladders.keys()):
            self._accrue_ladder(ek, now)
            self._update_ladder_level(ek)
        for c in self._contracts.values():
            if c.q != 0.0 and c.position_open_ts is not None:
                c.age_weighted_holding = (now - c.position_open_ts).total_seconds() / 3600.0

    # -- R3 lifetime-inventory metric ---------------------------------------

    def _ladder_gross(self, expiry_key: str) -> float:
        members = self._ladder_strikes.get(expiry_key, [])
        return float(sum(abs(self._contracts[m].q) for _, m in members if m in self._contracts))

    def _accrue_ladder(self, expiry_key: str, ts: datetime) -> None:
        ladder = self._ensure_ladder(expiry_key)
        if ladder.r3_last_ts is not None:
            elapsed_h = (ts - ladder.r3_last_ts).total_seconds() / 3600.0
            if elapsed_h > 0.0:
                ladder.r3_histogram[ladder.r3_last_level] = ladder.r3_histogram.get(ladder.r3_last_level, 0.0) + elapsed_h
        ladder.r3_last_ts = ts

    def _update_ladder_level(self, expiry_key: str) -> None:
        ladder = self._ensure_ladder(expiry_key)
        ladder.r3_last_level = int(round(self._ladder_gross(expiry_key)))

    # -- caps -----------------------------------------------------------------

    def cap_breached(self, market_id: str) -> bool:
        c = self._contracts.get(market_id)
        if c is None:
            return False
        return abs(c.q) > c.q_max

    def breaches(self) -> List[str]:
        return [m for m in self._contracts if self.cap_breached(m)]

    # -- ladder aggregation ---------------------------------------------------

    def net_band_exposure(self, expiry_key: str) -> List[float]:
        """Cumulative-sum bucket decomposition (N strikes -> N+1 buckets,
        including the two open tails, plan 2.3/2.6). A YES position at strike
        K_i pays in every bucket b >= i, so bucket b's exposure is the
        cumulative sum of effective q over strikes 1..b; bucket 0 (below the
        lowest strike) is always 0 by construction.
        """
        members = self._ladder_strikes.get(expiry_key, [])
        ladder = self._ladders.get(expiry_key)
        hedge_state = ladder.hedge_state if ladder is not None else {}
        q_eff = []
        for _strike, market_id in members:
            c = self._contracts.get(market_id)
            q = c.q if c is not None else 0.0
            q_eff.append(q + hedge_state.get(market_id, 0.0))
        if not q_eff:
            return [0.0]
        cum = np.cumsum(np.asarray(q_eff, dtype=float))
        return [0.0] + [float(v) for v in cum]

    # -- snapshot ---------------------------------------------------------------

    def snapshot(self, ts: datetime) -> InventoryState:
        per_contract: Dict[str, ContractInv] = {}
        for market_id, c in self._contracts.items():
            per_contract[market_id] = ContractInv(
                q=c.q,
                avg_cost=c.avg_cost,
                q_max=c.q_max,
                age_weighted_holding=c.age_weighted_holding,
            )
        per_ladder: Dict[str, LadderInv] = {}
        for ek, ladder in self._ladders.items():
            per_ladder[ek] = LadderInv(
                net_band_exposure=self.net_band_exposure(ek),
                gross=self._ladder_gross(ek),
                phi=ladder.phi,
                r3_histogram=dict(ladder.r3_histogram),
            )
        return InventoryState(ts=ts, per_contract=per_contract, per_ladder=per_ladder)

    # -- persistence hooks (state store reads/writes these; no sqlite here) ---

    def to_rows(self) -> Dict[str, List[Dict[str, Any]]]:
        inventory_rows = []
        for market_id, c in self._contracts.items():
            inventory_rows.append({
                "market_id": market_id,
                "expiry_key": c.expiry_key,
                "strike": c.strike,
                "q": c.q,
                "avg_cost": c.avg_cost,
                "fair_x": c.fair_x,
                "q_max": c.q_max,
                "age_weighted_holding": c.age_weighted_holding,
                "position_open_ts": c.position_open_ts.isoformat() if c.position_open_ts else None,
            })
        ladder_rows = []
        for ek, ladder in self._ladders.items():
            ladder_rows.append({
                "expiry_key": ek,
                "phi": ladder.phi,
                "hedge_state": dict(ladder.hedge_state),
                "r3_last_ts": ladder.r3_last_ts.isoformat() if ladder.r3_last_ts else None,
                "r3_last_level": ladder.r3_last_level,
                "r3_histogram": {str(k): v for k, v in ladder.r3_histogram.items()},
                "members": [[strike, mid] for strike, mid in self._ladder_strikes.get(ek, [])],
            })
        return {"inventory": inventory_rows, "ladder_state": ladder_rows}

    @classmethod
    def from_rows(cls, rows: Dict[str, List[Dict[str, Any]]], config: Optional[MMConfig] = None) -> "InventoryManager":
        mgr = cls(config)
        for row in rows.get("ladder_state", []):
            ek = row["expiry_key"]
            ladder = mgr._ensure_ladder(ek)
            ladder.phi = row.get("phi", 0.0)
            ladder.hedge_state = dict(row.get("hedge_state", {}) or {})
            r3_last_ts = row.get("r3_last_ts")
            ladder.r3_last_ts = datetime.fromisoformat(r3_last_ts) if r3_last_ts else None
            ladder.r3_last_level = row.get("r3_last_level", 0)
            ladder.r3_histogram = {int(k): v for k, v in (row.get("r3_histogram") or {}).items()}
            mgr._ladder_strikes[ek] = [(strike, mid) for strike, mid in row.get("members", [])]
        for row in rows.get("inventory", []):
            c = _ContractState(
                q=row.get("q", 0.0),
                avg_cost=row.get("avg_cost", 0.0),
                fair_x=row.get("fair_x", 0.0),
                q_max=row.get("q_max", 0.0),
                age_weighted_holding=row.get("age_weighted_holding", 0.0),
                expiry_key=row.get("expiry_key"),
                strike=row.get("strike"),
            )
            pos_ts = row.get("position_open_ts")
            c.position_open_ts = datetime.fromisoformat(pos_ts) if pos_ts else None
            mgr._contracts[row["market_id"]] = c
        return mgr
