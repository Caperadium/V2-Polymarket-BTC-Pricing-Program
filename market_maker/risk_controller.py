"""Risk controller / pull-quote switch (plan 2.10, task R1, contract 4.10).

Final authority over quote MODE per market: {TWO_SIDED, BID_ONLY, ASK_ONLY,
PULLED}. Owns the kill switch. Reuses the existing BTC vol gate
(core/strategy/vol_gate.py) as the jump/vol signal via an INJECTABLE callable so
tests can stub it -- it is never reimplemented here.

Trigger rules (plan 2.10 a-f), each producing a required mode plus eps_add /
kelly_mult / cancel_all contributions; the tick's directive is the combination
with the MOST RESTRICTIVE mode, summed eps_add, min kelly_mult, any cancel_all:
  (a) vol_gate shock or regime extreme -> PULLED; regime high -> widen
      (eps_add += edge_add_cents/100) + kelly_mult passthrough.
  (b) tte < near_resolution_pull_hours/24 days -> PULLED; spot within a
      threshold of strike while vol elevated (gap-through) -> PULLED.
  (c) inventory cap breach -> one-sided quoting AWAY from the breach (long/YES
      breach -> ASK_ONLY to sell down; short breach -> BID_ONLY to cover);
      extreme breach (ratio > 1.5x cap) -> PULLED.
  (d) feed stale/unhealthy -> PULLED + cancel_all=True (mandatory).
  (e) pricer snapshot stale -> widen first (eps_add += pricer_stale_eps_add);
      older than 2x max age -> PULLED.
  (f) liquidity DEGENERATE -> PULLED.
  manual override -> PULLED (MANUAL trigger).

Any transition INTO a restrictive mode latches for latch_seconds (default 60s);
transitions OUT require the latch expired AND the trigger cleared (no flapping).
Every mode transition is journaled. stdlib + numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

from market_maker.config import MMConfig
from market_maker.contracts import (
    ConfidenceTier,
    LiquidityRegime,
    PricerSnapshot,
    QuoteMode,
    RiskDirective,
    RiskTrigger,
)

_DEFAULT_LATCH_SECONDS = 60.0  # module default when config lacks it (plan 2.10)
_DEFAULT_GAP_STRIKE_FRAC = 0.005  # |spot-strike|/strike gap-through band (open Q8)
_DEFAULT_PRICER_STALE_EPS_ADD = 0.01  # widen on first-stage pricer staleness

# Mode restrictiveness rank (higher = more restrictive).
_RANK = {
    QuoteMode.TWO_SIDED: 0,
    QuoteMode.BID_ONLY: 1,
    QuoteMode.ASK_ONLY: 1,
    QuoteMode.PULLED: 2,
}


@dataclass
class InvBreach:
    """A per-market inventory cap breach handed to the controller.

    is_long True means excess YES (q > 0) -> reduce by quoting asks only.
    ratio = |q| / q_max (>= 1 is a breach; > 1.5 is extreme).
    """
    market_id: str
    is_long: bool
    ratio: float


def _more_restrictive(a: QuoteMode, b: QuoteMode) -> QuoteMode:
    """Combine two required modes: the more restrictive wins; two opposite
    one-sided requirements escalate to PULLED (cannot safely quote either)."""
    if a == b:
        return a
    ra, rb = _RANK[a], _RANK[b]
    if ra != rb:
        return a if ra > rb else b
    # equal rank, different one-sided sides -> pull (conservative)
    return QuoteMode.PULLED


def default_vol_gate_fn(btc_df, now_utc) -> Callable[[], object]:
    """Build a zero-arg vol-gate callable bound to data + time (reuses the
    existing module; imported lazily so the controller import stays light)."""
    from core.strategy.vol_gate import compute_vol_gate

    def _fn():
        return compute_vol_gate(btc_df, now_utc)

    return _fn


class RiskController:
    def __init__(self, config: Optional[MMConfig] = None,
                 vol_gate_fn: Optional[Callable[[], object]] = None,
                 latch_seconds: Optional[float] = None,
                 gap_strike_frac: float = _DEFAULT_GAP_STRIKE_FRAC,
                 pricer_stale_eps_add: float = _DEFAULT_PRICER_STALE_EPS_ADD) -> None:
        self._cfg = config or MMConfig()
        self._vol_gate_fn = vol_gate_fn
        self._latch_seconds = (
            latch_seconds if latch_seconds is not None
            else getattr(self._cfg, "risk_latch_seconds", _DEFAULT_LATCH_SECONDS)
        )
        self._gap_strike_frac = gap_strike_frac
        self._pricer_stale_eps_add = pricer_stale_eps_add
        # per-market latch state
        self._mode: dict = {}
        self._latched_until: dict = {}
        self._latched_mode: dict = {}
        self._journal: List[Tuple[datetime, str, QuoteMode, QuoteMode, List[RiskTrigger]]] = []

    # -- evaluation ------------------------------------------------------

    def evaluate(self, market_id: str, now: datetime, *,
                 tte_days: float,
                 pricer_snapshot: PricerSnapshot,
                 inventory_breaches: Optional[List[InvBreach]] = None,
                 liquidity_regime: LiquidityRegime = LiquidityRegime.NORMAL,
                 feed_healthy: bool = True,
                 spot: Optional[float] = None,
                 strike: Optional[float] = None,
                 manual_override: bool = False,
                 vol_gate_result: Optional[object] = None) -> RiskDirective:
        """Produce the RiskDirective for one market at one tick."""
        vg = vol_gate_result
        if vg is None and self._vol_gate_fn is not None:
            vg = self._vol_gate_fn()

        triggers: List[RiskTrigger] = []
        req_mode = QuoteMode.TWO_SIDED
        eps_add = 0.0
        cancel_all = False
        # kelly_mult always passes through from the vol gate (contract 4.10).
        kelly_mult = 1.0
        vol_elevated = False

        if vg is not None:
            kelly_mult = float(getattr(vg, "kelly_mult", 1.0))
            regime = str(getattr(vg, "regime", "normal"))
            shock = bool(getattr(vg, "shock", False))
            vol_elevated = shock or regime in ("high", "extreme")
            # (a) vol gate
            if shock or regime == "extreme":
                req_mode = _more_restrictive(req_mode, QuoteMode.PULLED)
                triggers.append(RiskTrigger.SPOT_JUMP)
            elif regime == "high":
                eps_add += max(0.0, float(getattr(vg, "edge_add_cents", 0.0))) / 100.0
                triggers.append(RiskTrigger.SPOT_JUMP)

        # (b) near resolution + gap-through-strike
        if tte_days < self._cfg.near_resolution_pull_hours / 24.0:
            req_mode = _more_restrictive(req_mode, QuoteMode.PULLED)
            triggers.append(RiskTrigger.NEAR_RESOLUTION)
        if (vol_elevated and spot is not None and strike is not None
                and strike > 0
                and abs(spot - strike) / strike < self._gap_strike_frac):
            req_mode = _more_restrictive(req_mode, QuoteMode.PULLED)
            triggers.append(RiskTrigger.SPOT_GAPPING_STRIKE)

        # (c) inventory cap breach
        breach = self._breach_for(market_id, inventory_breaches)
        if breach is not None:
            if breach.ratio > 1.5:
                req_mode = _more_restrictive(req_mode, QuoteMode.PULLED)
            else:
                side_mode = QuoteMode.ASK_ONLY if breach.is_long else QuoteMode.BID_ONLY
                req_mode = _more_restrictive(req_mode, side_mode)
            triggers.append(RiskTrigger.INV_CAP)

        # (d) feed stale / unhealthy -- mandatory pull + cancel-all
        if not feed_healthy:
            req_mode = _more_restrictive(req_mode, QuoteMode.PULLED)
            cancel_all = True
            triggers.append(RiskTrigger.FEED_STALE)

        # (e) pricer snapshot staleness
        age = (now - pricer_snapshot.ts).total_seconds()
        max_age = self._cfg.pricer_max_age_s
        if age > 2.0 * max_age:
            req_mode = _more_restrictive(req_mode, QuoteMode.PULLED)
            triggers.append(RiskTrigger.PRICER_STALE)
        elif age > max_age:
            eps_add += self._pricer_stale_eps_add
            triggers.append(RiskTrigger.PRICER_STALE)

        # (f) liquidity regime degenerate
        if liquidity_regime == LiquidityRegime.DEGENERATE:
            req_mode = _more_restrictive(req_mode, QuoteMode.PULLED)
            triggers.append(RiskTrigger.LIQ_DEGENERATE)

        # manual override
        if manual_override:
            req_mode = _more_restrictive(req_mode, QuoteMode.PULLED)
            triggers.append(RiskTrigger.MANUAL)

        effective, latched_until = self._apply_hysteresis(market_id, req_mode, now)

        # cancel_all whenever we end up PULLED (pulling means clear resting
        # quotes); feed-loss guarantees it above regardless.
        if effective == QuoteMode.PULLED:
            cancel_all = True

        self._record_transition(market_id, now, effective, triggers)

        return RiskDirective(
            ts=now, market_id=market_id, mode=effective,
            eps_add=max(0.0, eps_add),
            kelly_mult=min(1.0, max(0.0, kelly_mult)),
            triggers=triggers,
            latched_until=latched_until,
            cancel_all=cancel_all,
        )

    # -- hysteresis ------------------------------------------------------

    def _apply_hysteresis(self, market_id: str, req_mode: QuoteMode,
                          now: datetime) -> Tuple[QuoteMode, datetime]:
        latched_until = self._latched_until.get(market_id)
        latched_mode = self._latched_mode.get(market_id, QuoteMode.TWO_SIDED)
        latch_active = latched_until is not None and now < latched_until

        if latch_active:
            # Hold at least the latched mode; escalation re-arms the latch.
            effective = _more_restrictive(req_mode, latched_mode)
            if _RANK[effective] > _RANK[latched_mode] or (
                    effective != latched_mode and _RANK[effective] == _RANK[latched_mode]):
                latched_mode = effective
                latched_until = now + timedelta(seconds=self._latch_seconds)
        else:
            # Latch expired (or none): the raw requirement takes effect.
            effective = req_mode
            if _RANK[effective] > 0:
                latched_mode = effective
                latched_until = now + timedelta(seconds=self._latch_seconds)
            else:
                latched_mode = QuoteMode.TWO_SIDED
                latched_until = None

        self._latched_mode[market_id] = latched_mode
        self._latched_until[market_id] = latched_until
        out_latched_until = latched_until if latched_until is not None else now
        return effective, out_latched_until

    def _breach_for(self, market_id: str,
                    breaches: Optional[List[InvBreach]]) -> Optional[InvBreach]:
        if not breaches:
            return None
        worst = None
        for b in breaches:
            if b.market_id != market_id or b.ratio < 1.0:
                continue
            if worst is None or b.ratio > worst.ratio:
                worst = b
        return worst

    def _record_transition(self, market_id: str, now: datetime,
                           new_mode: QuoteMode, triggers: List[RiskTrigger]) -> None:
        old = self._mode.get(market_id, QuoteMode.TWO_SIDED)
        if new_mode != old:
            self._journal.append((now, market_id, old, new_mode, list(triggers)))
        self._mode[market_id] = new_mode

    # -- accessors -------------------------------------------------------

    def journal(self) -> List[Tuple[datetime, str, QuoteMode, QuoteMode, List[RiskTrigger]]]:
        return list(self._journal)

    def current_mode(self, market_id: str) -> QuoteMode:
        return self._mode.get(market_id, QuoteMode.TWO_SIDED)
