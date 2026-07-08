"""Liquidity monitor (plan Section 2.9, task M1, contract 4.9).

Computed purely from the MarketState stream (market_data_client, 2.14):
realized depth near touch, an impact-magnitude ("Kyle's lambda") estimate,
YES+NO arb-deviation half-life, headline-volume discount, and a liquidity
regime tag. NO order-flow direction signal is built anywhere in this module
(plan Finding 10: public-feed direction accuracy is ~59%, not worth the risk
of building toxicity metrics on it) -- the lambda estimate below is an
UNSIGNED impact-magnitude proxy (|dMid| regressed on unsigned trade size),
explicitly not a signed flow indicator.

Kyle's lambda estimator: rolling regression of |dMid| on unsigned trade size,
forced through the origin (lambda = sum(size*|dMid|) / sum(size^2)) -- exact
recovery when |dMid| = c*size with no noise, NaN before min_obs observations.

Arb half-life estimator: given a rolling series of |deviation| = |YES_mid +
NO_mid - 1|, the lag-1 "autocorrelation" coefficient is estimated the same
regression-through-origin way (phi = sum(x_t*x_{t+1}) / sum(x_t^2)), which is
the AR(1)/Ornstein-Uhlenbeck coefficient exp(-dt/tau); half_life = tau*ln(2)
recovered as -ln(2)*dt/ln(phi). NaN when phi is not in (0,1) (no decay) or
before min_obs observations or when no paired series has been supplied.
"""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from market_maker.config import MMConfig
from market_maker.contracts import LiquidityRegime, LiquidityState, MarketState

# Regime thresholds on combined (bid+ask) realized depth, shares (launch
# defaults, pending Stage A/B calibration -- no fill/depth history exists yet).
DEFAULT_REGIME_THRESHOLDS: Dict[str, float] = {
    "thick_depth": 500.0,
    "normal_depth": 100.0,
    "degenerate_floor": 5.0,
}


def _depth_within_ticks(levels, touch: Optional[float], ticks: int, tick_size: float, side: str) -> float:
    if touch is None or not levels:
        return 0.0
    band = ticks * tick_size
    total = 0.0
    for price, size in levels:
        if side == "bid" and price >= touch - band:
            total += size
        elif side == "ask" and price <= touch + band:
            total += size
    return float(total)


class LiquidityMonitor:
    """Per-market liquidity gauges, updated from a MarketState stream."""

    def __init__(
        self,
        config: Optional[MMConfig] = None,
        depth_ticks: int = 3,
        tick_size: float = 0.01,
        depth_window: int = 20,
        lambda_window: int = 200,
        min_obs: int = 30,
        regime_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config = config if config is not None else MMConfig()
        self.depth_ticks = depth_ticks
        self.tick_size = tick_size
        self.min_obs = min_obs

        self._depth_bid_hist: Deque[float] = deque(maxlen=depth_window)
        self._depth_ask_hist: Deque[float] = deque(maxlen=depth_window)
        self._lambda_obs: Deque[Tuple[float, float]] = deque(maxlen=lambda_window)
        self._prev_mid: Optional[float] = None
        self._dev_hist: Deque[Tuple[datetime, float]] = deque(maxlen=lambda_window)

        self.regime_thresholds: Dict[str, float] = dict(DEFAULT_REGIME_THRESHOLDS)
        if regime_thresholds:
            self.regime_thresholds.update(regime_thresholds)

        self._last_state: Optional[MarketState] = None

    # -- ingestion --------------------------------------------------------------

    def update(self, state: MarketState) -> None:
        """Feed one MarketState: updates depth history and the impact-lambda
        rolling sample from its (already drained-into-this-message) prints.
        """
        depth_bid = _depth_within_ticks(state.bid_depth, state.best_bid, self.depth_ticks, self.tick_size, "bid")
        depth_ask = _depth_within_ticks(state.ask_depth, state.best_ask, self.depth_ticks, self.tick_size, "ask")
        self._depth_bid_hist.append(depth_bid)
        self._depth_ask_hist.append(depth_ask)

        mid = None
        if state.best_bid is not None and state.best_ask is not None:
            mid = 0.5 * (state.best_bid + state.best_ask)
        total_size = float(sum(sz for (_ts, _p, sz) in state.last_prints))
        if mid is not None and self._prev_mid is not None and total_size > 0.0:
            dmid = abs(mid - self._prev_mid)
            self._lambda_obs.append((total_size, dmid))
        if mid is not None:
            self._prev_mid = mid

        self._last_state = state

    def update_pair(self, ts: datetime, deviation: float) -> None:
        """Feed one YES+NO arb deviation sample (YES_mid + NO_mid - 1).

        Stored SIGNED: Polymarket carries a structural fee wedge, so YES+NO
        does not converge to exactly 1 -- the deviation has a persistent
        nonzero baseline. The half-life estimator demeans against the rolling
        baseline and measures shock decay of the residual.
        """
        self._dev_hist.append((ts, float(deviation)))

    # -- gauges -----------------------------------------------------------------

    def realized_depth(self) -> Tuple[float, float]:
        bid = float(np.mean(self._depth_bid_hist)) if self._depth_bid_hist else 0.0
        ask = float(np.mean(self._depth_ask_hist)) if self._depth_ask_hist else 0.0
        return bid, ask

    def kyle_lambda(self) -> float:
        if len(self._lambda_obs) < self.min_obs:
            return float("nan")
        sizes = np.array([o[0] for o in self._lambda_obs], dtype=float)
        dmids = np.array([o[1] for o in self._lambda_obs], dtype=float)
        denom = float(np.sum(sizes * sizes))
        if denom <= 0.0:
            return float("nan")
        return float(np.sum(sizes * dmids) / denom)

    def arb_halflife_s(self) -> float:
        if len(self._dev_hist) < self.min_obs:
            return float("nan")
        ts_list = [t for t, _ in self._dev_hist]
        vals = np.array([v for _, v in self._dev_hist], dtype=float)
        # AR(1) WITH intercept: x_{t+1} = c + phi*x_t. The intercept absorbs
        # the structural fee wedge (YES+NO does not settle at exactly 1), so
        # phi measures shock decay toward the wedge baseline, not toward zero.
        # For deterministic decay toward any baseline the slope is exact.
        x, y = vals[:-1], vals[1:]
        mx, my = float(np.mean(x)), float(np.mean(y))
        denom = float(np.sum((x - mx) * (x - mx)))
        if denom <= 0.0:
            return float("nan")
        phi = float(np.sum((x - mx) * (y - my)) / denom)
        if not (0.0 < phi < 1.0):
            return float("nan")
        dts = [(ts_list[i + 1] - ts_list[i]).total_seconds() for i in range(len(ts_list) - 1)]
        dt = float(np.mean(dts)) if dts else float("nan")
        if not dts or dt <= 0.0 or math.isnan(dt):
            return float("nan")
        return -math.log(2.0) * dt / math.log(phi)

    def discount_volume(self, nominal: float) -> float:
        """Per plan 2.9: headline volume divided by ~2.5 (Tsang-Yang mint/burn
        inflation), MMConfig.volume_discount.
        """
        return float(nominal) / self.config.volume_discount

    def regime(
        self,
        depth_bid: float,
        depth_ask: float,
        feed_healthy: bool,
        best_bid: Optional[float],
        best_ask: Optional[float],
    ) -> LiquidityRegime:
        if not feed_healthy:
            return LiquidityRegime.DEGENERATE
        if best_bid is None or best_ask is None:  # empty or one-sided book
            return LiquidityRegime.DEGENERATE
        combined = depth_bid + depth_ask
        if combined < self.regime_thresholds["degenerate_floor"]:
            return LiquidityRegime.DEGENERATE
        if combined >= self.regime_thresholds["thick_depth"]:
            return LiquidityRegime.THICK
        if combined >= self.regime_thresholds["normal_depth"]:
            return LiquidityRegime.NORMAL
        return LiquidityRegime.THIN

    # -- emission -----------------------------------------------------------------

    def emit(self, window: str = "default") -> LiquidityState:
        if self._last_state is None:
            raise ValueError("emit() called before any update(state)")
        state = self._last_state
        depth_bid, depth_ask = self.realized_depth()
        reg = self.regime(depth_bid, depth_ask, state.feed_healthy, state.best_bid, state.best_ask)
        return LiquidityState(
            ts=state.ts,
            market_id=state.market_id,
            realized_depth_bid=depth_bid,
            realized_depth_ask=depth_ask,
            kyle_lambda=self.kyle_lambda(),
            arb_halflife_s=self.arb_halflife_s(),
            regime=reg,
            window=window,
            vol_discount=self.config.volume_discount,
        )
