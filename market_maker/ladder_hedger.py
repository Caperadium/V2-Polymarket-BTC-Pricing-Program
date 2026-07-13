"""Ladder hedger (plan Section 2.7, tasks L1 + L2).

Operates one expiry strike ladder as a single object (the risk-neutral CDF):

  (a) No-arb enforcement (L1): the quoted digital prices must be monotonically
      non-increasing in strike and the implied density (adjacent mid differences)
      non-negative. check() reports violations; repair() applies an isotonic
      pool-adjacent-violators (PAV) projection of the mid ladder onto the
      non-increasing cone (minimal L2 adjustment, toward the model CDF), then
      rebuilds bid/ask around the repaired mids preserving each contract's
      half-spread. Repair is an L2 projection and therefore idempotent.
  (b) Vertical-spread internal hedge (L1): when |q| in a strike exceeds a target
      fraction of its cap, emit a HedgeRecommendation to take the OPPOSITE side
      in an adjacent strike (converting naked binary risk into bounded band
      exposure). Also returns HedgeState (vertical offsets per (expiry_key,
      bucket)) -- an AUDIT-ONLY view that must never cross a module boundary;
      the module-level `hedge_offsets_by_market()` helper below builds the
      market_id-keyed offsets the inventory manager actually consumes,
      straight from the HedgeRecommendation list (plan W2.0).
  (c) Cross-strike beta hedge (L2, behind enable_beta_hedge, default False):
      instantaneous hedge ratio beta(i<-j), shrunk as S'(x) -> 0 and hard-clamped
      to +/- beta_max, zero outside the p-clamp band (risk 8.5 -- no explosive
      wing hedges).

Depends only on market_maker.contracts, market_maker.config, market_maker.logodds
and stdlib + numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from market_maker import logodds
from market_maker.config import MMConfig
from market_maker.contracts import (
    HedgeReason,
    HedgeRecommendation,
    InventoryState,
    QuoteSet,
    Side,
)

# Numeric tolerance for monotonicity/density violation detection (price units).
_ARB_TOL = 1e-9


@dataclass(frozen=True)
class NoArbVerdict:
    """Result of the pre-flight no-arb check on a quote-set ladder."""

    ok: bool
    violations: List[str]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _quantize(price: float, tick: float) -> float:
    """Round price to the nearest tick and clamp into the [0, 1] venue band."""
    if not math.isfinite(price):
        return 0.0
    if tick <= 0.0:
        q = price
    else:
        q = round(price / tick) * tick
    return min(1.0, max(0.0, q))


def _pav_nonincreasing(values: List[float]) -> List[float]:
    """Isotonic L2 projection of `values` onto the non-increasing cone via
    pool-adjacent-violators (unit weights). Idempotent: an already
    non-increasing sequence is returned unchanged.
    """
    # Negate so the constraint becomes non-decreasing, run standard PAV, negate
    # back. Each block carries (mean, weight, count).
    neg = [-float(v) for v in values]
    blocks: List[List[float]] = []
    for v in neg:
        blocks.append([v, 1.0, 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            nw = w1 + w2
            blocks.append([(m1 * w1 + m2 * w2) / nw, nw, c1 + c2])
    out: List[float] = []
    for mean, _w, cnt in blocks:
        out.extend([-mean] * int(cnt))
    return out


# ---------------------------------------------------------------------------
# LadderHedger
# ---------------------------------------------------------------------------


@dataclass
class LadderHedger:
    """Stateful over a persistable violation journal; otherwise pure per call."""

    config: MMConfig = field(default_factory=MMConfig)
    repair_or_reject: str = "repair"  # "repair" (default) or "reject"
    tick: float = 0.01  # venue price tick (Polymarket cent tick default)
    vertical_target_frac: float = 0.5  # emit when |q| > frac * q_max
    hedge_ttl_seconds: float = 300.0
    enable_beta_hedge: bool = False  # L2 flag, default OFF (risk 8.5)
    journal: List[dict] = field(default_factory=list)
    # Count of ladders that arrived violating no-arb (whether repaired or
    # rejected). The heartbeat reads this; the journal alone is in-memory
    # detail that never leaves the process.
    repair_count: int = 0

    # -- (a) no-arb -------------------------------------------------------

    def _mids(self, quote_sets: List[QuoteSet]) -> List[float]:
        return [0.5 * (qs.bid_price + qs.ask_price) for qs in quote_sets]

    def check(
        self,
        quote_sets: List[QuoteSet],
        strikes: List[float],
    ) -> NoArbVerdict:
        """Check bid/ask ladders are non-increasing in strike and the implied
        density (adjacent mid differences) is non-negative. Pure (no journal
        mutation)."""
        violations: List[str] = []
        n = len(quote_sets)
        if n != len(strikes):
            violations.append(
                "length_mismatch: %d quote_sets vs %d strikes" % (n, len(strikes))
            )
            return NoArbVerdict(False, violations)
        mids = self._mids(quote_sets)
        for i in range(n - 1):
            k0, k1 = strikes[i], strikes[i + 1]
            if quote_sets[i + 1].bid_price > quote_sets[i].bid_price + _ARB_TOL:
                violations.append(
                    "bid_monotonicity: bid(K=%s)=%.6f < bid(K=%s)=%.6f"
                    % (k1, quote_sets[i + 1].bid_price, k0, quote_sets[i].bid_price)
                )
            if quote_sets[i + 1].ask_price > quote_sets[i].ask_price + _ARB_TOL:
                violations.append(
                    "ask_monotonicity: ask(K=%s)=%.6f < ask(K=%s)=%.6f"
                    % (k1, quote_sets[i + 1].ask_price, k0, quote_sets[i].ask_price)
                )
            if mids[i] - mids[i + 1] < -_ARB_TOL:
                violations.append(
                    "negative_density: mid(K=%s)=%.6f mid(K=%s)=%.6f (diff %.6f<0)"
                    % (k0, mids[i], k1, mids[i + 1], mids[i] - mids[i + 1])
                )
        return NoArbVerdict(len(violations) == 0, violations)

    def repair(
        self,
        quote_sets: List[QuoteSet],
        strikes: List[float],
        model_cdf: Dict[float, float],
        expiry_key: Optional[str] = None,
    ) -> Optional[List[QuoteSet]]:
        """Enforce no-arb. If clean, mark noarb_checked and return unchanged.
        Otherwise, in "repair" mode apply the PAV isotonic projection toward the
        model CDF and rebuild; in "reject" mode journal and return None.
        Idempotent: repair(repair(L)) == repair(L).
        """
        verdict = self.check(quote_sets, strikes)
        if verdict.ok:
            return [replace(qs, noarb_checked=True) for qs in quote_sets]

        if self.repair_or_reject == "reject":
            self.repair_count += 1
            self.journal.append(
                {
                    "event": "reject",
                    "expiry_key": expiry_key,
                    "violations": list(verdict.violations),
                }
            )
            return None

        # Repair: PAV on mids, substituting the model CDF for any non-finite mid
        # (the model CDF supplies the monotone reference the projection targets).
        mids: List[float] = []
        for qs, k in zip(quote_sets, strikes):
            m = 0.5 * (qs.bid_price + qs.ask_price)
            if not math.isfinite(m):
                m = float(model_cdf.get(k, 0.5))
            mids.append(m)
        repaired_mids = _pav_nonincreasing(mids)

        out: List[QuoteSet] = []
        for qs, new_mid in zip(quote_sets, repaired_mids):
            hs = 0.5 * (qs.ask_price - qs.bid_price)  # preserved half-spread
            new_bid = _quantize(new_mid - hs, self.tick)
            new_ask = _quantize(new_mid + hs, self.tick)
            out.append(replace(qs, bid_price=new_bid, ask_price=new_ask, noarb_checked=True))

        self.repair_count += 1
        self.journal.append(
            {
                "event": "repair",
                "expiry_key": expiry_key,
                "violations": list(verdict.violations),
            }
        )
        return out

    # -- (b) vertical-spread internal hedge -------------------------------

    def _pick_neighbor(
        self,
        i: int,
        strikes: List[float],
        market_ids: List[str],
        depth_hint: Optional[Dict[str, float]],
    ) -> int:
        """Adjacent strike index to hedge into. Prefer the better-liquidity
        neighbor when a depth hint is given, else the strike-nearest neighbor;
        deterministic tie-break to the lower strike (i-1)."""
        cands = [j for j in (i - 1, i + 1) if 0 <= j < len(strikes)]
        if not cands:
            return -1
        if len(cands) == 1:
            return cands[0]
        if depth_hint is not None:
            d = {j: float(depth_hint.get(market_ids[j], 0.0)) for j in cands}
            best = max(cands, key=lambda j: d[j])
            if d[cands[0]] != d[cands[1]]:
                return best
        # nearest by strike distance, tie -> lower strike (i-1)
        dist = {j: abs(strikes[j] - strikes[i]) for j in cands}
        if dist[i - 1] <= dist[i + 1]:
            return i - 1
        return i + 1

    def vertical_hedges(
        self,
        inventory: InventoryState,
        expiry_key: str,
        strikes: List[float],
        market_ids: List[str],
        fair_p: Dict[str, float],
        ts: datetime,
        depth_hint: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[HedgeRecommendation], Dict[Tuple[str, int], float]]:
        """Emit vertical-offset hedges for strikes whose |q| exceeds
        vertical_target_frac * q_max. Returns (recommendations, hedge_state),
        where hedge_state maps (expiry_key, bucket_index) -> signed YES-equivalent
        offset shares the inventory manager consumes on the next tick.
        """
        recs: List[HedgeRecommendation] = []
        hedge_state: Dict[Tuple[str, int], float] = {}
        expires = ts + timedelta(seconds=self.hedge_ttl_seconds)
        for i, m in enumerate(market_ids):
            inv = inventory.per_contract.get(m)
            if inv is None:
                continue
            q, q_max = inv.q, inv.q_max
            target = self.vertical_target_frac * q_max
            excess = abs(q) - target
            if excess <= 0.0:
                continue
            j = self._pick_neighbor(i, strikes, market_ids, depth_hint)
            if j < 0:
                continue
            nbr_m = market_ids[j]
            nbr_inv = inventory.per_contract.get(nbr_m)
            nbr_cap = nbr_inv.q_max if nbr_inv is not None else excess
            size = min(excess, nbr_cap)
            if size <= 0.0:
                continue
            side = Side.BUY_NO if q > 0 else Side.BUY_YES
            fair = float(fair_p.get(nbr_m, 0.5))
            side_price = fair if side == Side.BUY_YES else 1.0 - fair
            max_price = _quantize(side_price + self.tick, self.tick)  # passive-preferred ceiling
            recs.append(
                HedgeRecommendation(
                    ts=ts,
                    expiry_key=expiry_key,
                    target_market_id=nbr_m,
                    side=side,
                    size=size,
                    max_price=max_price,
                    reason=HedgeReason.VERTICAL_OFFSET,
                    paired_market_id=m,
                    beta=None,
                    expires=expires,
                )
            )
            bucket = min(i, j)  # inter-strike bucket between strike[bucket], strike[bucket+1]
            signed = size if side == Side.BUY_YES else -size
            key = (expiry_key, bucket)
            hedge_state[key] = hedge_state.get(key, 0.0) + signed
        return recs, hedge_state

    # -- (c) cross-strike beta hedge (L2) --------------------------------

    def beta_ratio(
        self,
        p_i: float,
        p_j: float,
        sigma_bj: float,
        rho: float = 1.0,
        sigma_bi: Optional[float] = None,
    ) -> float:
        """Cross-strike instantaneous hedge ratio, per Dalen Section 4.4
        (verified against arXiv 2510.15205v2):

            beta(i<-j) = Cov(dp_i, dp_j) / Var(dp_j)
                       = (S'_i / S'_j) * (sigma_bi / sigma_bj) * rho

        Dalen's short form (equal belief vols) is (S'_i/S'_j)*rho; sigma_bi
        defaults to sigma_bj to reproduce it. NOTE: the synthesis transcribed
        this as S'_i*S'_j*rho/(S'_j^2*sigma_bj^2) — that drops sigma_bi from
        the covariance and inflates beta by ~1/sigma_b^2; corrected here.
        Shrunk toward 0 as either S'->0 and hard-clamped to +/- beta_max.
        Zero when either p is outside the p-clamp band (risk 8.5)."""
        p_lo, p_hi = self.config.p_clamp
        if p_i <= p_lo or p_i >= p_hi or p_j <= p_lo or p_j >= p_hi:
            return 0.0
        s_i = float(logodds.s_prime(logodds.logit(p_i)))
        s_j = float(logodds.s_prime(logodds.logit(p_j)))
        sig_j = max(abs(sigma_bj), 1e-9)
        sig_i = sig_j if sigma_bi is None else max(abs(sigma_bi), 1e-9)
        raw = (s_i / max(s_j, 1e-12)) * (sig_i / sig_j) * rho
        floor = self.config.s_prime_floor
        shrink = min(1.0, s_i / floor) * min(1.0, s_j / floor)
        val = raw * shrink
        bm = self.config.beta_max
        if not math.isfinite(val):
            return 0.0
        return max(-bm, min(bm, val))

    def beta_hedges(
        self,
        inventory: InventoryState,
        expiry_key: str,
        strikes: List[float],
        market_ids: List[str],
        fair_p: Dict[str, float],
        sigma_b: Dict[str, float],
        ts: datetime,
        rho: float = 1.0,
        depth_hint: Optional[Dict[str, float]] = None,
    ) -> List[HedgeRecommendation]:
        """Emit BETA_HEDGE recommendations sized by beta*q_j into each
        over-target strike's adjacent neighbor. No-op unless enable_beta_hedge."""
        if not self.enable_beta_hedge:
            return []
        recs: List[HedgeRecommendation] = []
        expires = ts + timedelta(seconds=self.hedge_ttl_seconds)
        for jdx, mj in enumerate(market_ids):
            inv = inventory.per_contract.get(mj)
            if inv is None:
                continue
            qj, q_max = inv.q, inv.q_max
            if abs(qj) <= self.vertical_target_frac * q_max:
                continue
            idx = self._pick_neighbor(jdx, strikes, market_ids, depth_hint)
            if idx < 0:
                continue
            mi = market_ids[idx]
            p_i = float(fair_p.get(mi, 0.5))
            p_j = float(fair_p.get(mj, 0.5))
            sig_bj = float(sigma_b.get(mj, self.config.sigma_b_floor))
            beta = self.beta_ratio(p_i, p_j, sig_bj, rho)
            size = abs(beta * qj)
            if size <= 0.0:
                continue
            side = Side.BUY_NO if qj > 0 else Side.BUY_YES
            side_price = p_i if side == Side.BUY_YES else 1.0 - p_i
            max_price = _quantize(side_price + self.tick, self.tick)
            recs.append(
                HedgeRecommendation(
                    ts=ts,
                    expiry_key=expiry_key,
                    target_market_id=mi,
                    side=side,
                    size=size,
                    max_price=max_price,
                    reason=HedgeReason.BETA_HEDGE,
                    paired_market_id=mj,
                    beta=beta,
                    expires=expires,
                )
            )
        return recs


# ---------------------------------------------------------------------------
# W2.0 -- market_id-keyed offset builder (reviewer finding 1, CRITICAL)
# ---------------------------------------------------------------------------


def hedge_offsets_by_market(recs: List[HedgeRecommendation]) -> Dict[str, float]:
    """Aggregate a list of HedgeRecommendations (vertical and/or beta) into a
    market_id-keyed signed offset dict, the shape `InventoryManager.
    set_hedge_state` / `net_band_exposure` and `store.upsert_ladder_state`
    actually consume.

    This is deliberately NOT the `(expiry_key, bucket)`-keyed second return of
    `vertical_hedges` -- that HedgeState is an inter-strike-bucket audit view
    for the hedger's own bookkeeping and must never cross a module boundary.
    The market_id-keyed view here is built directly from the recommendation
    list instead: `offsets[rec.target_market_id] += (+rec.size if
    rec.side == Side.BUY_YES else -rec.size)`.

    Semantics (plan W2.0): the returned offset is PENDING hedge demand on
    that market -- recommended, not yet filled. It is rebuilt from scratch
    each tick from that tick's fresh recs, so a recommendation's `expires`
    is honored implicitly by per-tick re-evaluation (a stale rec is simply
    not re-emitted, and the next tick's offsets dict no longer contains it).
    Once a hedge order actually fills, the fill enters `q` through the
    normal fill channel (InventoryManager.apply_fill) like any other fill --
    it does NOT get folded into this offset dict -- and the hedger's next
    excess computation (over the now-smaller |q|) naturally shrinks or drops
    the recommendation.
    """
    offsets: Dict[str, float] = {}
    for rec in recs:
        signed = rec.size if rec.side == Side.BUY_YES else -rec.size
        offsets[rec.target_market_id] = offsets.get(rec.target_market_id, 0.0) + signed
    return offsets
