"""Settlement handler (plan Section 2.13, task E1, contract 4.14).

Settlement authority in paper mode: at a ladder's expiry (12:00 ET on
expiry_key), resolves every held position to its 0/1 payoff against BTC spot.

Conventions REPLICATED (not imported) from
``core/backtesting/backtest_engine.py`` (that file is read-only, not
modified): ``_get_expiry_datetime`` (12:00 ET -> UTC), ``_expiry_is_settleable``
(intraday-data-range gate, +/-5min tolerance), ``_settlement_price`` (nearest
intraday close within tolerance), and ``_spot_as_of`` (prior-daily-close
fallback). Those are instance methods on ``BacktestEngine``, which would need
a full ``BacktestEngine`` constructed just to reach them, and its constructor
loads daily closes from disk unconditionally (no injection point for a daily
fixture DataFrame) -- incompatible with this module's injectable-data-provider
requirement (tests need to fix BOTH intraday and daily frames). Hence the
minimal replication below, with this comment naming the source functions.

Outcome convention: STRICT ``>`` (spot > strike -> YES), matching the venue's
confirmed resolution rule (Polymarket resolves YES only if price is strictly
above the strike) and the backtester's ``resolve_outcome_yes``. The pricing
engine's ``P(S_T >= K)`` differs only on the measure-zero boundary and is
left unchanged.

PnL authority (plan 2.13): realized PnL comes ONLY from folding the `fills`
table (the SETTLEMENT-tagged pseudo-fill's price-vs-avg_cost IS the
settlement PnL). ``SettlementEvent.pnl_realized`` computed here is
REPORT-ONLY -- never summed as an independent PnL source.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
    _ET_ZONE = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback for environments without tzdata
    _ET_ZONE = timezone(timedelta(hours=-5))

from market_maker.config import MMConfig
from market_maker.contracts import (
    ContractInv,
    Fill,
    LiquiditySource,
    SettlementEvent,
    SettlementOutcome,
    Side,
    SpotSource,
)
from market_maker.state_store import MMStateStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INTRADAY_PATH = _PROJECT_ROOT / "DATA" / "btc_intraday_1m.csv"
_DEFAULT_DAILY_PATH = _PROJECT_ROOT / "DATA" / "btc_daily.csv"

TERMINAL_OUTCOMES = (SettlementOutcome.YES, SettlementOutcome.NO)
_SETTLE_TOL = timedelta(minutes=5)


# ---------------------------------------------------------------------------
# BTC data access (injectable; default lazily reads the DATA/ CSVs)
# ---------------------------------------------------------------------------


def _load_close_csv(path: Path) -> pd.DataFrame:
    """Minimal replica of BacktestEngine._load_btc_prices / _load_close_csv:
    a timestamp/date + close/price CSV -> a UTC-DatetimeIndex single-`close`-
    column frame. Source: core/backtesting/backtest_engine.py.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(path)
    except Exception as e:
        logger.warning("failed to read %s: %s", path, e)
        return pd.DataFrame()
    cols = {c.lower(): c for c in d.columns}
    tcol = cols.get("timestamp", cols.get("time", cols.get("datetime", cols.get("date"))))
    ccol = cols.get("close", cols.get("price"))
    if tcol is None or ccol is None:
        return pd.DataFrame()
    d["datetime_utc"] = pd.to_datetime(d[tcol], utc=True, errors="coerce")
    d["close"] = pd.to_numeric(d[ccol], errors="coerce")
    d = d.dropna(subset=["datetime_utc", "close"]).set_index("datetime_utc").sort_index()
    return d[["close"]].copy()


class BTCDataProvider:
    """Injectable BTC price source. Tests construct one with fixture frames
    directly (``BTCDataProvider(intraday=df, daily=df2)``); the default
    lazily reads ``DATA/btc_intraday_1m.csv`` / ``DATA/btc_daily.csv`` (the
    same files ``core/data/data_fetcher.py`` maintains) on first use.
    """

    def __init__(
        self,
        intraday: Optional[pd.DataFrame] = None,
        daily: Optional[pd.DataFrame] = None,
        intraday_path: Path = _DEFAULT_INTRADAY_PATH,
        daily_path: Path = _DEFAULT_DAILY_PATH,
    ) -> None:
        self._intraday = intraday
        self._daily = daily
        self._intraday_path = intraday_path
        self._daily_path = daily_path
        # Whether each source is path-backed (constructor arg was None) --
        # only path-backed sources are eligible for refresh()/mtime tracking.
        # Injected frames are static for the provider's lifetime.
        self._intraday_is_path_backed = intraday is None
        self._daily_is_path_backed = daily is None
        self._intraday_mtime: Optional[float] = None
        self._daily_mtime: Optional[float] = None

    def _ensure_loaded(self) -> None:
        if self._intraday is None:
            self._intraday = _load_close_csv(self._intraday_path)
        if self._daily is None:
            self._daily = _load_close_csv(self._daily_path)

    def refresh(self) -> None:
        """Reload path-backed sources whose file mtime has changed since the
        last load. Injected frames (constructor arg not None) are never
        stat'd or reloaded here.

        This exists as a SEPARATE step from `_ensure_loaded` (which keeps its
        load-if-None-only semantics) because settlement retries must see
        freshly fetched CSVs -- the 24h UNSETTLEABLE retry window is useless
        if the provider's cache never invalidates -- while a single settlement
        resolution (`_resolve_settlement_spot`'s sequence of intraday_range /
        nearest_intraday_close / prior_daily_close calls) must see one
        consistent pair of frames throughout. Callers invalidate once per
        settle attempt (here) and then read via the pinned `_ensure_loaded`
        accessors for the rest of that attempt.
        """
        if self._intraday_is_path_backed:
            self._intraday = self._refresh_one(
                self._intraday_path, self._intraday, self._intraday_mtime, "_intraday_mtime"
            )
        if self._daily_is_path_backed:
            self._daily = self._refresh_one(
                self._daily_path, self._daily, self._daily_mtime, "_daily_mtime"
            )

    def _refresh_one(
        self, path: Path, cached: Optional[pd.DataFrame], last_mtime: Optional[float], mtime_attr: str
    ) -> Optional[pd.DataFrame]:
        # Independent try block per source: a stat failure here must not
        # abort the sibling source's refresh (caller invokes this per-source).
        try:
            mtime: Optional[float] = path.stat().st_mtime
        except OSError:
            mtime = None

        if cached is not None and mtime == last_mtime:
            return cached  # already loaded and unchanged

        reloaded = _load_close_csv(path)

        # Torn-read guard: an empty reload over a non-empty cache is a
        # transient read failure (mid-write truncation or an unreadable
        # file) -- keep serving the good cache and do NOT advance the
        # stored mtime, so the next refresh() retries the reload. An empty
        # reload over a None-or-empty cache (missing file, or a file that
        # is genuinely empty so far) IS accepted, mtime advances -- otherwise
        # a genuinely-empty-then-populated file would wedge forever.
        if reloaded.empty and cached is not None and not cached.empty:
            return cached

        setattr(self, mtime_attr, mtime)
        return reloaded

    def intraday_range(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        self._ensure_loaded()
        if self._intraday is None or self._intraday.empty:
            return None, None
        return self._intraday.index.min(), self._intraday.index.max()

    def nearest_intraday_close(self, ts: pd.Timestamp, tol: timedelta) -> Optional[float]:
        self._ensure_loaded()
        if self._intraday is None or self._intraday.empty:
            return None
        idx = self._intraday.index.get_indexer([ts], method="nearest")
        if idx[0] < 0:
            return None
        nearest = self._intraday.index[idx[0]]
        if abs(nearest - ts) <= tol:
            return float(self._intraday.iloc[idx[0]]["close"])
        return None

    def prior_daily_close(self, day: pd.Timestamp) -> Optional[float]:
        self._ensure_loaded()
        if self._daily is None or self._daily.empty:
            return None
        sl = self._daily.index[self._daily.index < day]
        if len(sl) == 0:
            return None
        return float(self._daily.loc[sl[-1], "close"])


# ---------------------------------------------------------------------------
# 12:00 ET -> UTC settlement instant + settleability/spot resolution
# ---------------------------------------------------------------------------


def settlement_instant_utc(expiry_key: str) -> datetime:
    """12:00 ET on expiry_key's (YYYY-MM-DD) date, converted to UTC. Same
    rule as BacktestEngine._get_expiry_datetime (source replicated above)."""
    d = pd.to_datetime(expiry_key)
    et_noon = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=_ET_ZONE)
    return et_noon.astimezone(timezone.utc)


def _resolve_settlement_spot(
    expiry_key: str, data: BTCDataProvider
) -> Tuple[Optional[float], SpotSource, datetime]:
    """Settleability gate mirrors _expiry_is_settleable (intraday data range
    must cover the settlement instant, +/-5min tolerance) -- "an intraday
    print at/after the instant must exist". Spot resolution then mirrors
    _settlement_price (nearest intraday close within tolerance) with a
    _spot_as_of-style prior-daily-close fallback for the rare case where the
    range check passes but no print lands within tolerance.
    """
    data.refresh()  # pick up fresh CSVs once per settle attempt (see refresh() docstring)

    settle_dt = settlement_instant_utc(expiry_key)
    settle_ts = pd.Timestamp(settle_dt)

    lo, hi = data.intraday_range()
    settleable = lo is not None and hi is not None and (lo - _SETTLE_TOL) <= settle_ts <= (hi + _SETTLE_TOL)
    if not settleable:
        return None, SpotSource.NONE, settle_dt

    spot = data.nearest_intraday_close(settle_ts, _SETTLE_TOL)
    if spot is not None:
        return spot, SpotSource.INTRADAY, settle_dt

    spot = data.prior_daily_close(settle_ts.normalize())
    if spot is not None:
        return spot, SpotSource.DAILY_PRIOR, settle_dt

    return None, SpotSource.NONE, settle_dt


# ---------------------------------------------------------------------------
# Inputs / results
# ---------------------------------------------------------------------------


@dataclass
class MarketPosition:
    """One market's settlement inputs (plan 2.13: "markets carry (market_id,
    strike, q, avg_cost)")."""
    market_id: str
    strike: float
    q: float
    avg_cost: float


@dataclass
class SettlementRunResult:
    events: List[SettlementEvent] = field(default_factory=list)
    escalated_market_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Settlement handler
# ---------------------------------------------------------------------------


class SettlementHandler:
    def __init__(
        self,
        store: MMStateStore,
        config: Optional[MMConfig] = None,
        data: Optional[BTCDataProvider] = None,
    ) -> None:
        self.store = store
        self.config = config if config is not None else MMConfig()
        self.data = data if data is not None else BTCDataProvider()

    # -- core settlement -----------------------------------------------------

    def settle_expiry(self, expiry_key: str, markets: List[MarketPosition], now: datetime) -> SettlementRunResult:
        """Settle every market in `markets` for one expiry_key at time `now`.
        Idempotent per (market_id, expiry_key): a market already terminal
        (YES/NO) is skipped entirely -- no re-check, no duplicate pseudo-fill.
        """
        spot, spot_source, settle_dt = _resolve_settlement_spot(expiry_key, self.data)

        events: List[SettlementEvent] = []
        escalated: List[str] = []

        for m in markets:
            existing = self.store.get_settlement(m.market_id, expiry_key)
            if existing is not None and existing.outcome in TERMINAL_OUTCOMES:
                continue  # terminal-only idempotency guard (plan 2.13/5)

            if spot is None:
                # UNSETTLEABLE: no pseudo-fill, position stays open, journaled,
                # excluded from gate PnL, retried until the escalation window.
                # `ts` is repurposed (documented deviation from the "processing
                # time" comment on contracts.SettlementEvent.ts) to carry the
                # FIRST-detected UNSETTLEABLE timestamp across retries, since
                # the settlements schema (not modified here) has no separate
                # column for it -- this is what the 24h retry-window clock
                # measures against.
                first_seen = (
                    existing.ts if (existing is not None and existing.outcome == SettlementOutcome.UNSETTLEABLE)
                    else now
                )
                event = SettlementEvent(
                    ts=first_seen, settlement_ts=settle_dt, market_id=m.market_id, expiry_key=expiry_key,
                    strike=m.strike, outcome=SettlementOutcome.UNSETTLEABLE, spot_used=None,
                    spot_source=SpotSource.NONE, q_settled=m.q, payoff=None, pnl_realized=None,
                    excluded_from_gate=True,
                )
                self.store.upsert_settlement(event)
                events.append(event)
                elapsed_h = (now - first_seen).total_seconds() / 3600.0
                if elapsed_h > self.config.settlement_retry_window_hours:
                    escalated.append(m.market_id)
                logger.warning(
                    "settlement UNSETTLEABLE market=%s expiry=%s elapsed=%.1fh window=%.1fh",
                    m.market_id, expiry_key, elapsed_h, self.config.settlement_retry_window_hours,
                )
                continue

            # Strict ">" per the venue-confirmed rule (Polymarket resolves
            # YES only if price is strictly above the strike; see module
            # docstring) and matching the backtester's resolve_outcome_yes.
            # The pricing engine's P(S_T >= K) differs only on the
            # measure-zero boundary and is intentionally left unchanged.
            outcome = SettlementOutcome.YES if spot > m.strike else SettlementOutcome.NO
            payoff_yes = 1.0 if outcome is SettlementOutcome.YES else 0.0

            if m.q == 0.0:
                event = SettlementEvent(
                    ts=now, settlement_ts=settle_dt, market_id=m.market_id, expiry_key=expiry_key,
                    strike=m.strike, outcome=outcome, spot_used=spot, spot_source=spot_source,
                    q_settled=0.0, payoff=0.0, pnl_realized=0.0, excluded_from_gate=False,
                )
                self.store.upsert_settlement(event)
                events.append(event)
                continue

            # SYNTHETIC CLOSING FILL (contract 4.11): side closes q; price is
            # the settlement value of a YES share (1.0 on YES, 0.0 on NO) --
            # the closing side (not the price) carries the sign, so the
            # fold's cost-basis, which uses this raw YES-scale price for
            # every side (C0 fix -- no per-side complement), yields the
            # correct economics for whichever direction was held.
            closing_side = Side.BUY_NO if m.q > 0.0 else Side.BUY_YES
            fill = Fill(
                ts=now, market_id=m.market_id, order_id=f"settlement:{m.market_id}:{expiry_key}",
                side=closing_side, price=payoff_yes, size=abs(m.q),
                liquidity=LiquiditySource.SETTLEMENT, venue_ts=now,
            )

            # PnL authority note: this is the report-only figure (see module
            # docstring); the AUTHORITATIVE realized PnL is whatever
            # store.fold_fills_to_inventory() derives from this same fill.
            payoff_total = abs(m.q) * (payoff_yes if m.q > 0.0 else (1.0 - payoff_yes))
            pnl_realized = m.q * (payoff_yes - m.avg_cost)

            prior_inv = self.store.get_inventory(m.market_id)
            resulting = ContractInv(
                q=0.0, avg_cost=0.0,
                q_max=prior_inv.q_max if prior_inv is not None else 0.0,
                age_weighted_holding=prior_inv.age_weighted_holding if prior_inv is not None else 0.0,
            )
            # Atomic: fill + inventory update in one transaction (plan Section
            # 5 write-ahead rule); fold(fills) == inventory survives resolution
            # with no special case (risk 8.2).
            self.store.record_fill_and_update_inventory(fill, resulting)

            event = SettlementEvent(
                ts=now, settlement_ts=settle_dt, market_id=m.market_id, expiry_key=expiry_key,
                strike=m.strike, outcome=outcome, spot_used=spot, spot_source=spot_source,
                q_settled=m.q, payoff=payoff_total, pnl_realized=pnl_realized, excluded_from_gate=False,
            )
            self.store.upsert_settlement(event)
            events.append(event)

        return SettlementRunResult(events=events, escalated_market_ids=escalated)

    # -- startup catch-up (plan Section 5 restart protocol step 4) -----------

    def catch_up(self, now: datetime, registry: Dict[str, Tuple[str, float]]) -> SettlementRunResult:
        """Scan for expired-but-unsettled ladders and settle them before
        quoting resumes.

        Deviation (documented): the `inventory` table (Section 5 schema, not
        modified here) persists q/avg_cost per market_id but NOT the market's
        (expiry_key, strike) -- nothing in the state store maps a market_id to
        its ladder. That mapping only exists in-memory (e.g.
        InventoryManager's ladder registry), so catch_up takes it as an
        explicit `registry: {market_id: (expiry_key, strike)}` argument
        instead of inferring it from the store alone. q/avg_cost are still
        read live from the store per market.
        """
        by_expiry: Dict[str, List[Tuple[str, float]]] = {}
        for market_id, (expiry_key, strike) in registry.items():
            by_expiry.setdefault(expiry_key, []).append((market_id, strike))

        all_events: List[SettlementEvent] = []
        all_escalated: List[str] = []
        for expiry_key, members in by_expiry.items():
            settle_dt = settlement_instant_utc(expiry_key)
            if now < settle_dt:
                continue  # not yet due

            pending: List[MarketPosition] = []
            for market_id, strike in members:
                existing = self.store.get_settlement(market_id, expiry_key)
                if existing is not None and existing.outcome in TERMINAL_OUTCOMES:
                    continue
                inv = self.store.get_inventory(market_id)
                q = inv.q if inv is not None else 0.0
                avg_cost = inv.avg_cost if inv is not None else 0.0
                pending.append(MarketPosition(market_id=market_id, strike=strike, q=q, avg_cost=avg_cost))

            if not pending:
                continue

            result = self.settle_expiry(expiry_key, pending, now)
            all_events.extend(result.events)
            all_escalated.extend(result.escalated_market_ids)

        return SettlementRunResult(events=all_events, escalated_market_ids=all_escalated)
