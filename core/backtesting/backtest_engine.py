#!/usr/bin/env python3
"""
backtest_engine.py — moved from scripts/backtesting/

Historical backtesting engine for the Auto-Reco trading strategy.
Simulates position entry/exit, settlement, and bankroll evolution.

This module is designed to be imported by dashboard.py for the Backtesting tab.

Key Features:
- Loads BTC intraday 1-minute data for settlement price lookups
- Iterates through market data batches chronologically
- Settles expired positions using actual BTC prices at expiry
- Executes new trades via auto_reco with position constraints
- Tracks equity curve and trade history
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    ET_ZONE = ZoneInfo("America/New_York")
except Exception:
    # Fallback for older Python
    ET_ZONE = timezone(timedelta(hours=-5))

from core.strategy.auto_reco import recommend_trades, recommendations_to_dataframe
from core.strategy.common import resolve_model_prob, MODEL_PROB_CANDIDATES

# Path resolution — robust regardless of CWD (Streamlit, scripts/, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_INTRADAY = _PROJECT_ROOT / "DATA" / "btc_intraday_1m.csv"

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """Represents a single open position in the backtest."""
    pricing_date: pd.Timestamp
    expiry_date: pd.Timestamp
    slug: str
    strike: float
    side: str  # "YES" or "NO"
    entry_price: float
    stake: float  # USD invested
    size_shares: float  # stake / entry_price
    model_prob: float = np.nan
    market_price: float = np.nan
    trade_id: str = ""
    kelly_applied: float = np.nan
    expiry_key: str = ""
    moneyness: float = np.nan
    momentum_6hr: float = np.nan
    edge: float = np.nan
    position_key: str = ""  # Unique key for position lookup (slug|expiry|strike|side)


def _generate_position_key(slug: str, expiry_key: str, strike: float, side: str) -> str:
    """Generate a unique key for position lookup."""
    return f"{slug}|{expiry_key}|{strike}|{side.upper()}"


def _positions_to_df(open_positions: List[OpenPosition]) -> pd.DataFrame:
    """Convert list of OpenPosition to DataFrame for auto_reco."""
    if not open_positions:
        return pd.DataFrame()
    return pd.DataFrame([asdict(p) for p in open_positions])


class BacktestEngine:
    """
    Main backtest engine for the Auto-Reco strategy.

    Inputs:
        market_data_batches: List of DataFrames loaded by dashboard.py
        initial_bankroll: Starting capital in USD
        strategy_params: Dictionary of strategy parameters for auto_reco
        btc_price_path: Path to BTC intraday 1-minute CSV (relative to program root)
        price_df: Optional pre-loaded BTC price DataFrame (takes precedence over file path)

    Usage:
        engine = BacktestEngine(batches, 1000.0, {'kelly_fraction': 0.15})
        trades_df, equity_df = engine.run()
    """

    def __init__(
        self,
        market_data_batches: List[pd.DataFrame],
        initial_bankroll: float,
        strategy_params: Dict,
        btc_price_path: str = str(_DEFAULT_INTRADAY),
        price_df: Optional[pd.DataFrame] = None,
    ):
        self.batches = market_data_batches
        self.initial_bankroll = initial_bankroll
        self.strategy_params = strategy_params
        self.btc_price_path = btc_price_path
        self._price_df_provided = price_df

        # Internal state
        self._btc_prices: Optional[pd.DataFrame] = None
        self._volgate_btc_df: Optional[pd.DataFrame] = None  # timestamp COLUMN + close, for vol gate
        self._btc_daily: Optional[pd.DataFrame] = None   # daily closes for spot fallback (not settlement)
        self._intraday_min: Optional[pd.Timestamp] = None
        self._intraday_max: Optional[pd.Timestamp] = None
        self._running_bankroll: float = initial_bankroll
        self._open_positions: List[OpenPosition] = []
        self._closed_trades: List[Dict] = []
        self._equity_snapshots: List[Dict] = []
        self._all_priced_contracts: List[Dict] = []  # Track ALL evaluated contracts
        self._excluded_trade_ids: set = set()  # trades dropped for missing settlement
        self._trade_counter: int = 0

    def _load_btc_prices(self) -> None:
        """
        Load BTC intraday 1-minute data and create DatetimeIndex for fast lookup.
        Called once at initialization.

        If price_df was provided in constructor, use that instead of loading from file.
        """
        # Use provided DataFrame if available
        if self._price_df_provided is not None and not self._price_df_provided.empty:
            df = self._price_df_provided.copy()
            logger.info(f"Using provided price DataFrame with {len(df)} rows")
        else:
            # Load from file — resolve relative paths against project root
            path = Path(self.btc_price_path)
            if not path.is_absolute():
                path = _PROJECT_ROOT / path
            if not path.exists():
                logger.warning(f"BTC price file not found: {path}")
                self._btc_prices = pd.DataFrame()
                return

            try:
                df = pd.read_csv(path)
            except Exception as e:
                logger.error(f"Failed to load BTC prices from file: {e}")
                self._btc_prices = pd.DataFrame()
                return

        try:
            # Normalize column names
            cols_lower = {c.lower(): c for c in df.columns}

            # Find timestamp column
            ts_col = None
            for cand in ["timestamp", "time", "datetime", "date"]:
                if cand in cols_lower:
                    ts_col = cols_lower[cand]
                    break

            if ts_col is None:
                logger.error("BTC price data missing timestamp column")
                self._btc_prices = pd.DataFrame()
                return

            # Parse timestamps to UTC
            df["datetime_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df.dropna(subset=["datetime_utc"])
            df = df.set_index("datetime_utc").sort_index()

            # Find close column
            close_col = None
            for cand in ["close", "price"]:
                if cand in cols_lower:
                    close_col = cols_lower[cand]
                    break

            if close_col is None:
                logger.error("BTC price data missing close/price column")
                self._btc_prices = pd.DataFrame()
                return

            df["close"] = pd.to_numeric(df[close_col], errors="coerce")
            self._btc_prices = df[["close"]].copy()
            if not self._btc_prices.empty:
                self._intraday_min = self._btc_prices.index.min()
                self._intraday_max = self._btc_prices.index.max()
                # Deterministic vol-gate-ready frame: timestamp COLUMN + close
                self._volgate_btc_df = pd.DataFrame({
                    "timestamp": self._btc_prices.index,
                    "close": self._btc_prices["close"].values,
                })
            logger.info(
                f"Loaded {len(self._btc_prices)} BTC price records "
                f"(coverage {self._intraday_min} → {self._intraday_max})"
            )

        except Exception as e:
            logger.error(f"Failed to process BTC prices: {e}")
            self._btc_prices = pd.DataFrame()

        # ---- daily closes (spot fallback only; settlement is intraday-only) ----
        # Resolve relative paths against project root (same as intraday load above),
        # not the CWD — under Streamlit the CWD is app/, which would yield app/DATA.
        _intraday_path = Path(self.btc_price_path)
        if not _intraday_path.is_absolute():
            _intraday_path = _PROJECT_ROOT / _intraday_path
        data_dir = _intraday_path.parent
        self._btc_daily = self._load_close_csv(data_dir / "btc_daily.csv", "daily")

    @staticmethod
    def _load_close_csv(path: Path, label: str) -> pd.DataFrame:
        """Load a ``date,close`` CSV into a UTC-indexed single-column frame."""
        try:
            if not path.exists():
                logger.info("%s BTC file not found (%s)", label, path)
                return pd.DataFrame()
            d = pd.read_csv(path)
            cols = {c.lower(): c for c in d.columns}
            dcol = cols.get("date", cols.get("timestamp", cols.get("datetime")))
            ccol = cols.get("close", cols.get("price"))
            if dcol is None or ccol is None:
                return pd.DataFrame()
            d["datetime_utc"] = pd.to_datetime(d[dcol], utc=True, errors="coerce")
            d["close"] = pd.to_numeric(d[ccol], errors="coerce")
            d = d.dropna(subset=["datetime_utc", "close"]).set_index("datetime_utc").sort_index()
            logger.info("Loaded %d %s BTC closes", len(d), label)
            return d[["close"]].copy()
        except Exception as e:
            logger.warning("Failed to load %s BTC closes: %s", label, e)
            return pd.DataFrame()

    @staticmethod
    def _nearest_close(frame: Optional[pd.DataFrame], target: pd.Timestamp,
                       tol: pd.Timedelta) -> Optional[float]:
        """Nearest close in *frame* to *target* within *tol*, else None."""
        if frame is None or frame.empty:
            return None
        idx = frame.index.get_indexer([target], method="nearest")
        if idx[0] < 0:
            return None
        nearest = frame.index[idx[0]]
        if abs(nearest - target) <= tol:
            return float(frame.iloc[idx[0]]["close"])
        return None

    def _settlement_price(self, settle_dt: pd.Timestamp) -> Tuple[Optional[float], Optional[str]]:
        """BTC price at the 12:00-ET settlement instant, from 1-minute data only.

        Settlement decides a YES/NO outcome against a strike, so it demands the
        highest available time resolution: a coarser bar (hourly/daily) can land
        on the wrong side of a near-the-money strike if BTC oscillated around it
        within the bar. We therefore settle exclusively from the 1-minute intraday
        series (tight ±5m tolerance to bridge the occasional missing bar). The
        intraday history must cover the contract range — run
        ``python core/data/data_fetcher.py`` to backfill. Returns (price, source)
        or (None, None) when no 1m print is within tolerance.
        """
        p = self._nearest_close(self._btc_prices, settle_dt, pd.Timedelta(minutes=5))
        if p is not None:
            return p, "intraday"
        return None, None

    def _spot_as_of(self, dt: datetime) -> Optional[float]:
        """Spot price known AT *dt*, leak-free, with a daily-close fallback.

        Single source of truth for "spot at snapshot" used by moneyness/momentum,
        consistent with the backrunner's S0 logic:
          1. last intraday close STRICTLY BEFORE dt (bars open-stamped → the
             close at/after dt is future), else
          2. the most recent daily close STRICTLY BEFORE dt's day (prior-day
             close ≈ the price at dt's midnight).
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        target = pd.Timestamp(dt).tz_convert("UTC")

        if self._btc_prices is not None and not self._btc_prices.empty:
            sl = self._btc_prices.index[self._btc_prices.index < target]
            if len(sl) > 0:
                return float(self._btc_prices.loc[sl[-1], "close"])

        if self._btc_daily is not None and not self._btc_daily.empty:
            day = target.normalize()
            sl = self._btc_daily.index[self._btc_daily.index < day]
            if len(sl) > 0:
                return float(self._btc_daily.loc[sl[-1], "close"])

        return None

    def _expiry_is_settleable(self, expiry_date: pd.Timestamp) -> bool:
        """True if 1-minute intraday data covers this expiry's 12:00-ET settlement.

        Backtest reliability gate: only enter contracts we can later settle from a
        real 1m print (settlement is intraday-only for strike-level precision).
        Avoids the old force-refund-to-PnL-0 path that biased aggregates. Backfill
        intraday (``data_fetcher.py``) to widen the tradeable/settleable range.
        """
        if self._intraday_min is None or self._intraday_max is None:
            return False
        try:
            settle_dt = pd.Timestamp(self._get_expiry_datetime(expiry_date))
        except Exception:
            return False
        tol = pd.Timedelta(minutes=5)
        return (self._intraday_min - tol) <= settle_dt <= (self._intraday_max + tol)

    def _get_btc_price_at(
        self,
        dt: datetime,
        tolerance_minutes: int = 5
    ) -> Optional[float]:
        """
        Lookup BTC close price at the specified datetime.

        Args:
            dt: Target datetime (should be UTC)
            tolerance_minutes: Maximum minutes to search for nearby price

        Returns:
            Close price if found, None otherwise
        """
        if self._btc_prices is None or self._btc_prices.empty:
            return None

        # Ensure dt is UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        target = pd.Timestamp(dt).tz_convert("UTC")

        # Try exact match first
        if target in self._btc_prices.index:
            return float(self._btc_prices.loc[target, "close"])

        # Try nearest within tolerance
        try:
            idx = self._btc_prices.index.get_indexer([target], method="nearest")
            if idx[0] >= 0:
                nearest_dt = self._btc_prices.index[idx[0]]
                delta_seconds = abs((nearest_dt - target).total_seconds())
                if delta_seconds <= tolerance_minutes * 60:
                    return float(self._btc_prices.loc[nearest_dt, "close"])
        except Exception:
            pass

        return None

    def _get_expiry_datetime(self, expiry_date: pd.Timestamp) -> datetime:
        """
        Convert expiry date to exact settlement time (12:00 ET on expiry day).
        Returns UTC datetime.
        """
        # Extract date components
        if isinstance(expiry_date, str):
            expiry_date = pd.to_datetime(expiry_date)

        # Settlement is at 12:00 ET (noon Eastern)
        et_noon = datetime(
            expiry_date.year,
            expiry_date.month,
            expiry_date.day,
            12, 0, 0,
            tzinfo=ET_ZONE
        )
        return et_noon.astimezone(timezone.utc)

    def resolve_outcome_yes(
        self,
        expiry_date: pd.Timestamp,
        strike: float
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Authoritative source for outcome resolution.

        Args:
            expiry_date: Contract expiry date
            strike: Contract strike price

        Returns:
            Tuple of (outcome_yes, btc_price_at_settlement, settlement_source)
            - outcome_yes: 1.0 if btc_price > strike (strict >), else 0.0
            - btc_price_at_settlement: BTC price at 12:00 ET on expiry day
            - settlement_source: 'intraday' | 'hourly' | 'daily_coarse'
            - All three are None if no price source covers the settlement instant
        """
        try:
            expiry_dt_utc = pd.Timestamp(self._get_expiry_datetime(expiry_date))
            btc_price, source = self._settlement_price(expiry_dt_utc)

            if btc_price is None:
                return None, None, None

            # Strict inequality: YES wins if btc_price > strike
            outcome_yes = 1.0 if btc_price > strike else 0.0
            return outcome_yes, btc_price, source

        except Exception:
            return None, None, None

    def _get_model_prob_col(self, df: pd.DataFrame) -> Optional[str]:
        """
        Get the model probability column name using preference order.
        Mirrors auto_reco's column selection logic.

        Preference: p_model_fit > p_real_mc > model_probability

        Returns:
            Column name if found, None otherwise
        """
        for col in ["p_model_fit", "p_real_mc", "model_probability"]:
            if col in df.columns:
                return col
        return None


    def _settle_positions(self, current_time: pd.Timestamp) -> float:
        """
        Settle positions whose expiry_date <= current_time.

        Returns:
            Total payout amount added to bankroll
        """
        if not self._open_positions:
            return 0.0

        total_payout = 0.0
        remaining: List[OpenPosition] = []

        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize("UTC")

        for pos in self._open_positions:
            # Parse expiry
            try:
                pos_expiry = pd.to_datetime(pos.expiry_date)
                if pos_expiry.tzinfo is None:
                    pos_expiry = pos_expiry.tz_localize("UTC")
            except Exception:
                remaining.append(pos)
                continue

            # Check if expired
            if current_time < pos_expiry:
                remaining.append(pos)
                continue

            # Use authoritative resolve_outcome_yes() for outcome determination
            outcome_yes, btc_price, settle_source = self.resolve_outcome_yes(pos_expiry, pos.strike)

            if outcome_yes is None:
                # No intraday print at the 12:00-ET settlement instant. The entry
                # gate (_expiry_is_settleable) should prevent this, so it is rare.
                # Wait a short grace period for late-arriving data, then EXCLUDE
                # the position: refund the stake to keep equity coherent, but do
                # NOT record a fake PnL=0 settlement that would pollute win-rate /
                # Sharpe / calibration. The trade is dropped from results entirely.
                days_past = (current_time - pos_expiry).days
                if days_past > 2:
                    logger.warning(
                        "Excluding %s from results: no BTC settlement price "
                        "(refunding stake $%.2f, no PnL recorded)",
                        pos.slug, pos.stake,
                    )
                    total_payout += pos.stake  # reverse the entry debit
                    # mark its entry row (if any) as excluded for later pruning
                    self._excluded_trade_ids.add(pos.trade_id)
                    continue
                else:
                    remaining.append(pos)
                    continue

            # Calculate payout using resolved outcome
            if pos.side.upper() == "YES":
                payout = pos.size_shares * outcome_yes
            else:
                payout = pos.size_shares * (1.0 - outcome_yes)

            pnl = payout - pos.stake
            total_payout += payout

            # Record closed trade
            self._closed_trades.append({
                "pricing_date": pos.pricing_date,
                "expiry_date": pos.expiry_date,
                "slug": pos.slug,
                "strike": pos.strike,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "stake": pos.stake,
                "size_shares": pos.size_shares,
                "model_prob": pos.model_prob,
                "market_price": pos.market_price,
                "moneyness": pos.moneyness,
                "momentum_6hr": pos.momentum_6hr,
                "edge": pos.edge,
                "btc_price_at_expiry": btc_price if btc_price is not None else np.nan,
                "outcome_yes": outcome_yes,
                "payout": payout,
                "pnl": pnl,
                "settled": True,
                "settlement_date": current_time,
                "settlement_source": settle_source,
                "trade_id": pos.trade_id,
                "kelly_applied": pos.kelly_applied,
            })

        self._open_positions = remaining
        return total_payout

    def _execute_trades(
        self,
        batch_df: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> None:
        """
        Run auto_reco on batch data and execute recommended trades.
        Deducts trade costs from bankroll and adds to open_positions.
        """
        if self._running_bankroll <= 0:
            return

        # Convert open positions to DataFrame for auto_reco
        positions_df = _positions_to_df(self._open_positions)

        # --- Log ALL priced contracts BEFORE calling recommend_trades ---
        # Row-level coalesce so a failed p_model_fit falls back to p_real_mc
        # (mirrors the strategy path) — all_priced then reflects the prob the
        # strategy actually used.
        resolved_model_prob = resolve_model_prob(batch_df)
        # Raw (never-calibrated) precedence, independent of USE_CALIBRATED_PROB, so
        # the signed-edge panel can show a true raw-vs-calibrated comparison even
        # when the backtest itself ran with the flag on (then resolved_model_prob
        # above already equals p_model_cal). FIX 7/M2.
        resolved_model_prob_raw = resolve_model_prob(
            batch_df, candidates=MODEL_PROB_CANDIDATES
        )
        if resolved_model_prob.notna().any():
            # Get spot price at snapshot time for moneyness calculation
            # (leak-free, with daily-close fallback for pre-intraday snapshots)
            snapshot_spot = self._spot_as_of(current_time)

            # Determine market price column
            market_col = None
            for col in ["market_price", "market_pr"]:
                if col in batch_df.columns:
                    market_col = col
                    break

            # Determine expiry key column
            expiry_key_col = None
            for col in ["expiry_key", "expiry_date"]:
                if col in batch_df.columns:
                    expiry_key_col = col
                    break

            # Determine DTE column (prefer existing, else compute)
            dte_col = None
            for col in ["t_days", "T_days", "dte_days"]:
                if col in batch_df.columns:
                    dte_col = col
                    break

            for idx, row in batch_df.iterrows():
                try:
                    strike = float(row.get("strike", np.nan))
                    model_prob = float(resolved_model_prob.loc[idx])
                    market_yes_price = float(row.get(market_col, np.nan)) if market_col else np.nan
                    expiry_key = str(row.get(expiry_key_col, "")) if expiry_key_col else ""
                    slug = str(row.get("slug", ""))

                    # Parse expiry date
                    raw_expiry = row.get("expiry_date")
                    try:
                        expiry_date = pd.to_datetime(raw_expiry)
                    except Exception:
                        expiry_date = pd.NaT

                    # Compute moneyness: (strike - spot) / spot
                    if snapshot_spot is not None and snapshot_spot > 0 and not np.isnan(strike):
                        moneyness = (strike - snapshot_spot) / snapshot_spot
                    else:
                        moneyness = np.nan

                    # Get DTE
                    if dte_col and dte_col in row.index:
                        dte_days = float(row.get(dte_col, np.nan))
                    elif pd.notna(expiry_date):
                        # Compute from dates
                        try:
                            dte_days = (expiry_date - current_time).total_seconds() / 86400.0
                        except Exception:
                            dte_days = np.nan
                    else:
                        dte_days = np.nan

                    # Skip already-expired contracts (snapshot_time > expiry_date)
                    if pd.notna(dte_days) and dte_days < 0:
                        continue

                    self._all_priced_contracts.append({
                        "snapshot_time": current_time,
                        "expiry_date": expiry_date,
                        "strike": strike,
                        "spot_price": snapshot_spot if snapshot_spot is not None else np.nan,
                        "market_yes_price": market_yes_price,
                        "model_prob_used": model_prob,
                        # Raw (never-calibrated) model prob — panel OFF state.
                        "model_prob_raw": float(resolved_model_prob_raw.loc[idx]),
                        # Carry the calibrated prob through (FIX 7/M2) so the
                        # signed-edge panel's "Use M2-calibrated p" toggle can see
                        # it. Only populated when the batch was fitted with
                        # USE_CALIBRATED_PROB on + a trusted shift table; NaN otherwise.
                        "p_model_cal": (
                            pd.to_numeric(row.get("p_model_cal"), errors="coerce")
                            if "p_model_cal" in batch_df.columns else np.nan
                        ),
                        "expiry_key": expiry_key,
                        "slug": slug,
                        "moneyness": moneyness,
                        "dte_days": dte_days,
                        "outcome_yes": np.nan,  # Resolved later
                        "btc_price_at_settlement": np.nan,  # Resolved later
                    })
                except Exception:
                    # Skip malformed rows
                    continue

        # Add moneyness column to batch_df if not present (needed for moneyness filter)
        if "moneyness" not in batch_df.columns and "strike" in batch_df.columns:
            if snapshot_spot is not None and snapshot_spot > 0:
                batch_df = batch_df.copy()
                batch_df["moneyness"] = (pd.to_numeric(batch_df["strike"], errors="coerce") - snapshot_spot) / snapshot_spot

        # Build strategy params with disable_staleness for backtest
        params = {
            "kelly_fraction": self.strategy_params.get("kelly_fraction", 0.15),
            "min_edge": self.strategy_params.get("min_edge", 0.06),
            "max_bets_per_expiry": self.strategy_params.get("max_bets_per_expiry", 3),
            "max_capital_per_expiry_frac": self.strategy_params.get(
                "max_capital_per_expiry_frac", 0.15
            ),
            "max_capital_total_frac": self.strategy_params.get(
                "max_capital_total_frac", 0.35
            ),
            "max_net_delta_frac": self.strategy_params.get("max_net_delta_frac", 0.20),
            "min_price": self.strategy_params.get("min_price", 0.03),
            "max_price": self.strategy_params.get("max_price", 0.95),
            "min_model_prob": self.strategy_params.get("min_model_prob", 0.0),
            "max_model_prob": self.strategy_params.get("max_model_prob", 1.0),
            "require_active": self.strategy_params.get("require_active", False),
            "use_stability_penalty": self.strategy_params.get(
                "use_stability_penalty", True
            ),
            "allow_no": self.strategy_params.get("allow_no", True),
            "correlation_penalty": self.strategy_params.get("correlation_penalty", 0.25),
            "disable_staleness": True,  # Always disable for backtest
        }

        # Handle min_trade_usd: None explicitly (dict.get returns None if key exists with None value)
        min_trade_usd = self.strategy_params.get("min_trade_usd")
        if min_trade_usd is None:
            min_trade_frac = self.strategy_params.get("min_trade_frac", 0.01)
            min_trade_usd = self._running_bankroll * min_trade_frac
        params["min_trade_usd"] = min_trade_usd

        # Fixed stake sizing params
        params["use_fixed_stake"] = self.strategy_params.get("use_fixed_stake", False)
        params["fixed_stake_amount"] = self.strategy_params.get("fixed_stake_amount", 10.0)

        # Max DTE filter
        params["max_dte"] = self.strategy_params.get("max_dte", None)

        # Probability threshold mode
        params["use_prob_threshold"] = self.strategy_params.get("use_prob_threshold", False)
        params["prob_threshold_yes"] = self.strategy_params.get("prob_threshold_yes", 0.7)
        params["prob_threshold_no"] = self.strategy_params.get("prob_threshold_no", 0.3)

        # Moneyness filter
        params["max_moneyness"] = self.strategy_params.get("max_moneyness", None)
        params["min_moneyness"] = self.strategy_params.get("min_moneyness", None)
        params["moneyness_mode"] = self.strategy_params.get("moneyness_mode", "abs")

        # Position size caps (critical for realistic backtesting)
        params["max_add_per_cycle_usd"] = self.strategy_params.get("max_add_per_cycle_usd", 50.0)
        params["max_reduce_per_cycle_usd"] = self.strategy_params.get("max_reduce_per_cycle_usd", 1000.0)

        # Call auto_reco with current open positions
        reco_list = recommend_trades(
            df=batch_df,
            bankroll=self._running_bankroll,
            positions_df=positions_df,
            current_open_positions=positions_df,
            btc_price_df=self._volgate_btc_df,   # cached once; avoids per-call CSV reload
            asof_utc=current_time,               # replay timestamp, not wall-clock now()
            **params
        )

        if not reco_list:
            return

        reco_df = recommendations_to_dataframe(reco_list)

        # Execute each trade
        for _, trade in reco_df.iterrows():
            action = str(trade.get("action", "BUY")).upper()
            side = str(trade.get("side", "")).upper()
            slug = str(trade.get("slug", ""))
            expiry_key_str = str(trade.get("expiry_key", ""))
            strike = float(trade.get("strike", np.nan))

            # Generate position key for lookup
            pos_key = _generate_position_key(slug, expiry_key_str, strike, side)

            # =====================================================================
            # SELL / EXIT Path - Close existing position at current market price
            # =====================================================================
            if action == "SELL":
                # Find matching open position
                matching_pos = None
                matching_idx = None
                for idx, pos in enumerate(self._open_positions):
                    if pos.position_key == pos_key:
                        matching_pos = pos
                        matching_idx = idx
                        break

                if matching_pos is None:
                    # No matching position to sell - skip
                    continue

                # Get exit price from trade recommendation
                exit_price = float(trade.get("market_price", 0.0))
                if exit_price <= 0:
                    continue

                # Calculate PnL
                # Proceeds = exit_price * shares
                # PnL = Proceeds - Stake (what we originally paid)
                proceeds = exit_price * matching_pos.size_shares
                pnl = proceeds - matching_pos.stake

                # Return proceeds to bankroll
                self._running_bankroll += proceeds

                # Log the exit trade
                self._closed_trades.append({
                    "pricing_date": matching_pos.pricing_date,  # Original entry time
                    "expiry_date": matching_pos.expiry_date,
                    "slug": matching_pos.slug,
                    "strike": matching_pos.strike,
                    "side": matching_pos.side,
                    "entry_price": matching_pos.entry_price,
                    "stake": matching_pos.stake,
                    "size_shares": matching_pos.size_shares,
                    "model_prob": matching_pos.model_prob,
                    "market_price": matching_pos.market_price,
                    "moneyness": matching_pos.moneyness,
                    "momentum_6hr": matching_pos.momentum_6hr,
                    "edge": matching_pos.edge,
                    "btc_price_at_expiry": np.nan,  # N/A for early exit
                    "outcome_yes": np.nan,  # N/A for early exit
                    "payout": proceeds,
                    "pnl": pnl,
                    "settled": True,
                    "settlement_date": current_time,
                    "settlement_type": "early_exit",
                    "exit_price": exit_price,
                    "bankroll_after": self._running_bankroll,
                    "kelly_applied": matching_pos.kelly_applied,
                    "trade_id": matching_pos.trade_id,
                })

                # Remove from open positions
                self._open_positions.pop(matching_idx)
                continue

            # =====================================================================
            # BUY Path - Open new position
            # =====================================================================
            if action != "BUY":
                # Skip HOLD or other actions
                continue

            # Reliability gate: only enter contracts we can later settle from a
            # real intraday print at 12:00 ET. Pre-/post-coverage expiries are
            # skipped here rather than force-refunded to PnL=0 at settlement
            # (which silently biased every aggregate).
            _raw_exp_gate = trade.get("expiry_date")
            _exp_for_gate = (
                expiry_key_str
                if (pd.isna(_raw_exp_gate) or _raw_exp_gate is None
                    or str(_raw_exp_gate) in ("", "NaT", "nan"))
                else str(_raw_exp_gate)
            )
            try:
                _exp_ts_gate = pd.to_datetime(_exp_for_gate)
            except Exception:
                _exp_ts_gate = pd.NaT
            if pd.isna(_exp_ts_gate) or not self._expiry_is_settleable(_exp_ts_gate):
                continue

            stake = float(
                trade.get("suggested_stake",
                    trade.get("stake_dollars",
                        trade.get("stake_usd", 0.0)
                    )
                )
            )

            # CRITICAL: Enforce hard cap on stake to prevent unrealistic compounding
            max_stake = self.strategy_params.get("max_add_per_cycle_usd", 50.0)
            stake = min(stake, max_stake)

            if stake <= 0 or stake > self._running_bankroll:
                continue

            price_yes = float(trade["market_price"])

            # Enforce price range filter - skip trades outside min/max price
            min_price = self.strategy_params.get("min_price", 0.03)
            max_price = self.strategy_params.get("max_price", 0.95)

            raw_execution_price = price_yes  # This is what we pay

            if side == "YES":
                if raw_execution_price < min_price or raw_execution_price > max_price:
                    continue
                entry_price = raw_execution_price
                stored_market_price = raw_execution_price  # Store YES price (q)
            else:
                # SIDE == NO
                if raw_execution_price < min_price or raw_execution_price > max_price:
                    continue

                entry_price = raw_execution_price
                # For analytics, store the implied YES price (q)
                stored_market_price = 1.0 - entry_price

            size_shares = stake / entry_price

            # Deduct stake from bankroll
            self._running_bankroll -= stake

            # Parse expiry date
            raw_expiry = trade.get("expiry_date")
            if pd.isna(raw_expiry) or raw_expiry is None or str(raw_expiry) in ("", "NaT", "nan"):
                expiry_str = expiry_key_str
            else:
                expiry_str = str(raw_expiry)

            try:
                expiry_ts = pd.to_datetime(expiry_str)
            except Exception:
                expiry_ts = pd.NaT

            # Compute moneyness and momentum for this trade
            trade_moneyness = np.nan
            momentum_6hr = np.nan
            snapshot_spot_at_trade = self._spot_as_of(current_time)
            if snapshot_spot_at_trade is not None and snapshot_spot_at_trade > 0:
                if not np.isnan(strike):
                    trade_moneyness = (strike - snapshot_spot_at_trade) / snapshot_spot_at_trade
                time_6h_ago = current_time - pd.Timedelta(hours=6)
                spot_6h_ago = self._spot_as_of(time_6h_ago)
                if spot_6h_ago is not None and spot_6h_ago > 0:
                    momentum_6hr = float(np.log(snapshot_spot_at_trade / spot_6h_ago))

            # Create position with moneyness, momentum, and position_key
            position = OpenPosition(
                pricing_date=current_time,
                expiry_date=expiry_ts,
                slug=slug,
                strike=strike,
                side=side,
                entry_price=entry_price,
                stake=stake,
                size_shares=size_shares,
                model_prob=float(trade.get("model_prob", np.nan)),
                market_price=stored_market_price,
                trade_id=f"T{self._trade_counter}",
                kelly_applied=float(trade.get("kelly_fraction_applied", np.nan)),
                expiry_key=expiry_key_str,
                moneyness=trade_moneyness,
                momentum_6hr=momentum_6hr,
                edge=float(trade.get("effective_edge", np.nan)),
                position_key=pos_key,
            )
            self._trade_counter += 1
            self._open_positions.append(position)

            # Log trade entry (use values from position)
            self._closed_trades.append({
                "pricing_date": current_time,
                "expiry_date": expiry_ts,
                "slug": position.slug,
                "strike": position.strike,
                "side": side,
                "entry_price": entry_price,
                "stake": stake,
                "size_shares": size_shares,
                "model_prob": position.model_prob,
                "market_price": position.market_price,
                "moneyness": position.moneyness,
                "momentum_6hr": position.momentum_6hr,
                "edge": position.edge,
                "btc_price_at_expiry": np.nan,
                "outcome_yes": np.nan,
                "payout": np.nan,
                "pnl": np.nan,
                "settled": False,
                "settlement_date": pd.NaT,
                "bankroll_after": self._running_bankroll,
                "kelly_applied": position.kelly_applied,
                "trade_id": position.trade_id,
            })

    def run(
        self,
        return_all_priced: bool = False
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Execute the backtest across all batches.

        Args:
            return_all_priced: If True, also return all_priced_df with every evaluated contract

        Returns:
            If return_all_priced=False: Tuple of (trades_df, equity_df)
            If return_all_priced=True: Tuple of (trades_df, equity_df, all_priced_df)
            - trades_df: All trade entries and settlements
            - equity_df: Equity curve snapshots
            - all_priced_df: Every evaluated contract with outcome resolution
        """
        # Load BTC prices
        self._load_btc_prices()

        # Reset state
        self._running_bankroll = self.initial_bankroll
        self._open_positions = []
        self._closed_trades = []
        self._equity_snapshots = []
        self._all_priced_contracts = []  # Reset all priced contracts
        self._excluded_trade_ids = set()

        # Sort batches chronologically by batch_timestamp or pricing_date
        sorted_batches = []
        for batch_df in self.batches:
            if batch_df is None or batch_df.empty:
                continue
            df = batch_df.copy()

            # Find timestamp column
            ts_col = None
            for cand in ["batch_timestamp", "pricing_date", "run_timestamp"]:
                if cand in df.columns:
                    ts_col = cand
                    break

            if ts_col is None:
                logger.warning("Batch missing timestamp column, skipping")
                continue

            # Parse timestamp
            try:
                ts_val = pd.to_datetime(df[ts_col].iloc[0])
                sorted_batches.append((ts_val, df))
            except Exception as e:
                logger.warning(f"Failed to parse batch timestamp: {e}")
                continue

        # Sort by timestamp
        sorted_batches.sort(key=lambda x: x[0])

        if not sorted_batches:
            logger.warning("No valid batches to process")
            if return_all_priced:
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            return pd.DataFrame(), pd.DataFrame()

        # Simulation loop
        for batch_ts, batch_df in sorted_batches:
            current_time = pd.Timestamp(batch_ts)
            if current_time.tzinfo is None:
                current_time = current_time.tz_localize("UTC")

            # STEP A: Settlement
            payout = self._settle_positions(current_time)
            self._running_bankroll += payout

            # STEP B: Execution
            self._execute_trades(batch_df, current_time)

            # Record equity snapshot (use pricing_date for dashboard compatibility)
            self._equity_snapshots.append({
                "pricing_date": current_time,
                "bankroll": self._running_bankroll,
                "open_position_count": len(self._open_positions),
                "total_stake_open": sum(p.stake for p in self._open_positions),
            })

        # Build output DataFrames
        trades_df = pd.DataFrame(self._closed_trades)
        # Drop entry rows for positions excluded at settlement (no settlement
        # price) so their stake-only rows don't masquerade as open/real trades.
        if (
            not trades_df.empty
            and self._excluded_trade_ids
            and "trade_id" in trades_df.columns
        ):
            trades_df = trades_df[~trades_df["trade_id"].isin(self._excluded_trade_ids)]
        if not trades_df.empty and "trade_id" in trades_df.columns:
            # Consolidate rows: keep last entry per trade_id (settled > unsettled)
            trades_df = trades_df.sort_values(["settled", "pricing_date"])
            trades_df = trades_df.drop_duplicates(subset=["trade_id"], keep="last")
            # Restore chronological order by entry time (approx)
            trades_df = trades_df.sort_values("trade_id", key=lambda x: x.str[1:].astype(int))
        equity_df = pd.DataFrame(self._equity_snapshots)

        if return_all_priced:
            all_priced_df = self._resolve_all_priced_contracts()
            # Run calibration metrics if outcomes are available
            self._run_calibration_if_possible(all_priced_df)
            return trades_df, equity_df, all_priced_df

        return trades_df, equity_df

    def _resolve_all_priced_contracts(self) -> pd.DataFrame:
        """
        Resolve outcomes for all priced contracts.

        Uses vectorized grouping by (expiry_date, strike) to minimize
        redundant price lookups.

        Returns:
            DataFrame with all priced contracts and resolved outcome_yes
        """
        if not self._all_priced_contracts:
            return pd.DataFrame()

        all_priced_df = pd.DataFrame(self._all_priced_contracts)

        # Group by unique (expiry_date, strike) pairs for efficient resolution
        # Create a resolution lookup
        resolution_cache: Dict[Tuple, Tuple[Optional[float], Optional[float], Optional[str]]] = {}

        for idx, row in all_priced_df.iterrows():
            expiry = row["expiry_date"]
            strike = row["strike"]

            # Skip if missing required fields
            if pd.isna(expiry) or pd.isna(strike):
                continue

            cache_key = (expiry, strike)

            if cache_key not in resolution_cache:
                # Resolve once per unique (expiry, strike) pair
                resolution_cache[cache_key] = self.resolve_outcome_yes(expiry, strike)

            outcome_yes, btc_price, settle_source = resolution_cache[cache_key]
            all_priced_df.at[idx, "outcome_yes"] = outcome_yes if outcome_yes is not None else np.nan
            all_priced_df.at[idx, "btc_price_at_settlement"] = btc_price if btc_price is not None else np.nan
            all_priced_df.at[idx, "settlement_source"] = settle_source

        return all_priced_df

    def _run_calibration_if_possible(self, all_priced_df: pd.DataFrame) -> None:
        """
        Run calibration metrics (Brier, ECE, reliability diagram) on
        all priced contracts with resolved outcomes.

        Logs summary. Full report accessible via CalibrationReport return
        if the caller wants it — here we just log to the backtest output.
        """
        from core.validation.calibration_metrics import run_calibration_report
        import tempfile, os

        # all_priced_df stores the strategy-used probability as 'model_prob_used'
        # (it is NOT one of the p_model_fit/p_real_mc precedence names). Use it
        # directly; the generic resolver would miss it and skip calibration.
        model_col = "model_prob_used"
        if model_col not in all_priced_df.columns:
            logger.info("Calibration skipped: no model probability column found.")
            return

        outcome_col = "outcome_yes"
        if outcome_col not in all_priced_df.columns:
            logger.info("Calibration skipped: no outcome column found.")
            return

        # Need finite outcomes
        valid = all_priced_df[[model_col, outcome_col]].dropna()
        if len(valid) < 20:
            logger.info(
                "Calibration skipped: only %d resolved contracts (need ≥20).",
                len(valid),
            )
            return

        # Write to temp CSV so run_calibration_report can read it
        tmp_path = os.path.join(tempfile.gettempdir(), f"_bt_cal_{id(self)}.csv")
        try:
            valid.to_csv(tmp_path, index=False)
            report = run_calibration_report(
                tmp_path,
                prob_col=model_col,
                outcome_col=outcome_col,
            )
            if report is not None:
                logger.info(
                    "Calibration — n=%d | Brier=%.4f | ECE=%.4f | "
                    "Mean_fc=%.4f | Mean_out=%.4f | Bias=%+.4f",
                    report.n_obs, report.brier, report.ece,
                    report.mean_forecast, report.mean_outcome,
                    report.calibration_bias,
                )
                # Log reliability bins at debug level
                logger.debug("Reliability bins:\n%s", report.bins.to_string(index=False))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# Convenience function for simple usage
def run_backtest(
    daily_batches: List[pd.DataFrame],
    initial_bankroll: float,
    strategy_params: Dict,
    btc_price_path: str = str(_DEFAULT_INTRADAY),
    price_df: Optional[pd.DataFrame] = None,
    return_all_priced: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Run backtest across sorted daily batches.

    This is a convenience wrapper around BacktestEngine for dashboard compatibility.

    Args:
        daily_batches: List of batch DataFrames
        initial_bankroll: Starting capital in USD
        strategy_params: Dictionary of strategy parameters
        btc_price_path: Path to BTC intraday data (used if price_df not provided)
        price_df: Optional pre-loaded BTC price DataFrame (takes precedence over file path)
        return_all_priced: If True, also return all_priced_df with every evaluated contract

    Returns:
        If return_all_priced=False: Tuple of (trades_df, equity_df)
        If return_all_priced=True: Tuple of (trades_df, equity_df, all_priced_df)
    """
    engine = BacktestEngine(
        market_data_batches=daily_batches,
        initial_bankroll=initial_bankroll,
        strategy_params=strategy_params,
        btc_price_path=btc_price_path,
        price_df=price_df,
    )
    return engine.run(return_all_priced=return_all_priced)
