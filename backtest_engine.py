#!/usr/bin/env python3
"""
backtest_engine.py

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

from auto_reco import recommend_trades, recommendations_to_dataframe

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
        btc_price_path: str = "DATA/btc_intraday_1m.csv",
        price_df: Optional[pd.DataFrame] = None,
    ):
        self.batches = market_data_batches
        self.initial_bankroll = initial_bankroll
        self.strategy_params = strategy_params
        self.btc_price_path = btc_price_path
        self._price_df_provided = price_df
        
        # Internal state
        self._btc_prices: Optional[pd.DataFrame] = None
        self._running_bankroll: float = initial_bankroll
        self._open_positions: List[OpenPosition] = []
        self._closed_trades: List[Dict] = []
        self._equity_snapshots: List[Dict] = []
        self._all_priced_contracts: List[Dict] = []  # Track ALL evaluated contracts
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
            # Load from file
            path = Path(self.btc_price_path)
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
            logger.info(f"Loaded {len(self._btc_prices)} BTC price records")
            
        except Exception as e:
            logger.error(f"Failed to process BTC prices: {e}")
            self._btc_prices = pd.DataFrame()
    
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
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Authoritative source for outcome resolution.
        
        Args:
            expiry_date: Contract expiry date
            strike: Contract strike price
            
        Returns:
            Tuple of (outcome_yes, btc_price_at_settlement)
            - outcome_yes: 1.0 if btc_price > strike (strict >), else 0.0
            - btc_price_at_settlement: BTC price at 12:00 ET on expiry day
            - Both are None if price data is unavailable
        """
        try:
            expiry_dt_utc = self._get_expiry_datetime(expiry_date)
            btc_price = self._get_btc_price_at(expiry_dt_utc)
            
            if btc_price is None:
                return None, None
            
            # Strict inequality: YES wins if btc_price > strike
            outcome_yes = 1.0 if btc_price > strike else 0.0
            return outcome_yes, btc_price
            
        except Exception:
            return None, None
    
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
            outcome_yes, btc_price = self.resolve_outcome_yes(pos_expiry, pos.strike)
            
            if outcome_yes is None:
                # No price data - log warning and skip (keep position open for now)
                # Or force settle if >2 days past expiry
                days_past = (current_time - pos_expiry).days
                if days_past > 2:
                    # Force settle as refund (return stake, PnL = 0)
                    logger.warning(
                        f"Force settling {pos.slug} as refund (missing BTC price data)"
                    )
                    payout = pos.stake
                    pnl = 0.0
                    outcome_yes = np.nan
                    btc_price = np.nan
                else:
                    remaining.append(pos)
                    continue
            else:
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
                "btc_price_at_expiry": btc_price if btc_price is not None else np.nan,
                "outcome_yes": outcome_yes,
                "payout": payout,
                "pnl": pnl,
                "settled": True,
                "settlement_date": current_time,
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
        model_prob_col = self._get_model_prob_col(batch_df)
        if model_prob_col is not None:
            # Get spot price at snapshot time for moneyness calculation
            snapshot_spot = self._get_btc_price_at(current_time)
            
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
            
            for _, row in batch_df.iterrows():
                try:
                    strike = float(row.get("strike", np.nan))
                    model_prob = float(row.get(model_prob_col, np.nan))
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
                    
                    self._all_priced_contracts.append({
                        "snapshot_time": current_time,
                        "expiry_date": expiry_date,
                        "strike": strike,
                        "spot_price": snapshot_spot if snapshot_spot is not None else np.nan,
                        "market_yes_price": market_yes_price,
                        "model_prob_used": model_prob,
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
        
        # Call auto_reco with current open positions
        reco_list = recommend_trades(
            df=batch_df,
            bankroll=self._running_bankroll,
            positions_df=positions_df,
            current_open_positions=positions_df,
            **params
        )
        
        if not reco_list:
            return
        
        reco_df = recommendations_to_dataframe(reco_list)
        
        # Execute each trade
        for _, trade in reco_df.iterrows():
            stake = float(
                trade.get("suggested_stake", 
                    trade.get("stake_dollars", 
                        trade.get("stake_usd", 0.0)
                    )
                )
            )
            
            if stake <= 0 or stake > self._running_bankroll:
                continue
            
            price_yes = float(trade["market_price"])
            side = str(trade["side"]).upper()
            
            # Calculate shares: stake / entry_price
            if side == "YES":
                entry_price = price_yes
            else:
                entry_price = max(1.0 - price_yes, 1e-6)
            
            size_shares = stake / entry_price
            
            # Deduct stake from bankroll
            self._running_bankroll -= stake
            
            # Parse expiry date
            raw_expiry = trade.get("expiry_date")
            if pd.isna(raw_expiry) or raw_expiry is None or str(raw_expiry) in ("", "NaT", "nan"):
                expiry_str = str(trade.get("expiry_key", ""))
            else:
                expiry_str = str(raw_expiry)
            
            try:
                expiry_ts = pd.to_datetime(expiry_str)
            except Exception:
                expiry_ts = pd.NaT

            expiry_key_str = str(trade.get("expiry_key", ""))
            
            # Create position
            position = OpenPosition(
                pricing_date=current_time,
                expiry_date=expiry_ts,
                slug=str(trade.get("slug", "")),
                strike=float(trade.get("strike", np.nan)),
                side=side,
                entry_price=entry_price,
                stake=stake,
                size_shares=size_shares,
                model_prob=float(trade.get("model_prob", np.nan)),
                market_price=price_yes,
                trade_id=f"T{self._trade_counter}",
                kelly_applied=float(trade.get("kelly_fraction_applied", np.nan)),
                expiry_key=expiry_key_str,
            )
            self._trade_counter += 1
            self._open_positions.append(position)
            
            # Log trade entry
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
                "market_price": price_yes,
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
        if not trades_df.empty and "trade_id" in trades_df.columns:
            # Consolidate rows: keep last entry per trade_id (settled > unsettled)
            trades_df = trades_df.sort_values(["settled", "pricing_date"])
            trades_df = trades_df.drop_duplicates(subset=["trade_id"], keep="last")
            # Restore chronological order by entry time (approx)
            trades_df = trades_df.sort_values("trade_id", key=lambda x: x.str[1:].astype(int))
        equity_df = pd.DataFrame(self._equity_snapshots)
        
        if return_all_priced:
            all_priced_df = self._resolve_all_priced_contracts()
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
        resolution_cache: Dict[Tuple, Tuple[Optional[float], Optional[float]]] = {}
        
        for idx, row in all_priced_df.iterrows():
            expiry = row["expiry_date"]
            strike = row["strike"]
            
            # Skip if missing required fields
            if pd.isna(expiry) or pd.isna(strike):
                continue
            
            cache_key = (expiry, strike)
            
            if cache_key not in resolution_cache:
                # Resolve once per unique (expiry, strike) pair
                outcome_yes, btc_price = self.resolve_outcome_yes(expiry, strike)
                resolution_cache[cache_key] = (outcome_yes, btc_price)
            
            outcome_yes, btc_price = resolution_cache[cache_key]
            all_priced_df.at[idx, "outcome_yes"] = outcome_yes if outcome_yes is not None else np.nan
            all_priced_df.at[idx, "btc_price_at_settlement"] = btc_price if btc_price is not None else np.nan
        
        return all_priced_df


# Convenience function for simple usage
def run_backtest(
    daily_batches: List[pd.DataFrame],
    initial_bankroll: float,
    strategy_params: Dict,
    btc_price_path: str = "DATA/btc_intraday_1m.csv",
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

