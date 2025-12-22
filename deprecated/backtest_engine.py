#!/usr/bin/env python3
"""Core backtest engine for Auto-Reco strategy.

This module is UI-agnostic and can be called from the Streamlit app or CLI.
It replays daily batch_with_fits data, applies the same recommend_trades logic,
and simulates bankroll evolution as positions resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo

    ET_ZONE = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    ET_ZONE = timezone(timedelta(hours=-5))

from auto_reco import recommend_trades, recommendations_to_dataframe


@dataclass
class OpenPosition:
    pricing_date: pd.Timestamp
    expiry_date: str
    slug: str
    strike: float
    side: str
    entry_price: float
    stake: float
    size_shares: float
    p_model_fit: float = np.nan
    market_price: float = np.nan


def _positions_to_df(open_positions: List[OpenPosition]) -> pd.DataFrame:
    if not open_positions:
        return pd.DataFrame()
    return pd.DataFrame([vars(p) for p in open_positions])


def _extract_outcome(row: pd.Series) -> float:
    """Extract YES outcome in [0,1]."""
    for col in ["outcome_yes", "outcome", "resolved_price", "final_price"]:
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except Exception:
                continue
    return np.nan


# Month name to number mapping
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_expiry_from_slug(slug: str, year: int = 2025) -> datetime:
    """
    Parse expiry date from slug like 'bitcoin-above-92k-on-november-14'.
    Returns datetime at 12:00 ET on the extracted date.
    """
    import re
    slug_lower = slug.lower()
    # Pattern: "on-{month}-{day}" at end of slug
    pattern = r"on-([a-z]+)-(\d+)"
    match = re.search(pattern, slug_lower)
    if match:
        month_name = match.group(1)
        day = int(match.group(2))
        month = MONTH_MAP.get(month_name)
        if month:
            return datetime(year, month, day, 12, 0, tzinfo=ET_ZONE)
    return None


def _infer_outcome_from_price(pos: OpenPosition, price_df: pd.DataFrame) -> float:
    """
    Infer YES outcome from BTC/USDT price at 12:00 ET on expiry date.
    Returns NaN if price not available.
    """
    if price_df is None or price_df.empty:
        return np.nan
    
    # Try parsing expiry_date directly
    expiry_dt = None
    try:
        expiry_date = pd.to_datetime(pos.expiry_date).date()
        expiry_dt = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 12, 0, tzinfo=ET_ZONE).astimezone(
            timezone.utc
        )
    except Exception:
        pass
    
    # Fallback: parse from slug if expiry_date parsing failed
    if expiry_dt is None and pos.slug:
        parsed = _parse_expiry_from_slug(pos.slug)
        if parsed:
            expiry_dt = parsed.astimezone(timezone.utc)
    
    # If still no expiry date, give up
    if expiry_dt is None:
        return np.nan
    
    # Normalize timestamps
    cols_lower = {c.lower(): c for c in price_df.columns}
    ts_col = None
    for cand in ["timestamp", "time", "datetime"]:
        if cand in cols_lower:
            ts_col = cols_lower[cand]
            break
    if ts_col is None:
        return np.nan
    close_col = None
    for cand in ["close", "price"]:
        if cand in cols_lower:
            close_col = cols_lower[cand]
            break
    if close_col is None:
        return np.nan
    target = expiry_dt.replace(second=0, microsecond=0)

    def _lookup(df_ts: pd.Series) -> float:
        df_local = price_df.copy()
        df_local[ts_col] = df_ts
        exact = df_local[df_local[ts_col] == target]
        if exact.empty:
            df_local["delta"] = (df_local[ts_col] - target).abs()
            df_local = df_local.sort_values("delta")
            nearest = df_local[df_local["delta"] <= pd.Timedelta(minutes=2)]
            if nearest.empty:
                return np.nan
            row = nearest.iloc[0]
        else:
            row = exact.iloc[0]
        return float(row[close_col])

    # First try interpreting timestamps as UTC
    ts_utc = pd.to_datetime(price_df[ts_col], utc=True, errors="coerce")
    close_price = _lookup(ts_utc)
    if not np.isnan(close_price):
        return 1.0 if close_price > float(pos.strike) else 0.0

    # Fallback: interpret as naive ET then convert to UTC
    ts_naive = pd.to_datetime(price_df[ts_col], errors="coerce")
    if ts_naive.notna().any():
        if ts_naive.dt.tz is None:
            ts_et = ts_naive.dt.tz_localize(ET_ZONE, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert(
                timezone.utc
            )
        else:
            ts_et = ts_naive.dt.tz_convert(timezone.utc)
        close_price = _lookup(ts_et)
        if not np.isnan(close_price):
            return 1.0 if close_price > float(pos.strike) else 0.0

    return np.nan


def _settle_positions(
    open_positions: List[OpenPosition],
    pricing_date: pd.Timestamp,
    trades_records: List[Dict],
    bankroll: float,
    price_df: pd.DataFrame = None,
) -> float:
    """Settle positions whose expiry_date <= pricing_date using BTC price inference."""
    if not open_positions:
        return bankroll
    keep: List[OpenPosition] = []
    for pos in open_positions:
        try:
            pos_expiry = pd.to_datetime(pos.expiry_date, utc=True)
        except Exception:
            pos_expiry = None
        # Settle on or after expiry (inclusive of expiry date)
        # Normalize pricing_date to UTC for comparison
        pricing_date_utc = pd.to_datetime(pricing_date, utc=True)
        if pos_expiry is not None and pricing_date_utc < pos_expiry:
            keep.append(pos)
            continue
        outcome_yes = _infer_outcome_from_price(pos, price_df)
        
        # Check for "Zombie" positions (Stuck > 2 days past expiry with no price data)
        if np.isnan(outcome_yes) and pos_expiry is not None:
            days_past_expiry = (pricing_date_utc - pos_expiry).days
            if days_past_expiry > 2:
                # FORCE SETTLE as Void/Refund to free up capital
                print(f"WARNING: Force settling zombie position {pos.slug} as Refund (missing price data).")
                payout = pos.stake  # Return original stake (PnL = 0)
                pnl = payout - pos.stake  # = 0
                bankroll += payout
                trades_records.append({
                    "pricing_date": pricing_date,
                    "settled": True,
                    "expiry_date": pos.expiry_date,
                    "slug": pos.slug,
                    "strike": pos.strike,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "stake": pos.stake,
                    "pnl": pnl,
                    "payout": payout,
                    "bankroll_after": bankroll,
                    "outcome_yes": np.nan,
                    "p_model_fit": getattr(pos, "p_model_fit", np.nan),
                    "market_price": getattr(pos, "market_price", np.nan),
                    "note": "FORCE_SETTLED_MISSING_DATA",
                })
                continue  # Skip to next position
        
        # If outcome still unknown and not a zombie, keep position open
        if np.isnan(outcome_yes):
            keep.append(pos)
            continue
        if pos.side.upper() == "YES":
            payout = pos.size_shares * outcome_yes
        else:
            payout = pos.size_shares * (1.0 - outcome_yes)
        pnl = payout - pos.stake
        bankroll += payout
        trades_records.append(
            {
                "pricing_date": pricing_date,
                "settled": True,
                "expiry_date": pos.expiry_date,
                "slug": pos.slug,
                "strike": pos.strike,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "stake": pos.stake,
                "pnl": pnl,
                "payout": payout,
                "bankroll_after": bankroll,
                "outcome_yes": outcome_yes,
                "p_model_fit": getattr(pos, "p_model_fit", np.nan),
                "market_price": getattr(pos, "market_price", np.nan),
            }
        )
    open_positions[:] = keep
    return bankroll


def run_backtest(
    daily_batches: List[pd.DataFrame],
    initial_bankroll: float,
    strategy_params: Dict,
    price_df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run backtest across sorted daily batches."""
    # Sort by pricing_date
    daily_sorted = []
    for df in daily_batches:
        if "pricing_date" not in df.columns:
            raise ValueError("Each batch must have a pricing_date column.")
        df = df.copy()
        df["pricing_date"] = pd.to_datetime(df["pricing_date"])
        daily_sorted.append(df)
    daily_sorted = sorted(daily_sorted, key=lambda d: d["pricing_date"].iloc[0])

    bankroll = float(initial_bankroll)
    open_positions: List[OpenPosition] = []
    trades_records: List[Dict] = []
    equity_records: List[Dict] = []

    for batch_df in daily_sorted:
        pricing_date = pd.to_datetime(batch_df["pricing_date"].iloc[0])

        # Settle positions due/resolved by this date
        bankroll = _settle_positions(open_positions, pricing_date, trades_records, bankroll, price_df=price_df)

        # Build positions_df for sizing (open positions only)
        positions_df = _positions_to_df(open_positions)

        # Run reco with current bankroll
        batch_for_reco = batch_df.copy()
        batch_for_reco["pricing_date"] = pd.to_datetime(batch_for_reco["pricing_date"], utc=True)

        reco_list = recommend_trades(
            batch_for_reco,
            bankroll=bankroll,
            positions_df=positions_df,
            kelly_fraction=strategy_params.get("kelly_fraction", 0.15),
            min_edge=strategy_params.get("min_edge", 0.06),
            max_bets_per_expiry=strategy_params.get("max_bets_per_expiry", 3),
            max_capital_per_expiry_frac=strategy_params.get("max_capital_per_expiry_frac", 0.15),
            max_capital_total_frac=strategy_params.get("max_capital_total_frac", 0.35),
            max_net_delta_frac=strategy_params.get("max_net_delta_frac", 0.20),
            min_price=strategy_params.get("min_price", 0.03),
            max_price=strategy_params.get("max_price", 0.95),
            min_model_prob=strategy_params.get("min_model_prob", 0.0),
            max_model_prob=strategy_params.get("max_model_prob", 1.0),
            require_active=strategy_params.get("require_active", True),
            use_stability_penalty=strategy_params.get("use_stability_penalty", True),
            allow_no=strategy_params.get("allow_no", True),
            correlation_penalty=strategy_params.get("correlation_penalty", 0.25),
            min_trade_usd=(
                strategy_params.get("min_trade_usd")
                if strategy_params.get("min_trade_usd") is not None
                else bankroll * strategy_params.get("min_trade_frac", 0.01)
            ),
            disable_staleness=strategy_params.get("disable_staleness", False),
        )
        reco_df = recommendations_to_dataframe(reco_list) if reco_list else pd.DataFrame()

        if not reco_df.empty:
            for _, trade in reco_df.iterrows():
                stake = float(trade.get("suggested_stake", trade.get("stake_dollars", trade.get("stake_usd", 0.0))))
                if stake <= 0 or stake > bankroll:
                    continue
                price_yes = float(trade["market_price"])
                side = str(trade["side"]).upper()
                model_prob = float(trade.get("model_prob", np.nan))
                bankroll_before = bankroll
                bankroll -= stake
                if side == "YES":
                    size_shares = stake / price_yes
                else:
                    size_shares = stake / max(1.0 - price_yes, 1e-6)
                # Extract expiry_date, falling back to expiry_key if expiry_date is missing/invalid
                raw_expiry = trade.get("expiry_date")
                if pd.isna(raw_expiry) or raw_expiry is None or str(raw_expiry) in ("", "NaT", "nan", "1.0"):
                    expiry_str = str(trade.get("expiry_key", ""))
                else:
                    expiry_str = str(raw_expiry)
                
                open_positions.append(
                    OpenPosition(
                        pricing_date=pricing_date,
                        expiry_date=expiry_str,
                        slug=str(trade.get("slug", "")),
                        strike=float(trade.get("strike", np.nan)),
                        side=side,
                        entry_price=price_yes if side == "YES" else (1.0 - price_yes),
                        stake=stake,
                        size_shares=size_shares,
                        p_model_fit=model_prob,
                        market_price=price_yes,
                    )
                )
                trades_records.append(
                    {
                        "pricing_date": pricing_date,
                        "expiry_date": expiry_str,
                        "slug": trade.get("slug", ""),
                        "strike": trade.get("strike", np.nan),
                        "side": side,
                        "p_model_fit": model_prob,
                        "market_price": price_yes,
                        "stake": stake,
                        "bankroll_before": bankroll_before,
                        "bankroll_after": bankroll,
                        "kelly_applied": trade.get("kelly_fraction_applied"),
                        "corr_multiplier": trade.get("corr_multiplier"),
                        "expected_value_dollars": trade.get("expected_value_dollars"),
                        "settled": False,
                    }
                )

        equity_records.append({"pricing_date": pricing_date, "bankroll": bankroll})

    # Final settlement block REMOVED to prevent leaking future outcomes.
    # Users expect the backtest to end exactly on the last loaded batch date.
    # Future positions should remain OPEN (settled=False).
    
    # if open_positions and price_df is not None:
    #     remaining = list(open_positions)
    #     open_positions.clear()
    #     for pos in remaining:
    #         expiry_dt = pd.to_datetime(pos.expiry_date, errors="coerce")
    #         pricing_date = expiry_dt if pd.notna(expiry_dt) else (daily_sorted[-1]["pricing_date"].iloc[0] if daily_sorted else pd.Timestamp.utcnow())
    #         bankroll = _settle_positions([pos], pricing_date, trades_records, bankroll, price_df=price_df)

    trades_df = pd.DataFrame(trades_records)
    equity_df = pd.DataFrame(equity_records)
    return trades_df, equity_df
