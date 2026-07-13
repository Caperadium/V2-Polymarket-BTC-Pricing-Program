"""Pure (streamlit-free) helpers for app/pages/mm_monitor.py's multi-expiry
tabs. Split out because mm_monitor.py executes main() at import time (it is a
Streamlit page), which makes it untestable to import -- these helpers carry
the tab-derivation and expiry-attribution logic so tests can cover it.

Expiry attribution rule: everything joins market_id -> expiry_key through the
state db's `markets` registry (never a fills.csv column -- the csv schema is
append-mode and cannot grow columns). The registry is never pruned and
accumulates every historical event's markets, so tab lists are always DERIVED
by filtering (current run's events + expiries actually present in the data),
never by enumerating the registry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from market_maker.config import MAKER_REBATE_SHARE_CRYPTO, TAKER_FEE_RATE_CRYPTO

UNKNOWN_EXPIRY = "unknown"


def run_meta_expiries(run_meta: Optional[Dict[str, Any]]) -> List[str]:
    """Expiries the current run is (or was) quoting: the multi-expiry
    `events` list when present, else the legacy singular `expiry_key`."""
    if not isinstance(run_meta, dict):
        return []
    events = run_meta.get("events")
    if isinstance(events, list) and events:
        out = []
        for e in events:
            ek = e.get("expiry_key") if isinstance(e, dict) else None
            if isinstance(ek, str) and ek and ek not in out:
                out.append(ek)
        if out:
            return out
    ek = run_meta.get("expiry_key")
    return [ek] if isinstance(ek, str) and ek else []


def registry_expiry_map(registry_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """market_id -> expiry_key from a `SELECT market_id, expiry_key, strike
    FROM markets` frame. Empty dict when the table is missing/empty."""
    if registry_df is None or registry_df.empty:
        return {}
    if "market_id" not in registry_df.columns or "expiry_key" not in registry_df.columns:
        return {}
    return {
        str(r["market_id"]): str(r["expiry_key"])
        for _, r in registry_df.iterrows()
        if pd.notna(r["market_id"]) and pd.notna(r["expiry_key"])
    }


def registry_strike_map(registry_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """market_id -> strike from the same registry frame."""
    if registry_df is None or registry_df.empty or "strike" not in getattr(registry_df, "columns", ()):
        return {}
    out: Dict[str, float] = {}
    for _, r in registry_df.iterrows():
        if pd.notna(r.get("market_id")) and pd.notna(r.get("strike")):
            try:
                out[str(r["market_id"])] = float(r["strike"])
            except (TypeError, ValueError):
                continue
    return out


def attach_expiry(
    df: Optional[pd.DataFrame], expiry_map: Dict[str, str], market_col: str
) -> pd.DataFrame:
    """Return a copy of `df` with an `expiry` column mapped from `market_col`
    through the registry map; unmapped markets get UNKNOWN_EXPIRY."""
    if df is None or df.empty or market_col not in df.columns:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    out["expiry"] = out[market_col].map(lambda m: expiry_map.get(str(m), UNKNOWN_EXPIRY))
    return out


def expiry_tabs_order(
    meta_expiries: List[str], data_expiries: List[str]
) -> List[str]:
    """Tab labels: sorted union of the run's own expiries and the expiries
    actually present in the rendered data; UNKNOWN_EXPIRY (if any) last. A
    single-expiry run yields exactly one tab."""
    known = {e for e in meta_expiries if e and e != UNKNOWN_EXPIRY}
    known |= {e for e in data_expiries if e and e != UNKNOWN_EXPIRY}
    tabs = sorted(known)
    if UNKNOWN_EXPIRY in (data_expiries or []):
        tabs.append(UNKNOWN_EXPIRY)
    return tabs


def split_by_expiry(df: pd.DataFrame, expiry_col: str = "expiry") -> Dict[str, pd.DataFrame]:
    """{expiry: sub-frame} for a frame that already carries `expiry_col`."""
    if df is None or df.empty or expiry_col not in df.columns:
        return {}
    return {str(ek): g for ek, g in df.groupby(expiry_col, sort=True)}


def event_meta_by_expiry(run_meta: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """expiry_key -> its event dict from run_meta['events'] (multi-expiry),
    or a synthesized single entry from the legacy fields."""
    if not isinstance(run_meta, dict):
        return {}
    events = run_meta.get("events")
    if isinstance(events, list) and events:
        return {
            e["expiry_key"]: e for e in events
            if isinstance(e, dict) and isinstance(e.get("expiry_key"), str)
        }
    ek = run_meta.get("expiry_key")
    if isinstance(ek, str) and ek:
        return {ek: {
            "expiry_key": ek,
            "event_slug": run_meta.get("event_slug", "?"),
            "strikes": run_meta.get("strikes") or [],
        }}
    return {}


def rebates_from_fills_df(fills_df: Optional[pd.DataFrame]) -> float:
    """Total estimated Polymarket maker rebate over a raw `fills`-table
    DataFrame (columns: price, size, liquidity) -- MAKER rows only (TAKER
    pays the fee instead of earning a rebate; SETTLEMENT pseudo-fills are not
    venue fills). Tolerant of None, empty, or missing-column frames (returns
    0.0): the PnL panel's secondary fills query is not allowed to block the
    rest of the panel from rendering (see app/pages/mm_monitor.py render_pnl).

    Vectorized pandas arithmetic using the two market_maker.config venue
    constants directly (same formula as market_maker.pnl_report.
    rebate_for_fill, applied per-row here instead of per-fill object) -- a
    light import that keeps this module dependency-light; deliberately does
    NOT import market_maker.pnl_report.

    This is a display-only ESTIMATE (pro-rata pool identity assumption, see
    pnl_report's module docstring "Maker rebates" section) -- never added to
    equity/realized/bankroll/sizing."""
    if fills_df is None or fills_df.empty:
        return 0.0
    required = {"price", "size", "liquidity"}
    if not required.issubset(fills_df.columns):
        return 0.0
    maker = fills_df[fills_df["liquidity"] == "MAKER"]
    if maker.empty:
        return 0.0
    price = pd.to_numeric(maker["price"], errors="coerce")
    size = pd.to_numeric(maker["size"], errors="coerce")
    rebate = MAKER_REBATE_SHARE_CRYPTO * TAKER_FEE_RATE_CRYPTO * price * (1.0 - price) * size
    return float(rebate.fillna(0.0).sum())
