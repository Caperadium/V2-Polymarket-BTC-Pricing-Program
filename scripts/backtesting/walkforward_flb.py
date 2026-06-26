#!/usr/bin/env python3
"""
walkforward_flb.py — standalone favorite-longshot-bias (FLB) walkforward test.

Model-free. Partitions the Polymarket contract history into N contiguous
calendar windows and, per window, measures the longshot *gap* (how much the
market over-prices OTM longshot YES) with a Wilson CI, then simulates the
deployable PnL of buying NO on every in-band contract and holding to
resolution.

NOT a predictor and NOT an in-sample/out-of-sample evaluation. Nothing is
*fitted* on outcomes here: the gap is a descriptive realized statistic per
window and the PnL is the realized result of a fixed mechanical rule. The
windows are descriptive calendar partitions — no trained parameter is carried
across windows — so there is no train/test leak to guard (unlike the M2
logit-shift in core/backtesting/in_sample_oos.py).

Two distinct time axes (kept separate on purpose):
  * entry_date    — midnight-UTC date of the price observation; drives the band
                    filter, moneyness spot, and the window partition.
  * settlement    — 12:00 ET on expiry day (≈17:00 UTC in winter), via the
                    engine's resolve_outcome_yes(); drives the binary outcome
                    ONLY. Never used for windowing.

Data sources (reused, no duplication of leak-free logic):
  * ContractPriceStore.load()  — daily mid YES prices, strike, expiry_date.
                                 (its `resolution` column is all-NaN, so it is
                                 NOT an outcome source.)
  * BacktestEngine helpers      — _spot_as_of() for leak-free moneyness spot,
                                 resolve_outcome_yes() / _expiry_is_settleable()
                                 for BTC-settled outcomes.

Spread note: the historical store keeps a single mid YES price — there is no
recorded bid/ask. The NO ask is modeled as (1 - mid_yes) + spread/2, with
`spread` an explicit assumption (default 0.02). Polymarket CLOB trading fees
are ~0, so `--fee` defaults to 0.

Usage:
    python scripts/backtesting/walkforward_flb.py --bankroll 1000 \
        [--stake 10] [--spread 0.02] [--fee 0.0] \
        [--band-lo 0.05 --band-hi 0.20] [--otm-threshold 0.0] \
        [--n-windows 4 | --window-months 2.75] \
        [--store DATA/historical_contract_prices.csv] \
        [--intraday DATA/btc_intraday_1m.csv] \
        [--out temp/flb_walkforward.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make `core` importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.backtesting.contract_store import ContractPriceStore, DEFAULT_CSV_PATH
from core.backtesting.backtest_engine import BacktestEngine, _DEFAULT_INTRADAY

_DAYS_PER_MONTH = 30.44  # average Gregorian month, for --window-months width


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (center - half, center + half)


def sharpe(returns: np.ndarray) -> float:
    """Per-trade Sharpe = mean / std (sample std, ddof=1). NaN if undefined.

    Un-annualized: each contract resolution is treated as one trade. There is
    no single sensible periodization across heterogeneous holding periods, so
    annualization is deliberately omitted.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd)


# ---------------------------------------------------------------------------
# Data build
# ---------------------------------------------------------------------------

def build_contract_frame(
    store_path: Path,
    intraday_path: Path,
    band_lo: float,
    band_hi: float,
    otm_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Build a one-row-per-contract frame of first-in-band NO entries.

    Returns (frame, stats). `frame` columns:
        clobTokenId, slug, entry_date, entry_mid_yes, moneyness, strike,
        expiry_date, outcome_yes.
    `stats` carries the drop accounting for transparency.
    """
    store = ContractPriceStore(csv_path=store_path)
    raw = store.load()
    stats: Dict[str, int] = {
        "store_rows": int(len(raw)),
        "store_contracts": int(raw["clobTokenId"].nunique()) if not raw.empty else 0,
    }
    if raw.empty:
        return pd.DataFrame(), stats

    # Engine instance purely for its leak-free BTC helpers (empty batches).
    engine = BacktestEngine(
        market_data_batches=[],
        initial_bankroll=0.0,
        strategy_params={},
        btc_price_path=str(intraday_path),
    )
    engine._load_btc_prices()
    if engine._btc_prices is None or engine._btc_prices.empty:
        raise SystemExit(
            f"No BTC intraday data loaded from {intraday_path}. "
            "Run `python core/data/data_fetcher.py` to backfill before this test."
        )

    # --- Pre-flight: does intraday cover the contract settlement span? ---
    exp = pd.to_datetime(raw["expiry_date"], utc=True, errors="coerce")
    cov_lo, cov_hi = engine._intraday_min, engine._intraday_max
    if exp.notna().any():
        e_lo, e_hi = exp.min(), exp.max()
        if e_lo < cov_lo or e_hi > cov_hi:
            print(
                f"[WARN] intraday coverage {cov_lo} -> {cov_hi} does NOT span "
                f"contract expiries {e_lo} -> {e_hi}; contracts outside coverage "
                f"are unsettleable and will be dropped.",
                file=sys.stderr,
            )

    df = raw.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["price", "strike", "date", "expiry_date", "clobTokenId"])

    # Moneyness per observation, memoized per unique observation date
    # (_spot_as_of is O(n) per call; ~hundreds of unique midnight dates).
    spot_cache: Dict[pd.Timestamp, Optional[float]] = {}

    def spot_for(ts: pd.Timestamp) -> Optional[float]:
        if ts not in spot_cache:
            spot_cache[ts] = engine._spot_as_of(ts.to_pydatetime())
        return spot_cache[ts]

    spots = df["date"].map(spot_for)
    df["spot"] = pd.to_numeric(spots, errors="coerce")
    df = df[df["spot"] > 0].copy()
    df["moneyness"] = (df["strike"] - df["spot"]) / df["spot"]

    # Band membership: market YES in [band_lo, band_hi] AND OTM (strict >).
    in_band = (
        (df["price"] >= band_lo)
        & (df["price"] <= band_hi)
        & (df["moneyness"] > otm_threshold)
    )
    band_df = df[in_band].copy()
    stats["contracts_with_spot"] = int(df["clobTokenId"].nunique())
    stats["contracts_ever_in_band"] = int(band_df["clobTokenId"].nunique())
    stats["contracts_never_in_band"] = (
        stats["contracts_with_spot"] - stats["contracts_ever_in_band"]
    )
    if band_df.empty:
        return pd.DataFrame(), stats

    # First in-band observation per contract = the entry.
    band_df = band_df.sort_values("date")
    entry = band_df.groupby("clobTokenId", as_index=False).first()
    entry = entry.rename(columns={"date": "entry_date", "price": "entry_mid_yes"})

    # Resolve outcome via BTC settlement (memoized per expiry+strike). Drop
    # contracts that are unsettleable / unresolved (no 12:00-ET intraday print).
    res_cache: Dict[Tuple, Optional[float]] = {}

    def outcome_for(expiry, strike: float) -> Optional[float]:
        key = (expiry, strike)
        if key not in res_cache:
            if not engine._expiry_is_settleable(pd.Timestamp(expiry)):
                res_cache[key] = None
            else:
                o, _, _ = engine.resolve_outcome_yes(expiry, strike)
                res_cache[key] = o
        return res_cache[key]

    entry["outcome_yes"] = [
        outcome_for(e, s) for e, s in zip(entry["expiry_date"], entry["strike"])
    ]
    n_in_band = len(entry)
    entry = entry.dropna(subset=["outcome_yes"]).copy()
    stats["contracts_dropped_unsettleable"] = n_in_band - len(entry)
    stats["contracts_final"] = len(entry)

    entry["outcome_yes"] = entry["outcome_yes"].astype(float)
    cols = [
        "clobTokenId", "slug", "entry_date", "entry_mid_yes", "moneyness",
        "strike", "expiry_date", "outcome_yes",
    ]
    return entry[cols].reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def assign_windows(
    entry: pd.DataFrame,
    n_windows: int,
    window_months: Optional[float],
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """Partition by entry_date into contiguous calendar windows.

    Default: `n_windows` equal-width windows over [min, max] entry date.
    If `window_months` is given: fixed-width windows of that many months,
    deriving the count to cover the span (the requested n_windows is ignored).
    Returns a list of (start, end_inclusive, sub_df), chronological.
    """
    if entry.empty:
        return []
    t = pd.to_datetime(entry["entry_date"], utc=True)
    t_min, t_max = t.min(), t.max()
    span = t_max - t_min

    if window_months is not None:
        width = pd.Timedelta(days=window_months * _DAYS_PER_MONTH)
        n = max(1, int(math.ceil((span / width)))) if span > pd.Timedelta(0) else 1
    else:
        n = max(1, int(n_windows))
        width = span / n if span > pd.Timedelta(0) else pd.Timedelta(days=1)

    windows = []
    for i in range(n):
        start = t_min + i * width
        # Last window is inclusive of t_max; others are half-open [start, end).
        if i == n - 1:
            end = t_max
            mask = (t >= start) & (t <= end)
        else:
            end = t_min + (i + 1) * width
            mask = (t >= start) & (t < end)
        windows.append((start, end, entry[mask.values].copy()))
    return windows


# ---------------------------------------------------------------------------
# Per-window metrics + PnL
# ---------------------------------------------------------------------------

def window_metrics(
    sub: pd.DataFrame,
    stake: float,
    spread: float,
    fee_frac: float,
    z: float,
    small_n: int,
) -> dict:
    """Gap + Wilson CI and flat-stake buy-NO PnL/Sharpe for one window."""
    n = len(sub)
    if n == 0:
        return {"n": 0, "small_sample": True}

    p = sub["entry_mid_yes"].to_numpy(dtype=float)
    y = sub["outcome_yes"].to_numpy(dtype=float)  # 1 = YES happened
    k = int(np.nansum(y))

    mean_market_p = float(np.mean(p))
    realized_yes_rate = k / n
    gap = mean_market_p - realized_yes_rate

    w_lo, w_hi = wilson_ci(k, n, z)
    # gap = mean_p - rate, so subtracting the rate's CI inverts the bounds.
    gap_lo = mean_market_p - w_hi
    gap_hi = mean_market_p - w_lo

    # --- Buy NO, flat $ stake, hold to resolution ---
    no_ask = np.clip((1.0 - p) + spread / 2.0, 1e-6, 1.0 - 1e-6)
    shares = stake / no_ask
    gross = shares * (1.0 - y)               # NO pays $1 if YES did not happen
    fee = fee_frac * stake
    net_pnl = gross - stake - fee
    per_trade_return = net_pnl / stake

    pnl_total = float(np.sum(net_pnl))
    capital_deployed = stake * n
    roi = pnl_total / capital_deployed if capital_deployed > 0 else float("nan")

    return {
        "n": n,
        "mean_market_p": mean_market_p,
        "realized_yes_rate": realized_yes_rate,
        "gap": gap,
        "gap_ci_lo": gap_lo,
        "gap_ci_hi": gap_hi,
        "wilson_rate_lo": w_lo,
        "wilson_rate_hi": w_hi,
        "pnl": pnl_total,
        "capital_deployed": capital_deployed,
        "roi": roi,
        "sharpe": sharpe(per_trade_return),
        "small_sample": n < small_n,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "   n/a"
    return f"{x:.{nd}f}"


def print_report(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]],
    rows: List[dict],
    overall: dict,
    stats: Dict[str, int],
    args: argparse.Namespace,
) -> None:
    print("=" * 96)
    print("FAVORITE-LONGSHOT WALKFORWARD - buy NO on OTM longshots, hold to resolution")
    print("=" * 96)
    print(
        f"band=[{args.band_lo},{args.band_hi}]  OTM>{args.otm_threshold}  "
        f"spread={args.spread}  fee_frac={args.fee}  stake=${args.stake:g}  "
        f"bankroll=${args.bankroll:g}"
    )
    print(
        f"contracts: store={stats.get('store_contracts',0)}  "
        f"with_spot={stats.get('contracts_with_spot',0)}  "
        f"ever_in_band={stats.get('contracts_ever_in_band',0)}  "
        f"never_in_band={stats.get('contracts_never_in_band',0)}  "
        f"dropped_unsettleable={stats.get('contracts_dropped_unsettleable',0)}  "
        f"final={stats.get('contracts_final',0)}"
    )
    print("-" * 96)
    hdr = (
        f"{'window':<26}{'n':>5}{'mean_p':>9}{'yes_rt':>9}{'gap':>9}"
        f"{'gap_CI':>20}{'PnL$':>11}{'ROI%':>9}{'Shrp':>8}"
    )
    print(hdr)
    print("-" * 96)

    for (start, end, _), r in zip(windows, rows):
        label = f"{start:%Y-%m-%d}..{end:%Y-%m-%d}"
        flag = " *" if r.get("small_sample") else ""
        if r["n"] == 0:
            print(f"{label:<26}{0:>5}  (empty)")
            continue
        ci = f"[{_fmt(r['gap_ci_lo'],3)},{_fmt(r['gap_ci_hi'],3)}]"
        roi_pct = r["roi"] * 100.0 if math.isfinite(r["roi"]) else float("nan")
        print(
            f"{label:<26}{r['n']:>5}{_fmt(r['mean_market_p'],4):>9}"
            f"{_fmt(r['realized_yes_rate'],4):>9}{_fmt(r['gap'],4):>9}"
            f"{ci:>20}{_fmt(r['pnl'],2):>11}{_fmt(roi_pct,2):>9}"
            f"{_fmt(r['sharpe'],3):>8}{flag}"
        )

    print("-" * 96)
    if overall["n"] > 0:
        ci = f"[{_fmt(overall['gap_ci_lo'],3)},{_fmt(overall['gap_ci_hi'],3)}]"
        roi_pct = overall["roi"] * 100.0 if math.isfinite(overall["roi"]) else float("nan")
        print(
            f"{'TOTAL':<26}{overall['n']:>5}{_fmt(overall['mean_market_p'],4):>9}"
            f"{_fmt(overall['realized_yes_rate'],4):>9}{_fmt(overall['gap'],4):>9}"
            f"{ci:>20}{_fmt(overall['pnl'],2):>11}{_fmt(roi_pct,2):>9}"
            f"{_fmt(overall['sharpe'],3):>8}"
        )
    print("=" * 96)
    print("gap = mean market YES - realized YES rate (positive => longshots overpriced => NO edge).")
    print("Sharpe = per-trade net-return mean/std (un-annualized). '*' = small-sample window "
          f"(n < {args.small_n}).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone favorite-longshot-bias walkforward test "
                    "(buy NO on OTM longshots, hold to resolution).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bankroll", type=float, required=True,
                   help="Starting bankroll in USD (context / ROI denominator).")
    p.add_argument("--stake", type=float, default=10.0,
                   help="Flat $ stake per NO trade.")
    p.add_argument("--spread", type=float, default=0.02,
                   help="Assumed total bid-ask spread; NO ask = (1-mid)+spread/2. "
                        "No historical spread is recorded in the store.")
    p.add_argument("--fee", type=float, default=0.0,
                   help="Fee as a fraction of stake per trade (Polymarket CLOB ~0).")
    p.add_argument("--band-lo", type=float, default=0.05,
                   help="Lower bound of the market YES price band.")
    p.add_argument("--band-hi", type=float, default=0.20,
                   help="Upper bound of the market YES price band.")
    p.add_argument("--otm-threshold", type=float, default=0.0,
                   help="OTM filter: keep contracts with moneyness strictly above this.")
    p.add_argument("--n-windows", type=int, default=4,
                   help="Number of equal-width calendar windows.")
    p.add_argument("--window-months", type=float, default=None,
                   help="Override: fixed window width in months (derives count; "
                        "ignores --n-windows).")
    p.add_argument("--small-n", type=int, default=30,
                   help="Window flagged small-sample below this contract count.")
    p.add_argument("--z", type=float, default=1.96,
                   help="Z-score for the Wilson interval (1.96 = 95%%).")
    p.add_argument("--store", type=str, default=str(DEFAULT_CSV_PATH),
                   help="Path to historical_contract_prices.csv.")
    p.add_argument("--intraday", type=str, default=str(_DEFAULT_INTRADAY),
                   help="Path to btc_intraday_1m.csv (settlement + spot).")
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to write the full result as JSON.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    entry, stats = build_contract_frame(
        store_path=Path(args.store),
        intraday_path=Path(args.intraday),
        band_lo=args.band_lo,
        band_hi=args.band_hi,
        otm_threshold=args.otm_threshold,
    )

    if entry.empty:
        print("No in-band, settleable contracts found. Stats:", file=sys.stderr)
        print(json.dumps(stats, indent=2), file=sys.stderr)
        return 1

    windows = assign_windows(entry, args.n_windows, args.window_months)
    rows = [
        window_metrics(sub, args.stake, args.spread, args.fee, args.z, args.small_n)
        for (_, _, sub) in windows
    ]
    overall = window_metrics(
        entry, args.stake, args.spread, args.fee, args.z, args.small_n
    )

    print_report(windows, rows, overall, stats, args)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": vars(args),
            "stats": stats,
            "windows": [
                {
                    "start": f"{s:%Y-%m-%d}",
                    "end": f"{e:%Y-%m-%d}",
                    **{k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                       for k, v in r.items()},
                }
                for (s, e, _), r in zip(windows, rows)
            ],
            "overall": {
                k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in overall.items()
            },
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
