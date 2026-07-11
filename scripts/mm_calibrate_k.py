"""Fit the Dalen arrival decay k from recorded trade prints (no fills needed).

The market-maker's launch k_arrival was a placeholder ("quote wide, calibrate
from fills") -- but wide quotes produced zero fills, so there was never fill
data to calibrate from. This script breaks that loop by fitting the arrival
intensity from the venue's OWN trade-print stream, which the paper runner
records to the state db (`trade_prints` table, 2026-07-11) regardless of
whether we ever get filled:

    lambda(delta_x) = A * exp(-k * delta_x)

where delta_x is the print's log-odds distance from the prevailing mid
(mid_log table, per-tick). Method:

1. Join each print to the last mid at or before its timestamp (per market,
   backward asof-join, staleness-capped).
2. delta_x = |logit(print_price) - logit(mid)|, p clamped to the p-band.
3. Bin delta_x; per-bin arrival rate = count / observed_hours, where
   observed_hours is the mid_log coverage span summed per market.
4. Weighted least squares on ln(rate) vs bin center (weights = counts):
   slope = -k, intercept = ln(A).

Caveats (deliberate, first-pass):
- Print intensity at distance delta from mid is a PROXY for "our resting
  quote at distance delta trades". The paper fill sim is queue-behind
  conservative, so realized fill rates will be lower than lambda; k (the
  decay SHAPE) transfers better than A (the level).
- Extreme-p markets are excluded by default (--p-min/--p-max): near the
  clamps, logit distances explode and books are degenerate.

Usage (VPS):
    python scripts/mm_calibrate_k.py --state-db market_maker/mm_paper_state.db
    python scripts/mm_calibrate_k.py --state-db ... --days 3 --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

DEFAULT_P_LO = 0.001
DEFAULT_P_HI = 0.999


def _logit(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    p = np.clip(p, lo, hi)
    return np.log(p / (1.0 - p))


def load_frames(db_path: str, since: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    try:
        since_s = since.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        prints = pd.read_sql_query(
            "SELECT ts, market_id, price, size FROM trade_prints WHERE ts >= ? ORDER BY ts",
            con, params=(since_s,),
        )
        mids = pd.read_sql_query(
            "SELECT ts, market_id, mid FROM mid_log WHERE ts >= ? ORDER BY ts",
            con, params=(since_s,),
        )
    finally:
        con.close()
    for df, col in ((prints, "ts"), (mids, "ts")):
        df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)
    return prints, mids


def join_prints_to_mids(
    prints: pd.DataFrame, mids: pd.DataFrame, max_mid_age_s: float
) -> pd.DataFrame:
    """Backward asof-join each print to the last mid <= its ts, per market."""
    out = []
    for market_id, grp in prints.groupby("market_id"):
        m = mids[mids["market_id"] == market_id]
        if m.empty:
            continue
        joined = pd.merge_asof(
            grp.sort_values("ts"),
            m[["ts", "mid"]].sort_values("ts").rename(columns={"ts": "mid_ts"}),
            left_on="ts", right_on="mid_ts",
            direction="backward",
            tolerance=pd.Timedelta(seconds=max_mid_age_s),
        )
        out.append(joined.dropna(subset=["mid"]))
    if not out:
        return pd.DataFrame(columns=["ts", "market_id", "price", "size", "mid"])
    return pd.concat(out, ignore_index=True)


def coverage_hours(mids: pd.DataFrame, markets: pd.Index) -> float:
    """Sum of per-market mid_log coverage spans, in hours. This is the
    market-hours denominator for per-bin arrival rates (each market's book
    is an independent arrival stream)."""
    total = 0.0
    for market_id in markets:
        m = mids[mids["market_id"] == market_id]
        if len(m) >= 2:
            total += (m["ts"].max() - m["ts"].min()).total_seconds() / 3600.0
    return total


def fit_k(
    joined: pd.DataFrame, hours: float, bin_x: float, max_x: float,
    p_lo: float, p_hi: float,
) -> dict:
    x_print = _logit(joined["price"].to_numpy(float), p_lo, p_hi)
    x_mid = _logit(joined["mid"].to_numpy(float), p_lo, p_hi)
    delta = np.abs(x_print - x_mid)
    delta = delta[delta <= max_x]

    edges = np.arange(0.0, max_x + bin_x, bin_x)
    counts, _ = np.histogram(delta, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = counts > 0
    if keep.sum() < 3:
        raise SystemExit(
            f"only {int(keep.sum())} non-empty delta_x bins -- not enough to fit; "
            "collect more prints or widen --bin-x"
        )
    rate = counts[keep] / max(hours, 1e-9)  # prints per market-hour per bin
    ln_rate = np.log(rate)
    # WLS: weight by counts (Poisson-ish variance of ln(count) ~ 1/count)
    w = counts[keep].astype(float)
    slope, intercept = np.polyfit(centers[keep], ln_rate, 1, w=np.sqrt(w))
    k = -float(slope)
    A = float(math.exp(intercept))

    bins_table = [
        {"delta_x_center": round(float(c), 4), "count": int(n),
         "rate_per_hour": round(float(n / max(hours, 1e-9)), 4)}
        for c, n in zip(centers[keep], counts[keep])
    ]
    return {"k": k, "A_per_hour": A, "n_prints_used": int(counts.sum()),
            "bins": bins_table}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state-db", required=True, help="MMStateStore sqlite path")
    ap.add_argument("--days", type=float, default=7.0, help="lookback window (default 7)")
    ap.add_argument("--bin-x", type=float, default=0.02, help="delta_x bin width (log-odds units)")
    ap.add_argument("--max-x", type=float, default=1.0, help="max delta_x considered")
    ap.add_argument("--p-min", type=float, default=0.05, help="exclude prints with mid below this")
    ap.add_argument("--p-max", type=float, default=0.95, help="exclude prints with mid above this")
    ap.add_argument("--max-mid-age-s", type=float, default=60.0, help="max mid staleness at print ts")
    ap.add_argument("--gamma", type=float, default=0.10, help="MMConfig.gamma, for the implied arrival-term report")
    ap.add_argument("--json", default=None, help="also write the result dict to this path")
    args = ap.parse_args(argv)

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    prints, mids = load_frames(args.state_db, since)
    print(f"loaded {len(prints)} prints, {len(mids)} mids since {since.isoformat()}")
    if prints.empty or mids.empty:
        raise SystemExit("no prints or no mids in window -- run the paper runner longer first")

    joined = join_prints_to_mids(prints, mids, args.max_mid_age_s)
    joined = joined[(joined["mid"] >= args.p_min) & (joined["mid"] <= args.p_max)]
    print(f"{len(joined)} prints joined to a fresh mid inside p-band "
          f"[{args.p_min}, {args.p_max}] across {joined['market_id'].nunique()} markets")
    if joined.empty:
        raise SystemExit("no usable prints after join/p-band filter")

    hours = coverage_hours(mids, joined["market_id"].unique())
    result = fit_k(joined, hours, args.bin_x, args.max_x, DEFAULT_P_LO, DEFAULT_P_HI)
    result.update({
        "window_days": args.days,
        "market_hours_observed": round(hours, 2),
        "p_band": [args.p_min, args.p_max],
        "bin_x": args.bin_x,
    })

    k, g = result["k"], args.gamma
    if k > 0:
        arr_x = (1.0 / k) * math.log1p(g / k)
        result["implied_arrival_halfspread_x"] = round(arr_x, 6)
        result["implied_arrival_halfspread_cents_atm"] = round(100 * arr_x * 0.25, 3)

    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.json}")

    if k <= 0:
        print("WARNING: fitted k <= 0 (arrival rate not decaying with distance) "
              "-- do not use; inspect the bins table")
        return 1
    print(f"suggested MMConfig.k_arrival = {k:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
