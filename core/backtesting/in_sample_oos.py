#!/usr/bin/env python3
"""
in_sample_oos.py — In-sample / out-of-sample evaluation window for the
Backtesting dashboard tab.

Design (see temp/is_oos_plan.md for the full plan + plan-review resolutions):

The ONLY active, outcome-consuming, pooled fitted component in the pricing path is
the M2 logit-shift ``B`` (``fit_probability_curves.fit_calibration``). Every
BTC-return-process component (GARCH/FIGARCH, jump calibration, regime HMM) and the
per-expiry logistic fit are already fit per-snapshot on strictly ``< snapshot_time``
truncated slices by the backrunner — they are walk-forward leak-free and are NOT
frozen at the cutoff. XGBoost is excluded (dead in the hot path; when re-enabled it
lives per-snapshot in the backrunner, covered by the verifier's BTC arm).

Leak model (§9): the M2 "label" is a contract's *outcome*, timestamped at its
*settlement* (12:00 ET on expiry), NOT its pricing time. So:
  - M2 training population  = contracts whose settlement_time < cutoff (resolved).
  - OOS evaluation population = contracts whose snapshot_time   >= cutoff.
Then every training label resolves ``< cutoff <=`` every OOS pricing time, so the
single cached B never sees a label that post-dates an OOS prediction.

``all_priced_df`` has no settlement_time column; it is derived here from
``expiry_date`` via the same 12:00-ET rule the engine settled outcomes at
(``BacktestEngine._get_expiry_datetime``), reused so training/verifier cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Eastern timezone for the 12:00-ET settlement rule (mirror backtest_engine).
try:  # pragma: no cover - platform dependent
    from zoneinfo import ZoneInfo
    ET_ZONE = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    ET_ZONE = timezone(timedelta(hours=-5))

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = _PROJECT_ROOT / "DATA" / "is_oos_cache"

# Minimum resolved obs per DTE bucket before a fitted M2 shift is trusted. Mirrors
# fit_probability_curves.CALIBRATION_MIN_OBS — re-exported so the page banner and
# this module agree.
from core.pricing.fit_probability_curves import (
    CALIBRATION_MIN_OBS,
    dte_bucket,
    fit_calibration,
)

N_MIN_SAMPLE = 200  # §8 small-sample threshold


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OOSHygieneError(Exception):
    """Raised when an outcome-conditioning query is attempted in OOS mode (§7)."""


class OOSLeakError(AssertionError):
    """Raised by verify_oos_leak_free when a leak is detected (§9)."""


# ---------------------------------------------------------------------------
# Window spec
# ---------------------------------------------------------------------------

class WindowMode(str, Enum):
    IS = "in_sample"
    OOS = "out_of_sample"
    ALL = "all"

    @classmethod
    def from_label(cls, label: str) -> "WindowMode":
        s = str(label).strip().lower()
        if s.startswith("in"):
            return cls.IS
        if s.startswith("out"):
            return cls.OOS
        return cls.ALL


@dataclass(frozen=True)
class WindowSpec:
    cutoff: pd.Timestamp  # midnight-UTC boundary
    mode: WindowMode


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _to_utc(series_or_val):
    """Coerce to tz-aware UTC datetime (Series or scalar)."""
    return pd.to_datetime(series_or_val, utc=True, errors="coerce")


def normalize_cutoff(cutoff) -> pd.Timestamp:
    """Coerce a cutoff (str / date / Timestamp) to a midnight-UTC Timestamp.

    [REVIEW N2] Midnight-floored so the cutoff lands on a snapshot boundary
    (snapshot_time is midnight-floored at backrunner.py:518), never mid-cluster.
    """
    ts = pd.to_datetime(cutoff, utc=True)
    return ts.floor("D")


def derive_settlement_time(expiry_date) -> Optional[pd.Timestamp]:
    """Settlement instant = 12:00 ET on expiry day, as UTC Timestamp.

    Mirrors BacktestEngine._get_expiry_datetime (backtest_engine.py:351-368) so the
    M2 training filter / manifest / §9 verifier use the exact rule outcomes were
    resolved at. Returns None on NaT / unparseable expiry (those rows have NaN
    outcome and are excluded from training anyway).
    """
    ts = pd.to_datetime(expiry_date, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    et_noon = datetime(ts.year, ts.month, ts.day, 12, 0, 0, tzinfo=ET_ZONE)
    return pd.Timestamp(et_noon.astimezone(timezone.utc))


def add_settlement_time(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``settlement_time`` column derived from expiry_date.

    [REVIEW B1] all_priced_df carries no settlement timestamp; derive it.
    """
    out = df.copy()
    if "expiry_date" not in out.columns:
        out["settlement_time"] = pd.NaT
        return out
    uniq = pd.Index(out["expiry_date"].dropna().unique())
    mapping = {e: derive_settlement_time(e) for e in uniq}
    out["settlement_time"] = pd.to_datetime(
        out["expiry_date"].map(mapping), utc=True, errors="coerce"
    )
    return out


def contract_ids(df: pd.DataFrame) -> pd.Series:
    """Stable per-contract id.

    [REVIEW N1] clobTokenId is dropped upstream (contract_store.to_market_df:191)
    so it never reaches all_priced_df — the id is always slug|strike here. The
    clobTokenId branch is kept only as a defensive fallback for other callers.
    """
    if "clobTokenId" in df.columns and df["clobTokenId"].notna().any():
        return df["clobTokenId"].astype(str)
    slugs = df["slug"].astype(str) if "slug" in df.columns else pd.Series("", index=df.index)
    strikes = df["strike"].astype(str) if "strike" in df.columns else pd.Series("", index=df.index)
    return slugs + "|" + strikes


# ---------------------------------------------------------------------------
# Cutoff selection & partitioning
# ---------------------------------------------------------------------------

def compute_default_cutoff(
    all_priced_df: pd.DataFrame,
    target_is_frac: float = 0.7,
    time_col: str = "snapshot_time",
) -> Optional[pd.Timestamp]:
    """Midnight-UTC cutoff that puts ~``target_is_frac`` of unique contracts in IS.

    Split by UNIQUE-CONTRACT count (each contract counted once, at its earliest
    snapshot), not row count — one contract priced across N daily snapshots must
    not weigh N times. Returns None if the frame is empty / lacks the time column.
    """
    if all_priced_df is None or all_priced_df.empty or time_col not in all_priced_df.columns:
        return None
    df = all_priced_df.copy()
    df["_id"] = contract_ids(df)
    df["_ts"] = _to_utc(df[time_col]).dt.floor("D")
    first_snap = df.dropna(subset=["_ts"]).groupby("_id")["_ts"].min().sort_values()
    if first_snap.empty:
        return None
    # IS gets the earliest `target_is_frac` of contracts; cutoff is the first
    # snapshot day NOT in IS (so IS = priced strictly before cutoff).
    n = len(first_snap)
    idx = int(round(target_is_frac * n))
    idx = min(max(idx, 1), n - 1) if n > 1 else 1
    days = list(first_snap.unique())
    # Find the day at the idx-th contract; cutoff = that contract's first-snapshot day.
    cutoff_day = first_snap.iloc[idx]
    # If that day equals the prior contract's day, the split lands mid-day; advance
    # to the next distinct day so a whole snapshot's ladder stays on one side.
    if idx > 0 and first_snap.iloc[idx - 1] == cutoff_day:
        later = [d for d in days if d > cutoff_day]
        if later:
            cutoff_day = later[0]
    return pd.Timestamp(cutoff_day).floor("D")


def partition_contracts(
    all_priced_df: pd.DataFrame,
    cutoff,
    time_col: str = "snapshot_time",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into (is_eval, oos_eval, straddlers) by snapshot_time vs cutoff.

    - is_eval:    snapshot_time <  cutoff
    - oos_eval:   snapshot_time >= cutoff
    - straddlers: snapshot_time <  cutoff AND settlement_time >= cutoff
                  (priced IS, settle OOS — excluded from M2 training, still IS-eval)
    """
    cutoff = normalize_cutoff(cutoff)
    df = add_settlement_time(all_priced_df)
    snap = _to_utc(df[time_col])
    is_mask = snap < cutoff
    oos_mask = snap >= cutoff
    straddle_mask = is_mask & (df["settlement_time"] >= cutoff)
    return df[is_mask].copy(), df[oos_mask].copy(), df[straddle_mask].copy()


def apply_window(
    all_priced_df: pd.DataFrame,
    spec: WindowSpec,
    time_col: str = "snapshot_time",
) -> pd.DataFrame:
    """Filter the contract-level eval frame to the active window by snapshot_time."""
    if spec.mode == WindowMode.ALL or all_priced_df is None or all_priced_df.empty:
        return all_priced_df.copy() if all_priced_df is not None else all_priced_df
    snap = _to_utc(all_priced_df[time_col])
    if spec.mode == WindowMode.IS:
        return all_priced_df[snap < spec.cutoff].copy()
    return all_priced_df[snap >= spec.cutoff].copy()


def apply_window_trades(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    spec: WindowSpec,
    trade_time_col: str = "pricing_date",
    equity_time_col: str = "pricing_date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter trade-sim panels by entry time (Decision 3 = window everything).

    [REVIEW S3] Trades are partitioned by entry (``pricing_date``). Daily PnL /
    Sharpe still bucket by settlement_date downstream — that is intentional and
    captioned in the UI. Rows with NaT entry time survive only under ALL mode.
    """
    if spec.mode == WindowMode.ALL:
        return trades_df, equity_df

    def _filt(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if df is None or df.empty or col not in df.columns:
            return df
        ts = _to_utc(df[col])
        if spec.mode == WindowMode.IS:
            return df[ts < spec.cutoff].copy()
        return df[ts >= spec.cutoff].copy()

    return _filt(trades_df, trade_time_col), _filt(equity_df, equity_time_col)


# ---------------------------------------------------------------------------
# M2 training population (axis = settlement_time)
# ---------------------------------------------------------------------------

def m2_training_set(all_priced_df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Contracts eligible to TRAIN the M2 shift: settlement_time < cutoff & resolved.

    Guarantees every training label resolves before the cutoff (and thus before any
    OOS contract's pricing time) — the §9 / §1.6 leak guard.
    """
    cutoff = normalize_cutoff(cutoff)
    df = add_settlement_time(all_priced_df)
    outcome = pd.to_numeric(df.get("outcome_yes", pd.Series(dtype=float)), errors="coerce")
    mask = (df["settlement_time"] < cutoff) & outcome.notna()
    return df[mask].copy()


# ---------------------------------------------------------------------------
# OOS hygiene guard (§7)
# ---------------------------------------------------------------------------

def guarded_filter(
    df: pd.DataFrame,
    spec: WindowSpec,
    *,
    conditions_on_outcome: bool,
    desc: str = "",
) -> pd.DataFrame:
    """Pass-through gate: raise if an outcome-conditioning query runs in OOS mode.

    [REVIEW N3] ALL and IS explicitly short-circuit (allowed); only OOS with
    ``conditions_on_outcome`` raises. The caller applies its own predicate; this
    only enforces the hygiene rule at the data layer.
    """
    if spec.mode in (WindowMode.ALL, WindowMode.IS):
        return df
    if conditions_on_outcome:
        raise OOSHygieneError(
            f"Outcome-conditioning filter blocked in OOS mode: {desc or 'unnamed query'}"
        )
    return df


# ---------------------------------------------------------------------------
# Small-sample handling (§8)
# ---------------------------------------------------------------------------

def small_sample_state(n: int, threshold: int = N_MIN_SAMPLE) -> dict:
    """Return {suppress, banner}. n < threshold → suppress panel summary stats."""
    if n < threshold:
        return {
            "suppress": True,
            "banner": (
                f"N = {n} (< {threshold}) — small-sample window. Summary statistics "
                "suppressed; underlying data still shown."
            ),
        }
    return {"suppress": False, "banner": None}


# ---------------------------------------------------------------------------
# Cache fingerprint
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:  # pragma: no cover - environment dependent
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT), text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _params_hash(strategy_params: Optional[dict], n_sims) -> str:
    payload = json.dumps(
        {"strategy_params": strategy_params or {}, "n_sims": n_sims},
        sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _fingerprint(
    cutoff: pd.Timestamp,
    train_df: pd.DataFrame,
    *,
    code_version: str,
    strategy_params: Optional[dict],
    n_sims,
) -> dict:
    """Identity used for cache invalidation [REVIEW S5]."""
    n_train = int(contract_ids(train_df).nunique()) if not train_df.empty else 0
    settle_max = (
        train_df["settlement_time"].max() if "settlement_time" in train_df.columns and not train_df.empty
        else pd.NaT
    )
    return {
        "cutoff_iso": cutoff.isoformat(),
        "n_is_train_contracts": n_train,
        "is_label_max_ts": None if pd.isna(settle_max) else pd.Timestamp(settle_max).isoformat(),
        "code_version": code_version,
        "params_hash": _params_hash(strategy_params, n_sims),
    }


def _cache_dir(cutoff: pd.Timestamp, cache_root: Path) -> Path:
    return Path(cache_root) / f"cutoff_{cutoff:%Y-%m-%d}"


# ---------------------------------------------------------------------------
# train_pipeline / load
# ---------------------------------------------------------------------------

def train_pipeline(
    cutoff_date,
    all_priced_df: pd.DataFrame,
    *,
    components: Sequence[str] = ("m2",),
    cache_root: Path = CACHE_ROOT,
    strategy_params: Optional[dict] = None,
    n_sims=None,
    min_obs: int = CALIBRATION_MIN_OBS,
) -> dict:
    """Fit every cutoff-sensitive component on the IS population and cache it.

    Only ``m2`` is fitted (XGBoost excluded, §1.2). The M2 shift is fit on
    ``m2_training_set`` (settlement < cutoff) and written to
    ``<cache>/cutoff_<date>/calibration_shift.csv`` — NEVER the global DATA file
    [REVIEW B2]. A manifest.json with the invalidation fingerprint is written too.

    Returns the artifacts dict (also retrievable via load_artifacts).
    """
    if "m2" not in components:
        raise ValueError("train_pipeline currently supports only the 'm2' component")

    cutoff = normalize_cutoff(cutoff_date)
    cache_dir = _cache_dir(cutoff, cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shift_csv = cache_dir / "calibration_shift.csv"

    is_eval, oos_eval, straddlers = partition_contracts(all_priced_df, cutoff)
    train = m2_training_set(all_priced_df, cutoff)

    # Fit B on the never-calibrated raw prob [REVIEW N4]; train_frac=1.0 because the
    # IS/OOS split IS the holdout (no further internal split). Explicit output_path
    # so the global DATA/calibration_shift.csv is untouched [REVIEW B2].
    m2 = fit_calibration(
        train,
        prob_col="model_prob_raw",
        outcome_col="outcome_yes",
        dte_col="dte_days",
        time_col="snapshot_time",
        train_frac=1.0,
        output_path=str(shift_csv),
        min_obs=min_obs,
    )

    code_version = _git_sha()
    fp = _fingerprint(
        cutoff, train, code_version=code_version,
        strategy_params=strategy_params, n_sims=n_sims,
    )

    applied_buckets = [b for b, e in m2.items() if e.get("applied")]
    manifest = {
        **fp,
        "target_is_frac": None,
        "n_is_eval_contracts": int(contract_ids(is_eval).nunique()) if not is_eval.empty else 0,
        "n_oos_eval_contracts": int(contract_ids(oos_eval).nunique()) if not oos_eval.empty else 0,
        "n_straddlers": int(contract_ids(straddlers).nunique()) if not straddlers.empty else 0,
        "oos_pricing_min_ts": (
            _to_utc(oos_eval["snapshot_time"]).min().isoformat()
            if not oos_eval.empty else None
        ),
        "components": {
            "m2": {
                "applied_buckets": applied_buckets,
                "B": {b: e.get("B_fitted", 0.0) for b, e in m2.items()},
                "n_obs": {b: e.get("n_obs", 0) for b, e in m2.items()},
                "inert": len(applied_buckets) == 0,
                "min_obs": min_obs,
            },
        },
        "engine_truncation": "strict_lt_snapshot",
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    # gap_ok: max training label resolves before the earliest OOS pricing time.
    gap_ok = True
    if manifest["is_label_max_ts"] and manifest["oos_pricing_min_ts"]:
        gap_ok = pd.Timestamp(manifest["is_label_max_ts"]) < pd.Timestamp(manifest["oos_pricing_min_ts"])
    manifest["gap_ok"] = bool(gap_ok)

    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "cutoff": cutoff,
        "cache_dir": cache_dir,
        "manifest": manifest,
        "shift_table": _load_shift_table(shift_csv),
        "m2": m2,
    }


def _load_shift_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_artifacts(cutoff_date, cache_root: Path = CACHE_ROOT) -> Optional[dict]:
    """Load cached artifacts for a cutoff, or None if absent/corrupt."""
    cutoff = normalize_cutoff(cutoff_date)
    cache_dir = _cache_dir(cutoff, cache_root)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    return {
        "cutoff": cutoff,
        "cache_dir": cache_dir,
        "manifest": manifest,
        "shift_table": _load_shift_table(cache_dir / "calibration_shift.csv"),
        "m2": None,
    }


def _fingerprint_matches(manifest: dict, fp: dict) -> bool:
    keys = ("cutoff_iso", "n_is_train_contracts", "is_label_max_ts",
            "code_version", "params_hash")
    return all(manifest.get(k) == fp.get(k) for k in keys)


def load_or_train(
    cutoff_date,
    all_priced_df: pd.DataFrame,
    *,
    mode: WindowMode,
    cache_root: Path = CACHE_ROOT,
    strategy_params: Optional[dict] = None,
    n_sims=None,
    min_obs: int = CALIBRATION_MIN_OBS,
) -> dict:
    """Return artifacts for the cutoff, training only when allowed.

    HARD RULE (§5): in OOS mode never fit. A cache miss or fingerprint mismatch in
    OOS raises — the canonical IS cutoff must be (re)trained first. IS/ALL may train.
    """
    cutoff = normalize_cutoff(cutoff_date)
    train = m2_training_set(all_priced_df, cutoff)
    fp = _fingerprint(
        cutoff, train, code_version=_git_sha(),
        strategy_params=strategy_params, n_sims=n_sims,
    )
    existing = load_artifacts(cutoff, cache_root)
    if existing is not None and _fingerprint_matches(existing["manifest"], fp):
        return existing

    if mode == WindowMode.OOS:
        raise OOSLeakError(
            "OOS mode requires a cached IS artifact matching the current data "
            f"fingerprint for cutoff {cutoff:%Y-%m-%d}; none found (missing or stale). "
            "Re-run in In-sample/All mode at this cutoff to (re)train first — OOS "
            "never refits."
        )
    return train_pipeline(
        cutoff, all_priced_df, cache_root=cache_root,
        strategy_params=strategy_params, n_sims=n_sims, min_obs=min_obs,
    )


# ---------------------------------------------------------------------------
# OOS calibration overlay
# ---------------------------------------------------------------------------

def m2_shifts_applied(shift_table: pd.DataFrame) -> Dict[str, float]:
    """Bucket -> B for buckets whose fit was trusted (applied=True)."""
    out: Dict[str, float] = {}
    if shift_table is None or shift_table.empty:
        return out
    for _, r in shift_table.iterrows():
        if bool(r.get("applied", False)):
            out[str(r["bucket"])] = float(r["B_fitted"])
    return out


def is_m2_inert(shift_table: pd.DataFrame) -> bool:
    """True when no DTE bucket has a trusted shift [REVIEW S1 banner]."""
    return len(m2_shifts_applied(shift_table)) == 0


def apply_oos_calibration(window_df: pd.DataFrame, shift_table: pd.DataFrame) -> pd.DataFrame:
    """Overlay p_model_cal on OOS rows using the cached IS shift table.

    p_model_cal = sigmoid(logit(model_prob_raw) + B_bucket), applied buckets only
    [REVIEW N4]. No-op (identity column copy) when the table is inert.
    """
    from scipy.special import expit, logit

    df = window_df.copy()
    shifts = m2_shifts_applied(shift_table)
    base = pd.to_numeric(df.get("model_prob_raw", df.get("model_prob_used")), errors="coerce")
    if not shifts or "dte_days" not in df.columns:
        df["p_model_cal"] = base
        return df
    # apply_calibration_shift takes a scalar B; vectorize per-row B over DTE buckets.
    B = df["dte_days"].apply(dte_bucket).map(shifts).fillna(0.0).to_numpy(dtype=float)
    eps = 1e-6
    p = np.clip(base.to_numpy(dtype=float), eps, 1 - eps)
    df["p_model_cal"] = expit(logit(p) + B)
    return df


# ---------------------------------------------------------------------------
# Verification (§9)
# ---------------------------------------------------------------------------

def _strict_lt(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.DatetimeIndex:
    """The backrunner's truncation rule: bars STRICTLY before the snapshot."""
    return idx[idx < ts]


def verify_oos_leak_free(
    all_priced_df: pd.DataFrame,
    cutoff,
    artifacts: dict,
    *,
    n_samples: int = 3,
    seed: int = 0,
    include_btc_arm: bool = False,
    data_dir: Optional[Path] = None,
    btc_truncate=_strict_lt,
) -> None:
    """Assert every parameter touching a sampled OOS contract's prediction was fit
    on labels strictly before that contract's pricing timestamp. Raises OOSLeakError.

    Arms:
      M2          — manifest.is_label_max_ts < contract.snapshot_time.
      no-other-pool — manifest declares only the m2 pooled component.
      BTC (opt)   — [REVIEW S2] heavy (re-reads BTC CSVs); pytest-only. Re-applies the
                    truncation rule (``btc_truncate``, default strict ``<``) the
                    backrunner uses and asserts the resulting slice holds NO bar
                    ``>= snapshot_time``. Passing a ``<=`` truncation (the regression
                    this guards) makes the midnight bar leak and the arm raise.
    """
    cutoff = normalize_cutoff(cutoff)
    manifest = artifacts["manifest"]

    # no-other-pool arm
    comps = set(manifest.get("components", {}).keys())
    if not comps.issubset({"m2"}):
        raise OOSLeakError(f"Unexpected pooled components in manifest: {comps - {'m2'}}")

    df = add_settlement_time(all_priced_df)
    snap = _to_utc(df["snapshot_time"])
    oos = df[snap >= cutoff].copy()
    oos["_snap"] = _to_utc(oos["snapshot_time"])
    oos = oos.dropna(subset=["_snap"])
    if oos.empty:
        raise OOSLeakError("No OOS contracts to verify (snapshot_time >= cutoff is empty).")

    oos["_id"] = contract_ids(oos)
    uniq = oos.drop_duplicates(subset=["_id"])
    rng = np.random.default_rng(seed)
    k = min(n_samples, len(uniq))
    sample = uniq.iloc[rng.choice(len(uniq), size=k, replace=False)]

    label_max = manifest.get("is_label_max_ts")
    label_max_ts = pd.Timestamp(label_max) if label_max else None

    btc = _load_btc_max_index(data_dir) if include_btc_arm else None

    for _, row in sample.iterrows():
        snap_ts = row["_snap"]
        # M2 arm: every training label resolved before this contract was priced.
        if label_max_ts is not None and not (label_max_ts < snap_ts):
            raise OOSLeakError(
                f"M2 leak: training label max {label_max_ts} not < OOS pricing time "
                f"{snap_ts} (contract {row['_id']})."
            )
        # BTC arm (optional, pytest-only): re-apply the truncation rule and assert
        # the slice the engine would feed contains no bar at/after the snapshot.
        if btc is not None:
            for name, idx in btc.items():
                sl = btc_truncate(idx, snap_ts)
                if len(sl) and sl.max() >= snap_ts:
                    raise OOSLeakError(
                        f"BTC leak: {name} slice max {sl.max()} >= OOS pricing time "
                        f"{snap_ts} (contract {row['_id']})."
                    )


def _load_btc_max_index(data_dir: Optional[Path]) -> Dict[str, pd.DatetimeIndex]:
    """Load UTC datetime indexes for BTC CSVs (verification BTC arm only)."""
    data_dir = Path(data_dir) if data_dir else (_PROJECT_ROOT / "DATA")
    out: Dict[str, pd.DatetimeIndex] = {}
    for name, fname in (("hourly", "btc_hourly.csv"),
                        ("intraday", "btc_intraday_1m.csv"),
                        ("daily", "btc_daily.csv")):
        p = data_dir / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        cmap = {c.lower(): c for c in d.columns}
        tcol = cmap.get("timestamp", cmap.get("date", cmap.get("datetime")))
        if tcol is None:
            continue
        out[name] = pd.DatetimeIndex(pd.to_datetime(d[tcol], utc=True, errors="coerce")).dropna()
    return out
