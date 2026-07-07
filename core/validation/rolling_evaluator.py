"""
rolling_evaluator.py

Wave 3 / T7 (Journal Articles/improvement_plan.md section 3.1): rolling-window
evaluation framework. Rolls a fixed-length training window forward through the
hourly BTC history, re-fits the BASE pricing engine (jump-filtered GARCH/FIGARCH
+ calibrated Kou/SVCJ jumps) at each anchor, forecasts a strike grid across
several horizons, and scores the forecasts against realized outcomes with a
naive Gaussian baseline (Brier score) and Basel-style VaR exceedance tracking.

SCOPE (read before using): this evaluator drives `simulate_paths()` DIRECTLY,
not `calculate_probabilities()`. It therefore exercises only the BASE engine --
GARCH/FIGARCH + Kou/SVCJ jumps + skewed-t, all per `engine_flags` -- and does
NOT include the regime-switching layer (`RegimeDetector` / `use_regime_switching`)
or the XGBoost directional drift shift. `calculate_probabilities()` returns only
`{strike: probability}` (no path array), so it cannot feed the VaR arm, which
needs the full terminal-price distribution; hence the direct `simulate_paths`
call. A future full-engine mode (using `core.pricing.engine_config.build_engine_kwargs`)
would need a `calculate_probabilities` variant that also returns paths.

Leak rules: at anchor t_w, the GARCH fit, jump calibration, and trailing
realized-vol estimate all use hourly data strictly `< t_w` (same convention as
`core.backtesting.backrunner`). Realized outcomes (step 4) come from the FULL
hourly series -- this is the only place future data is touched, and only to
grade a forecast already made from `< t_w` data.

Anchoring: candidate anchors are generated over the full data range; when
`max_windows` caps the run, the MOST RECENT N anchors are kept (tail-taking),
so bounded runs evaluate the latest regime rather than the start of the data.
Runs that produce zero forecasts log a WARNING and write no CSV.

Usage:
    from core.validation.rolling_evaluator import RollingEvaluator
    ev = RollingEvaluator(window_days=90, step_days=7, horizons=(1, 14, 28))
    df = ev.run()
    print(ev.summary(df))

CLI:
    python core/validation/rolling_evaluator.py --window-days 90 --step-days 7 \\
        --horizons 1,14,28 --n-sims 5000 --max-windows 40
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest

# Guard: ensure repo root is on sys.path when invoked as a script (e.g.
# `python core/validation/rolling_evaluator.py --help`), mirroring
# core/backtesting/backrunner.py. Without this, module-level `core.*` imports
# below raise ModuleNotFoundError because sys.path[0] is this script's own
# directory, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pricing.btc_pricing_engine import fit_garch_model, simulate_paths
from core.pricing.jump_calibration import calibrate_jumps
from core.validation.basel_backtest import basel_traffic_light

logger = logging.getLogger(__name__)

# Minimum trailing hourly observations required to attempt a GARCH fit at an
# anchor point. Below this the fit is unreliable / prone to non-convergence;
# the window is skipped rather than fit on too little data.
MIN_TRAIN_HOURS = 500

# Strike grid: S0 * exp(k * sigma_h). Chosen so the naive Gaussian baseline
# probability at each grid point is exactly 1 - Phi(k), independent of the
# window's actual sigma_h (the grid is defined IN vol-units).
STRIKE_K_GRID = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

# Trailing window (hours) used for the realized-vol estimate that sizes the
# strike grid and the naive baseline (30 days).
TRAILING_VOL_HOURS = 30 * 24

DEFAULT_ENGINE_FLAGS = {"use_svcj": True, "use_skewed_t": True, "use_figarch": True}


@dataclass
class WindowResult:
    """One row of rolling-evaluation output: a single (window, horizon) forecast."""
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    horizon_days: int
    brier_model: float           # mean Brier over the strike grid (model)
    brier_naive: float           # same grid, naive Gaussian baseline
    var_hit_5: Optional[int]     # 0/1 indicator: realized < 5% MC quantile
    var_hit_1: Optional[int]     # 0/1 indicator: realized < 1% MC quantile
    n_forecasts: int


class RollingEvaluator:
    """Rolling re-fit + forecast + score harness for the base pricing engine.

    See module docstring for scope (base engine only -- no regime layer, no XGB).
    """

    def __init__(
        self,
        hourly_csv: str = "DATA/btc_hourly.csv",
        window_days: int = 90,
        step_days: int = 7,
        horizons: Sequence[int] = (1, 14, 28),
        n_sims: int = 5000,
        engine_flags: Optional[dict] = None,
        seed: int = 42,
        max_windows: Optional[int] = 40,
        hourly_df: Optional[pd.DataFrame] = None,
        out_dir: str = "DATA/rolling_eval",
    ):
        """Args of note:
            max_windows: keep only the MOST RECENT N anchors (the anchor list
                is generated over the full data range first, then tail-taken),
                so a bounded run evaluates the latest -- most relevant --
                regime rather than the start of the data.
            out_dir: directory for the results CSV. Tests should pass a temp
                directory so they do not pollute the real DATA/rolling_eval/.
        """
        self.hourly_csv = hourly_csv
        self.window_days = window_days
        self.step_days = step_days
        self.horizons = tuple(int(h) for h in horizons)
        self.n_sims = n_sims
        self.engine_flags = dict(engine_flags) if engine_flags else dict(DEFAULT_ENGINE_FLAGS)
        for key in ("use_svcj", "use_skewed_t", "use_figarch"):
            self.engine_flags.setdefault(key, DEFAULT_ENGINE_FLAGS[key])
        self.seed = seed
        self.max_windows = max_windows
        self._hourly_df = hourly_df
        self.out_dir = out_dir

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load(self) -> Tuple[pd.DataFrame, str, str]:
        if self._hourly_df is not None:
            df = self._hourly_df.copy()
        else:
            df = pd.read_csv(self.hourly_csv)

        col_map = {c.lower(): c for c in df.columns}
        if "close" not in col_map:
            raise ValueError("hourly data must contain a 'Close'/'close' column.")
        close_col = col_map["close"]

        ts_col = col_map.get("timestamp", col_map.get("date"))
        if ts_col is None:
            raise ValueError(
                "hourly data must contain a 'timestamp' or 'date' column for "
                "rolling-window anchoring."
            )

        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col, close_col]).sort_values(ts_col).reset_index(drop=True)
        return df, close_col, ts_col

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Roll the training window forward, forecast, and score. Returns one
        row per (window, horizon) as a DataFrame, and also writes a CSV to
        `self.out_dir` (default DATA/rolling_eval/) -- but ONLY when at least
        one forecast was produced (no silent empty CSVs)."""
        df, close_col, ts_col = self._load()

        if df.empty:
            logger.warning("RollingEvaluator: empty input data -- nothing to do.")
            return pd.DataFrame([])

        first_ts = df[ts_col].iloc[0]
        last_ts = df[ts_col].iloc[-1]

        # Generate ALL candidate anchors over the data range first, then --
        # when max_windows caps the run -- keep the MOST RECENT N. Taking the
        # first N instead would evaluate only the oldest (least relevant)
        # regime in the data on every bounded run.
        anchors = []
        anchor = first_ts + pd.Timedelta(days=self.window_days)
        while anchor <= last_ts:
            anchors.append(anchor)
            anchor += pd.Timedelta(days=self.step_days)

        n_total_anchors = len(anchors)
        if self.max_windows is not None and n_total_anchors > self.max_windows:
            anchors = anchors[-self.max_windows:]
            logger.info(
                "RollingEvaluator: %d candidate anchors; keeping the most "
                "recent %d (max_windows).", n_total_anchors, self.max_windows,
            )

        rows = []

        for window_idx, anchor in enumerate(anchors):
            window_start = anchor - pd.Timedelta(days=self.window_days)
            train_df = df.loc[df[ts_col] < anchor]

            if len(train_df) < MIN_TRAIN_HOURS:
                logger.debug(
                    "Window %d (anchor=%s): only %d training hours (<%d) -- skipping.",
                    window_idx, anchor, len(train_df), MIN_TRAIN_HOURS,
                )
                continue

            returns_slice = np.log(train_df[close_col] / train_df[close_col].shift(1)).dropna()
            S0 = float(train_df[close_col].iloc[-1])

            try:
                garch_params = fit_garch_model(
                    returns_slice,
                    use_figarch=self.engine_flags["use_figarch"],
                    filter_jumps=True,
                )
            except Exception as e:
                logger.warning(
                    "Window %d (anchor=%s): GARCH fit failed (%s) -- skipping window.",
                    window_idx, anchor, e,
                )
                continue

            try:
                cal = calibrate_jumps(returns=returns_slice.to_numpy(), detection_method="bipower")
                jp = (
                    {
                        "lambda": cal.lam,
                        "crash_prob": cal.p_crash,
                        "eta_up": cal.eta_up,
                        "eta_down": cal.eta_down,
                        "mu_v": cal.mu_v,
                        "rho_J": cal.rho_J,
                        "rho_j_slope": cal.rho_j_slope,
                    }
                    if cal.fit_converged
                    else None
                )
            except Exception as e:
                logger.warning(
                    "Window %d (anchor=%s): jump calibration failed (%s) -- "
                    "using engine default jumps.", window_idx, anchor, e,
                )
                jp = None

            # Trailing 30d realized vol (leak-free -- from the < t_w slice only).
            trail_n = min(TRAILING_VOL_HOURS, len(returns_slice))
            trail_returns = returns_slice.to_numpy()[-trail_n:]
            hourly_std = float(np.std(trail_returns)) if trail_n > 1 else float("nan")
            daily_vol = hourly_std * np.sqrt(24.0)

            for h in self.horizons:
                target_time = anchor + pd.Timedelta(days=h)
                future = df.loc[df[ts_col] >= target_time]
                if future.empty:
                    # Realization falls beyond the available data -- skip.
                    continue

                if not np.isfinite(daily_vol) or daily_vol <= 0:
                    continue

                S_real = float(future[close_col].iloc[0])
                sigma_h = daily_vol * np.sqrt(h)

                seed_h = self.seed + window_idx * 1000 + h
                paths = simulate_paths(
                    S0, garch_params, jp,
                    hours_to_expiry=h * 24.0,
                    n_sims=self.n_sims,
                    seed=seed_h,
                    use_naive_prior=True,
                    use_svcj=self.engine_flags["use_svcj"],
                    use_skewed_t=self.engine_flags["use_skewed_t"],
                    use_figarch=self.engine_flags["use_figarch"],
                )

                strikes = S0 * np.exp(STRIKE_K_GRID * sigma_h)
                outcomes = (S_real > strikes).astype(float)

                p_model = np.array([float(np.mean(paths >= K)) for K in strikes])
                # NOTE: strikes sit at fixed z-multiples (STRIKE_K_GRID) of the
                # SAME trailing sigma_h the naive Gaussian uses, so p_naive per
                # strike is the CONSTANT 1 - Phi(k) for every (window, horizon).
                # brier_naive therefore depends only on the binary outcome
                # pattern and takes values from a small discrete set --
                # repeated identical naive-Brier cells across rows are
                # expected, not a bug.
                p_naive = 1.0 - norm.cdf(STRIKE_K_GRID)

                brier_model = float(np.mean((p_model - outcomes) ** 2))
                brier_naive = float(np.mean((p_naive - outcomes) ** 2))

                realized_log_ret = float(np.log(S_real / S0))
                log_ret_paths = np.log(paths / S0)
                q05 = float(np.quantile(log_ret_paths, 0.05))
                q01 = float(np.quantile(log_ret_paths, 0.01))

                rows.append(WindowResult(
                    window_start=window_start,
                    window_end=anchor,
                    horizon_days=h,
                    brier_model=brier_model,
                    brier_naive=brier_naive,
                    var_hit_5=int(realized_log_ret < q05),
                    var_hit_1=int(realized_log_ret < q01),
                    n_forecasts=len(strikes),
                ))

        result_df = pd.DataFrame([asdict(r) for r in rows])

        if result_df.empty:
            # Do NOT write an empty CSV -- a 2-byte header-less file in
            # DATA/rolling_eval/ is silent noise. Explain why instead.
            logger.warning(
                "RollingEvaluator: produced ZERO forecasts -- no CSV written. "
                "Likely causes: not enough data for any anchor (need > %d "
                "days + %d training hours; data spans %s to %s), or every "
                "anchor's realization (max horizon %dd) falls beyond the end "
                "of the data.",
                self.window_days, MIN_TRAIN_HOURS, first_ts, last_ts,
                max(self.horizons) if self.horizons else 0,
            )
            return result_df

        out_dir = Path(self.out_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"rolling_eval_{stamp}.csv"
            result_df.to_csv(out_path, index=False)
            logger.info("Rolling evaluation results written to %s (%d rows)", out_path, len(result_df))
        except OSError as e:
            logger.warning("Could not write rolling eval CSV: %s", e)

        return result_df

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self, df: pd.DataFrame) -> dict:
        """Aggregate model-vs-naive Brier comparison (with a paired sign test)
        and Basel traffic-light VaR exceedance classification per horizon."""
        if df is None or df.empty:
            return {"n_rows": 0, "note": "no forecasts produced"}

        out: Dict[str, object] = {}
        out["n_rows"] = int(len(df))
        out["brier_model_mean"] = float(df["brier_model"].mean())
        out["brier_naive_mean"] = float(df["brier_naive"].mean())

        wins = int((df["brier_model"] < df["brier_naive"]).sum())
        losses = int((df["brier_model"] > df["brier_naive"]).sum())
        n_compared = wins + losses
        if n_compared > 0:
            sign_test_p = float(binomtest(wins, n_compared, p=0.5).pvalue)
        else:
            sign_test_p = float("nan")

        out["model_wins"] = wins
        out["model_losses"] = losses
        out["ties"] = int(len(df) - n_compared)
        out["sign_test_p_value"] = sign_test_p

        var_backtest = {}
        for h in sorted(df["horizon_days"].unique()):
            sub = df[df["horizon_days"] == h]
            for alpha, col in ((0.05, "var_hit_5"), (0.01, "var_hit_1")):
                valid = sub[col].dropna()
                n_obs = int(len(valid))
                if n_obs == 0:
                    continue
                n_exceed = int(valid.sum())
                zone, p_value = basel_traffic_light(n_exceed, alpha, n_obs)
                var_backtest[f"h={int(h)}d,alpha={alpha:.0%}"] = {
                    "n_obs": n_obs,
                    "n_exceed": n_exceed,
                    "observed_rate": n_exceed / n_obs,
                    "expected_rate": alpha,
                    "zone": zone,
                    "p_value": p_value,
                }
        out["var_backtest"] = var_backtest

        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Rolling-window evaluation of the BASE BTC pricing engine "
            "(GARCH/FIGARCH + Kou/SVCJ jumps + skewed-t; no regime layer, no XGB) "
            "against a naive Gaussian baseline, with Basel-style VaR exceedance "
            "tracking."
        )
    )
    parser.add_argument("--hourly-csv", default="DATA/btc_hourly.csv", help="Path to hourly BTC data")
    parser.add_argument("--window-days", type=int, default=90, help="Trailing training window length (days)")
    parser.add_argument("--step-days", type=int, default=7, help="Days between successive anchors")
    parser.add_argument("--horizons", default="1,14,28", help="Comma-separated forecast horizons (days)")
    parser.add_argument("--n-sims", type=int, default=5000, help="Monte Carlo paths per (window, horizon)")
    parser.add_argument("--max-windows", type=int, default=40,
                        help="Keep only the MOST RECENT N anchors (bounded runs "
                             "evaluate the latest regime, not the start of the data)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--no-svcj", action="store_true", help="Disable SVCJ vol jumps")
    parser.add_argument("--no-skewed-t", action="store_true", help="Disable skewed-t innovations")
    parser.add_argument("--no-figarch", action="store_true", help="Disable FIGARCH (use GARCH(1,1))")
    args = parser.parse_args()

    horizons = tuple(int(x) for x in args.horizons.split(","))
    engine_flags = {
        "use_svcj": not args.no_svcj,
        "use_skewed_t": not args.no_skewed_t,
        "use_figarch": not args.no_figarch,
    }

    evaluator = RollingEvaluator(
        hourly_csv=args.hourly_csv,
        window_days=args.window_days,
        step_days=args.step_days,
        horizons=horizons,
        n_sims=args.n_sims,
        engine_flags=engine_flags,
        seed=args.seed,
        max_windows=args.max_windows,
    )

    result_df = evaluator.run()
    summ = evaluator.summary(result_df)

    print(f"\n{'='*72}")
    print(f"ROLLING-WINDOW EVALUATION -- {len(result_df)} (window, horizon) forecasts")
    print(f"{'='*72}")
    print(f"Brier (model): {summ.get('brier_model_mean', float('nan')):.4f}")
    print(f"Brier (naive): {summ.get('brier_naive_mean', float('nan')):.4f}")
    print(
        f"Model wins/losses vs naive: {summ.get('model_wins', 0)}/{summ.get('model_losses', 0)} "
        f"(sign-test p={summ.get('sign_test_p_value', float('nan')):.4f})"
    )
    print("\nVaR backtest (Basel traffic light):")
    for key, v in summ.get("var_backtest", {}).items():
        print(
            f"  {key}: exceed={v['n_exceed']}/{v['n_obs']} "
            f"(observed={v['observed_rate']:.4f} vs expected={v['expected_rate']:.4f}) "
            f"-> {v['zone']} (p={v['p_value']:.4f})"
        )
    print(f"{'='*72}\n")
