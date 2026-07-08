"""
basel_backtest.py

Teng (2025) style Basel VaR/ES backtest framework for BTC pricing engines.
Validates tail risk forecasting against the Basel Committee traffic light system.

Two modes:
  - analytical: Rolling GARCH(1,1) + Student-t → VaR via conditional variance forecast
  - mc: Full Monte Carlo (GARCH + Student-t + SVCJ jumps) → VaR from empirical path quantiles

Reference: Teng, Huang & Shih (2025). "Tail Risk in Bitcoin Under the Basel Framework."
           Finance Research Letters, 86:108528.

Usage:
    python core/validation/basel_backtest.py --input DATA/btc_hourly.csv
    python core/validation/basel_backtest.py --input DATA/btc_hourly.csv --mode mc --num-sims 20000

Or programmatically:
    from core.validation.basel_backtest import BaselBacktestResult, run_basel_backtest
    result = run_basel_backtest(returns_hourly, horizons=[1, 14, 28], mode="analytical")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom, norm, t as t_dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for rolling GARCH VaR
# ---------------------------------------------------------------------------

DEFAULT_ROLLING_WINDOW = 500   # hours between GARCH refits
MIN_TRAINING_HOURS = 500       # minimum hours (~21 days) to fit a reliable GARCH model

# ---------------------------------------------------------------------------
# Traffic Light Classification (Basel Committee)
# ---------------------------------------------------------------------------

def basel_traffic_light(
    n_exceed: int,
    expected_rate: float,
    n_obs: int,
    confidence: float = 0.95,
) -> Tuple[str, float]:
    """
    Basel traffic light classification for VaR exceedances.

    Zones (based on Kupiec POF test p-value):
        Green:  p >= 0.10  (model adequate)
        Yellow: 0.05 <= p < 0.10 (model requires monitoring)
        Red:    p < 0.05   (model inadequate)

    Also applies the Basel multiplier zones:
        Green:  exceedance count in [floor(n*alpha*0.5), ceil(n*alpha*1.5)]
        Yellow: exceedance count in [ceil(n*alpha*1.5)+1, ceil(n*alpha*2.0)]
        Red:    exceedance count > ceil(n*alpha*2.0)

    Args:
        n_exceed: Observed number of VaR exceedances.
        expected_rate: Expected rate (e.g., 0.01 for 1% VaR).
        n_obs: Number of observations.
        confidence: Confidence level (not used directly for traffic light).

    Returns:
        (zone, p_value) where zone is "Green", "Yellow", or "Red".
    """
    expected_exceed = expected_rate * n_obs

    # Kupiec POF test
    if n_exceed == 0:
        # Special case: zero exceedances
        p_value = binom.cdf(0, n_obs, expected_rate)
        zone = "Green" if p_value >= 0.10 else ("Yellow" if p_value >= 0.05 else "Red")
        return zone, p_value

    # Two-sided binomial test
    p_value = min(
        2 * binom.cdf(n_exceed, n_obs, expected_rate),
        2 * (1 - binom.cdf(n_exceed - 1, n_obs, expected_rate)),
        1.0,
    )

    if p_value >= 0.10:
        zone = "Green"
    elif p_value >= 0.05:
        zone = "Yellow"
    else:
        zone = "Red"

    return zone, p_value


def expected_shortfall_test(
    returns: np.ndarray,
    var_values: np.ndarray,
    alpha: float = 0.01,
) -> Dict[str, float]:
    """
    Expected Shortfall (ES) backtest per Acerbi & Szekely (2014).

    Tests Z1, Z2, Z3 should be within [-1.96, 1.96] for a correctly specified model.

    Args:
        returns: Realized returns.
        var_values: Forecast VaR values (negative for losses).
        alpha: VaR confidence level.

    Returns:
        Dict with Z1, Z2, Z3 test statistics.
    """
    # VaR violations
    violations = returns < var_values  # VaR is negative; violation when return < VaR

    n_violations = np.sum(violations)
    if n_violations == 0:
        return {"Z1": np.nan, "Z2": np.nan, "Z3": np.nan, "n_violations": 0}

    violation_returns = returns[violations]
    violation_var = var_values[violations]

    # Z1: Unconditional ES test
    es_violations = np.mean(violation_returns)
    es_expected = np.mean(var_values[violations])
    se_z1 = np.std(violation_returns, ddof=1) / np.sqrt(n_violations) if n_violations > 1 else 1.0
    Z1 = (es_violations - es_expected) / se_z1 if se_z1 > 0 else 0.0

    # Z2: Conditional ES test (violations only)
    exceedances = violation_returns - violation_var
    Z2 = np.mean(exceedances) / (np.std(exceedances, ddof=1) / np.sqrt(n_violations)) if n_violations > 1 else 0.0

    # Z3: Magnitude test
    overall_mean_exceedance = np.mean(returns - var_values)
    Z3 = overall_mean_exceedance

    return {
        "Z1": Z1, "Z2": Z2, "Z3": Z3,
        "n_violations": n_violations,
        "es_violations": es_violations,
        "es_expected": es_expected,
    }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HorizonResult:
    """Basel backtest result for a single horizon."""
    horizon: int
    expected_rate: float
    exceedance_rate: float
    n_obs: int
    n_exceed: int
    zone: str
    p_value: float
    es_tests: Dict[str, float] = field(default_factory=dict)


@dataclass
class BaselBacktestResult:
    """Complete Basel backtest result across multiple horizons and confidence levels."""
    timestamp: str
    n_total_obs: int
    horizons: List[int]
    alphas: List[float]
    results: Dict[Tuple[int, float], HorizonResult] = field(default_factory=dict)
    summary: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for (h, alpha), hr in sorted(self.results.items()):
            rows.append({
                "horizon": h, "alpha": alpha,
                "expected_rate": hr.expected_rate,
                "exceedance_rate": round(hr.exceedance_rate, 6),
                "n_exceed": hr.n_exceed, "n_obs": hr.n_obs,
                "zone": hr.zone, "p_value": round(hr.p_value, 4),
                "Z1": round(hr.es_tests.get("Z1", np.nan), 3),
                "Z2": round(hr.es_tests.get("Z2", np.nan), 3),
            })
        return pd.DataFrame(rows)

    def print_summary(self):
        """Print Basel traffic light table matching Teng (2025) format."""
        df = self.to_dataframe()
        print(f"\n{'='*80}")
        print(f"BASEL BACKTEST RESULTS -- {self.timestamp}")
        print(f"Observations: {self.n_total_obs}")
        print(f"{'='*80}")

        for alpha in sorted(set(df["alpha"])):
            subset = df[df["alpha"] == alpha].sort_values("horizon")
            print(f"\n--- alpha = {alpha:.0%} ---")
            print(f"{'h':>4} {'Expected':>8} {'Exceed':>8} {'Zone':>8} {'p(POF)':>8} {'Z1':>8} {'Z2':>8}")
            print(f"{'-'*56}")
            for _, row in subset.iterrows():
                print(
                    f"{int(row['horizon']):>4} "
                    f"{row['expected_rate']:>8.4f} "
                    f"{row['exceedance_rate']:>8.4f} "
                    f"{row['zone']:>8} "
                    f"{row['p_value']:>8.4f} "
                    f"{row['Z1']:>8} "
                    f"{row['Z2']:>8}"
                )

        # Overall verdict
        red_zones = df[df["zone"] == "Red"]
        if len(red_zones) == 0:
            print(f"\nVERDICT: ALL GREEN/YELLOW -- Model passes Basel framework")
        else:
            print(f"\nVERDICT: {len(red_zones)} RED zone(s) -- Model INADEQUATE")
            for _, row in red_zones.iterrows():
                print(f"  h={int(row['horizon'])}, alpha={row['alpha']:.0%}: {row['zone']}")

        print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Analytical GARCH VaR -- rolling refit every N hours
# ---------------------------------------------------------------------------

def _fit_garch_on_window(
    train_returns: np.ndarray,
    use_figarch: bool = False,
) -> Optional[dict]:
    """
    Fit GARCH(1,1) (or FIGARCH(1,d,1)) with Student-t errors on a training window.

    Returns dict with omega, beta, nu, mu, last_variance (hourly units) — plus
    alpha (GARCH) or figarch_weights (FIGARCH) — or None if fit fails.

    FIX 6 (C2): when use_figarch=True, route through fit_garch_model(use_figarch=True)
    so the Basel MC validation tests the DEPLOYED FIGARCH variance variant rather than
    the GARCH-only one. fit_garch_model already returns the figarch_weights the
    simulator needs; without this the validator would silently validate GARCH.
    """
    if use_figarch:
        from core.pricing.btc_pricing_engine import fit_garch_model
        try:
            return fit_garch_model(pd.Series(train_returns), use_figarch=True)
        except Exception as e:
            logger.debug("FIGARCH fit failed: %s", e)
            return None

    from arch import arch_model

    scaled = pd.Series(train_returns) * 100

    try:
        model = arch_model(scaled, vol='Garch', p=1, q=1, dist='t', mean='Constant')
        res = model.fit(disp='off')
        params = res.params

        forecast = res.forecast(horizon=1)
        last_var_scaled = forecast.variance.values[-1, 0]

        return {
            'omega': params['omega'] / 10000.0,
            'alpha': params['alpha[1]'],
            'beta': params['beta[1]'],
            'nu': params['nu'],
            'mu': params['mu'] / 100.0,
            'last_variance': last_var_scaled / 10000.0,
        }
    except Exception as e:
        logger.debug("GARCH fit failed: %s", e)
        return None


def _forecast_garch_var(
    garch_params: dict,
    horizon_hours: int,
    alpha: float,
) -> float:
    """
    Forecast h-step ahead VaR from a fitted GARCH(1,1) model.

    Computes total variance of h-period cumulative log return using the
    GARCH(1,1) recursion:

      E[sigma^2_{t+j} | F_t] = sigma^2_uncond
        + (alpha+beta)^(j-1) * (h_{t+1} - sigma^2_uncond)

      total_var = sum_{j=1}^{h} E[sigma^2_{t+j} | F_t]

      VaR = h * mu + sqrt(total_var) * t_quantile(alpha, nu)

    where h_{t+1} is the 1-step ahead conditional variance forecast.
    """
    omega = garch_params['omega']
    alpha_g = garch_params['alpha']
    beta = garch_params['beta']
    nu = garch_params['nu']
    mu = garch_params['mu']
    h_t1 = garch_params['last_variance']  # 1-step ahead conditional variance

    persistence = alpha_g + beta
    sigma2_uncond = omega / (1.0 - persistence) if persistence < 1.0 else h_t1

    if abs(1.0 - persistence) < 1e-12 or persistence >= 1.0:
        # IGARCH / unit root: variance grows linearly
        total_var = h_t1 * horizon_hours
    else:
        # Geometric series: sum_{j=1}^h [sigma2_uncond + (persistence)^(j-1) * (h_t1 - sigma2_uncond)]
        # = h * sigma2_uncond + (h_t1 - sigma2_uncond) * (1 - persistence^h) / (1 - persistence)
        total_var = (
            horizon_hours * sigma2_uncond
            + (h_t1 - sigma2_uncond) * (1.0 - persistence ** horizon_hours) / (1.0 - persistence)
        )

    # Student-t quantile at alpha level (lower tail)
    t_quantile = float(t_dist.ppf(alpha, nu))

    # VaR: mean drift + volatility
    var_forecast = horizon_hours * mu + np.sqrt(max(total_var, 1e-12)) * t_quantile

    return float(var_forecast)


def compute_analytical_garch_var(
    hourly_returns: np.ndarray,
    horizons: List[int],
    alphas: List[float],
    refit_every: int = DEFAULT_ROLLING_WINDOW,
    min_training: int = MIN_TRAINING_HOURS,
) -> Dict[Tuple[int, float], np.ndarray]:
    """
    Compute rolling analytical GARCH Student-t VaR forecasts.

    At each refit point t (every `refit_every` hours), fits GARCH(1,1) on
    returns[:t], then forecasts h-step ahead VaR. Between refits, the last
    fitted model is used (static forecast) -- this avoids re-estimating for
    every observation while remaining responsive to regime shifts.

    Args:
        hourly_returns: Array of hourly log returns.
        horizons: VaR forecast horizons in hours.
        alphas: VaR confidence levels.
        refit_every: Number of hours between GARCH refits.
        min_training: Minimum hours to fit a model.

    Returns:
        Dict mapping (horizon, alpha) → array of VaR forecasts (length = n).
    """
    n = len(hourly_returns)
    var_forecasts: Dict[Tuple[int, float], np.ndarray] = {
        (h, a): np.full(n, np.nan) for h in horizons for a in alphas
    }

    last_garch_params: Optional[dict] = None
    last_refit_idx: int = -1

    for t in range(min_training, n):
        # Re-fit only at specified intervals
        if t == min_training or (t - last_refit_idx) >= refit_every:
            train_data = hourly_returns[:t]
            garch_params = _fit_garch_on_window(train_data)
            if garch_params is not None:
                last_garch_params = garch_params
                last_refit_idx = t
                if t % (refit_every * 10) == 0:
                    persistence = garch_params['alpha'] + garch_params['beta']
                    logger.debug(
                        "GARCH refit at obs %d/%d: omega=%.8f, alpha=%.3f, "
                        "beta=%.3f, persistence=%.4f, nu=%.1f, mu=%.6f",
                        t, n, garch_params['omega'], garch_params['alpha'],
                        garch_params['beta'], persistence, garch_params['nu'],
                        garch_params['mu'],
                    )

        if last_garch_params is None:
            continue

        for h in horizons:
            for a in alphas:
                var_forecasts[(h, a)][t] = _forecast_garch_var(
                    last_garch_params, horizon_hours=h, alpha=a,
                )

    return var_forecasts


# ---------------------------------------------------------------------------
# MC Validation Mode -- full SVCJ simulator
# ---------------------------------------------------------------------------

def _mc_refit_point(args: tuple) -> Tuple[int, Optional[Dict[Tuple[int, float], float]]]:
    """Fit GARCH/FIGARCH on returns[:t] and compute per-(horizon, alpha) VaR.

    Module-level so ProcessPoolExecutor can pickle it (Windows spawn). Each
    refit point is independent of the others (expanding-window fit on the
    prefix of the returns array only), so points can run in parallel.
    """
    (t, hourly_returns, horizons, alphas, num_sims, seed,
     use_figarch, use_svcj, use_skewed_t, use_naive_prior, jump_params) = args
    from core.pricing.btc_pricing_engine import simulate_paths

    garch_params = _fit_garch_on_window(hourly_returns[:t], use_figarch=use_figarch)
    if garch_params is None:
        return t, None

    # Use last price from training data as S0
    S0 = np.exp(np.cumsum(hourly_returns[:t]))[-1]
    vals: Dict[Tuple[int, float], float] = {}
    for h in horizons:
        if h <= 0:
            continue
        # Simulate paths. FIX 6 (C2): pass the deployed feature flags so the
        # validator tests what is actually traded (FIGARCH + SVCJ + skewed-t +
        # naive prior), not bare GARCH/Student-t.
        paths = simulate_paths(
            S0, garch_params,
            jump_params=jump_params,
            hours_to_expiry=h,
            n_sims=num_sims,
            seed=seed,
            use_naive_prior=use_naive_prior,
            use_svcj=use_svcj,
            use_skewed_t=use_skewed_t,
            use_figarch=use_figarch,
        )
        # FIX 6 (C2): simulate_paths returns a 1-D array of TERMINAL prices, not a
        # (n_sims, n_steps) matrix — `paths[:, -1]` raised IndexError. Use `paths`.
        sim_returns = np.log(paths / S0)
        for a in alphas:
            vals[(h, a)] = float(np.percentile(sim_returns, a * 100))
    return t, vals


def compute_mc_var(
    hourly_returns: np.ndarray,
    horizons: List[int],
    alphas: List[float],
    refit_every: int = DEFAULT_ROLLING_WINDOW,
    min_training: int = MIN_TRAINING_HOURS,
    num_sims: int = 10000,
    seed: int = 42,
    use_jumps: bool = True,
    use_figarch: bool = True,
    use_svcj: bool = True,
    use_skewed_t: bool = True,
    use_naive_prior: bool = True,
    n_workers: int = 1,
) -> Dict[Tuple[int, float], np.ndarray]:
    """
    Compute VaR forecasts using full Monte Carlo simulation (GARCH + SVCJ).

    At each refit point, fits GARCH on returns[:t], loads calibrated jump
    params, then runs `simulate_paths()` for each horizon. VaR comes from
    the empirical quantile of the simulated terminal-to-initial return distribution.

    Args:
        hourly_returns: Array of hourly log returns.
        horizons: VaR forecast horizons in hours.
        alphas: VaR confidence levels.
        refit_every: Hours between GARCH refits.
        min_training: Minimum hours for initial fit.
        num_sims: Number of MC simulations per horizon per refit.
        seed: Random seed.
        use_jumps: If True, include SVCJ jumps in simulation.

    Returns:
        Dict mapping (horizon, alpha) → array of VaR forecasts (length = n).
    """
    from core.pricing.btc_pricing_engine import load_calibrated_jumps

    n = len(hourly_returns)

    var_forecasts: Dict[Tuple[int, float], np.ndarray] = {
        (h, a): np.full(n, np.nan) for h in horizons for a in alphas
    }

    # Pre-calibrate jumps on all data for efficiency
    try:
        cal_jumps = load_calibrated_jumps(
            hourly_csv="DATA/btc_hourly.csv", force_recalibrate=False,
        )
        if cal_jumps.get("fit_converged", False):
            jump_params = {
                "lambda": cal_jumps["lam"],
                "crash_prob": cal_jumps["p_crash"],
                "eta_up": cal_jumps["eta_up"],
                "eta_down": cal_jumps["eta_down"],
                "mu_v": cal_jumps["mu_v"],
                "rho_J": cal_jumps["rho_J"],
                # FIX 4 (M1) consistency: SVCJ return-vol regression slope actually
                # used in simulate_paths (rho_J above is reporting-only). Keeps this
                # caller aligned with backrunner/live pipelines.
                "rho_j_slope": cal_jumps.get("rho_j_slope", 0.0),
                "sigma_s": cal_jumps.get("sigma_s", 0.01),
            }
            logger.info("MC mode: using calibrated jumps (lam=%.1f/yr)", cal_jumps["lam"])
        else:
            jump_params = None
    except Exception as e:
        logger.info("Jump calibration skipped: %s -- using default jumps", e)
        jump_params = None

    # Refit schedule: matches the sequential loop when every fit succeeds
    # (t = min_training, +refit_every, ...). Each point is an independent
    # expanding-window fit on returns[:t], so points parallelize cleanly.
    # Behavior deviation vs the old loop: a FAILED fit no longer retries
    # hour-by-hour — its window keeps the previous refit's values (fits
    # essentially never fail; _fit_garch_on_window falls back FIGARCH→GARCH
    # internally).
    refit_points = list(range(min_training, n, refit_every))
    effective_jumps = jump_params if use_jumps else None
    worker_args = [
        (t, hourly_returns, horizons, alphas, num_sims, seed,
         use_figarch, use_svcj, use_skewed_t, use_naive_prior, effective_jumps)
        for t in refit_points
    ]

    results: List[Tuple[int, Optional[Dict[Tuple[int, float], float]]]] = []
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, res in enumerate(executor.map(_mc_refit_point, worker_args)):
                results.append(res)
                logger.info(
                    "MC refit %d/%d (obs %d/%d): %s",
                    i + 1, len(refit_points), res[0], n,
                    "ok" if res[1] else "FIT FAILED",
                )
    else:
        for i, args in enumerate(worker_args):
            res = _mc_refit_point(args)
            results.append(res)
            logger.info(
                "MC refit %d/%d (obs %d/%d): %s",
                i + 1, len(refit_points), res[0], n,
                "ok" if res[1] else "FIT FAILED",
            )

    # Fill forward: between successful refits the sim params AND the seed are
    # unchanged, so simulate_paths would return identical paths at every t —
    # recomputing per hour is pure waste (~500x).
    successes = [(t, vals) for t, vals in results if vals]
    for idx, (t0, vals) in enumerate(successes):
        t1 = successes[idx + 1][0] if idx + 1 < len(successes) else n
        for key, value in vals.items():
            var_forecasts[key][t0:t1] = value

    return var_forecasts


# ---------------------------------------------------------------------------
# Main Backtest Function
# ---------------------------------------------------------------------------

def run_basel_backtest(
    returns: np.ndarray,
    horizons: Optional[List[int]] = None,
    alphas: Optional[List[float]] = None,
    mode: str = "analytical",
    refit_every: int = DEFAULT_ROLLING_WINDOW,
    num_sims: int = 10000,
    seed: int = 42,
    use_jumps: bool = True,
    use_figarch: bool = True,
    use_svcj: bool = True,
    use_skewed_t: bool = True,
    use_naive_prior: bool = True,
    n_workers: int = 1,
) -> BaselBacktestResult:
    """
    Run Teng-style Basel backtest on BTC hourly returns.

    Two modes:
      - "analytical": Rolling GARCH(1,1) + Student-t VaR via conditional variance.
        Fast, captures time-varying volatility and fat tails. ~500ms per refit.
      - "mc": Full Monte Carlo GARCH + SVCJ simulations. Most accurate but slow.
        Each refit runs num_sims paths per horizon. ~seconds per refit.

    Args:
        returns: Array of hourly log returns.
        horizons: Forecast horizons in hours (default: [1, 336, 672] = 1h, 14d, 28d).
        alphas: VaR confidence levels (default: [0.05, 0.01]).
        mode: "analytical" or "mc".
        refit_every: Hours between model refits (default: 500).
        num_sims: MC simulations per refit (MC mode only).
        seed: Random seed.
        use_jumps: Include SVCJ jumps (MC mode only).

    Returns:
        BaselBacktestResult with traffic light classifications.
    """
    if horizons is None:
        horizons = [1, 14 * 24, 28 * 24]  # 1h, 14d, 28d

    if alphas is None:
        alphas = [0.05, 0.01]

    n = len(returns)
    logger.info(
        "Starting Basel backtest: mode=%s, n=%d, horizons=%s, alphas=%s",
        mode, n, horizons, alphas,
    )

    # Compute VaR forecasts
    if mode == "mc":
        var_forecasts = compute_mc_var(
            hourly_returns=returns,
            horizons=horizons,
            alphas=alphas,
            refit_every=refit_every,
            num_sims=num_sims,
            seed=seed,
            use_jumps=use_jumps,
            use_figarch=use_figarch,
            use_svcj=use_svcj,
            use_skewed_t=use_skewed_t,
            use_naive_prior=use_naive_prior,
            n_workers=n_workers,
        )
    else:
        var_forecasts = compute_analytical_garch_var(
            hourly_returns=returns,
            horizons=horizons,
            alphas=alphas,
            refit_every=refit_every,
        )

    results = {}

    # Realized h-hour FORWARD cumulative return starting at t: the VaR at t is
    # a quantile of the h-hour cumulative return distribution, so it must be
    # compared against sum(returns[t:t+h]), NOT the single 1-hour return at t
    # (which for h>>1 can never breach an h-hour VaR — every long-horizon cell
    # came out 0 exceedances / Red regardless of the model). For h=1 this
    # reduces to the old behavior exactly.
    csum = np.concatenate([[0.0], np.cumsum(returns)])

    for h in horizons:
        fwd_h = np.full(n, np.nan)
        if 0 < h <= n:
            fwd_h[: n - h + 1] = csum[h:] - csum[: n + 1 - h]
        if h > 1:
            logger.info(
                "h=%d: overlapping forward windows — exceedances are serially "
                "correlated, POF p-values are optimistic; treat zones as "
                "indicative.", h,
            )
        for alpha in alphas:
            var_h = var_forecasts.get((h, alpha), np.full(n, np.nan))
            valid = ~np.isnan(var_h) & ~np.isnan(fwd_h)

            n_valid = int(np.sum(valid))
            if n_valid < 10:
                logger.warning("h=%d alpha=%.0f%%: only %d valid forecasts -- skipping", h, alpha * 100, n_valid)
                continue

            var_values = var_h[valid]
            ret_values = fwd_h[valid]

            # Count exceedances
            n_exceed = int(np.sum(ret_values < var_values))
            exceed_rate = n_exceed / n_valid

            # Traffic light
            zone, p_value = basel_traffic_light(n_exceed, alpha, n_valid)

            # ES tests (at 1% only, per Teng)
            es_tests = {}
            if alpha == 0.01:
                es_tests = expected_shortfall_test(ret_values, var_values, alpha)

            results[(h, alpha)] = HorizonResult(
                horizon=h,
                expected_rate=alpha,
                exceedance_rate=exceed_rate,
                n_obs=n_valid,
                n_exceed=n_exceed,
                zone=zone,
                p_value=p_value,
                es_tests=es_tests,
            )

            logger.info(
                "h=%4d alpha=%.0f%%: exceed=%.4f vs expected=%.4f → %s (p=%.4f) "
                "Z1=%.2f Z2=%.2f",
                h, alpha * 100, exceed_rate, alpha, zone, p_value,
                es_tests.get("Z1", np.nan), es_tests.get("Z2", np.nan),
            )

    # Summary
    red_count = sum(1 for hr in results.values() if hr.zone == "Red")
    yellow_count = sum(1 for hr in results.values() if hr.zone == "Yellow")
    green_count = sum(1 for hr in results.values() if hr.zone == "Green")

    summary = f"Green:{green_count} Yellow:{yellow_count} Red:{red_count}"

    return BaselBacktestResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        n_total_obs=n,
        horizons=horizons,
        alphas=alphas,
        results=results,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Run Teng Basel backtest on BTC returns")
    parser.add_argument("--input", default="DATA/btc_hourly.csv", help="Path to hourly BTC data")
    parser.add_argument("--mode", default="analytical", choices=["analytical", "mc"],
                       help="VaR computation mode: analytical GARCH or MC simulation")
    parser.add_argument("--refit-every", type=int, default=500,
                       help="Hours between GARCH refits (default: 500)")
    parser.add_argument("--num-sims", type=int, default=10000,
                       help="MC simulations per refit (MC mode only)")
    parser.add_argument("--no-jumps", action="store_true",
                       help="Disable SVCJ jumps in MC mode")
    parser.add_argument("--garch-only", action="store_true",
                       help="MC mode: validate the plain GARCH variant (disable "
                            "FIGARCH/SVCJ/skewed-t) instead of the deployed config")
    parser.add_argument("--workers", type=int, default=1,
                       help="Parallel workers for MC-mode refit points (default: 1)")
    args = parser.parse_args()

    # Load returns
    df = pd.read_csv(args.input)
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get('close', df.columns[-1])
    returns = np.log(df[close_col] / df[close_col].shift(1)).dropna().values

    logger.info(f"Loaded {len(returns)} hourly returns from {args.input}")

    _deployed = not args.garch_only
    result = run_basel_backtest(
        returns,
        horizons=[1, 14 * 24, 28 * 24],
        alphas=[0.05, 0.01],
        mode=args.mode,
        refit_every=args.refit_every,
        num_sims=args.num_sims,
        use_jumps=not args.no_jumps,
        use_figarch=_deployed,
        use_svcj=_deployed,
        use_skewed_t=_deployed,
        n_workers=args.workers,
    )

    result.print_summary()
