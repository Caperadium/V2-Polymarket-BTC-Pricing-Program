"""
bayesian_estimation.py

Wave 3 / T8 (Journal Articles/improvement_plan.md section 3.2): Bayesian
posterior distributions for the key BTC pricing-engine parameters, replacing
point estimates with credible intervals for P(S_T >= K).

No new dependencies: numpy/scipy/pandas only.
  - GARCH(1,1)-t parameters (omega, alpha, beta, nu): custom 2-chain
    random-walk Metropolis-Hastings on a transformed (unconstrained) scale,
    with the log-Jacobian folded into the target so the chain samples the
    correct posterior in the NATURAL parameter scale.
  - Jump parameters (lambda, p_crash, eta_up, eta_down): CLOSED-FORM
    Gamma/Beta conjugate posteriors on top of the existing bipower jump
    detector (`core.pricing.jump_calibration.detect_jumps_bipower`) -- no
    sampling loop needed, these are exact.

SCOPE:
  - FIGARCH is OUT OF SCOPE for this module. The FIGARCH(1,d,1) likelihood
    (fractionally-integrated ARCH(infinity) recursion, ~1000 lag terms) is not
    a small extension of the GARCH(1,1) MH sampler implemented here -- it is
    Phase 4 material. `garch_posterior` always fits/samples plain GARCH(1,1);
    FIGARCH keeps its point-estimate-only path via `fit_garch_model(use_figarch=True)`.
  - Regime probabilities already have posteriors from the HMM
    (`core.pricing.regime_detector.RegimeDetector` exposes the fitted
    posterior weights directly) -- not duplicated here.
  - `posterior_probability_bands` draws GARCH and jump parameters
    INDEPENDENTLY (no joint/cross-parameter posterior) and runs the BASE
    engine only (`use_naive_prior=True`, no regime layer, no XGB directional
    shift) -- the bands quantify PARAMETER uncertainty of the base engine,
    not model-structure uncertainty.

PERFORMANCE NOTE: the GARCH-t log-likelihood recursion is inherently
sequential (each step's conditional variance depends on the previous step),
so it cannot be fully vectorized across time. To keep 2 chains x n_iter MH
iterations tractable in pure Python, the likelihood is truncated to the most
recent 15,000 hourly returns (~1.7 years) -- consistent with this module's
rolling-window philosophy (`core.validation.rolling_evaluator`) -- and the
per-step recursion uses plain scalar arithmetic (fast), with the Student-t
density evaluated ONCE per likelihood call via a single vectorized
`scipy.stats.t.logpdf` over the whole return series (slow if done per-step,
fast when batched). Measured runtime is reported in the module's test suite
and CHANGES.md; if a future environment finds the default n_iter=4000 over
the ~5 minute CLI budget, lower `--n-iter` (or the `garch_posterior` default)
to 2500 -- see CHANGES.md Wave 3 entry for the measurement this project used.

CLI:
    python core/pricing/bayesian_estimation.py --strikes 90000,100000 \\
        --hours 336 --n-posterior 50
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, halfnorm, beta as beta_dist, expon

# Guard: ensure repo root is on sys.path when invoked as a script (e.g.
# `python core/pricing/bayesian_estimation.py --help`), mirroring
# core/backtesting/backrunner.py. Without this, module-level `core.*` imports
# below raise ModuleNotFoundError because sys.path[0] is this script's own
# directory, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pricing.btc_pricing_engine import (
    fit_garch_model,
    filter_jump_returns,
    simulate_paths,
)
from core.pricing.jump_calibration import detect_jumps_bipower

logger = logging.getLogger(__name__)

# Truncate the GARCH-t likelihood to the most recent N hourly returns (see
# PERFORMANCE NOTE above). Consistent with the rolling-window philosophy --
# older data contributes less to current volatility dynamics anyway.
MAX_LIKELIHOOD_OBS = 15000

# Simplex cap: alpha + beta < GARCH_SIMPLEX_CAP (< 1 for stationarity, with
# headroom so the recursion never sits exactly at the unit-root boundary).
GARCH_SIMPLEX_CAP = 0.999

# nu (Student-t degrees of freedom) is sampled as nu = NU_FLOOR + exp(c);
# NU_FLOOR keeps variance finite (nu>2) with margin for numerical stability.
NU_FLOOR = 2.1

_HOURS_PER_YEAR = 365.0 * 24.0


# ==============================================================================
# GARCH(1,1)-t POSTERIOR (random-walk Metropolis-Hastings)
# ==============================================================================

@dataclass
class GarchPosterior:
    """Posterior draws for a GARCH(1,1)-Student-t model.

    `draws` columns: omega, alpha, beta, nu (post burn-in, thinned, both
    chains pooled). `mu` (drift) is held fixed at the MLE point estimate --
    the naive-prior engine zeroes drift downstream anyway (Wave 1 M3 note),
    so it is not part of the sampled parameter vector.
    """
    draws: pd.DataFrame
    acceptance_rate: float
    rhat: Dict[str, float]
    converged: bool
    point_estimate: dict


def _garch_t_loglik(r_scaled: np.ndarray, omega: float, alpha: float, beta: float,
                     mu: float, nu: float) -> float:
    """GARCH(1,1)-Student-t log-likelihood on returns*100 (scaled units, same
    convention as `fit_garch_model`). Variance recursion is a plain Python
    loop (sequential dependency, cannot vectorize); the Student-t density is
    evaluated in ONE batched `scipy.stats.t.logpdf` call over the full series.

    Initial variance = sample variance (matches plan spec).
    """
    n = len(r_scaled)
    if n == 0:
        return -np.inf
    eps = r_scaled - mu
    var = np.empty(n)
    var[0] = max(float(np.var(r_scaled)), 1e-10)
    eps_sq = eps * eps
    for tt in range(1, n):
        v = omega + alpha * eps_sq[tt - 1] + beta * var[tt - 1]
        var[tt] = v if v > 1e-10 else 1e-10

    sigma = np.sqrt(var)
    scale = np.sqrt((nu - 2.0) / nu) if nu > 2.0 else 1.0
    z = eps / (sigma * scale)
    ll = np.sum(t_dist.logpdf(z, nu) - np.log(sigma * scale))
    return float(ll)


def _natural_to_transformed(omega: float, alpha: float, beta: float, nu: float) -> np.ndarray:
    """Inverse of the constrained bijection used by the sampler (see
    `_transformed_to_natural`). Used to initialize chains from the MLE."""
    log_omega = np.log(max(omega, 1e-12))
    s = min(max((alpha + beta) / GARCH_SIMPLEX_CAP, 1e-9), 1 - 1e-9)
    frac = min(max(alpha / max(alpha + beta, 1e-12), 1e-9), 1 - 1e-9)
    a_prime = np.log(s / (1 - s))
    b_prime = np.log(frac / (1 - frac))
    log_nu_c = np.log(max(nu - NU_FLOOR, 1e-6))
    return np.array([log_omega, a_prime, b_prime, log_nu_c])


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _transformed_to_natural(theta: np.ndarray) -> Tuple[float, float, float, float]:
    """Constrained bijection: unconstrained R^4 -> (omega>0, alpha>0, beta>0,
    alpha+beta<GARCH_SIMPLEX_CAP, nu>NU_FLOOR).

    (alpha, beta) via a stick-breaking style map: sa = sigmoid(a'), sb =
    sigmoid(b'); alpha = CAP*sa*sb, beta = CAP*sa*(1-sb). alpha+beta = CAP*sa
    < CAP < 1 by construction, so the stationarity constraint always holds.
    """
    log_omega, a_prime, b_prime, log_nu_c = theta
    omega = np.exp(log_omega)
    sa = _sigmoid(a_prime)
    sb = _sigmoid(b_prime)
    alpha = GARCH_SIMPLEX_CAP * sa * sb
    beta = GARCH_SIMPLEX_CAP * sa * (1.0 - sb)
    nu = NU_FLOOR + np.exp(log_nu_c)
    return float(omega), float(alpha), float(beta), float(nu)


def _log_jacobian(theta: np.ndarray) -> float:
    """log|d(omega,alpha,beta,nu)/d(theta)| for the bijection above.

    omega: d(omega)/d(log_omega) = omega  -> log|J| = log_omega.
    nu: d(nu)/d(log_nu_c) = exp(log_nu_c) = nu - NU_FLOOR -> log|J| = log_nu_c.
    (alpha,beta) 2x2 Jacobian determinant (derived from the stick-breaking
    map): |det| = CAP^2 * sa^2*(1-sa) * sb*(1-sb).
    """
    log_omega, a_prime, b_prime, log_nu_c = theta
    sa = _sigmoid(a_prime)
    sb = _sigmoid(b_prime)
    # Guard against exact 0/1 saturation (log(0) = -inf poisons the chain).
    eps = 1e-300
    log_j_ab = (
        2 * np.log(GARCH_SIMPLEX_CAP)
        + 2 * np.log(max(sa, eps)) + np.log(max(1 - sa, eps))
        + np.log(max(sb, eps)) + np.log(max(1 - sb, eps))
    )
    return float(log_omega + log_nu_c + log_j_ab)


def _log_prior(omega: float, alpha: float, beta: float, nu: float, omega_prior_scale: float) -> float:
    """Weakly-informative priors on the NATURAL scale (see module docstring /
    plan for exact hyperparameters): omega ~ HalfNormal, alpha ~ Beta(2,10),
    beta ~ Beta(10,2), (nu - NU_FLOOR) ~ Exponential(mean=8)."""
    lp_omega = halfnorm.logpdf(omega, scale=omega_prior_scale)
    lp_alpha = beta_dist.logpdf(min(max(alpha, 1e-12), 1 - 1e-12), 2.0, 10.0)
    lp_beta = beta_dist.logpdf(min(max(beta, 1e-12), 1 - 1e-12), 10.0, 2.0)
    lp_nu = expon.logpdf(nu - NU_FLOOR, scale=8.0)
    return float(lp_omega + lp_alpha + lp_beta + lp_nu)


def _log_target(theta: np.ndarray, r_scaled: np.ndarray, mu_scaled: float, omega_prior_scale: float) -> float:
    omega, alpha, beta, nu = _transformed_to_natural(theta)
    if not (omega > 0 and alpha > 0 and beta > 0 and (alpha + beta) < 1.0 and nu > NU_FLOOR):
        return -np.inf
    try:
        ll = _garch_t_loglik(r_scaled, omega, alpha, beta, mu_scaled, nu)
        lp = _log_prior(omega, alpha, beta, nu, omega_prior_scale)
        lj = _log_jacobian(theta)
    except (FloatingPointError, ValueError, OverflowError):
        return -np.inf
    total = ll + lp + lj
    return total if np.isfinite(total) else -np.inf


def _run_mh_chain(
    r_scaled: np.ndarray, mu_scaled: float, omega_prior_scale: float,
    init_theta: np.ndarray, n_iter: int, burn_in: int, seed: int,
) -> Tuple[np.ndarray, float]:
    """Single-chain random-walk Metropolis-Hastings on the transformed scale.
    Step sizes are adapted (x1.1 / x0.9 toward 30% acceptance) every 100
    iterations DURING burn-in only, then frozen. Returns (theta_draws[n_iter,4],
    acceptance_rate)."""
    rng = np.random.default_rng(seed)
    step_sizes = np.array([0.25, 0.4, 0.4, 0.3])

    theta = init_theta.copy()
    log_target_val = _log_target(theta, r_scaled, mu_scaled, omega_prior_scale)

    draws = np.empty((n_iter, 4))
    n_accept = 0
    window_accept = 0

    for i in range(n_iter):
        proposal = theta + rng.normal(0.0, step_sizes)
        lt_new = _log_target(proposal, r_scaled, mu_scaled, omega_prior_scale)
        log_alpha = lt_new - log_target_val
        if np.log(rng.uniform()) < log_alpha:
            theta = proposal
            log_target_val = lt_new
            n_accept += 1
            window_accept += 1
        draws[i] = theta

        if i < burn_in and (i + 1) % 100 == 0:
            rate = window_accept / 100.0
            if rate > 0.30:
                step_sizes = step_sizes * 1.1
            else:
                step_sizes = step_sizes * 0.9
            window_accept = 0

    return draws, n_accept / n_iter


def _split_rhat(chain_samples: Sequence[np.ndarray]) -> float:
    """Split-Rhat (Gelman et al.): each input chain is split into two halves,
    treated as independent chains, and the standard Rhat formula is applied."""
    split_chains = []
    for c in chain_samples:
        n = len(c)
        half = n // 2
        if half < 2:
            return float("nan")
        split_chains.append(np.asarray(c[:half]))
        split_chains.append(np.asarray(c[half:2 * half]))

    m = len(split_chains)
    n = len(split_chains[0])
    chain_means = np.array([np.mean(c) for c in split_chains])
    chain_vars = np.array([np.var(c, ddof=1) for c in split_chains])

    W = float(np.mean(chain_vars))
    B = float(n * np.var(chain_means, ddof=1))
    if W <= 0:
        return float("nan")
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(max(var_hat, 0.0) / W))


def garch_posterior(
    returns: pd.Series,
    n_iter: int = 4000,
    burn_in: int = 1000,
    thin: int = 3,
    seed: int = 42,
    filter_jumps: bool = True,
) -> GarchPosterior:
    """2-chain random-walk Metropolis posterior for GARCH(1,1)-Student-t
    parameters (omega, alpha, beta, nu). mu is fixed at the MLE (see
    GarchPosterior docstring). FIGARCH is out of scope (module docstring).

    Truncates the likelihood to the most recent MAX_LIKELIHOOD_OBS returns.
    """
    returns = returns if isinstance(returns, pd.Series) else pd.Series(returns)
    n_available = len(returns)
    if n_available > MAX_LIKELIHOOD_OBS:
        returns = returns.iloc[-MAX_LIKELIHOOD_OBS:]
        logger.info(
            "garch_posterior: truncating likelihood to most recent %d hourly "
            "returns (of %d available) -- see module PERFORMANCE NOTE.",
            MAX_LIKELIHOOD_OBS, n_available,
        )

    if filter_jumps:
        returns = filter_jump_returns(returns)

    # Point estimate (MLE fallback + dispersed-start anchor). Already filtered
    # above if requested, so avoid double-filtering here.
    point_estimate = fit_garch_model(returns, filter_jumps=False, use_figarch=False)

    r_scaled = returns.to_numpy() * 100.0
    mu_scaled = point_estimate["mu"] * 100.0
    sample_var = float(np.var(r_scaled))
    omega_prior_scale = 5.0 * sample_var * (1.0 - 0.95)  # = 0.25 * sample_var

    mle_theta = _natural_to_transformed(
        point_estimate["omega"] * 10000.0,
        point_estimate["alpha"],
        point_estimate["beta"],
        point_estimate["nu"],
    )

    # Chain 2: dispersed start (+50% omega, -20% beta), alpha unchanged,
    # renormalized if it would violate the simplex constraint.
    omega2 = point_estimate["omega"] * 10000.0 * 1.5
    alpha2 = point_estimate["alpha"]
    beta2 = point_estimate["beta"] * 0.8
    if alpha2 + beta2 >= GARCH_SIMPLEX_CAP:
        scale_down = (GARCH_SIMPLEX_CAP * 0.95) / (alpha2 + beta2)
        alpha2 *= scale_down
        beta2 *= scale_down
    dispersed_theta = _natural_to_transformed(omega2, alpha2, beta2, point_estimate["nu"])

    draws1, acc1 = _run_mh_chain(r_scaled, mu_scaled, omega_prior_scale, mle_theta, n_iter, burn_in, seed)
    draws2, acc2 = _run_mh_chain(r_scaled, mu_scaled, omega_prior_scale, dispersed_theta, n_iter, burn_in, seed + 1)

    post1 = draws1[burn_in:]
    post2 = draws2[burn_in:]

    # Convert to natural scale for Rhat + output draws.
    def _to_natural_matrix(theta_mat: np.ndarray) -> np.ndarray:
        out = np.empty_like(theta_mat)
        for i in range(len(theta_mat)):
            omega, alpha, beta, nu = _transformed_to_natural(theta_mat[i])
            out[i] = [omega / 10000.0, alpha, beta, nu]
        return out

    nat1 = _to_natural_matrix(post1)
    nat2 = _to_natural_matrix(post2)

    param_names = ["omega", "alpha", "beta", "nu"]
    rhat = {
        name: _split_rhat([nat1[:, j], nat2[:, j]])
        for j, name in enumerate(param_names)
    }

    thinned = np.vstack([nat1[::thin], nat2[::thin]])
    draws_df = pd.DataFrame(thinned, columns=param_names)

    acceptance_rate = float((acc1 + acc2) / 2.0)
    converged = bool(
        all(np.isfinite(v) and v < 1.1 for v in rhat.values())
        and 0.1 <= acceptance_rate <= 0.5
    )
    if not converged:
        logger.warning(
            "garch_posterior: chains did not meet convergence criteria "
            "(rhat=%s, acceptance=%.3f) -- downstream callers should treat "
            "`point_estimate` as authoritative.", rhat, acceptance_rate,
        )

    return GarchPosterior(
        draws=draws_df,
        acceptance_rate=acceptance_rate,
        rhat=rhat,
        converged=converged,
        point_estimate=point_estimate,
    )


# ==============================================================================
# JUMP PARAMETER POSTERIOR (closed-form Gamma/Beta conjugacy)
# ==============================================================================

@dataclass
class JumpPosterior:
    """Closed-form conjugate posterior draws for the Kou jump parameters."""
    lam_draws: np.ndarray        # Gamma posterior draws, annualized jump intensity
    eta_up_draws: np.ndarray
    eta_down_draws: np.ndarray
    p_crash_draws: np.ndarray    # Beta posterior draws


def _lam_posterior_shape_rate(
    n_jumps: int, n_obs: int, hours_per_year: float = _HOURS_PER_YEAR,
    prior_shape: float = 2.0, prior_rate: float = 2.0 / 25.0,
) -> Tuple[float, float]:
    """Gamma(2, rate=2/25) prior on annual lambda (mean 25/yr, weak) ->
    posterior Gamma(2 + n_jumps, rate=2/25 + n_obs/hours_per_year)."""
    return prior_shape + n_jumps, prior_rate + n_obs / hours_per_year


def _p_crash_posterior_ab(n_down: int, n_up: int,
                           prior_a: float = 3.0, prior_b: float = 2.0) -> Tuple[float, float]:
    """Beta(3,2) prior (mild down-bias) -> posterior Beta(3+n_down, 2+n_up)."""
    return prior_a + n_down, prior_b + n_up


def _eta_posterior_shape_rate(n_events: int, sum_sizes: float,
                               a0: float, b0: float) -> Tuple[float, float]:
    """Gamma(a0, b0) prior on the exponential jump-size rate eta ->
    posterior Gamma(a0 + n_events, b0 + sum_sizes)."""
    return a0 + n_events, b0 + sum_sizes


def jump_posterior(returns: np.ndarray, n_draws: int = 2000, seed: int = 42) -> JumpPosterior:
    """Closed-form conjugate posterior for lambda, p_crash, eta_up, eta_down
    given the bipower jump mask on `returns`. See module docstring hyperparams.
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    n_obs = len(returns)

    jump_mask, _sigma_local = detect_jumps_bipower(returns, return_sigma=True)
    jump_returns = returns[jump_mask]
    n_jumps = int(len(jump_returns))
    n_up = int(np.sum(jump_returns > 0))
    n_down = int(np.sum(jump_returns < 0))
    up_sizes = jump_returns[jump_returns > 0]
    down_sizes = -jump_returns[jump_returns < 0]  # positive magnitudes

    lam_shape, lam_rate = _lam_posterior_shape_rate(n_jumps, n_obs)
    lam_draws = rng.gamma(lam_shape, scale=1.0 / lam_rate, size=n_draws)

    a_crash, b_crash = _p_crash_posterior_ab(n_down, n_up)
    p_crash_draws = rng.beta(a_crash, b_crash, size=n_draws)

    up_shape, up_rate = _eta_posterior_shape_rate(n_up, float(np.sum(up_sizes)), a0=2.0, b0=2.0 / 50.0)
    eta_up_draws = rng.gamma(up_shape, scale=1.0 / up_rate, size=n_draws)

    down_shape, down_rate = _eta_posterior_shape_rate(n_down, float(np.sum(down_sizes)), a0=2.0, b0=2.0 / 25.0)
    eta_down_draws = rng.gamma(down_shape, scale=1.0 / down_rate, size=n_draws)

    return JumpPosterior(
        lam_draws=lam_draws,
        eta_up_draws=eta_up_draws,
        eta_down_draws=eta_down_draws,
        p_crash_draws=p_crash_draws,
    )


# ==============================================================================
# POSTERIOR PROBABILITY BANDS
# ==============================================================================

def _load_hourly_returns_and_s0(
    hourly_df: Optional[pd.DataFrame] = None, hourly_csv: str = "DATA/btc_hourly.csv",
) -> Tuple[pd.Series, float]:
    """Load hourly log returns and S0 (last hourly close) from `hourly_df` (DI
    for tests) or `hourly_csv`. NOTE: unlike `load_and_prep_data` in
    btc_pricing_engine.py, this does NOT touch intraday data -- the plan's
    `posterior_probability_bands` signature takes only hourly_df/hourly_csv,
    so S0 here is the last HOURLY close, not the higher-frequency intraday
    close the live engine uses for its spot mark. Fine for posterior-band
    estimation (a few-percent-of-a-day granularity difference is immaterial
    next to multi-day MC noise); documented as a deliberate scope choice."""
    if hourly_df is not None:
        df = hourly_df.copy()
    else:
        df = pd.read_csv(hourly_csv)

    col_map = {c.lower(): c for c in df.columns}
    if "close" not in col_map:
        raise ValueError("hourly data must contain a 'Close'/'close' column.")
    close_col = col_map["close"]

    returns = np.log(df[close_col] / df[close_col].shift(1)).dropna()
    s0 = float(df[close_col].iloc[-1])
    return returns, s0


def _garch_filter_terminal_variance(returns: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    """Run the GARCH(1,1) variance recursion under a GIVEN (omega, alpha, beta)
    draw over the full return history, in the same *100 scaled units as
    `fit_garch_model`, and return the terminal conditional variance in RAW
    (unscaled) units. Used by `posterior_probability_bands` -- each posterior
    draw needs ITS OWN last_variance; reusing the MLE's last_variance across
    draws would silently collapse the vol-path uncertainty back to a point
    estimate."""
    r = np.asarray(returns, dtype=float) * 100.0
    n = len(r)
    if n == 0:
        return 1e-10
    omega_scaled = omega * 10000.0
    var = max(float(np.var(r)), 1e-10)
    r_sq = r * r
    for tt in range(1, n):
        v = omega_scaled + alpha * r_sq[tt - 1] + beta * var
        var = v if v > 1e-10 else 1e-10
    return float(var / 10000.0)


def posterior_probability_bands(
    strikes: Sequence[float],
    hours_to_expiry: float,
    hourly_df: Optional[pd.DataFrame] = None,
    hourly_csv: str = "DATA/btc_hourly.csv",
    n_posterior: int = 100,
    n_sims_per_draw: int = 2000,
    seed: int = 42,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict:
    """Credible bands for P(S_T >= K) from independent GARCH x jump posterior
    draws, run through the BASE engine (`use_naive_prior=True`, no regime
    layer, no XGB). Returns {strike: {'q05':.., 'q50':.., 'q95':.., 'point':..}}
    plus '_meta'.

    GARCH and jump parameters are drawn INDEPENDENTLY (no joint posterior --
    this ignores cross-parameter dependence; documented limitation). Each
    draw's `last_variance` is recomputed under ITS OWN (omega, alpha, beta)
    via `_garch_filter_terminal_variance` -- never the MLE's last_variance.
    """
    hourly_returns, S0 = _load_hourly_returns_and_s0(hourly_df=hourly_df, hourly_csv=hourly_csv)
    r_arr = hourly_returns.to_numpy()

    garch_post = garch_posterior(hourly_returns, seed=seed)
    jump_post = jump_posterior(r_arr, n_draws=max(n_posterior, 200), seed=seed + 1)

    rng = np.random.default_rng(seed + 2)
    n_garch_draws = len(garch_post.draws)
    n_jump_draws = len(jump_post.lam_draws)
    garch_idx = rng.integers(0, n_garch_draws, size=n_posterior)
    jump_idx = rng.integers(0, n_jump_draws, size=n_posterior)

    strikes = list(strikes)
    prob_samples = {K: np.empty(n_posterior) for K in strikes}
    point_estimate = garch_post.point_estimate

    for i in range(n_posterior):
        row = garch_post.draws.iloc[garch_idx[i]]
        omega, alpha, beta, nu = float(row["omega"]), float(row["alpha"]), float(row["beta"]), float(row["nu"])
        last_var = _garch_filter_terminal_variance(r_arr, omega, alpha, beta)

        garch_params_i = {
            "omega": omega, "alpha": alpha, "beta": beta, "nu": nu,
            "mu": point_estimate["mu"], "last_variance": last_var,
        }
        j = jump_idx[i]
        jp_i = {
            "lambda": float(jump_post.lam_draws[j]),
            "crash_prob": float(jump_post.p_crash_draws[j]),
            "eta_up": float(jump_post.eta_up_draws[j]),
            "eta_down": float(jump_post.eta_down_draws[j]),
        }

        paths = simulate_paths(
            S0, garch_params_i, jp_i, hours_to_expiry=hours_to_expiry,
            n_sims=n_sims_per_draw, seed=seed + 10_000 + i, use_naive_prior=True,
        )
        for K in strikes:
            prob_samples[K][i] = float(np.mean(paths >= K))

    # Single standard run for the point estimate (MLE GARCH, default jumps).
    point_paths = simulate_paths(
        S0, point_estimate, None, hours_to_expiry=hours_to_expiry,
        n_sims=max(n_sims_per_draw * 5, 10000), seed=seed, use_naive_prior=True,
    )
    point_probs = {K: float(np.mean(point_paths >= K)) for K in strikes}

    result: Dict[object, object] = {}
    for K in strikes:
        samples = prob_samples[K]
        band = {f"q{int(round(q * 100)):02d}": float(np.quantile(samples, q)) for q in quantiles}
        band["point"] = point_probs[K]
        result[K] = band

    result["_meta"] = {
        "S0": S0,
        "n_posterior": n_posterior,
        "n_sims_per_draw": n_sims_per_draw,
        "garch_converged": garch_post.converged,
        "garch_acceptance_rate": garch_post.acceptance_rate,
        "garch_rhat": garch_post.rhat,
        "note": (
            "Bands quantify PARAMETER uncertainty of the BASE engine only "
            "(no regime layer, no XGB); GARCH and jump draws sampled "
            "independently (no joint posterior)."
        ),
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Bayesian posterior bands for P(S_T >= K) from the base BTC "
            "pricing engine (GARCH(1,1)-t + Kou jumps). FIGARCH out of scope."
        )
    )
    parser.add_argument("--hourly-csv", default="DATA/btc_hourly.csv", help="Path to hourly BTC data")
    parser.add_argument("--strikes", default="90000,100000", help="Comma-separated strike prices")
    parser.add_argument("--hours", type=float, default=336.0, help="Hours to expiry")
    parser.add_argument("--n-posterior", type=int, default=50, help="Number of joint posterior draws")
    parser.add_argument("--n-sims-per-draw", type=int, default=2000, help="MC paths per posterior draw")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    args = parser.parse_args()

    strikes = [float(x) for x in args.strikes.split(",")]

    t0 = time.time()
    bands = posterior_probability_bands(
        strikes, args.hours, hourly_csv=args.hourly_csv,
        n_posterior=args.n_posterior, n_sims_per_draw=args.n_sims_per_draw,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*72}")
    print(f"POSTERIOR PROBABILITY BANDS (elapsed {elapsed:.1f}s)")
    print(f"{'='*72}")
    print(f"{'Strike':>12} {'q05':>8} {'q50':>8} {'q95':>8} {'point':>8}")
    for K in strikes:
        b = bands[K]
        print(f"{K:>12.0f} {b['q05']:>8.4f} {b['q50']:>8.4f} {b['q95']:>8.4f} {b['point']:>8.4f}")
    meta = bands["_meta"]
    print(f"\nGARCH converged: {meta['garch_converged']} (acceptance={meta['garch_acceptance_rate']:.3f})")
    print(f"rhat: {meta['garch_rhat']}")
    print(f"{'='*72}\n")
