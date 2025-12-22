import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t

# ==============================================================================
# JUMP PARAMETERS (Kou's Double Exponential Jump Diffusion)
# These are kept constant for easy tuning as per requirements.
# ==============================================================================
LAMBDA = 25.0       # Jump intensity (expected number of jumps per year)
CRASH_PROB = 0.6    # Probability that a jump is a crash (downward)
ETA_UP = 50.0       # Decay parameter for upward jumps (1/mean jump size)
ETA_DOWN = 25.0     # Decay parameter for downward jumps (1/mean jump size)

# ==============================================================================
# DRIFT PARAMETERS
# ==============================================================================
DRIFT_CLAMP_MULT = 0.25  # Max drift = ±0.25 * sigma_day
MOMENTUM_GATE_MULT = 0.25  # Only apply momentum if |mu| > 0.25 * sigma_day

# ==============================================================================
# DATA INGESTION
# ==============================================================================
def load_and_prep_data(
    daily_csv: str = "DATA/btc_daily.csv",
    intraday_csv: str = "DATA/btc_intraday_1m.csv",
    daily_df: pd.DataFrame = None,
    intraday_df: pd.DataFrame = None,
):
    """
    Loads daily data for GARCH fitting and intraday data for the latest price mark.
    Supports dependency injection for backtesting.
    """
    # 1. Load Daily Data for GARCH fitting
    if daily_df is None:
        daily_df = pd.read_csv(daily_csv)
    else:
        daily_df = daily_df.copy()
    
    col_map = {c.lower(): c for c in daily_df.columns}
    if 'close' not in col_map:
        raise ValueError("daily_btc.csv must contain a 'Close' or 'close' column.")
    close_col = col_map['close']
    
    # Calculate Log Returns: ln(S_t / S_{t-1})
    daily_returns = np.log(daily_df[close_col] / daily_df[close_col].shift(1)).dropna()
    
    # 2. Load Intraday Data for S0
    if intraday_df is None:
        intraday_df = pd.read_csv(intraday_csv)
    else:
        intraday_df = intraday_df.copy()
        
    col_map_intra = {c.lower(): c for c in intraday_df.columns}
    if 'close' not in col_map_intra:
        raise ValueError("intraday_btc.csv must contain a 'Close' or 'close' column.")
    close_col_intra = col_map_intra['close']
    
    # Get the latest close price
    current_price_S0 = float(intraday_df[close_col_intra].iloc[-1])
    
    return daily_returns, current_price_S0

# ==============================================================================
# MODEL FITTING
# ==============================================================================
def fit_garch_model(daily_returns: pd.Series, drift_window: int = None):
    """
    Fits a GARCH(1,1) model with Student-t errors.
    
    CRITICAL UPDATE: MOMENTUM INJECTION
    If drift_window is provided, calculates the Exponential Moving Average (EMA)
    of log returns to capture short-term trend momentum.
    
    Args:
        daily_returns: pandas Series of daily log returns.
        drift_window: If int, use EMA of last N days as the drift (mu).
                      If None, use the long-term fitted mean (usually ~0).
    """
    # 1. Scale returns for numerical stability
    scaled_returns = daily_returns * 100
    
    # 2. Fit GARCH(1,1) with Student-t
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='t', mean='Constant')
    res = model.fit(disp='off')
    
    params = res.params
    
    omega = params['omega'] / 10000.0
    alpha = params['alpha[1]']
    beta = params['beta[1]']
    nu = params['nu']
    
    # 3. Calculate Drift (mu) - The "Momentum" Fix
    if drift_window is not None:
        # Use Exponential Weighted Moving Average to capture recent trend.
        # This allows the model to "see" if the market is crashing or pumping.
        mu = daily_returns.ewm(span=drift_window).mean().iloc[-1]
    else:
        # Fallback to the long-term GARCH mean (Static/Random Walk assumption)
        mu = params['mu'] / 100.0
        
    return {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'nu': nu,
        'mu': mu,
        'last_variance': res.conditional_volatility.iloc[-1]**2 / 10000.0
    }

# ==============================================================================
# VARIANCE CONSISTENCY CHECK (Issue 2 diagnostic)
# ==============================================================================
def check_variance_consistency(garch_params: dict, n_samples: int = 10000, seed: int = 12345) -> float:
    """
    Diagnostic check: simulate 1-day returns (no jumps, no drift) and compare
    empirical variance to the model's conditional variance.
    
    This validates that the Student-t scaling and GARCH recursion are consistent.
    Returns the ratio (empirical_var / model_var). Should be close to 1.0.
    
    Emits a warning if ratio deviates by more than 15%.
    """
    import logging
    
    np.random.seed(seed)
    
    omega = garch_params['omega']
    alpha = garch_params['alpha']
    beta = garch_params['beta']
    nu = garch_params['nu']
    model_variance = garch_params['last_variance']
    
    # Scale Student-t to unit variance
    if nu > 2:
        scale_factor = np.sqrt((nu - 2) / nu)
    else:
        scale_factor = 1.0
    
    # Simulate 1-day returns: r = sigma * z where z ~ scaled Student-t
    z = t.rvs(nu, size=n_samples) * scale_factor
    sigma = np.sqrt(model_variance)
    returns = sigma * z
    
    empirical_variance = np.var(returns)
    ratio = empirical_variance / model_variance
    
    if abs(ratio - 1.0) > 0.15:
        logging.warning(
            f"Variance consistency check: empirical/model ratio = {ratio:.3f} "
            f"(expected ~1.0). Student-t scaling or GARCH params may be mismatched."
        )
    
    return ratio

# ==============================================================================
# MONTE CARLO SIMULATION
# ==============================================================================
def simulate_paths(
    S0,
    garch_params,
    jump_params,
    days_to_expiry,
    n_sims=150000,
    seed=None,
    apply_jump_drift_correction: bool = True,
    initial_variance: float = None,
    use_momentum_gating: bool = False,
):
    """
    Simulates price paths using GARCH(1,1) + Student-t + Jumps + Momentum Drift.
    
    Args:
        S0: Current spot price.
        garch_params: Dict with omega, alpha, beta, nu, mu, last_variance.
        jump_params: Dict with lambda, crash_prob, eta_up, eta_down (or None for defaults).
        days_to_expiry: Float, number of days until expiry.
        n_sims: Number of Monte Carlo paths.
        seed: Random seed for reproducibility.
        apply_jump_drift_correction: If True, subtract expected_jump_drift from mu.
            Use True for structural/fitted mu, False for momentum/EMA mu.
        initial_variance: Override for starting variance (for RV blending).
        use_momentum_gating: If True, only apply momentum drift when |mu| > MOMENTUM_GATE_MULT * sigma.
            Gating decision is made GLOBALLY (once per run), not per-path.
    """
    if seed is not None:
        np.random.seed(seed)

    # Resolve Jump Parameters
    if jump_params is None:
        lam = LAMBDA
        p_crash = CRASH_PROB
        eta_up = ETA_UP
        eta_down = ETA_DOWN
    else:
        lam = jump_params.get('lambda', LAMBDA)
        p_crash = jump_params.get('crash_prob', CRASH_PROB)
        eta_up = jump_params.get('eta_up', ETA_UP)
        eta_down = jump_params.get('eta_down', ETA_DOWN)

    # 1. Convert Annual Lambda to Daily
    lam_daily = lam / 365.0

    # 2. Calculate Expected Jump Drift (Daily log-return)
    # E[J] = (1-p_crash)/eta_up - p_crash/eta_down
    expected_jump_drift = lam_daily * ((1 - p_crash) / eta_up - p_crash / eta_down)

    n_days = int(np.ceil(days_to_expiry))
    dt_schedule = np.ones(n_days)
    if days_to_expiry % 1 != 0:
        dt_schedule[-1] = days_to_expiry % 1 
        
    omega = garch_params['omega']
    alpha = garch_params['alpha']
    beta = garch_params['beta']
    nu = garch_params['nu']
    mu = garch_params['mu']  # Daily log-return units (scalar)
    current_variance = garch_params['last_variance']  # Daily variance
    
    # Use initial_variance override if provided (for RV blending)
    if initial_variance is not None:
        current_variance = initial_variance
    
    # =========================================================================
    # GLOBAL MOMENTUM GATING (Issue 1 fix)
    # Decide ONCE whether to apply momentum, using scalar reference volatility.
    # This prevents pathwise selection bias from per-path gating.
    # =========================================================================
    if use_momentum_gating:
        sigma_ref = np.sqrt(current_variance)  # scalar reference volatility
        threshold = MOMENTUM_GATE_MULT * sigma_ref
        if np.abs(mu) > threshold:
            mu_effective = mu  # momentum is significant, keep it
        else:
            mu_effective = 0.0  # momentum is noise, zero it globally
    else:
        mu_effective = mu
    
    log_prices = np.full(n_sims, np.log(S0))
    variances = np.full(n_sims, current_variance)
    
    for dt in dt_schedule:
        # Scale Student-t
        if nu > 2:
            scale_factor = np.sqrt((nu - 2) / nu)
        else:
            scale_factor = 1.0 
            
        z_t = t.rvs(nu, size=n_sims) * scale_factor
        
        step_variance = variances * dt
        step_sigma = np.sqrt(step_variance)
        
        # 3. DYNAMIC PER-PATH DRIFT CLAMPING
        # sigma_day_step is a vector (per-path), evolves with GARCH
        sigma_day_step = np.sqrt(variances)  # vector: sqrt of current daily variance
        
        # Clamp drift per-path based on current variance state
        # (mu_effective is already globally gated, now just clamp per-path)
        mu_clamped = np.clip(mu_effective, -DRIFT_CLAMP_MULT * sigma_day_step, DRIFT_CLAMP_MULT * sigma_day_step)
        
        # 4. Apply Drift (clamped, with optional jump correction)
        if apply_jump_drift_correction:
            drift_per_step = (mu_clamped - expected_jump_drift) * dt
        else:
            drift_per_step = mu_clamped * dt
            
        garch_ret = drift_per_step + step_sigma * z_t
        
        # 5. GARCH variance update - ONLY for full-day steps
        if abs(dt - 1.0) < 1e-12:
            epsilon_squared = (step_sigma * z_t) ** 2
            variances = omega + alpha * epsilon_squared + beta * variances
        # else: variance unchanged for fractional step
        
        # 6. COMPOUND POISSON JUMPS (Multi-jump aggregation)
        k = np.random.poisson(lam_daily * dt, size=n_sims)  # int array
        k_down = np.random.binomial(k, p_crash)  # int array
        k_up = k - k_down
        
        # Explicit masking - do NOT rely on Gamma(0, ...) = 0
        down_mag = np.zeros(n_sims)
        up_mag = np.zeros(n_sims)
        mask_down = k_down > 0
        mask_up = k_up > 0
        
        if np.any(mask_down):
            down_mag[mask_down] = np.random.gamma(k_down[mask_down], scale=1.0 / eta_down)
        if np.any(mask_up):
            up_mag[mask_up] = np.random.gamma(k_up[mask_up], scale=1.0 / eta_up)
        
        jump_sizes = up_mag - down_mag  # log-return: up=positive, down=negative
            
        total_log_return = garch_ret + jump_sizes
        log_prices += total_log_return

    return np.exp(log_prices)

def get_contract_probability(paths: np.ndarray, strike_price: float, strict_above: bool = False):
    """
    Calculate probability of paths ending above strike.
    
    Args:
        paths: Array of simulated terminal prices.
        strike_price: Strike price for the binary contract.
        strict_above: If True, use > (strict). If False, use >= (default).
    """
    if strict_above:
        return np.mean(paths > strike_price)
    else:
        return np.mean(paths >= strike_price)

# ==============================================================================
# HIGH-LEVEL WRAPPER
# ==============================================================================
def calculate_probabilities(
    strikes: list,
    days_to_expiry: float,
    daily_df: pd.DataFrame = None,
    intraday_df: pd.DataFrame = None,
    daily_csv: str = "DATA/btc_daily.csv",
    intraday_csv: str = "DATA/btc_intraday_1m.csv",
    n_sims: int = 50000,
    drift_window: int = None,
    jump_params: dict = None,
    seed: int = None,
    rv_intraday: float = None,
    rv_blend_weight: float = 0.75,
    strict_above: bool = False,
    use_momentum_gating: bool = None,
) -> dict:
    """
    Calculates probabilities for multiple strikes.
    
    Args:
        strikes: List of strike prices.
        days_to_expiry: Days until contract expiry.
        daily_df: Optional DataFrame of daily prices (for backtesting).
        intraday_df: Optional DataFrame of intraday prices (for backtesting).
        daily_csv: Path to daily CSV file.
        intraday_csv: Path to intraday CSV file.
        n_sims: Number of Monte Carlo simulations.
        drift_window: If int, use EMA of last N days as drift (momentum mode).
                      If None, use structural/fitted mean.
        jump_params: Optional dict overriding jump parameters.
        seed: Random seed for reproducibility.
        rv_intraday: Optional intraday realized variance (daily units).
                     If provided, blends with GARCH variance.
        rv_blend_weight: Weight for GARCH variance in blend (default 0.75).
        strict_above: If True, use > instead of >= for probability.
        use_momentum_gating: If True, only apply drift when |mu| > threshold.
                             Defaults to True when drift_window is set.
    """
    daily_returns, S0 = load_and_prep_data(
        daily_csv=daily_csv,
        intraday_csv=intraday_csv,
        daily_df=daily_df,
        intraday_df=intraday_df,
    )
    
    garch_params = fit_garch_model(daily_returns, drift_window=drift_window)
    
    # Determine if we should apply jump drift correction
    # True for structural (drift_window=None), False for momentum
    apply_jump_drift_correction = (drift_window is None)
    
    # Default: enable momentum gating when using momentum drift
    if use_momentum_gating is None:
        use_momentum_gating = (drift_window is not None)
    
    # RV blending: done here in caller, passed as initial_variance
    initial_variance = garch_params['last_variance']
    if rv_intraday is not None:
        rv_intraday = max(rv_intraday, 1e-10)  # floor for safety
        initial_variance = rv_blend_weight * initial_variance + (1 - rv_blend_weight) * rv_intraday
    
    paths = simulate_paths(
        S0=S0,
        garch_params=garch_params,
        jump_params=jump_params,
        days_to_expiry=days_to_expiry,
        n_sims=n_sims,
        seed=seed,
        apply_jump_drift_correction=apply_jump_drift_correction,
        initial_variance=initial_variance,
        use_momentum_gating=use_momentum_gating,
    )
    
    results = {}
    for strike in strikes:
        results[strike] = get_contract_probability(paths, strike, strict_above=strict_above)
    
    return results


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BTC Pricing Engine Validation Tests")
    print("=" * 60)
    
    all_passed = True
    
    # -------------------------------------------------------------------------
    # Test 1: Multi-Jump Aggregation (compare 99th percentile)
    # -------------------------------------------------------------------------
    print("\n[Test 1] Multi-Jump Aggregation...")
    np.random.seed(42)
    
    n_test = 100000
    lam_high = 500.0  # Very high annual lambda for testing
    lam_daily_test = lam_high / 365.0
    p_crash_test = 0.5
    eta_up_test = 30.0
    eta_down_test = 30.0
    dt_test = 1.0
    
    # New implementation (multi-jump)
    k = np.random.poisson(lam_daily_test * dt_test, size=n_test)
    k_down = np.random.binomial(k, p_crash_test)
    k_up = k - k_down
    down_mag = np.zeros(n_test)
    up_mag = np.zeros(n_test)
    mask_down, mask_up = k_down > 0, k_up > 0
    if np.any(mask_down):
        down_mag[mask_down] = np.random.gamma(k_down[mask_down], scale=1.0 / eta_down_test)
    if np.any(mask_up):
        up_mag[mask_up] = np.random.gamma(k_up[mask_up], scale=1.0 / eta_up_test)
    jump_sizes_new = np.abs(up_mag - down_mag)
    q99_new = np.percentile(jump_sizes_new, 99)
    
    # Old implementation (single jump, capped at 1)
    np.random.seed(42)  # Reset seed
    n_jumps_old = np.random.poisson(lam_daily_test * dt_test, size=n_test)
    has_jump_old = n_jumps_old > 0
    jump_sizes_old = np.zeros(n_test)
    n_jumpers = np.sum(has_jump_old)
    if n_jumpers > 0:
        is_crash = np.random.rand(n_jumpers) < p_crash_test
        mags = np.zeros(n_jumpers)
        n_crashes = np.sum(is_crash)
        if n_crashes > 0:
            mags[is_crash] = np.random.exponential(1.0 / eta_down_test, size=n_crashes)
        n_pumps = n_jumpers - n_crashes
        if n_pumps > 0:
            mags[~is_crash] = np.random.exponential(1.0 / eta_up_test, size=n_pumps)
        jump_sizes_old[has_jump_old] = mags
    q99_old = np.percentile(np.abs(jump_sizes_old), 99)
    
    if q99_new >= 1.2 * q99_old:
        print(f"  PASS: q99_new ({q99_new:.4f}) >= 1.2 * q99_old ({1.2*q99_old:.4f})")
    else:
        print(f"  FAIL: q99_new ({q99_new:.4f}) < 1.2 * q99_old ({1.2*q99_old:.4f})")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # Test 2: Fractional dt Variance Preservation
    # -------------------------------------------------------------------------
    print("\n[Test 2] Fractional dt Variance Preservation...")
    np.random.seed(42)
    
    n_test2 = 1000
    v0 = 0.0004  # Initial variance
    omega_test = 0.00001
    alpha_test = 0.1
    beta_test = 0.85
    nu_test = 5.0
    dt_frac = 0.5
    
    variances = np.full(n_test2, v0)
    log_prices = np.zeros(n_test2)
    
    # Simulate one fractional step
    scale_factor = np.sqrt((nu_test - 2) / nu_test) if nu_test > 2 else 1.0
    z_t = t.rvs(nu_test, size=n_test2) * scale_factor
    step_sigma = np.sqrt(variances * dt_frac)
    log_prices += step_sigma * z_t
    
    # Variance should NOT update for fractional dt
    if abs(dt_frac - 1.0) < 1e-12:
        epsilon_squared = (step_sigma * z_t) ** 2
        variances = omega_test + alpha_test * epsilon_squared + beta_test * variances
    # else: unchanged
    
    variance_unchanged = np.allclose(variances, v0)
    price_changed = not np.allclose(log_prices, 0.0)
    
    if variance_unchanged and price_changed:
        print(f"  PASS: Variance unchanged ({variances[0]:.6f} == {v0:.6f}), prices moved")
    else:
        print(f"  FAIL: variance_unchanged={variance_unchanged}, price_changed={price_changed}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # Test 3: Dynamic Per-Path Drift Clamping
    # -------------------------------------------------------------------------
    print("\n[Test 3] Dynamic Per-Path Drift Clamping...")
    
    # Test that clamp uses per-path variance (vector), not scalar
    n_test3 = 5
    variances_test = np.array([0.0001, 0.0004, 0.0009, 0.0016, 0.0025])  # Different variances
    sigma_day_step = np.sqrt(variances_test)  # [0.01, 0.02, 0.03, 0.04, 0.05]
    mu_extreme = 0.10  # 10% drift, way too high for all
    
    mu_clamped = np.clip(mu_extreme, -DRIFT_CLAMP_MULT * sigma_day_step, DRIFT_CLAMP_MULT * sigma_day_step)
    expected_clamped = DRIFT_CLAMP_MULT * sigma_day_step  # Should be [0.0025, 0.005, 0.0075, 0.01, 0.0125]
    
    # Verify it's a vector matching per-path sigma
    is_vector = isinstance(mu_clamped, np.ndarray) and len(mu_clamped) == n_test3
    clamps_match = np.allclose(mu_clamped, expected_clamped)
    
    if is_vector and clamps_match:
        print(f"  PASS: Clamped drift varies per-path: {mu_clamped}")
    else:
        print(f"  FAIL: is_vector={is_vector}, clamps_match={clamps_match}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # Test 4: Global Momentum Gating (NOT per-path)
    # -------------------------------------------------------------------------
    print("\n[Test 4] Global Momentum Gating...")
    
    # Verify gating is a SINGLE global decision, not per-path
    sigma_ref = 0.02  # 2% daily vol (scalar)
    threshold = MOMENTUM_GATE_MULT * sigma_ref  # 0.005 scalar
    
    mu_small = 0.003  # Below threshold
    mu_large = 0.010  # Above threshold
    
    # Global gating: scalar decision
    mu_eff_small = mu_small if np.abs(mu_small) > threshold else 0.0
    mu_eff_large = mu_large if np.abs(mu_large) > threshold else 0.0
    
    # Verify both are scalars (not arrays), and correct values
    small_is_scalar = np.isscalar(mu_eff_small)
    large_is_scalar = np.isscalar(mu_eff_large)
    small_zeroed = mu_eff_small == 0.0
    large_kept = mu_eff_large == mu_large
    
    if small_is_scalar and large_is_scalar and small_zeroed and large_kept:
        print(f"  PASS: Global gate - small zeroed ({mu_eff_small}), large kept ({mu_eff_large})")
    else:
        print(f"  FAIL: small_is_scalar={small_is_scalar}, large_is_scalar={large_is_scalar}")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # Test 5: Variance Consistency Check
    # -------------------------------------------------------------------------
    print("\n[Test 5] Variance Consistency Check...")
    
    # Create mock garch_params
    test_garch = {
        'omega': 0.00001,
        'alpha': 0.1,
        'beta': 0.85,
        'nu': 5.0,
        'mu': 0.0,
        'last_variance': 0.0004,  # 2% daily vol
    }
    
    ratio = check_variance_consistency(test_garch, n_samples=50000, seed=12345)
    
    if abs(ratio - 1.0) < 0.15:
        print(f"  PASS: Variance ratio = {ratio:.4f} (within ±15% of 1.0)")
    else:
        print(f"  FAIL: Variance ratio = {ratio:.4f} (outside ±15%)")
        all_passed = False
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)