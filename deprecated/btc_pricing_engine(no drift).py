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
    
    Supports dependency injection for backtesting (skip disk I/O by passing DataFrames).
    
    Args:
        daily_csv: Path to daily OHLCV CSV (used if daily_df is None).
        intraday_csv: Path to intraday OHLCV CSV (used if intraday_df is None).
        daily_df: Optional DataFrame with daily data (skip disk read if provided).
        intraday_df: Optional DataFrame with intraday data (skip disk read if provided).
        
    Returns:
        tuple: (daily_returns (pd.Series), current_price_S0 (float))
    """
    # 1. Load Daily Data for GARCH fitting
    # Use injected DataFrame or load from disk
    if daily_df is None:
        daily_df = pd.read_csv(daily_csv)
    else:
        daily_df = daily_df.copy()  # Don't mutate injected data
    
    # Normalize columns to title case (Close) or lowercase to ensure consistency
    col_map = {c.lower(): c for c in daily_df.columns}
    if 'close' not in col_map:
        raise ValueError("daily_btc.csv must contain a 'Close' or 'close' column.")
    close_col = col_map['close']
    
    # Calculate Log Returns: ln(S_t / S_{t-1})
    # We drop the first NaN value created by differencing.
    daily_returns = np.log(daily_df[close_col] / daily_df[close_col].shift(1)).dropna()
    
    # 2. Load Intraday Data for S0
    # Use injected DataFrame or load from disk
    if intraday_df is None:
        intraday_df = pd.read_csv(intraday_csv)
    else:
        intraday_df = intraday_df.copy()  # Don't mutate injected data
        
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
    Fits a GARCH(1,1) model with Student-t errors to the daily log returns.
    
    Args:
        daily_returns: pandas Series of daily log returns.
        drift_window: If int, calculate mean return (drift) from the last N rows.
                      If None, use full dataset mean.
                      
    Returns:
        dict: Parameters {omega, alpha, beta, nu, mu}
    """
    # 1. Scale returns by 100 for better numerical stability during fitting
    # (Common practice with GARCH models as returns are small numbers)
    scaled_returns = daily_returns * 100
    
    # 2. Define GARCH(1,1) model with Student-t innovations (dist='t')
    # mean='Constant' implies r_t = mu + epsilon_t
    # vol='Garch', p=1, q=1 implies GARCH(1,1)
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='t', mean='Constant')
    
    # 3. Fit the model
    # disp='off' suppresses console output
    res = model.fit(disp='off')
    
    # 4. Extract Parameters
    # Note: Parameters are for the scaled returns (return * 100).
    # We need to be careful with scaling when using them.
    # omega is variance scale, so it scales by 100^2 = 10000.
    # mu is return scale, so it scales by 100.
    # alpha, beta, nu are dimensionless.
    
    params = res.params
    
    omega = params['omega'] / 10000.0  # Rescale back to raw units
    alpha = params['alpha[1]']
    beta = params['beta[1]']
    nu = params['nu']             # Degrees of freedom for Student-t
    
    # 5. Calculate Drift (mu)
    if drift_window is not None:
        # Use simple moving average of the last N returns for drift
        mu = daily_returns.iloc[-drift_window:].mean()
    else:
        # Use the fitted mean from the GARCH model (rescaled)
        mu = params['mu'] / 100.0
        
    return {
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'nu': nu,
        'mu': mu,
        'last_variance': res.conditional_volatility.iloc[-1]**2 / 10000.0 # Rescaled
    }

# ==============================================================================
# MONTE CARLO SIMULATION
# ==============================================================================
def simulate_paths(S0, garch_params, jump_params, days_to_expiry, n_sims=150000, seed=None):
    """
    Simulates price paths using a Hybrid Volatility Model:
    GARCH(1,1) Variance + Student-t Shocks + Kou's Double Exponential Jumps.
    
    Args:
        S0: Current price.
        garch_params: Dict with {omega, alpha, beta, nu, mu, last_variance}.
        jump_params: Dict or None. If None, uses module constants.
        days_to_expiry: Float, number of days to simulate (e.g. 1.5).
        n_sims: Number of paths to simulate.
        seed: Random seed for reproducibility.
        
    Returns:
        np.array: Final prices for each path (shape: (n_sims,))
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

    # --- CORRECTION 1: Convert Annual Lambda to Daily ---
    lam_daily = lam / 365.0

    # --- CORRECTION 2: Calculate Expected Jump Drift (Daily) ---
    # E[J] = lambda_daily * ( (1-p)*E[Pump] + p*E[Crash] )
    # E[Pump] = 1/eta_up, E[Crash] = -1/eta_down
    expected_jump_drift = lam_daily * ( (1 - p_crash) * (1/eta_up) + p_crash * (-1/eta_down) )

    n_days = int(np.ceil(days_to_expiry))
    dt_schedule = np.ones(n_days)
    if days_to_expiry % 1 != 0:
        dt_schedule[-1] = days_to_expiry % 1 # Last step is fractional
        
    omega = garch_params['omega']
    alpha = garch_params['alpha']
    beta = garch_params['beta']
    nu = garch_params['nu']
    mu = garch_params['mu']
    current_variance = garch_params['last_variance']
    
    log_prices = np.full(n_sims, np.log(S0))
    variances = np.full(n_sims, current_variance)
    
    for dt in dt_schedule:
        # Scale Student-t to unit variance
        if nu > 2:
            scale_factor = np.sqrt((nu - 2) / nu)
        else:
            scale_factor = 1.0 
            
        z_t = t.rvs(nu, size=n_sims) * scale_factor
        
        step_variance = variances * dt
        step_sigma = np.sqrt(step_variance)
        
        # --- CORRECTION 3: Subtract Jump Drift to prevent Double Counting ---
        # We want the total drift to be 'mu', so we remove the drift that the jumps will add.
        garch_ret = (mu * dt) - (expected_jump_drift * dt) + step_sigma * z_t
        
        epsilon_squared = (step_sigma * z_t)**2
        next_variances = omega + alpha * epsilon_squared + beta * variances
        
        # Use Daily Lambda
        n_jumps = np.random.poisson(lam_daily * dt, size=n_sims)
        
        has_jump = n_jumps > 0
        jump_sizes = np.zeros(n_sims)
        
        if np.any(has_jump):
            n_jumpers = np.sum(has_jump)
            is_crash = np.random.rand(n_jumpers) < p_crash
            
            mags = np.zeros(n_jumpers)
            n_crashes = np.sum(is_crash)
            if n_crashes > 0:
                mags[is_crash] = np.random.exponential(1.0 / eta_down, size=n_crashes)
            
            n_pumps = n_jumpers - n_crashes
            if n_pumps > 0:
                mags[~is_crash] = np.random.exponential(1.0 / eta_up, size=n_pumps)
            
            signs = np.where(is_crash, -1.0, 1.0)
            jump_sizes[has_jump] = signs * mags
            
        total_log_return = garch_ret + jump_sizes
        log_prices += total_log_return
        variances = next_variances

    return np.exp(log_prices)

def get_contract_probability(paths: np.array, strike_price: float):
    """
    Calculates the probability of expiry price >= strike.
    
    Args:
        paths: Array of final simulated prices.
        strike_price: The contract strike price.
        
    Returns:
        float: Probability (0.0 to 1.0).
    """
    return np.mean(paths >= strike_price)


# ==============================================================================
# HIGH-LEVEL CONVENIENCE WRAPPER (for backtesting)
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
) -> dict:
    """
    High-level wrapper to calculate probabilities for multiple strikes.
    
    Combines data loading, GARCH fitting, simulation, and probability calculation.
    Supports dependency injection for backtesting (skip disk I/O by passing DataFrames).
    
    Args:
        strikes: List of strike prices to calculate probabilities for.
        days_to_expiry: Days until contract expiry.
        daily_df: Optional DataFrame with daily data (skip disk read if provided).
        intraday_df: Optional DataFrame with intraday data (skip disk read if provided).
        daily_csv: Path to daily OHLCV CSV (used if daily_df is None).
        intraday_csv: Path to intraday OHLCV CSV (used if intraday_df is None).
        n_sims: Number of Monte Carlo paths (default: 50000 for speed).
        drift_window: If int, use last N days for drift estimation.
        jump_params: Optional jump parameters dict.
        seed: Random seed for reproducibility.
        
    Returns:
        dict: {strike: probability} mapping.
    """
    # Load and prep data (with optional injection)
    daily_returns, S0 = load_and_prep_data(
        daily_csv=daily_csv,
        intraday_csv=intraday_csv,
        daily_df=daily_df,
        intraday_df=intraday_df,
    )
    
    # Fit GARCH model
    garch_params = fit_garch_model(daily_returns, drift_window=drift_window)
    
    # Simulate paths
    paths = simulate_paths(
        S0=S0,
        garch_params=garch_params,
        jump_params=jump_params,
        days_to_expiry=days_to_expiry,
        n_sims=n_sims,
        seed=seed,
    )
    
    # Calculate probability for each strike
    results = {}
    for strike in strikes:
        results[strike] = get_contract_probability(paths, strike)
    
    return results

