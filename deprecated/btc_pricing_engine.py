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
# MONTE CARLO SIMULATION
# ==============================================================================
def simulate_paths(S0, garch_params, jump_params, days_to_expiry, n_sims=150000, seed=None):
    """
    Simulates price paths using GARCH(1,1) + Student-t + Jumps + Momentum Drift.
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

    # 2. Calculate Expected Jump Drift (Daily)
    # We remove this from the total drift 'mu' so we don't double-count the jump trend.
    expected_jump_drift = lam_daily * ( (1 - p_crash) * (1/eta_up) + p_crash * (-1/eta_down) )

    n_days = int(np.ceil(days_to_expiry))
    dt_schedule = np.ones(n_days)
    if days_to_expiry % 1 != 0:
        dt_schedule[-1] = days_to_expiry % 1 
        
    omega = garch_params['omega']
    alpha = garch_params['alpha']
    beta = garch_params['beta']
    nu = garch_params['nu']
    mu = garch_params['mu']  # This now contains the Momentum (EMA) drift if configured
    current_variance = garch_params['last_variance']
    
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
        
        # 3. Apply Drift Correction
        # Total Drift = mu (Momentum)
        # Diffusion Drift = mu - Jump Drift
        garch_ret = (mu * dt) - (expected_jump_drift * dt) + step_sigma * z_t
        
        epsilon_squared = (step_sigma * z_t)**2
        next_variances = omega + alpha * epsilon_squared + beta * variances
        
        # Jumps
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
    drift_window: int = None,  # DEFAULT: 5-day Momentum Window (Was None)
    jump_params: dict = None,
    seed: int = None,
) -> dict:
    """
    Calculates probabilities for multiple strikes.
    
    Defaults:
        drift_window=5: This enables the Momentum Injection by default.
                        Set to None to revert to standard Random Walk.
    """
    daily_returns, S0 = load_and_prep_data(
        daily_csv=daily_csv,
        intraday_csv=intraday_csv,
        daily_df=daily_df,
        intraday_df=intraday_df,
    )
    
    garch_params = fit_garch_model(daily_returns, drift_window=drift_window)
    
    paths = simulate_paths(
        S0=S0,
        garch_params=garch_params,
        jump_params=jump_params,
        days_to_expiry=days_to_expiry,
        n_sims=n_sims,
        seed=seed,
    )
    
    results = {}
    for strike in strikes:
        results[strike] = get_contract_probability(paths, strike)
    
    return results