"""
regime_detector.py

3-state HMM regime detection for BTC daily returns. Identifies Bear, Sideways,
and Bull market regimes with weekly re-estimation.

Based on: Oprea & Bâra (2026), Malekinezhad & Rafati (2026), Pakstaite et al. (2025).

Uses hmmlearn.GaussianHMM for production stability (per plan-reviewer recommendation).
Homogeneous HMM preferred over NH-HMM for production per Pakstaite convergence evidence.

Usage:
    from core.pricing.regime_detector import RegimeDetector, RegimeLabels
    detector = RegimeDetector()
    weights, labels = detector.fit_predict(daily_returns)
    print(f"Current regime: {labels.dominant} ({weights})")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regime labeling based on annualized mean return thresholds
BEAR_THRESHOLD = -0.10   # Annualized return < -10% → Bear
BULL_THRESHOLD = 0.10    # Annualized return > +10% → Bull
# Between these: Sideways

# Default re-estimation frequency
DEFAULT_REESTIMATE_DAYS = 7  # Weekly

# Training window (daily observations)
TRAINING_WINDOW_DAYS = 730  # 2 years


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegimeLabels:
    """Labeled regime states with human-readable names."""
    state_order: List[int]         # [bear_state_idx, sideways_state_idx, bull_state_idx]
    state_names: List[str]         # ["bear", "sideways", "bull"]
    state_means: List[float]       # Mean daily return per state

    @property
    def bear_idx(self) -> int:
        return self.state_order[0]

    @property
    def sideways_idx(self) -> int:
        return self.state_order[1]

    @property
    def bull_idx(self) -> int:
        return self.state_order[2]

    def label_state(self, state: int) -> str:
        """Map HMM state integer to regime name."""
        if state == self.bear_idx:
            return "bear"
        elif state == self.bull_idx:
            return "bull"
        else:
            return "sideways"


@dataclass
class RegimeResult:
    """Complete regime detection output for a single date."""
    timestamp: datetime
    probabilities: Dict[str, float]   # {"bear": w, "sideways": w, "bull": w}
    dominant: str                     # "bear", "sideways", or "bull"
    transition_matrix: np.ndarray     # 3x3 daily transition matrix
    state_means: List[float]
    log_likelihood: float
    coverage_warning: bool = False

    @property
    def bear_weight(self) -> float:
        return self.probabilities.get("bear", 0.0)

    @property
    def sideways_weight(self) -> float:
        return self.probabilities.get("sideways", 1.0)

    @property
    def bull_weight(self) -> float:
        return self.probabilities.get("bull", 0.0)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "dominant": self.dominant,
            "bear_weight": self.bear_weight,
            "sideways_weight": self.sideways_weight,
            "bull_weight": self.bull_weight,
            "bear_mean": self.state_means[0] if len(self.state_means) > 0 else np.nan,
            "sideways_mean": self.state_means[1] if len(self.state_means) > 1 else np.nan,
            "bull_mean": self.state_means[2] if len(self.state_means) > 2 else np.nan,
            "log_likelihood": self.log_likelihood,
        }


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    3-state homogeneous HMM for BTC regime detection.

    States labeled by annualized mean return:
      - Bear:  lowest mean (typically negative)
      - Bull:  highest mean (typically positive)
      - Sideways: middle (near zero)

    Re-estimated weekly using 2-year rolling window.
    """

    def __init__(
        self,
        n_states: int = 3,
        training_window: int = TRAINING_WINDOW_DAYS,
        reestimate_days: int = DEFAULT_REESTIMATE_DAYS,
        bear_threshold: float = BEAR_THRESHOLD,
        bull_threshold: float = BULL_THRESHOLD,
        random_state: int = 42,
        covariance_type: str = "full",
    ):
        self.n_states = n_states
        self.training_window = training_window
        self.reestimate_days = reestimate_days
        self.bear_threshold = bear_threshold
        self.bull_threshold = bull_threshold
        self.random_state = random_state
        self.covariance_type = covariance_type

        # State
        self._model: Optional[hmm.GaussianHMM] = None
        self._labels: Optional[RegimeLabels] = None
        self._last_fit_date: Optional[datetime] = None
        self._last_weights: Dict[str, float] = {"bear": 0.0, "sideways": 1.0, "bull": 0.0}
        self._last_dominant: str = "sideways"
        self._last_transmat: np.ndarray = np.ones((3, 3)) / 3.0
        self._last_fit_obs: int = 0

    def _label_states(
        self,
        model: hmm.GaussianHMM,
        daily_returns: np.ndarray,
    ) -> RegimeLabels:
        """Label HMM states by annualized mean return."""
        # Get posterior state assignments
        _, hidden_states = model.decode(daily_returns.reshape(-1, 1))

        # Compute mean return per state
        state_means = []
        for s in range(model.n_components):
            mask = hidden_states == s
            if np.any(mask):
                state_means.append(float(np.mean(daily_returns[mask])))
            else:
                state_means.append(0.0)

        # Sort by mean → [bear_idx, sideways_idx, bull_idx]
        state_order = np.argsort(state_means).tolist()

        return RegimeLabels(
            state_order=state_order,
            state_names=["bear", "sideways", "bull"],
            state_means=[state_means[i] for i in state_order],
        )

    def _needs_refit(self, now: datetime) -> bool:
        """Check if model needs re-estimation based on frequency."""
        if self._model is None:
            return True
        if self._last_fit_date is None:
            return True
        delta = (now - self._last_fit_date).total_seconds() / 86400
        return delta >= self.reestimate_days

    def fit(
        self,
        daily_returns: np.ndarray,
        now: Optional[datetime] = None,
        force: bool = False,
    ) -> Optional[RegimeResult]:
        """
        Fit (or re-fit) the HMM on daily log returns.

        Args:
            daily_returns: Array of daily log returns, sorted chronologically.
            now: Current timestamp (for re-estimation gating).
            force: If True, refit regardless of reestimate_days.

        Returns:
            RegimeResult with current regime probabilities, or None if insufficient data.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if not force and not self._needs_refit(now):
            return RegimeResult(
                timestamp=now,
                probabilities=self._last_weights,
                dominant=self._last_dominant,
                transition_matrix=self._last_transmat,
                state_means=self._labels.state_means if self._labels else [0.0] * 3,
                log_likelihood=0.0,
            )

        n = len(daily_returns)

        # Use trailing window
        window = min(self.training_window, n)
        if window < 60:
            logger.warning(f"Insufficient data for HMM fit: {window} obs, need ≥60")
            return None

        train_data = daily_returns[-window:].reshape(-1, 1)

        try:
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=200,
                tol=1e-4,
                random_state=self.random_state,
                init_params="stmc",
                params="stmc",
            )
            model.fit(train_data)
        except Exception as e:
            logger.warning(f"HMM fit failed: {e}")
            if self._model is not None:
                # Use stale model
                logger.info("Using previously fitted HMM model (stale)")
                model = self._model
            else:
                return None

        self._model = model
        self._labels = self._label_states(model, train_data)
        self._last_fit_date = now
        self._last_fit_obs = window

        # Get current regime probabilities
        # Use the last observation's smoothed probability
        try:
            posterior = model.predict_proba(train_data)
            last_probs = posterior[-1]  # Shape: (n_components,)
        except Exception:
            # Fallback to stationary distribution
            last_probs = np.ones(self.n_states) / self.n_states

        # Map to labeled probabilities
        probs = {}
        for i, (name, idx) in enumerate(zip(
            ["bear", "sideways", "bull"],
            self._labels.state_order,
        )):
            probs[name] = float(last_probs[idx])

        # Ensure sum to 1
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        dominant = max(probs, key=probs.get)

        self._last_weights = probs
        self._last_dominant = dominant
        self._last_transmat = model.transmat_

        logger.info(
            f"HMM fit: {window} obs, dominant={dominant}, "
            f"weights: bear={probs['bear']:.3f}, sideways={probs['sideways']:.3f}, "
            f"bull={probs['bull']:.3f}"
        )

        return RegimeResult(
            timestamp=now,
            probabilities=probs,
            dominant=dominant,
            transition_matrix=model.transmat_,
            state_means=self._labels.state_means,
            log_likelihood=float(model.score(train_data)),
        )

    def predict_weights(
        self,
        n_days_ahead: int = 0,
    ) -> Dict[str, float]:
        """
        Predict regime weights n_days ahead using transition matrix.

        Args:
            n_days_ahead: Days to forecast forward.

        Returns:
            Dict with regime weights {"bear", "sideways", "bull"}.
        """
        if self._model is None or self._labels is None:
            return {"bear": 0.0, "sideways": 1.0, "bull": 0.0}

        # Current probability vector (in model state order)
        current = np.zeros(self.n_states)
        for name, idx in zip(
            ["bear", "sideways", "bull"],
            self._labels.state_order,
        ):
            current[idx] = self._last_weights.get(name, 0.0)

        # Step forward
        if n_days_ahead > 0:
            forward = current @ np.linalg.matrix_power(self._last_transmat, n_days_ahead)
        else:
            forward = current

        result = {}
        for name, idx in zip(
            ["bear", "sideways", "bull"],
            self._labels.state_order,
        ):
            result[name] = float(forward[idx])

        return result

    def fit_predict(
        self,
        daily_returns: np.ndarray,
        now: Optional[datetime] = None,
        force: bool = False,
    ) -> Tuple[Dict[str, float], str]:
        """
        Convenience: fit and return (weights, dominant_regime).

        Returns:
            (weights_dict, dominant_label)
        """
        result = self.fit(daily_returns, now=now, force=force)
        if result is None:
            return {"bear": 0.0, "sideways": 1.0, "bull": 0.0}, "sideways"
        return result.probabilities, result.dominant

    def get_params(self) -> dict:
        """Get fitted model parameters as dict."""
        if self._model is None:
            return {}

        return {
            "means": self._model.means_.flatten().tolist(),
            "covars": self._model.covars_.flatten().tolist(),
            "transmat": self._model.transmat_.tolist(),
            "startprob": self._model.startprob_.tolist(),
            "labels": self._labels.state_names if self._labels else [],
        }


# ---------------------------------------------------------------------------
# Utility: Load daily returns from hourly data
# ---------------------------------------------------------------------------

def hourly_to_daily_returns(
    hourly_path: str = "DATA/btc_hourly.csv",
    df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Convert hourly BTC data to daily log returns.

    Args:
        hourly_path: Path to hourly CSV (date, close columns).
        df: Optional pre-loaded DataFrame.

    Returns:
        Array of daily log returns.
    """
    if df is None:
        df = pd.read_csv(hourly_path)

    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get('close', df.columns[-1])

    # Parse date column
    date_col = col_map.get('date', col_map.get('timestamp'))
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], utc=True)

    # Resample to daily: last price of day
    if date_col:
        df = df.set_index(date_col)
        daily_close = df[close_col].resample('D').last().dropna()
    else:
        # Fallback: use every 24th observation
        daily_close = df[close_col].iloc[::24]

    daily_returns = np.log(daily_close / daily_close.shift(1)).dropna().values
    return daily_returns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Detect BTC market regimes using 3-state HMM")
    parser.add_argument("--input", default="DATA/btc_hourly.csv", help="Path to hourly BTC data")
    parser.add_argument("--output", default=None, help="Optional CSV output path for regime history")
    args = parser.parse_args()

    # Load data
    daily_returns = hourly_to_daily_returns(args.input)

    if len(daily_returns) < 60:
        print(f"Error: only {len(daily_returns)} daily observations, need ≥60")
        exit(1)

    # Fit detector
    detector = RegimeDetector()
    result = detector.fit(daily_returns)

    if result is None:
        print("HMM fit failed")
        exit(1)

    print(f"\n=== HMM Regime Detection ===")
    print(f"Observations:    {len(daily_returns)}")
    print(f"Dominant regime: {result.dominant}")
    print(f"Bear weight:     {result.bear_weight:.4f}")
    print(f"Sideways weight: {result.sideways_weight:.4f}")
    print(f"Bull weight:     {result.bull_weight:.4f}")
    print(f"\nState means (annualized):")
    for name, mean in zip(["Bear", "Sideways", "Bull"], result.state_means):
        ann_mean = mean * 365
        print(f"  {name}: {ann_mean:.4f} ({ann_mean*100:.2f}%)")
    print(f"\nTransition matrix:")
    names = ["Bear", "Sideways", "Bull"]
    print(f"       {' '.join(f'{n:>8}' for n in names)}")
    for i, name in enumerate(names):
        row = result.transition_matrix[i]
        print(f"  {name}: {' '.join(f'{v:8.4f}' for v in row)}")

    # Optionally output regime history
    if args.output:
        # Run fit over rolling windows to generate history
        history = []
        window = detector.training_window
        for t in range(window, len(daily_returns)):
            train = daily_returns[:t]
            r = detector.fit(train, force=True)
            if r:
                history.append(r.to_dict())

        if history:
            df_out = pd.DataFrame(history)
            df_out.to_csv(args.output, index=False)
            print(f"\nRegime history saved to {args.output} ({len(history)} rows)")
