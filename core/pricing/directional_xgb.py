"""
directional_xgb.py

XGBoost directional classifier for BTC price movement prediction.
Provides a probability modifier that adjusts the base SVCJ+HMM distribution
with directional signals from macro and on-chain features.

Based on: Paskaleva & Vasenska (2025) — XGBoost 81% directional accuracy
          Kim et al. (2025) — asymmetric features for up vs down
          Oprea & Bâra (2026) — meta-learning architecture

Phase 2.3 of improvement plan. Loaded weight: 30% XGBoost + 70% SVCJ model,
per Shelton (2024) OOS evidence that individual predictors are weak.

Usage:
    from core.pricing.directional_xgb import DirectionalXGB
    xgb = DirectionalXGB()
    xgb.train(btc_daily_returns, macro_df)
    adjustment = xgb.predict_direction_adjustment(S0=95000, hours_to_expiry=720)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XGB_WEIGHT = 0.3  # Weight of XGBoost prediction in final blended probability
MIN_TRAIN_SAMPLES = 200  # Minimum samples required to train
DEFAULT_FORECAST_HORIZONS = [7, 14, 30]  # Days

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DirectionalResult:
    """Directional prediction output."""
    prob_up: float         # P(up) from XGBoost
    confidence: float      # Model confidence (0-1)
    horizon_days: int
    features_used: list
    trained: bool = True


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_features(
    btc_returns: np.ndarray,
    macro_df: Optional[pd.DataFrame] = None,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """
    Build feature matrix for XGBoost directional prediction.

    Features (ranked by synthesis evidence):
      1. Realized volatility (multi-window)
      2. BTC momentum (past N-day returns)
      3. Gold returns + rolling BTC-Gold correlation
      4. DXY level + trend
      5. VIX level
      6. SPX returns

    Args:
        btc_returns: Array of daily BTC log returns.
        macro_df: DataFrame with macro features (gold, dxy, vix, spx).
        horizon_days: Forecast horizon in days.

    Returns:
        DataFrame of features with target column (future_return_direction).
    """
    n = len(btc_returns)

    if n < 60:
        return pd.DataFrame()

    # Create base DataFrame
    features = pd.DataFrame(index=range(n))

    # --- BTC-derived features ---
    # Realized volatility (multi-window)
    for window in [7, 14, 30, 90]:
        if n > window:
            features[f"vol_{window}d"] = pd.Series(btc_returns).rolling(window).std().values

    # Momentum (past returns)
    for window in [1, 3, 5, 10, 21]:
        if n > window:
            features[f"ret_{window}d"] = pd.Series(btc_returns).rolling(window).sum().values

    # Max drawdown over lookback
    if n > 30:
        cumsum = pd.Series(btc_returns).rolling(30).sum().values
        features["drawdown_30d"] = pd.Series(btc_returns).rolling(30).min().values

    # Volatility of volatility
    if n > 90:
        # Vol of 7-day vol over 90 days
        vol_7d = pd.Series(btc_returns).rolling(7).std()
        features["vol_of_vol"] = vol_7d.rolling(90).std().values

    # --- Target: future direction ---
    future_ret = pd.Series(btc_returns).rolling(horizon_days).sum().shift(-horizon_days)
    features["target"] = (future_ret > 0).astype(int).values

    # --- Macro features (if available) ---
    if macro_df is not None and not macro_df.empty:
        # Align macro data to same length
        macro_recent = macro_df.tail(n)

        # Gold features
        if "gold_ret" in macro_recent.columns:
            features["gold_ret_30d"] = macro_recent["gold_ret"].rolling(30).sum().values[-n:]
        if "gold" in macro_recent.columns:
            features["gold_level"] = macro_recent["gold"].values[-n:] if len(macro_recent) >= n else np.nan

        # DXY features
        if "dxy_ret" in macro_recent.columns:
            features["dxy_ret_30d"] = macro_recent["dxy_ret"].rolling(30).sum().values[-n:]
        if "dxy_trend" in macro_recent.columns:
            features["dxy_trend"] = macro_recent["dxy_trend"].values[-n:]

        # VIX features
        if "vix" in macro_recent.columns:
            vix_vals = macro_recent["vix"].values[-n:] if len(macro_recent) >= n else np.nan
            features["vix_level"] = vix_vals

        # SPX features
        if "spx_ret" in macro_recent.columns:
            features["spx_ret_30d"] = macro_recent["spx_ret"].rolling(30).sum().values[-n:]

        # Correlation features
        if "btc_gold_corr_30d" in macro_recent.columns:
            features["btc_gold_corr"] = macro_recent["btc_gold_corr_30d"].values[-n:]
        if "btc_dxy_corr_30d" in macro_recent.columns:
            features["btc_dxy_corr"] = macro_recent["btc_dxy_corr_30d"].values[-n:]

    # Drop NaN from rolling windows
    features = features.dropna()

    return features


# ---------------------------------------------------------------------------
# Directional XGBoost Model
# ---------------------------------------------------------------------------

class DirectionalXGB:
    """
    XGBoost classifier for directional probability adjustment.

    Provides a lightweight probability modifier that blends with the
    SVCJ+HMM distribution. Weight: 30% per Shelton (2024) OOS evidence.

    Features are BTC-derived + macro (Gold, DXY, VIX, SPX).
    Target: P(price_up) over forecast horizon.
    """

    def __init__(
        self,
        weight: float = XGB_WEIGHT,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.weight = weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self._model = None
        self._feature_names: list = []
        self._trained = False
        self._train_date: Optional[datetime] = None
        self._accuracy: Optional[float] = None

    def train(
        self,
        btc_returns: np.ndarray,
        macro_df: Optional[pd.DataFrame] = None,
        horizon_days: int = 30,
    ) -> bool:
        """
        Train the XGBoost classifier on historical data.

        Args:
            btc_returns: Array of daily BTC log returns.
            macro_df: Optional macro feature DataFrame.
            horizon_days: Forecast horizon in days.

        Returns:
            True if training succeeded.
        """
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score
        except ImportError:
            logger.error("xgboost or sklearn not installed")
            return False

        # Build features
        df = build_features(btc_returns, macro_df, horizon_days)

        if len(df) < MIN_TRAIN_SAMPLES:
            logger.warning(f"Insufficient training data: {len(df)} < {MIN_TRAIN_SAMPLES}")
            return False

        # Separate features and target
        feature_cols = [c for c in df.columns if c not in ("target",)]
        self._feature_names = feature_cols

        X = df[feature_cols].values
        y = df["target"].values

        # Train/test split (time-series aware)
        split_idx = int(0.8 * len(X))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(np.unique(y_train)) < 2:
            logger.warning("Only one class in training data; skipping XGBoost")
            return False

        # Train model
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X_train, y_train)

        # Evaluate
        if len(X_test) > 0:
            y_pred = self._model.predict(X_test)
            self._accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"XGBoost trained: accuracy={self._accuracy:.3f}, n_features={len(feature_cols)}")
        else:
            self._accuracy = None

        self._trained = True
        self._train_date = datetime.now(timezone.utc)

        return True

    def predict_direction_adjustment(
        self,
        S0: float = None,
        hours_to_expiry: float = None,
        btc_returns: Optional[np.ndarray] = None,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Predict directional probability adjustment.

        Returns P(up) estimate from XGBoost, which is blended with the
        base SVCJ distribution at self.weight.

        Args:
            S0: Current BTC price (unused currently; placeholder for price features).
            hours_to_expiry: Hours to expiry (converted to days for horizon).
            btc_returns: Daily BTC returns array.
            macro_df: Macro features DataFrame.

        Returns:
            Estimated P(up) from XGBoost (0-1), or 0.5 if untrained.
        """
        if not self._trained or self._model is None:
            return 0.5  # Neutral

        horizon_days = max(7, min(30, int(hours_to_expiry / 24))) if hours_to_expiry else 30

        # Build current features from input data
        if btc_returns is not None and len(btc_returns) > 0:
            df = build_features(btc_returns, macro_df, horizon_days)
            if df.empty or len(df) < 1:
                return 0.5
            current_features = df[self._feature_names].iloc[-1:].values
        else:
            # Fallback: return neutral
            return 0.5

        # Predict
        try:
            prob_up = float(self._model.predict_proba(current_features)[0, 1])
            return prob_up
        except Exception as e:
            logger.debug(f"XGBoost prediction failed: {e}")
            return 0.5

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def accuracy(self) -> Optional[float]:
        return self._accuracy

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self._model is None or not self._feature_names:
            return {}
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="Train XGBoost directional classifier")
    parser.add_argument("--btc", default="DATA/btc_hourly.csv", help="BTC hourly data")
    parser.add_argument("--macro", default=None, help="Macro data CSV")
    parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon in days")
    args = parser.parse_args()

    # Load BTC daily returns
    from core.pricing.regime_detector import hourly_to_daily_returns
    btc_ret = hourly_to_daily_returns(args.btc)

    # Load macro data
    macro_df = None
    if args.macro:
        macro_df = pd.read_csv(args.macro, index_col=0, parse_dates=True)

    # Train
    xgb = DirectionalXGB()
    success = xgb.train(btc_ret, macro_df, args.horizon)

    if success:
        print(f"\n=== XGBoost Directional Model ===")
        print(f"Trained: {success}")
        print(f"Accuracy: {xgb.accuracy:.3f}" if xgb.accuracy else "Accuracy: N/A")
        print(f"Features: {len(xgb._feature_names)}")

        # Feature importance
        imp = xgb.get_feature_importance()
        if imp:
            print("\nTop 10 features:")
            for name, score in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {name}: {score:.4f}")

        # Make a sample prediction
        adj = xgb.predict_direction_adjustment(
            hours_to_expiry=720,
            btc_returns=btc_ret,
            macro_df=macro_df,
        )
        print(f"\nSample P(up) prediction (30-day): {adj:.4f}")
    else:
        print("Training failed — insufficient data or missing features")
