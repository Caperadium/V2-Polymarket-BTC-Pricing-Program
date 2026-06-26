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

# DEPRECATED (old additive blend, FIX 3 — unused). The re-enabled design uses a
# distribution-level drift shift with tilt strength XGB_TILT_LAMBDA in
# core.pricing.btc_pricing_engine, NOT this probability-space mix weight.
XGB_WEIGHT = 0.3  # Weight of XGBoost prediction in final blended probability
MIN_TRAIN_SAMPLES = 200  # Minimum samples required to train
# DEPRECATED alongside XGB_WEIGHT — horizon is now chosen per DTE bucket (C2-a)
# via core.pricing.btc_pricing_engine.XGB_DTE_BUCKETS / dte_bucket_horizon.
DEFAULT_FORECAST_HORIZONS = [7, 14, 30]  # Days

_MACRO_POSITIONAL_WARNED = False  # one-shot warn for legacy positional macro align


def to_daily_log_return_series(hourly_df: pd.DataFrame) -> pd.Series:
    """
    Daily log returns as a DATE-INDEXED Series (leak-free; caller pre-truncates).

    Unlike regime_detector.hourly_to_daily_returns (which drops the index and
    returns a bare ndarray), this keeps the DatetimeIndex so build_features can
    date-join macro safely (C3). Resamples to daily last-close, logs successive
    ratios. Returns an empty Series if no usable close/date columns.
    """
    if hourly_df is None or len(hourly_df) == 0:
        return pd.Series(dtype=float)
    col_map = {c.lower(): c for c in hourly_df.columns}
    close_col = col_map.get("close", hourly_df.columns[-1])
    date_col = col_map.get("date", col_map.get("timestamp"))
    df = hourly_df.copy()
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df = df.set_index(date_col)
        daily_close = df[close_col].resample("D").last().dropna()
    else:
        daily_close = df[close_col].iloc[::24]
    ret = np.log(daily_close / daily_close.shift(1)).dropna()
    ret.name = "btc_ret"
    return ret

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
    include_target: bool = True,
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
        btc_returns: Daily BTC log returns. PREFERRED: a date-indexed pd.Series
            (DatetimeIndex) so macro is joined by date (leak-safe, C3). A bare
            ndarray triggers the legacy POSITIONAL macro alignment (mis-registers
            on weekend gaps and can forward-leak) — used only for the CLI path.
        macro_df: DataFrame with macro features (gold, dxy, vix, spx); date-indexed.
        horizon_days: Forecast horizon in days.

    Returns:
        DataFrame of features with target column (future_return_direction).
    """
    global _MACRO_POSITIONAL_WARNED

    # Detect date-indexed Series (preferred) vs bare array (legacy).
    if isinstance(btc_returns, pd.Series) and isinstance(btc_returns.index, pd.DatetimeIndex):
        btc_dates = btc_returns.index
        ret_values = btc_returns.to_numpy()
    else:
        btc_dates = None
        ret_values = np.asarray(btc_returns)

    n = len(ret_values)
    if n < 60:
        return pd.DataFrame()

    ret_s = pd.Series(ret_values)  # positional Series for rolling features

    # Create base DataFrame (positional index; rows align to ret_values order).
    features = pd.DataFrame(index=range(n))

    # --- BTC-derived features ---
    # Realized volatility (multi-window)
    for window in [7, 14, 30, 90]:
        if n > window:
            features[f"vol_{window}d"] = ret_s.rolling(window).std().values

    # Momentum (past returns)
    for window in [1, 3, 5, 10, 21]:
        if n > window:
            features[f"ret_{window}d"] = ret_s.rolling(window).sum().values

    # Max drawdown over lookback
    if n > 30:
        features["drawdown_30d"] = ret_s.rolling(30).min().values

    # Volatility of volatility
    if n > 90:
        # Vol of 7-day vol over 90 days
        vol_7d = ret_s.rolling(7).std()
        features["vol_of_vol"] = vol_7d.rolling(90).std().values

    # --- Target: future direction ---
    # include_target=False at PREDICT time: the shift(-horizon_days) nulls the
    # most recent `horizon_days` rows, and the trailing dropna() would then drop
    # the latest row the predictor needs (.iloc[-1]). Skip the target entirely so
    # only leading rolling-NaN rows are dropped and the latest features survive.
    if include_target:
        future_ret = ret_s.rolling(horizon_days).sum().shift(-horizon_days)
        # NaN-preserving: the last `horizon_days` rows have no realized future
        # return, so their label is UNKNOWN. Keep NaN (not 0) so the trailing
        # dropna() excludes them from training — `(future_ret > 0).astype(int)`
        # would silently mislabel them as 0/"down".
        target = (future_ret > 0).astype(float)
        target[future_ret.isna()] = np.nan
        features["target"] = target.values

    # --- Macro features (if available) ---
    if macro_df is not None and not macro_df.empty:
        if btc_dates is not None and isinstance(macro_df.index, pd.DatetimeIndex):
            # C3: DATE JOIN with past-only forward-fill. After sorting + de-dup,
            # reindex onto the BTC dates with method='ffill' so the macro row glued
            # to a BTC day is the latest macro row with macro.date <= btc.date
            # (no positional misalignment, no forward leak). Resulting frame is
            # length n and row-aligned to `features` positionally.
            macro = macro_df.sort_index()
            macro = macro[~macro.index.duplicated(keep="last")]
            macro_aligned = macro.reindex(btc_dates, method="ffill")
        else:
            # Legacy POSITIONAL alignment (CLI / no dates). Mis-registers on
            # weekend gaps; warn once.
            if not _MACRO_POSITIONAL_WARNED:
                logger.warning(
                    "build_features: macro aligned POSITIONALLY (no DatetimeIndex on "
                    "btc_returns and/or macro_df). Pass a date-indexed btc_returns "
                    "Series for leak-safe date-join (C3)."
                )
                _MACRO_POSITIONAL_WARNED = True
            macro_aligned = macro_df.tail(n).reset_index(drop=True)
            macro_aligned = macro_aligned.reindex(range(n))

        def _col(name):
            return macro_aligned[name].to_numpy() if name in macro_aligned.columns else None

        def _roll_sum(name, w=30):
            if name in macro_aligned.columns:
                return macro_aligned[name].rolling(w).sum().to_numpy()
            return None

        def _corr(precomputed, macro_ret_col, w=30):
            """BTC-macro rolling correlation feature.

            Prefer a precomputed `btc_*_corr_30d` column if the macro frame
            carries one (the merge_with_btc path). Otherwise compute it here
            from the date-joined BTC returns (`ret_s`) and the macro return
            column — both reindexed to 0..n-1, so the rolling-`w` window is
            positionally aligned and past-only (leak-safe). The saved
            DATA/macro_daily.csv has no correlation columns, so this branch is
            what actually fires in the pipeline (the highest-value Köse
            gold/dxy features would otherwise be silently dropped).
            """
            if precomputed in macro_aligned.columns:
                return macro_aligned[precomputed].to_numpy()
            if macro_ret_col not in macro_aligned.columns:
                return None
            macro_ret = pd.Series(macro_aligned[macro_ret_col].to_numpy())
            return ret_s.rolling(w).corr(macro_ret).to_numpy()

        for feat_name, vals in (
            ("gold_ret_30d", _roll_sum("gold_ret")),
            ("gold_level", _col("gold")),
            ("dxy_ret_30d", _roll_sum("dxy_ret")),
            ("dxy_trend", _col("dxy_trend")),
            ("vix_level", _col("vix")),
            ("spx_ret_30d", _roll_sum("spx_ret")),
            ("btc_gold_corr", _corr("btc_gold_corr_30d", "gold_ret")),
            ("btc_dxy_corr", _corr("btc_dxy_corr_30d", "dxy_ret")),
        ):
            if vals is not None:
                features[feat_name] = vals

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
        self._horizon_days: int = 30  # set by train(); used as predict default

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
        y = df["target"].astype(int).values  # float 0.0/1.0 → int class labels

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
        self._horizon_days = int(horizon_days)

        return True

    def train_from_slice(
        self,
        daily_returns,
        macro_df: Optional[pd.DataFrame] = None,
        horizon_days: int = 30,
    ) -> bool:
        """
        Leak-free training entry for callers that have ALREADY truncated the data
        (backrunner per-snapshot, live). Accepts a date-indexed pd.Series (preferred,
        for macro date-join) or a bare ndarray of daily log returns, plus a macro
        DataFrame the caller has sliced to dates < as_of. Never reads a file.
        """
        return self.train(daily_returns, macro_df, horizon_days)

    def predict_direction_adjustment(
        self,
        S0: float = None,
        hours_to_expiry: float = None,
        btc_returns: Optional[np.ndarray] = None,
        macro_df: Optional[pd.DataFrame] = None,
        horizon_days: Optional[int] = None,
    ) -> float:
        """
        Predict P(up) over the horizon from the XGBoost classifier.

        Returns a strike-agnostic P(up) the engine converts into a single
        distribution drift shift (apply_xgb_drift_shift). NOT the old per-strike
        blend (FIX 3).

        Args:
            S0: Current BTC price. NO-OP — the model has no price feature
                (placeholder). Passed for API symmetry only.
            hours_to_expiry: Hours to expiry; used only if horizon_days is None.
            btc_returns: Daily BTC returns — PREFER a date-indexed pd.Series for
                leak-safe macro date-join (C3).
            macro_df: Macro features DataFrame (caller pre-truncated for leak safety).
            horizon_days: Explicit forecast horizon (the DTE-bucket horizon, C2-a).
                When given, overrides the hours_to_expiry-derived clamp so a <7d
                bucket is not silently floored to 7. Falls back to the trained
                horizon, then the hours-based clamp.

        Returns:
            Estimated P(up) (0-1), or 0.5 if untrained / insufficient data / failure.
        """
        if not self._trained or self._model is None:
            return 0.5  # Neutral

        if horizon_days is None:
            if hours_to_expiry:
                horizon_days = max(7, min(30, int(hours_to_expiry / 24)))
            else:
                horizon_days = self._horizon_days

        # Build current features from input data. include_target=False so the
        # latest row survives the dropna (target's shift(-h) would null the tail).
        if btc_returns is not None and len(btc_returns) > 0:
            df = build_features(btc_returns, macro_df, horizon_days, include_target=False)
            if df.empty or len(df) < 1:
                return 0.5
            missing = [c for c in self._feature_names if c not in df.columns]
            if missing:
                logger.debug("XGB predict: missing features %s; returning neutral", missing)
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

    # -- Serialization (B1) --------------------------------------------------
    # joblib is not a direct repo dependency; it ships transitively with
    # scikit-learn (already used). Import it lazily and fall back to stdlib
    # pickle so save/load never forces a new dependency.
    @staticmethod
    def _dump(obj, path):
        try:
            import joblib
            joblib.dump(obj, path)
        except ImportError:
            import pickle
            with open(path, "wb") as fh:
                pickle.dump(obj, fh)

    @staticmethod
    def _load_blob(path):
        try:
            import joblib
            return joblib.load(path)
        except ImportError:
            import pickle
            with open(path, "rb") as fh:
                return pickle.load(fh)

    def save(self, path) -> None:
        """Persist the trained model + metadata to `path` (DATA/xgb_models/...)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._dump(
            {
                "model": self._model,
                "feature_names": self._feature_names,
                "train_date": self._train_date,
                "accuracy": self._accuracy,
                "horizon_days": self._horizon_days,
                "params": {
                    "weight": self.weight,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "random_state": self.random_state,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "DirectionalXGB":
        """Load a model previously saved with `save`. Raises if the file is missing."""
        blob = cls._load_blob(Path(path))
        p = blob.get("params", {})
        inst = cls(
            weight=p.get("weight", XGB_WEIGHT),
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.05),
            random_state=p.get("random_state", 42),
        )
        inst._model = blob.get("model")
        inst._feature_names = blob.get("feature_names", [])
        inst._train_date = blob.get("train_date")
        inst._accuracy = blob.get("accuracy")
        inst._horizon_days = blob.get("horizon_days", 30)
        inst._trained = inst._model is not None
        return inst


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
