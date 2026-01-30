# src/models/ml_signal_scorer.py
"""
ML-based Signal Scorer for Pairs Trading

This module replaces the hand-tuned weighted scoring system with a machine learning
model that learns optimal entry signals from historical trade outcomes.

How It Works:
=============

1. FEATURE EXTRACTION
   For each potential entry signal (bar × pair), we extract features that capture:
   - Signal strength: z-score magnitude, distance from entry threshold
   - Signal freshness: velocity, acceleration, is_fresh_extreme vs is_stale
   - Profit potential: OU-model expected profit, expected holding time
   - Stability metrics: Kalman gain, beta uncertainty, recent beta drift
   - Market context: spread volatility, half-life, rolling performance

2. TRAINING DATA GENERATION
   We label historical trades as:
   - Target = 1 (positive): Trade was profitable (net return > 0)
   - Target = 0 (negative): Trade was unprofitable (net return <= 0)

   For regression mode, target = actual trade return (clipped to avoid outliers)

3. MODEL ARCHITECTURE
   We use LightGBM (gradient boosting) because:
   - Handles tabular data extremely well
   - Fast training and inference
   - Built-in feature importance for interpretability
   - Handles missing values gracefully
   - Less prone to overfitting than neural nets on small datasets

4. WALK-FORWARD TRAINING
   To avoid lookahead bias:
   - Train on windows 1..N-1
   - Predict on window N
   - Retrain monthly as new data arrives

5. PREDICTION OUTPUT
   The model outputs P(profitable | features) ∈ [0, 1]
   This probability is used as the signal score for:
   - Entry filtering (score > threshold)
   - Conviction sizing (higher score = larger position)

Feature Importance (typical):
=============================
- z_score_magnitude: 15-25% (how extreme the signal is)
- expected_profit: 15-20% (OU model's profit estimate)
- is_fresh_extreme: 10-15% (timing quality)
- spread_volatility: 10-15% (opportunity size)
- half_life_normalized: 8-12% (mean-reversion speed)
- kalman_gain: 5-10% (filter stability)
- beta_drift_recent: 5-10% (relationship stability)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_signal_scorer")

# Try to import LightGBM, fall back to sklearn if not available
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed. Falling back to sklearn GradientBoosting.")

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class MLScorerConfig:
    """Configuration for ML Signal Scorer."""

    # Model type: 'classifier' (predict win/loss) or 'regressor' (predict return)
    model_type: str = "classifier"

    # Minimum samples required to train
    min_training_samples: int = 50

    # LightGBM parameters
    lgb_params: Dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 15,  # Keep small to avoid overfitting
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    })

    # Feature engineering params
    velocity_lookback: int = 3
    drift_lookback_bars: int = 168  # 7 days at hourly
    rolling_perf_lookback: int = 240  # 10 days at hourly

    # Profit threshold for classification target
    profit_threshold: float = 0.0  # Trade is "good" if return > this

    # Return clipping for regression target
    return_clip_pct: float = 0.05  # Clip returns to [-5%, +5%]


@dataclass
class FeatureSet:
    """Container for extracted features."""
    features: pd.DataFrame  # (n_samples, n_features)
    feature_names: List[str]
    timestamps: pd.Index
    pairs: List[str]

    # Original shape info for reshaping predictions back
    original_shape: Tuple[int, int] = (0, 0)


class MLSignalScorer:
    """
    Machine Learning based signal scorer for pairs trading.

    Learns to predict trade profitability from historical outcomes,
    replacing hand-tuned scoring weights with data-driven predictions.
    """

    def __init__(self, config: Optional[MLScorerConfig] = None):
        self.config = config or MLScorerConfig()
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_stats: Dict = {}

    def extract_features(
        self,
        z_score: pd.DataFrame,
        spread_volatility: pd.DataFrame,
        expected_profit: pd.DataFrame,
        expected_hold_bars: pd.DataFrame,
        half_life_bars: Union[float, Dict[str, float], pd.Series],
        *,
        kalman_gain: Optional[pd.DataFrame] = None,
        beta: Optional[pd.DataFrame] = None,
        beta_history: Optional[pd.DataFrame] = None,
        entry_z: float = 2.0,
        max_entry_z: float = 4.0,
    ) -> FeatureSet:
        """
        Extract ML features for each bar × pair combination.

        Parameters
        ----------
        z_score : pd.DataFrame
            Z-scores (T × P)
        spread_volatility : pd.DataFrame
            Spread volatility (T × P)
        expected_profit : pd.DataFrame
            OU-model expected profit (T × P)
        expected_hold_bars : pd.DataFrame
            Expected holding time in bars (T × P)
        half_life_bars : float, dict, or Series
            Half-life per pair
        kalman_gain : pd.DataFrame, optional
            Kalman filter gain (T × P)
        beta : pd.DataFrame, optional
            Current beta estimates (T × P)
        beta_history : pd.DataFrame, optional
            Historical beta for drift calculation
        entry_z : float
            Entry z-score threshold
        max_entry_z : float
            Maximum entry z-score

        Returns
        -------
        FeatureSet with extracted features
        """
        pairs = list(z_score.columns)
        T, P = z_score.shape

        # Convert half_life to Series
        if isinstance(half_life_bars, (int, float)):
            hl_series = pd.Series(half_life_bars, index=pairs)
        elif isinstance(half_life_bars, dict):
            hl_series = pd.Series(half_life_bars).reindex(pairs).fillna(500)
        else:
            hl_series = half_life_bars.reindex(pairs).fillna(500)

        # Initialize feature storage
        feature_dict = {}

        # === 1. Z-SCORE FEATURES ===
        abs_z = z_score.abs()

        # Raw z magnitude
        feature_dict["z_score_abs"] = abs_z

        # Normalized z (relative to entry threshold)
        z_range = max_entry_z - entry_z
        feature_dict["z_normalized"] = (abs_z - entry_z) / z_range

        # Distance from optimal entry (middle of range)
        optimal_z = (entry_z + max_entry_z) / 2
        feature_dict["z_dist_from_optimal"] = (abs_z - optimal_z).abs() / z_range

        # === 2. VELOCITY/TIMING FEATURES ===
        lookback = self.config.velocity_lookback

        # Z-score velocity (rate of change)
        velocity = (z_score - z_score.shift(lookback)) / lookback
        feature_dict["z_velocity"] = velocity.abs()

        # Velocity direction relative to z (positive = expanding, negative = reverting)
        expanding = ((z_score > 0) & (velocity > 0)) | ((z_score < 0) & (velocity < 0))
        feature_dict["is_expanding"] = expanding.astype(float)

        # Is reverting (velocity opposite to z direction)
        reverting = ((z_score > 0) & (velocity < 0)) | ((z_score < 0) & (velocity > 0))
        feature_dict["is_reverting"] = reverting.astype(float)

        # Fresh extreme detection
        abs_velocity = abs_z - abs_z.shift(lookback)
        is_fresh = (abs_velocity >= -0.1 * lookback)
        feature_dict["is_fresh_extreme"] = is_fresh.astype(float)

        # Stale signal detection
        past_abs_z = abs_z.shift(2 * lookback)
        is_stale = (past_abs_z > abs_z) & reverting
        feature_dict["is_stale_signal"] = is_stale.astype(float)

        # Acceleration (second derivative)
        velocity_prev = (z_score.shift(lookback) - z_score.shift(2 * lookback)) / lookback
        acceleration = (velocity - velocity_prev) / lookback
        feature_dict["z_acceleration"] = acceleration.abs()

        # === 3. PROFIT POTENTIAL FEATURES ===
        feature_dict["expected_profit"] = expected_profit.clip(-0.1, 0.1)
        feature_dict["expected_hold_bars"] = expected_hold_bars.clip(0, 2000)

        # Profit per bar (efficiency)
        profit_per_bar = expected_profit / expected_hold_bars.replace(0, np.nan)
        feature_dict["profit_per_bar"] = profit_per_bar.clip(-0.001, 0.001)

        # === 4. VOLATILITY FEATURES ===
        feature_dict["spread_volatility"] = spread_volatility

        # Rolling volatility trend
        vol_ma = spread_volatility.rolling(24).mean()
        feature_dict["vol_trend"] = (spread_volatility / vol_ma.replace(0, np.nan)).clip(0.5, 2.0)

        # === 5. HALF-LIFE FEATURES ===
        # Broadcast half-life to all rows
        hl_df = pd.DataFrame(
            np.tile(hl_series.values, (T, 1)),
            index=z_score.index,
            columns=pairs
        )
        feature_dict["half_life_bars"] = hl_df

        # Normalized half-life (optimal around 240 hours)
        optimal_hl = 240
        feature_dict["half_life_normalized"] = (hl_df / optimal_hl).clip(0.1, 5.0)

        # Mean reversion rate (lambda = ln(2) / half_life)
        feature_dict["mean_reversion_rate"] = np.log(2) / hl_df.clip(60, 2000)

        # === 6. STABILITY FEATURES ===
        if kalman_gain is not None:
            kalman_gain_clipped = kalman_gain.clip(0, 1)
            # Check if ML_NORMALIZE_FEATURES is enabled for rank normalization
            from src.backtest import config_backtest as cfg
            use_rank_normalization = getattr(cfg, "ML_NORMALIZE_FEATURES", False)

            if use_rank_normalization:
                # NOTE: kalman_gain distribution can shift significantly when ENABLE_ADAPTIVE_KALMAN
                # is True, causing ML model trained on one regime to fail on another.
                # Solution: Use rank normalization instead of raw values.
                rank_window = min(96, len(kalman_gain_clipped))
                if rank_window > 10:
                    kalman_gain_rank = kalman_gain_clipped.rolling(rank_window).apply(
                        lambda x: (x.iloc[-1:].values >= x.iloc[:-1].values).mean() if len(x) > 1 else 0.5,
                        raw=False
                    ).fillna(0.5)
                else:
                    kalman_gain_rank = kalman_gain_clipped
                feature_dict["kalman_gain"] = kalman_gain_rank
            else:
                # Use raw kalman_gain (original behavior before ML_NORMALIZE_FEATURES)
                feature_dict["kalman_gain"] = kalman_gain_clipped
        else:
            feature_dict["kalman_gain"] = pd.DataFrame(0.5, index=z_score.index, columns=pairs)

        if beta is not None:
            feature_dict["beta_abs"] = beta.abs().clip(0, 5)

            # Beta drift (recent change in beta)
            if beta_history is not None and len(beta_history) > self.config.drift_lookback_bars:
                drift_lookback = self.config.drift_lookback_bars
                beta_old = beta_history.shift(drift_lookback)
                beta_drift = (beta - beta_old).abs()
                feature_dict["beta_drift_recent"] = beta_drift.clip(0, 1)
            else:
                # Use rolling beta change as proxy
                beta_drift = (beta - beta.shift(24)).abs()
                feature_dict["beta_drift_recent"] = beta_drift.clip(0, 1)
        else:
            feature_dict["beta_abs"] = pd.DataFrame(1.0, index=z_score.index, columns=pairs)
            feature_dict["beta_drift_recent"] = pd.DataFrame(0.0, index=z_score.index, columns=pairs)

        # === 7. ROLLING PERFORMANCE FEATURES ===
        # Recent z-score behavior (mean-reversion quality)
        z_crosses_zero = ((z_score > 0) & (z_score.shift(1) < 0)) | \
                         ((z_score < 0) & (z_score.shift(1) > 0))
        rolling_crosses = z_crosses_zero.rolling(self.config.rolling_perf_lookback).sum()
        feature_dict["rolling_zero_crosses"] = rolling_crosses.fillna(0)

        # Compile feature names
        self.feature_names = list(feature_dict.keys())

        # Stack all features into a single DataFrame
        # Shape: (T * P, n_features) for model input
        feature_frames = []
        for fname in self.feature_names:
            fdf = feature_dict[fname]
            # Stack: convert (T, P) to (T*P,) with multi-index
            stacked = fdf.stack()
            stacked.name = fname
            feature_frames.append(stacked)

        features_df = pd.concat(feature_frames, axis=1)
        features_df = features_df.reset_index()
        features_df.columns = ["timestamp", "pair"] + self.feature_names

        # Create FeatureSet
        return FeatureSet(
            features=features_df,
            feature_names=self.feature_names,
            timestamps=z_score.index,
            pairs=pairs,
            original_shape=(T, P),
        )

    def prepare_training_data(
        self,
        feature_set: FeatureSet,
        trade_returns: pd.DataFrame,
        entry_mask: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare training data from historical trades.

        Parameters
        ----------
        feature_set : FeatureSet
            Extracted features
        trade_returns : pd.DataFrame
            Actual trade returns (T × P), non-zero at exit bars
        entry_mask : pd.DataFrame
            Boolean mask of entry signals (T × P)

        Returns
        -------
        X : np.ndarray
            Feature matrix for training
        y : np.ndarray
            Target labels (1=profitable, 0=unprofitable) or returns
        info_df : pd.DataFrame
            Additional info (timestamp, pair, return) for analysis
        """
        features_df = feature_set.features.copy()

        # Stack trade returns and entry mask to match features format
        # Use reset_index to avoid multi-index join issues
        returns_stacked = trade_returns.stack().reset_index()
        returns_stacked.columns = ["timestamp", "pair", "trade_return"]

        entry_stacked = entry_mask.stack().reset_index()
        entry_stacked.columns = ["timestamp", "pair", "is_entry"]

        # Merge with features using explicit column merge
        features_df = features_df.merge(
            returns_stacked,
            on=["timestamp", "pair"],
            how="left"
        )
        features_df = features_df.merge(
            entry_stacked,
            on=["timestamp", "pair"],
            how="left"
        )

        # Filter to only entry points that resulted in trades
        # A trade is identified by having a non-zero return somewhere after entry
        # For simplicity, we use entries where expected_profit was computed
        train_mask = features_df["is_entry"].fillna(False).astype(bool)
        train_df = features_df[train_mask].copy()

        if len(train_df) < self.config.min_training_samples:
            logger.warning(
                f"Insufficient training samples: {len(train_df)} < {self.config.min_training_samples}"
            )
            return np.array([]), np.array([]), pd.DataFrame()

        # Prepare X (features only)
        X = train_df[self.feature_names].values

        # Prepare y (target)
        if self.config.model_type == "classifier":
            # Binary: was trade profitable?
            y = (train_df["trade_return"] > self.config.profit_threshold).astype(int).values
        else:
            # Regression: predict actual return (clipped)
            clip = self.config.return_clip_pct
            y = train_df["trade_return"].clip(-clip, clip).values

        # Info for analysis
        info_df = train_df[["timestamp", "pair", "trade_return"]].copy()

        # Handle NaN in features
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        y = np.nan_to_num(y, nan=0.0)

        return X, y, info_df

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Train the ML model on historical trade data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target labels
        validation_split : float
            Fraction of data for validation

        Returns
        -------
        Dict with training statistics
        """
        if len(X) < self.config.min_training_samples:
            logger.error(f"Cannot train: only {len(X)} samples available")
            return {"error": "insufficient_samples"}

        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # Split for validation
        n_val = int(len(X) * validation_split)
        X_train, X_val = X_scaled[:-n_val], X_scaled[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        # Train model
        if HAS_LIGHTGBM:
            self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif HAS_SKLEARN:
            self._train_sklearn(X_train, y_train)
        else:
            raise ImportError("Neither LightGBM nor sklearn available")

        self.is_trained = True

        # Compute training stats
        train_pred = self.predict_proba(X_train, scale=False)
        val_pred = self.predict_proba(X_val, scale=False)

        if self.config.model_type == "classifier":
            train_acc = ((train_pred > 0.5) == y_train).mean()
            val_acc = ((val_pred > 0.5) == y_val).mean()

            # Class balance
            pos_rate_train = y_train.mean()
            pos_rate_val = y_val.mean()

            self.training_stats = {
                "n_samples": len(X),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "pos_rate_train": float(pos_rate_train),
                "pos_rate_val": float(pos_rate_val),
                "feature_names": self.feature_names,
            }
        else:
            # Regression metrics
            train_mse = ((train_pred - y_train) ** 2).mean()
            val_mse = ((val_pred - y_val) ** 2).mean()

            self.training_stats = {
                "n_samples": len(X),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "feature_names": self.feature_names,
            }

        # Feature importance
        if HAS_LIGHTGBM and self.model is not None:
            importance = self.model.feature_importance(importance_type="gain")
            importance_dict = dict(zip(self.feature_names, importance))
            # Normalize to percentages
            total = sum(importance)
            if total > 0:
                importance_dict = {k: v / total for k, v in importance_dict.items()}
            self.training_stats["feature_importance"] = importance_dict

        logger.info(f"ML Scorer trained: {self.training_stats}")
        return self.training_stats

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Train LightGBM model."""
        params = self.config.lgb_params.copy()

        if self.config.model_type == "regressor":
            params["objective"] = "regression"
            params["metric"] = "mse"

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

    def _train_sklearn(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train sklearn GradientBoosting model (fallback)."""
        if self.config.model_type == "classifier":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )
        self.model.fit(X_train, y_train)

    def predict_proba(
        self,
        X: np.ndarray,
        scale: bool = True,
    ) -> np.ndarray:
        """
        Predict probability scores for given features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        scale : bool
            Whether to apply scaling (use False if already scaled)

        Returns
        -------
        np.ndarray of probabilities (n_samples,)
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained, returning default scores")
            return np.full(len(X), 0.5)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        # Scale if needed
        if scale and self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict
        if HAS_LIGHTGBM:
            pred = self.model.predict(X)
            if self.config.model_type == "classifier":
                # LightGBM returns probabilities directly for binary
                return pred
            else:
                # Regressor: convert return prediction to 0-1 score
                # Clip to reasonable range and normalize
                pred_clipped = np.clip(pred, -0.05, 0.05)
                return (pred_clipped + 0.05) / 0.10  # Map [-5%, +5%] to [0, 1]
        else:
            # sklearn
            if self.config.model_type == "classifier":
                return self.model.predict_proba(X)[:, 1]
            else:
                pred = self.model.predict(X)
                pred_clipped = np.clip(pred, -0.05, 0.05)
                return (pred_clipped + 0.05) / 0.10

    def predict_scores(
        self,
        z_score: pd.DataFrame,
        spread_volatility: pd.DataFrame,
        expected_profit: pd.DataFrame,
        expected_hold_bars: pd.DataFrame,
        half_life_bars: Union[float, Dict[str, float], pd.Series],
        *,
        kalman_gain: Optional[pd.DataFrame] = None,
        beta: Optional[pd.DataFrame] = None,
        entry_z: float = 2.0,
        max_entry_z: float = 4.0,
    ) -> pd.DataFrame:
        """
        Predict signal scores for all bar × pair combinations.

        This is the main entry point for using the trained model
        during backtesting or live trading.

        Parameters
        ----------
        [Same as extract_features]

        Returns
        -------
        pd.DataFrame of scores (T × P), values in [0, 1]
        """
        # Extract features
        feature_set = self.extract_features(
            z_score=z_score,
            spread_volatility=spread_volatility,
            expected_profit=expected_profit,
            expected_hold_bars=expected_hold_bars,
            half_life_bars=half_life_bars,
            kalman_gain=kalman_gain,
            beta=beta,
            entry_z=entry_z,
            max_entry_z=max_entry_z,
        )

        # Get feature matrix
        features_df = feature_set.features
        X = features_df[self.feature_names].values

        # Predict
        scores = self.predict_proba(X)

        # Reshape back to (T, P) DataFrame
        T, P = feature_set.original_shape
        pairs = feature_set.pairs
        timestamps = feature_set.timestamps

        scores_2d = scores.reshape(T, P)
        scores_df = pd.DataFrame(scores_2d, index=timestamps, columns=pairs)

        return scores_df

    def save(self, path: Union[str, Path]):
        """Save trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "training_stats": self.training_stats,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"ML Scorer saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MLSignalScorer":
        """Load trained model from disk."""
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        scorer = cls(config=state["config"])
        scorer.model = state["model"]
        scorer.scaler = state["scaler"]
        scorer.feature_names = state["feature_names"]
        scorer.is_trained = state["is_trained"]
        scorer.training_stats = state["training_stats"]

        logger.info(f"ML Scorer loaded from {path}")
        return scorer


def create_training_labels_from_returns(
    returns_matrix: pd.DataFrame,
    entry_mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create per-entry trade return labels from returns matrix.

    The returns_matrix has non-zero values at exit bars.
    We need to map these back to entry bars for training labels.

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Trade returns (T × P), non-zero at exit bars
    entry_mask : pd.DataFrame
        Entry signals (T × P), True at entry bars

    Returns
    -------
    pd.DataFrame with trade returns aligned to entry bars
    """
    trade_labels = pd.DataFrame(
        np.nan, index=returns_matrix.index, columns=returns_matrix.columns
    )

    for pair in returns_matrix.columns:
        ret_col = returns_matrix[pair]
        entry_col = entry_mask[pair]

        # Find exit bars (non-zero returns)
        exit_bars = ret_col[ret_col != 0].index

        # Find entry bars
        entry_bars = entry_col[entry_col].index

        # Match entries to exits (entry before exit)
        for exit_bar in exit_bars:
            # Find most recent entry before this exit
            prior_entries = entry_bars[entry_bars < exit_bar]
            if len(prior_entries) > 0:
                entry_bar = prior_entries[-1]
                trade_labels.loc[entry_bar, pair] = ret_col[exit_bar]

    return trade_labels
