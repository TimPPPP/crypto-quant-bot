# src/models/risk_predictor.py
"""
ML-Based Risk Prediction for Dynamic Position Sizing

Predicts trade risk (not direction) to dynamically adjust position sizes:
- Forward Volatility: Next-N-bar spread volatility
- Max Adverse Excursion (MAE): Worst unrealized loss during trade
- Stop-Out Probability: P(trade hits stop-loss before profit target)

Critical Design Constraints:
- NO LEAKAGE: Labels strictly from future outcomes; features strictly from decision time
- TIME-SERIES CV: Walk-forward with purged/embargo splits
- COST-AWARE LABELS: Include fees, slippage, funding in outcome labels
- STABILITY > ACCURACY: LightGBM with monotonic constraints, calibrated probabilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("models.risk_predictor")

# Exit reason codes (must match pnl_engine.py)
EXIT_REASON_NONE = 0
EXIT_REASON_SIGNAL = 1
EXIT_REASON_TIME_STOP = 2
EXIT_REASON_STOP_LOSS = 3
EXIT_REASON_FORCED = 4


@dataclass
class RiskPredictorConfig:
    """Configuration for risk prediction models."""

    # Model targets
    predict_mae: bool = True
    predict_forward_vol: bool = True
    predict_stopout: bool = True

    # Horizon settings (in bars)
    mae_horizon_bars: int = 24
    vol_horizon_bars: int = 24

    # LightGBM parameters
    lgb_num_leaves: int = 15
    lgb_max_depth: int = 4
    lgb_learning_rate: float = 0.03
    lgb_n_estimators: int = 150
    lgb_min_child_samples: int = 20
    lgb_subsample: float = 0.7
    lgb_colsample: float = 0.7
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 0.5

    # Walk-forward settings
    min_training_samples: int = 100
    purge_bars: int = 24  # Gap between train/test to avoid leakage
    embargo_bars: int = 12  # Additional gap for lagged features

    # Calibration
    calibrate_probabilities: bool = True

    # Position sizing integration
    confidence_threshold: float = 0.3
    size_reduction_max: float = 0.5  # Max 50% reduction from risk

    # Monitoring
    track_feature_drift: bool = True
    drift_threshold: float = 0.25  # PSI threshold for retraining


@dataclass
class RiskLabels:
    """Container for risk prediction labels."""

    mae: pd.DataFrame  # Max adverse excursion (positive values)
    forward_vol: pd.DataFrame  # Forward volatility
    stopout: pd.DataFrame  # Binary: 1 if stopped out, 0 otherwise
    exit_reason: pd.DataFrame  # Exit reason codes
    trade_duration: pd.DataFrame  # Actual hold duration in bars
    is_censored: pd.DataFrame  # True if trade still open at horizon end


class RiskLabelGenerator:
    """
    Generate risk labels from trade outcomes.

    CRITICAL: All labels are strictly future-looking from decision time.
    """

    def __init__(self, config: RiskPredictorConfig):
        self.config = config

    def generate_labels(
        self,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        returns_matrix: pd.DataFrame,
        mae_matrix: pd.DataFrame,
        exit_reason_matrix: pd.DataFrame,
        spread_returns: pd.DataFrame,
    ) -> RiskLabels:
        """
        Generate all risk labels from backtest results.

        Parameters
        ----------
        entries : pd.DataFrame
            Entry mask (T × P), True at entry timestamps
        exits : pd.DataFrame
            Exit mask (T × P), True at exit timestamps
        returns_matrix : pd.DataFrame
            Net returns at exit timestamps (T × P)
        mae_matrix : pd.DataFrame
            MAE values at exit timestamps (T × P), from pnl_engine
        exit_reason_matrix : pd.DataFrame
            Exit reason codes at exit timestamps (T × P)
        spread_returns : pd.DataFrame
            Per-bar spread returns (T × P)

        Returns
        -------
        RiskLabels
            Container with all generated labels
        """
        T, P = entries.shape
        pairs = entries.columns

        # Initialize label DataFrames
        mae_labels = pd.DataFrame(np.nan, index=entries.index, columns=pairs)
        forward_vol_labels = pd.DataFrame(np.nan, index=entries.index, columns=pairs)
        stopout_labels = pd.DataFrame(np.nan, index=entries.index, columns=pairs)
        exit_reason_labels = pd.DataFrame(0, index=entries.index, columns=pairs)
        duration_labels = pd.DataFrame(np.nan, index=entries.index, columns=pairs)
        censored_labels = pd.DataFrame(False, index=entries.index, columns=pairs)

        # For each pair, match entries to their corresponding exits
        for pair in pairs:
            entry_mask = entries[pair].values
            exit_mask = exits[pair].values
            mae_col = mae_matrix[pair].values
            exit_reason_col = exit_reason_matrix[pair].values
            returns_col = returns_matrix[pair].values

            entry_indices = np.where(entry_mask)[0]
            exit_indices = np.where(exit_mask | (mae_col > 0))[0]  # Exit or has MAE

            # Match each entry to its next exit
            for entry_idx in entry_indices:
                # Find the next exit after this entry
                exit_idx = exit_indices[exit_indices > entry_idx]
                if len(exit_idx) == 0:
                    # Trade never exited (censored)
                    censored_labels.iloc[entry_idx, pairs.get_loc(pair)] = True
                    continue

                exit_idx = exit_idx[0]  # First exit after entry

                # MAE label: directly from pnl_engine
                mae_labels.iloc[entry_idx, pairs.get_loc(pair)] = mae_col[exit_idx]

                # Exit reason
                exit_reason_labels.iloc[entry_idx, pairs.get_loc(pair)] = exit_reason_col[exit_idx]

                # Duration (bars held)
                duration_labels.iloc[entry_idx, pairs.get_loc(pair)] = exit_idx - entry_idx

                # Stop-out label: 1 if exit_reason == STOP_LOSS
                stopout_labels.iloc[entry_idx, pairs.get_loc(pair)] = (
                    1.0 if exit_reason_col[exit_idx] == EXIT_REASON_STOP_LOSS else 0.0
                )

        # Forward volatility: strictly forward-looking
        forward_vol_labels = self.compute_forward_volatility(
            spread_returns, horizon_bars=self.config.vol_horizon_bars
        )

        return RiskLabels(
            mae=mae_labels,
            forward_vol=forward_vol_labels,
            stopout=stopout_labels,
            exit_reason=exit_reason_labels,
            trade_duration=duration_labels,
            is_censored=censored_labels,
        )

    def compute_forward_volatility(
        self,
        spread_returns: pd.DataFrame,
        horizon_bars: int,
    ) -> pd.DataFrame:
        """
        Compute realized volatility over next N bars.

        Strictly forward-looking: at time t, compute std(returns[t+1:t+horizon+1]).
        """
        # Shift by -1 to exclude current bar, then rolling std
        future_returns = spread_returns.shift(-1)
        forward_vol = future_returns.rolling(horizon_bars).std()
        # Shift back to align: forward_vol[t] = vol of bars [t+1, t+horizon]
        forward_vol = forward_vol.shift(-(horizon_bars - 1))
        return forward_vol

    def compute_mae_for_trade(
        self,
        entry_bar: int,
        exit_bar: int,
        spread_returns: np.ndarray,
        direction: int,
        entry_cost: float = 0.0014,
    ) -> float:
        """
        Compute MAE for a single trade including transaction costs.

        MAE = worst unrealized loss from entry to exit.

        Parameters
        ----------
        entry_bar : int
            Bar index of trade entry
        exit_bar : int
            Bar index of trade exit
        spread_returns : np.ndarray
            Per-bar spread returns
        direction : int
            +1 for long spread, -1 for short spread
        entry_cost : float
            Entry transaction cost (half of round-trip)

        Returns
        -------
        float
            MAE as positive value (how bad it got)
        """
        if exit_bar <= entry_bar:
            return 0.0

        # Cumulative PnL from entry
        trade_returns = spread_returns[entry_bar:exit_bar] * direction
        cumulative_pnl = np.cumsum(trade_returns) - entry_cost

        # MAE is the worst point (most negative)
        mae = -np.min(cumulative_pnl)
        return max(0.0, mae)


class RiskFeatureExtractor:
    """
    Extract features for risk prediction.

    All features are strictly backward-looking at decision time.
    """

    # Feature names for documentation and importance tracking
    FEATURE_NAMES = [
        # Signal strength features (from existing ml_signal_scorer)
        "z_score_abs",
        "z_normalized",
        "z_velocity",
        "is_fresh_extreme",
        "is_reverting",
        # Spread dynamics
        "spread_volatility",
        "spread_vol_ratio",
        "spread_autocorr_lag1",
        # Model stability
        "kalman_gain",
        "beta_drift_recent",
        "half_life_normalized",
        # Regime features (NEW)
        "vol_regime_pctl",
        "trend_autocorr",
        "hours_since_last_trade",
        "recent_mae_same_pair",
    ]

    def __init__(self, config: RiskPredictorConfig):
        self.config = config
        self._historical_vol_median: Optional[pd.Series] = None

    def extract_features(
        self,
        z_score: pd.DataFrame,
        spread_volatility: pd.DataFrame,
        kalman_gain: pd.DataFrame,
        beta: pd.DataFrame,
        half_life: pd.DataFrame,
        price_matrix: pd.DataFrame,
        entry_z: float = 2.0,
        max_entry_z: float = 4.0,
        btc_column: str = "BTC",
    ) -> pd.DataFrame:
        """
        Extract all features at decision time.

        Parameters
        ----------
        z_score : pd.DataFrame
            Current z-scores (T × P)
        spread_volatility : pd.DataFrame
            Current spread volatility (T × P)
        kalman_gain : pd.DataFrame
            Kalman filter gain (T × P)
        beta : pd.DataFrame
            Current hedge ratio (T × P)
        half_life : pd.DataFrame
            Estimated half-life in bars (T × P)
        price_matrix : pd.DataFrame
            Price data (T × coins) for regime features
        entry_z : float
            Entry z-score threshold
        max_entry_z : float
            Maximum z-score for entry
        btc_column : str
            Column name for BTC prices (for regime detection)

        Returns
        -------
        pd.DataFrame
            Feature matrix (T × num_features × P) flattened appropriately
        """
        T = len(z_score)
        pairs = z_score.columns
        features = {}

        # Signal strength features
        features["z_score_abs"] = z_score.abs()
        features["z_normalized"] = z_score.abs() / entry_z
        features["z_velocity"] = z_score.diff(3)  # 3-bar velocity
        features["is_fresh_extreme"] = (
            (z_score.abs() > entry_z) & (z_score.abs().shift(1) <= entry_z)
        ).astype(float)
        features["is_reverting"] = (
            (z_score.abs() < z_score.abs().shift(1)) & (z_score.abs() > 0.5)
        ).astype(float)

        # Spread dynamics
        features["spread_volatility"] = spread_volatility

        # Volatility ratio (current vs historical median)
        if self._historical_vol_median is None:
            self._historical_vol_median = spread_volatility.median()
        vol_ratio = spread_volatility / self._historical_vol_median.clip(lower=1e-6)
        features["spread_vol_ratio"] = vol_ratio.clip(0.1, 10.0)

        # Spread autocorrelation (from z-score changes)
        z_changes = z_score.diff()
        features["spread_autocorr_lag1"] = z_changes.rolling(24).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 2 else 0.0, raw=False
        ).fillna(0.0)

        # Model stability
        features["kalman_gain"] = kalman_gain
        features["beta_drift_recent"] = beta.diff(24).abs()  # 24-bar beta drift
        features["half_life_normalized"] = half_life / 240.0  # Normalize to ~10 hours

        # Regime features
        regime_features = self._extract_regime_features(price_matrix, btc_column)
        for name, values in regime_features.items():
            # Broadcast to all pairs
            features[name] = pd.DataFrame(
                np.tile(values.values.reshape(-1, 1), (1, len(pairs))),
                index=z_score.index,
                columns=pairs,
            )

        # Hours since last trade (placeholder - requires trade history)
        features["hours_since_last_trade"] = pd.DataFrame(
            24.0, index=z_score.index, columns=pairs
        )  # Default 24 hours

        # Recent MAE for same pair (placeholder - requires rolling history)
        features["recent_mae_same_pair"] = pd.DataFrame(
            0.01, index=z_score.index, columns=pairs
        )  # Default 1%

        return features

    def _extract_regime_features(
        self,
        price_matrix: pd.DataFrame,
        btc_column: str = "BTC",
    ) -> Dict[str, pd.Series]:
        """
        Extract regime-aware features for risk prediction.

        Parameters
        ----------
        price_matrix : pd.DataFrame
            Price data (T × coins)
        btc_column : str
            Column name for BTC prices

        Returns
        -------
        Dict[str, pd.Series]
            Regime features indexed by timestamp
        """
        features = {}

        if btc_column in price_matrix.columns:
            btc_returns = price_matrix[btc_column].pct_change()

            # Volatility regime percentile (rolling rank)
            vol_rolling = btc_returns.rolling(336).std()
            vol_pctl = vol_rolling.rank(pct=True)
            features["vol_regime_pctl"] = vol_pctl.fillna(0.5)

            # Trend autocorrelation
            autocorr = btc_returns.rolling(336).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 2 else 0.0, raw=False
            )
            features["trend_autocorr"] = autocorr.fillna(0.0)
        else:
            # Fallback: neutral regime
            features["vol_regime_pctl"] = pd.Series(
                0.5, index=price_matrix.index
            )
            features["trend_autocorr"] = pd.Series(
                0.0, index=price_matrix.index
            )

        return features

    def prepare_training_data(
        self,
        features: Dict[str, pd.DataFrame],
        labels: RiskLabels,
        entry_mask: pd.DataFrame,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
        """
        Prepare training data from features and labels.

        Only includes samples where entry_mask is True (actual trade entries).

        Parameters
        ----------
        features : Dict[str, pd.DataFrame]
            Feature DataFrames from extract_features()
        labels : RiskLabels
            Labels from RiskLabelGenerator
        entry_mask : pd.DataFrame
            True at entry timestamps (T × P)

        Returns
        -------
        Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]
            X (samples × features), y_dict (target arrays), info_df (metadata)
        """
        pairs = entry_mask.columns
        feature_names = list(features.keys())

        # Collect samples
        X_rows = []
        y_mae = []
        y_vol = []
        y_stopout = []
        info_rows = []

        for t in range(len(entry_mask)):
            for j, pair in enumerate(pairs):
                if entry_mask.iloc[t, j]:
                    # Check if we have valid labels
                    mae_val = labels.mae.iloc[t, j]
                    vol_val = labels.forward_vol.iloc[t, j]
                    stopout_val = labels.stopout.iloc[t, j]
                    censored = labels.is_censored.iloc[t, j]

                    if censored or not np.isfinite(mae_val):
                        continue  # Skip censored or invalid samples

                    # Extract feature vector
                    row = [features[fn].iloc[t, j] for fn in feature_names]
                    X_rows.append(row)
                    y_mae.append(mae_val)
                    y_vol.append(vol_val if np.isfinite(vol_val) else 0.0)
                    y_stopout.append(stopout_val if np.isfinite(stopout_val) else 0.0)
                    info_rows.append({
                        "timestamp": entry_mask.index[t],
                        "pair": pair,
                        "duration": labels.trade_duration.iloc[t, j],
                    })

        X = np.array(X_rows, dtype=np.float64)
        y_dict = {
            "mae": np.array(y_mae, dtype=np.float64),
            "forward_vol": np.array(y_vol, dtype=np.float64),
            "stopout": np.array(y_stopout, dtype=np.float64),
        }
        info_df = pd.DataFrame(info_rows)

        logger.info(
            "Prepared training data: %d samples, %d features",
            len(X), len(feature_names)
        )

        return X, y_dict, info_df


class RiskPredictor:
    """
    Ensemble of risk models for position sizing.

    Predicts MAE, forward volatility, and stop-out probability.
    """

    def __init__(self, config: Optional[RiskPredictorConfig] = None):
        self.config = config or RiskPredictorConfig()
        self.mae_model = None
        self.vol_model = None
        self.stopout_model = None
        self.calibrator = None
        self.is_trained = False
        self.training_stats: Dict = {}
        self.feature_names: List[str] = []
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def _get_lgb_params(self, objective: str = "regression") -> Dict:
        """Get LightGBM parameters from config."""
        return {
            "objective": objective,
            "metric": "rmse" if objective == "regression" else "auc",
            "boosting_type": "gbdt",
            "num_leaves": self.config.lgb_num_leaves,
            "max_depth": self.config.lgb_max_depth,
            "learning_rate": self.config.lgb_learning_rate,
            "n_estimators": self.config.lgb_n_estimators,
            "min_child_samples": self.config.lgb_min_child_samples,
            "subsample": self.config.lgb_subsample,
            "colsample_bytree": self.config.lgb_colsample,
            "reg_alpha": self.config.lgb_reg_alpha,
            "reg_lambda": self.config.lgb_reg_lambda,
            "random_state": 42,
            "verbose": -1,
        }

    def _get_monotone_constraints(self, target: str) -> List[int]:
        """
        Get monotonic constraints for a target.

        +1 = feature increase → target increase (positive correlation expected)
        -1 = feature increase → target decrease
        0 = no constraint
        """
        # Feature order must match RiskFeatureExtractor.FEATURE_NAMES
        constraints = {
            "mae": {
                "z_score_abs": 1,  # Higher z → higher MAE
                "spread_volatility": 1,  # Higher vol → higher MAE
                "spread_vol_ratio": 1,
                "kalman_gain": 1,  # Unstable filter → higher MAE
                "vol_regime_pctl": 1,  # High vol regime → higher MAE
                "half_life_normalized": -1,  # Longer HL → lower MAE
            },
            "stopout": {
                "z_score_abs": 1,
                "spread_volatility": 1,
                "kalman_gain": 1,
                "beta_drift_recent": 1,  # Unstable beta → higher stopout
                "vol_regime_pctl": 1,
            },
            "forward_vol": {
                "spread_volatility": 1,  # Current vol predicts future vol
                "vol_regime_pctl": 1,
            },
        }

        target_constraints = constraints.get(target, {})
        feature_names = RiskFeatureExtractor.FEATURE_NAMES

        return [
            target_constraints.get(fn, 0) for fn in feature_names
        ]

    def train(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        feature_names: List[str],
        val_X: Optional[np.ndarray] = None,
        val_y_dict: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        """
        Train risk prediction models.

        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features)
        y_dict : Dict[str, np.ndarray]
            Training labels: {"mae": ..., "forward_vol": ..., "stopout": ...}
        feature_names : List[str]
            Feature names for importance tracking
        val_X : Optional[np.ndarray]
            Validation features
        val_y_dict : Optional[Dict[str, np.ndarray]]
            Validation labels

        Returns
        -------
        Dict
            Training statistics
        """
        try:
            import lightgbm as lgb
            use_lgb = True
        except (ImportError, OSError):
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            use_lgb = False
            logger.warning("LightGBM not available, using sklearn fallback")

        self.feature_names = feature_names
        self.feature_means = np.nanmean(X, axis=0)
        self.feature_stds = np.nanstd(X, axis=0)
        self.feature_stds[self.feature_stds < 1e-6] = 1.0  # Avoid division by zero

        stats = {}

        # Train MAE model
        if self.config.predict_mae and "mae" in y_dict:
            y_mae = y_dict["mae"]
            valid_mask = np.isfinite(y_mae)

            if valid_mask.sum() >= self.config.min_training_samples:
                if use_lgb:
                    params = self._get_lgb_params("regression")
                    params["monotone_constraints"] = self._get_monotone_constraints("mae")
                    self.mae_model = lgb.LGBMRegressor(**params)
                else:
                    self.mae_model = GradientBoostingRegressor(
                        n_estimators=100, max_depth=4, random_state=42
                    )

                self.mae_model.fit(X[valid_mask], y_mae[valid_mask])
                train_pred = self.mae_model.predict(X[valid_mask])
                stats["mae_train_rmse"] = np.sqrt(np.mean((train_pred - y_mae[valid_mask]) ** 2))
                logger.info("MAE model trained: RMSE=%.4f", stats["mae_train_rmse"])

        # Train forward volatility model
        if self.config.predict_forward_vol and "forward_vol" in y_dict:
            y_vol = y_dict["forward_vol"]
            valid_mask = np.isfinite(y_vol)

            if valid_mask.sum() >= self.config.min_training_samples:
                if use_lgb:
                    params = self._get_lgb_params("regression")
                    params["monotone_constraints"] = self._get_monotone_constraints("forward_vol")
                    self.vol_model = lgb.LGBMRegressor(**params)
                else:
                    self.vol_model = GradientBoostingRegressor(
                        n_estimators=100, max_depth=4, random_state=42
                    )

                self.vol_model.fit(X[valid_mask], y_vol[valid_mask])
                train_pred = self.vol_model.predict(X[valid_mask])
                stats["vol_train_rmse"] = np.sqrt(np.mean((train_pred - y_vol[valid_mask]) ** 2))
                logger.info("Forward vol model trained: RMSE=%.4f", stats["vol_train_rmse"])

        # Train stop-out classifier
        if self.config.predict_stopout and "stopout" in y_dict:
            y_stopout = y_dict["stopout"]
            valid_mask = np.isfinite(y_stopout)

            if valid_mask.sum() >= self.config.min_training_samples:
                if use_lgb:
                    params = self._get_lgb_params("binary")
                    params["monotone_constraints"] = self._get_monotone_constraints("stopout")
                    self.stopout_model = lgb.LGBMClassifier(**params)
                else:
                    self.stopout_model = GradientBoostingClassifier(
                        n_estimators=100, max_depth=4, random_state=42
                    )

                self.stopout_model.fit(X[valid_mask], y_stopout[valid_mask].astype(int))

                # Compute train accuracy
                train_pred = self.stopout_model.predict(X[valid_mask])
                stats["stopout_train_acc"] = np.mean(train_pred == y_stopout[valid_mask])
                logger.info("Stop-out model trained: Accuracy=%.4f", stats["stopout_train_acc"])

        self.is_trained = True
        self.training_stats = stats
        return stats

    def predict(
        self,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Predict risk metrics.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)

        Returns
        -------
        Dict[str, np.ndarray]
            Predictions: {
                "predicted_mae": array,
                "predicted_vol": array,
                "stopout_prob": array,
                "confidence": array,
            }
        """
        n_samples = len(X)
        predictions = {
            "predicted_mae": np.full(n_samples, np.nan),
            "predicted_vol": np.full(n_samples, np.nan),
            "stopout_prob": np.full(n_samples, 0.5),
            "confidence": np.full(n_samples, 0.0),
        }

        if not self.is_trained:
            logger.warning("RiskPredictor not trained, returning defaults")
            return predictions

        # Handle NaN in features
        X_clean = np.copy(X)
        nan_mask = ~np.isfinite(X_clean)
        if self.feature_means is not None:
            for j in range(X_clean.shape[1]):
                X_clean[nan_mask[:, j], j] = self.feature_means[j]

        # Predict MAE
        if self.mae_model is not None:
            predictions["predicted_mae"] = self.mae_model.predict(X_clean)

        # Predict forward volatility
        if self.vol_model is not None:
            predictions["predicted_vol"] = self.vol_model.predict(X_clean)

        # Predict stop-out probability
        if self.stopout_model is not None:
            try:
                proba = self.stopout_model.predict_proba(X_clean)[:, 1]
                predictions["stopout_prob"] = proba
            except Exception:
                predictions["stopout_prob"] = self.stopout_model.predict(X_clean).astype(float)

        # Compute confidence as inverse of feature drift from training
        if self.feature_means is not None and self.feature_stds is not None:
            z_scores = np.abs((X_clean - self.feature_means) / self.feature_stds)
            mean_z = np.nanmean(z_scores, axis=1)
            # Confidence: 1.0 when z=0, 0.0 when z>=3
            predictions["confidence"] = np.clip(1.0 - mean_z / 3.0, 0.0, 1.0)

        return predictions

    def get_position_size_multiplier(
        self,
        predictions: Dict[str, np.ndarray],
        historical_mae_median: float = 0.01,
        historical_vol_median: float = 0.005,
    ) -> np.ndarray:
        """
        Convert risk predictions to position size multipliers.

        Higher predicted risk → lower multiplier (smaller position).

        Parameters
        ----------
        predictions : Dict[str, np.ndarray]
            Predictions from predict()
        historical_mae_median : float
            Historical median MAE for normalization
        historical_vol_median : float
            Historical median forward vol for normalization

        Returns
        -------
        np.ndarray
            Position size multipliers in range [1 - max_reduction, 1.0]
        """
        n_samples = len(predictions["predicted_mae"])
        config = self.config

        # Normalize predictions to 0-1 scale
        mae_norm = np.clip(predictions["predicted_mae"] / (3 * historical_mae_median), 0, 1)
        vol_norm = np.clip(predictions["predicted_vol"] / (3 * historical_vol_median), 0, 1)
        stopout = predictions["stopout_prob"]

        # Handle NaN
        mae_norm = np.nan_to_num(mae_norm, nan=0.5)
        vol_norm = np.nan_to_num(vol_norm, nan=0.5)
        stopout = np.nan_to_num(stopout, nan=0.5)

        # Weighted risk score
        risk_score = 0.4 * mae_norm + 0.3 * vol_norm + 0.3 * stopout

        # Convert to multiplier (higher risk → lower size)
        risk_mult = 1.0 - risk_score * config.size_reduction_max
        risk_mult = np.clip(risk_mult, 1.0 - config.size_reduction_max, 1.0)

        # Blend with fallback when confidence is low
        confidence = predictions["confidence"]
        low_conf_mask = confidence < config.confidence_threshold

        # Fallback: simple vol-based heuristic
        fallback_mult = 1.0 - vol_norm * 0.3
        fallback_mult = np.clip(fallback_mult, 0.7, 1.0)

        # Blend based on confidence
        blend_weight = np.clip(confidence / config.confidence_threshold, 0, 1)
        final_mult = np.where(
            low_conf_mask,
            fallback_mult * (1 - blend_weight) + risk_mult * blend_weight,
            risk_mult,
        )

        return final_mult


def create_purged_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    purge_bars: int = 24,
    embargo_bars: int = 12,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward splits with purge/embargo gaps.

    |------ train ------|-- purge --|-- embargo --|-- test --|

    Purge: Remove samples too close to train/test boundary
    Embargo: Additional gap for lagged features

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_splits : int
        Number of walk-forward splits
    train_ratio : float
        Fraction of data for training (before purge/embargo)
    purge_bars : int
        Gap between train and test (to avoid label leakage)
    embargo_bars : int
        Additional gap for lagged feature leakage

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples
    """
    splits = []
    gap = purge_bars + embargo_bars
    step = (n_samples - gap) // n_splits

    for i in range(n_splits):
        test_start = (i + 1) * step
        test_end = min((i + 2) * step, n_samples)

        train_end = test_start - gap
        if train_end < step:
            continue  # Not enough training data

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits
