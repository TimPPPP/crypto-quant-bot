# src/models/risk_monitor.py
"""
Risk Model Monitoring and Drift Detection

Tracks model performance and detects when retraining is needed:
- Feature drift detection (PSI)
- Prediction accuracy monitoring
- Calibration quality tracking
- Hit rate by regime analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.risk_calibrator import compute_brier_score, compute_calibration_error

logger = logging.getLogger("models.risk_monitor")


@dataclass
class RiskMonitoringMetrics:
    """Metrics for risk model monitoring dashboard."""

    # Prediction accuracy
    mae_rmse_rolling: float = 0.0
    vol_rmse_rolling: float = 0.0
    stopout_brier_score: float = 0.0

    # Calibration quality
    stopout_calibration_error: float = 0.0

    # Feature drift
    feature_psi_scores: Dict[str, float] = field(default_factory=dict)
    max_feature_psi: float = 0.0
    top_drifting_features: List[str] = field(default_factory=list)

    # Hit rates by regime
    hit_rate_high_vol: float = 0.0
    hit_rate_low_vol: float = 0.0
    hit_rate_trending: float = 0.0
    hit_rate_mean_revert: float = 0.0

    # Model confidence
    confidence_p10: float = 0.0
    confidence_p50: float = 0.0
    confidence_p90: float = 0.0
    pct_using_fallback: float = 0.0

    # Staleness
    days_since_train: int = 0
    samples_since_train: int = 0


class RiskMonitor:
    """
    Monitor risk model performance in production.

    Tracks:
    - Feature drift using Population Stability Index (PSI)
    - Prediction accuracy vs actual outcomes
    - Calibration quality of probability predictions
    - Performance breakdown by market regime
    """

    def __init__(
        self,
        reference_features: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.25,
        n_bins: int = 10,
    ):
        """
        Initialize risk monitor.

        Parameters
        ----------
        reference_features : pd.DataFrame, optional
            Reference feature distribution from training
        drift_threshold : float
            PSI threshold above which to flag feature drift
        n_bins : int
            Number of bins for PSI calculation
        """
        self.drift_threshold = drift_threshold
        self.n_bins = n_bins

        # Store reference distribution
        self._reference_quantiles: Optional[Dict[str, np.ndarray]] = None
        self._reference_bin_counts: Optional[Dict[str, np.ndarray]] = None

        if reference_features is not None:
            self._compute_reference_distribution(reference_features)

        # History tracking
        self.prediction_history: List[Dict] = []
        self.outcome_history: List[Dict] = []
        self._samples_since_train = 0
        self._days_since_train = 0

    def _compute_reference_distribution(self, features: pd.DataFrame) -> None:
        """Compute reference quantiles and bin counts for PSI."""
        self._reference_quantiles = {}
        self._reference_bin_counts = {}

        for col in features.columns:
            values = features[col].dropna().values
            if len(values) < self.n_bins:
                continue

            # Compute quantile bin edges
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(values, quantiles)
            self._reference_quantiles[col] = bin_edges

            # Compute reference bin counts (should be uniform)
            counts, _ = np.histogram(values, bins=bin_edges)
            self._reference_bin_counts[col] = counts / len(values)

    def compute_feature_drift(
        self,
        current_features: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute Population Stability Index (PSI) for feature drift.

        PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))

        Interpretation:
        - PSI < 0.10: No significant drift
        - 0.10 <= PSI < 0.25: Moderate drift
        - PSI >= 0.25: Significant drift (consider retraining)

        Parameters
        ----------
        current_features : pd.DataFrame
            Current feature values

        Returns
        -------
        Dict[str, float]
            PSI score per feature
        """
        if self._reference_quantiles is None:
            logger.warning("No reference distribution, cannot compute drift")
            return {}

        psi_scores = {}

        for col, bin_edges in self._reference_quantiles.items():
            if col not in current_features.columns:
                continue

            values = current_features[col].dropna().values
            if len(values) < self.n_bins:
                psi_scores[col] = 0.0
                continue

            # Compute current bin counts
            counts, _ = np.histogram(values, bins=bin_edges)
            current_pct = counts / max(len(values), 1)

            # Get reference percentages
            ref_pct = self._reference_bin_counts.get(col)
            if ref_pct is None:
                continue

            # Compute PSI (with small epsilon to avoid log(0))
            eps = 1e-6
            current_pct = np.clip(current_pct, eps, 1 - eps)
            ref_pct = np.clip(ref_pct, eps, 1 - eps)

            psi = np.sum((current_pct - ref_pct) * np.log(current_pct / ref_pct))
            psi_scores[col] = float(psi)

        return psi_scores

    def evaluate_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        actual_mae: np.ndarray,
        actual_vol: np.ndarray,
        actual_stopout: np.ndarray,
        regime_labels: Optional[np.ndarray] = None,
    ) -> RiskMonitoringMetrics:
        """
        Evaluate prediction accuracy on recent data.

        Parameters
        ----------
        predictions : Dict[str, np.ndarray]
            Model predictions: {predicted_mae, predicted_vol, stopout_prob, confidence}
        actual_mae : np.ndarray
            Actual MAE outcomes
        actual_vol : np.ndarray
            Actual forward volatility
        actual_stopout : np.ndarray
            Actual stop-out events (0 or 1)
        regime_labels : np.ndarray, optional
            Regime labels for breakdown (e.g., "high_vol", "trending")

        Returns
        -------
        RiskMonitoringMetrics
            Comprehensive monitoring metrics
        """
        metrics = RiskMonitoringMetrics()

        # MAE prediction accuracy
        pred_mae = predictions.get("predicted_mae")
        if pred_mae is not None and len(actual_mae) > 0:
            valid = np.isfinite(pred_mae) & np.isfinite(actual_mae)
            if valid.sum() > 0:
                metrics.mae_rmse_rolling = float(np.sqrt(
                    np.mean((pred_mae[valid] - actual_mae[valid]) ** 2)
                ))

        # Forward vol prediction accuracy
        pred_vol = predictions.get("predicted_vol")
        if pred_vol is not None and len(actual_vol) > 0:
            valid = np.isfinite(pred_vol) & np.isfinite(actual_vol)
            if valid.sum() > 0:
                metrics.vol_rmse_rolling = float(np.sqrt(
                    np.mean((pred_vol[valid] - actual_vol[valid]) ** 2)
                ))

        # Stop-out calibration
        stopout_prob = predictions.get("stopout_prob")
        if stopout_prob is not None and len(actual_stopout) > 0:
            valid = np.isfinite(stopout_prob) & np.isfinite(actual_stopout)
            if valid.sum() > 0:
                metrics.stopout_brier_score = compute_brier_score(
                    stopout_prob[valid], actual_stopout[valid]
                )
                ece, _, _ = compute_calibration_error(
                    stopout_prob[valid], actual_stopout[valid]
                )
                metrics.stopout_calibration_error = ece

        # Confidence statistics
        confidence = predictions.get("confidence")
        if confidence is not None and len(confidence) > 0:
            valid_conf = confidence[np.isfinite(confidence)]
            if len(valid_conf) > 0:
                metrics.confidence_p10 = float(np.percentile(valid_conf, 10))
                metrics.confidence_p50 = float(np.percentile(valid_conf, 50))
                metrics.confidence_p90 = float(np.percentile(valid_conf, 90))
                # Assuming fallback threshold of 0.3
                metrics.pct_using_fallback = float(np.mean(valid_conf < 0.3) * 100)

        # Hit rates by regime
        if regime_labels is not None and stopout_prob is not None:
            # "Hit" = correctly predicted stop-out direction
            predicted_stop = stopout_prob > 0.5
            correct = predicted_stop == (actual_stopout > 0.5)

            for regime in ["high_vol", "low_vol", "trending", "mean_revert"]:
                mask = regime_labels == regime
                if mask.sum() > 0:
                    hit_rate = float(np.mean(correct[mask]))
                    setattr(metrics, f"hit_rate_{regime}", hit_rate)

        # Staleness tracking
        metrics.days_since_train = self._days_since_train
        metrics.samples_since_train = self._samples_since_train

        return metrics

    def should_retrain(
        self,
        current_features: Optional[pd.DataFrame] = None,
        current_metrics: Optional[RiskMonitoringMetrics] = None,
        baseline_mae_rmse: float = 0.01,
        max_days: int = 30,
    ) -> bool:
        """
        Determine if model should be retrained.

        Triggers retraining if:
        - Feature drift exceeds threshold
        - Prediction accuracy degraded significantly
        - Model is too old

        Parameters
        ----------
        current_features : pd.DataFrame, optional
            Current features for drift detection
        current_metrics : RiskMonitoringMetrics, optional
            Current performance metrics
        baseline_mae_rmse : float
            Baseline MAE RMSE for comparison
        max_days : int
            Maximum days before forced retraining

        Returns
        -------
        bool
            True if retraining is recommended
        """
        reasons = []

        # Check feature drift
        if current_features is not None:
            psi_scores = self.compute_feature_drift(current_features)
            max_psi = max(psi_scores.values()) if psi_scores else 0.0
            if max_psi > self.drift_threshold:
                reasons.append(f"Feature drift: max PSI={max_psi:.3f}")

        # Check prediction accuracy
        if current_metrics is not None:
            if current_metrics.mae_rmse_rolling > 1.5 * baseline_mae_rmse:
                reasons.append(
                    f"MAE RMSE degraded: {current_metrics.mae_rmse_rolling:.4f} "
                    f"vs baseline {baseline_mae_rmse:.4f}"
                )

            if current_metrics.stopout_calibration_error > 0.15:
                reasons.append(
                    f"Calibration error high: {current_metrics.stopout_calibration_error:.3f}"
                )

        # Check staleness
        if self._days_since_train > max_days:
            reasons.append(f"Model age: {self._days_since_train} days")

        if reasons:
            logger.warning("Retraining recommended: %s", "; ".join(reasons))
            return True

        return False

    def update_staleness(self, days: int = 0, samples: int = 0) -> None:
        """Update staleness counters."""
        self._days_since_train += days
        self._samples_since_train += samples

    def reset_staleness(self) -> None:
        """Reset staleness counters after retraining."""
        self._days_since_train = 0
        self._samples_since_train = 0

    def log_metrics(self, metrics: RiskMonitoringMetrics) -> None:
        """Log monitoring metrics."""
        logger.info(
            "Risk monitoring: mae_rmse=%.4f, vol_rmse=%.4f, "
            "brier=%.4f, ece=%.4f, fallback_pct=%.1f%%",
            metrics.mae_rmse_rolling,
            metrics.vol_rmse_rolling,
            metrics.stopout_brier_score,
            metrics.stopout_calibration_error,
            metrics.pct_using_fallback,
        )

        if metrics.max_feature_psi > 0.1:
            logger.warning(
                "Feature drift detected: max_psi=%.3f, top_drifting=%s",
                metrics.max_feature_psi,
                metrics.top_drifting_features[:3],
            )
