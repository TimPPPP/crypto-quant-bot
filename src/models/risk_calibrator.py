# src/models/risk_calibrator.py
"""
Probability Calibration for Risk Predictions

Ensures stop-out probability predictions are well-calibrated:
- If model predicts 70% stop-out probability, ~70% of such trades should stop out

Methods:
- Platt Scaling: Fits sigmoid to map raw scores to calibrated probabilities
- Isotonic Regression: Non-parametric monotonic calibration
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("models.risk_calibrator")


class PlattScaler:
    """
    Platt scaling for probability calibration.

    Fits a sigmoid function: P(y=1|f) = 1 / (1 + exp(A*f + B))
    where f is the raw model output (logits or probabilities).

    This transforms potentially overconfident/underconfident predictions
    into well-calibrated probabilities.
    """

    def __init__(self):
        self.A: float = 0.0
        self.B: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> "PlattScaler":
        """
        Fit Platt scaling parameters using Newton's method.

        Parameters
        ----------
        y_pred : np.ndarray
            Raw predictions (probabilities or logits)
        y_true : np.ndarray
            True binary labels (0 or 1)
        max_iter : int
            Maximum iterations for optimization
        tol : float
            Convergence tolerance

        Returns
        -------
        PlattScaler
            Self, for method chaining
        """
        n = len(y_pred)
        if n == 0:
            logger.warning("Empty calibration data, using identity transform")
            self.A = -1.0
            self.B = 0.0
            self.is_fitted = True
            return self

        # Prior for Bayes' rule
        prior1 = np.sum(y_true == 1) + 1
        prior0 = np.sum(y_true == 0) + 1

        # Target probabilities with regularization (Platt's approach)
        hi_target = (prior1 + 1) / (prior1 + 2)
        lo_target = 1 / (prior0 + 2)
        target = np.where(y_true == 1, hi_target, lo_target)

        # Initialize A, B
        A, B = 0.0, np.log((prior0 + 1) / (prior1 + 1))

        # Newton's method
        for iteration in range(max_iter):
            # Compute predictions
            fApB = y_pred * A + B
            # Avoid numerical overflow
            fApB = np.clip(fApB, -500, 500)
            p = 1 / (1 + np.exp(fApB))
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # First and second derivatives
            d1 = np.sum((target - p) * y_pred)
            d2 = np.sum((target - p))
            h11 = np.sum(p * (1 - p) * y_pred * y_pred)
            h22 = np.sum(p * (1 - p))
            h12 = np.sum(p * (1 - p) * y_pred)

            # Avoid singular Hessian
            det = h11 * h22 - h12 * h12
            if abs(det) < 1e-10:
                break

            # Newton step
            dA = -(h22 * d1 - h12 * d2) / det
            dB = -(-h12 * d1 + h11 * d2) / det

            A += dA
            B += dB

            # Check convergence
            if abs(dA) < tol and abs(dB) < tol:
                break

        self.A = A
        self.B = B
        self.is_fitted = True

        logger.info("Platt scaling fitted: A=%.4f, B=%.4f", A, B)
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            Raw predictions

        Returns
        -------
        np.ndarray
            Calibrated probabilities in [0, 1]
        """
        if not self.is_fitted:
            logger.warning("PlattScaler not fitted, returning raw predictions")
            return np.clip(y_pred, 0, 1)

        fApB = y_pred * self.A + self.B
        fApB = np.clip(fApB, -500, 500)
        calibrated = 1 / (1 + np.exp(fApB))
        return calibrated

    def fit_transform(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_pred, y_true)
        return self.transform(y_pred)


class IsotonicCalibrator:
    """
    Isotonic regression calibration.

    Non-parametric monotonic calibration that preserves the
    ranking of predictions while mapping to calibrated probabilities.

    More flexible than Platt scaling but requires more data.
    """

    def __init__(self):
        self._x_values: Optional[np.ndarray] = None
        self._y_values: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic regression.

        Parameters
        ----------
        y_pred : np.ndarray
            Raw predictions
        y_true : np.ndarray
            True binary labels

        Returns
        -------
        IsotonicCalibrator
            Self, for method chaining
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(y_pred, y_true)
            self._x_values = iso.X_thresholds_
            self._y_values = iso.y_thresholds_
            self.is_fitted = True
            logger.info(
                "Isotonic calibration fitted: %d thresholds",
                len(self._x_values) if self._x_values is not None else 0
            )
        except ImportError:
            logger.warning("sklearn not available, using linear interpolation fallback")
            # Simple fallback: sort and bin
            sorted_idx = np.argsort(y_pred)
            n = len(y_pred)
            n_bins = min(20, n // 5)
            if n_bins < 2:
                self._x_values = np.array([0, 1])
                self._y_values = np.array([y_true.mean(), y_true.mean()])
            else:
                bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
                self._x_values = np.zeros(n_bins)
                self._y_values = np.zeros(n_bins)
                for i in range(n_bins):
                    mask = slice(bin_edges[i], bin_edges[i + 1])
                    bin_preds = y_pred[sorted_idx[mask]]
                    bin_labels = y_true[sorted_idx[mask]]
                    self._x_values[i] = bin_preds.mean()
                    self._y_values[i] = bin_labels.mean()
            self.is_fitted = True

        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Parameters
        ----------
        y_pred : np.ndarray
            Raw predictions

        Returns
        -------
        np.ndarray
            Calibrated probabilities
        """
        if not self.is_fitted or self._x_values is None:
            logger.warning("IsotonicCalibrator not fitted, returning raw predictions")
            return np.clip(y_pred, 0, 1)

        # Linear interpolation between fitted points
        calibrated = np.interp(y_pred, self._x_values, self._y_values)
        return np.clip(calibrated, 0, 1)


def compute_calibration_error(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum(|acc(bin) - conf(bin)| * weight(bin))

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted probabilities
    y_true : np.ndarray
        True binary labels
    n_bins : int
        Number of calibration bins

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        (ECE, bin_accuracies, bin_confidences)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper bound for last bin
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])

        if mask.sum() > 0:
            bin_accuracies[i] = y_true[mask].mean()
            bin_confidences[i] = y_pred[mask].mean()
            bin_counts[i] = mask.sum()

    # Weighted ECE
    total = bin_counts.sum()
    if total > 0:
        ece = np.sum(np.abs(bin_accuracies - bin_confidences) * bin_counts) / total
    else:
        ece = 0.0

    return ece, bin_accuracies, bin_confidences


def compute_brier_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Compute Brier score (mean squared error for probabilities).

    Lower is better. Perfect calibration = 0.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted probabilities
    y_true : np.ndarray
        True binary labels

    Returns
    -------
    float
        Brier score
    """
    return np.mean((y_pred - y_true) ** 2)
