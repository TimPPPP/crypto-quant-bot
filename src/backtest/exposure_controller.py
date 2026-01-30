# src/backtest/exposure_controller.py
"""
Exposure Controller - Systematic capital deployment.

This module manages target exposure and position scaling to minimize idle capital.
When the strategy has valid signals but low exposure, it scales up positions.
When under-diversified (few valid pairs), it scales down to manage risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.backtest import config_backtest as cfg

logger = logging.getLogger("backtest.exposure_controller")


@dataclass
class ExposureController:
    """
    Manage target exposure and position scaling.

    Rules:
    1. If under-diversified (few pairs), scale down positions
    2. If below target exposure and signals available, scale up
    3. Never exceed per-position cap
    """

    # Target gross exposure (sum of |position sizes|)
    target_gross_exposure: float = 0.6  # 60% notional deployed

    # Minimum pairs for full-size positions
    min_pairs_for_full_size: int = 8

    # Scale factor when under-diversified
    under_diversified_scale: float = 0.5

    # Maximum scale-up multiplier when below target exposure
    max_scale_up: float = 1.5

    # Minimum scale (never scale below this)
    min_scale: float = 0.3

    @classmethod
    def from_config(cls) -> "ExposureController":
        """Create an ExposureController using configuration parameters."""
        return cls(
            target_gross_exposure=float(getattr(cfg, "TARGET_GROSS_EXPOSURE", 0.6)),
            min_pairs_for_full_size=int(getattr(cfg, "MIN_PAIRS_FOR_FULL_SIZE", 8)),
            under_diversified_scale=float(getattr(cfg, "UNDER_DIVERSIFIED_SCALE", 0.5)),
            max_scale_up=float(getattr(cfg, "EXPOSURE_MAX_SCALE_UP", 1.5)),
            min_scale=float(getattr(cfg, "EXPOSURE_MIN_SCALE", 0.3)),
        )

    def compute_position_scale(
        self,
        current_exposure: float,
        n_active_positions: int,
        n_valid_pairs: int,
        pending_entry_count: int,
    ) -> float:
        """
        Compute position size scale factor.

        Parameters
        ----------
        current_exposure : float
            Current gross exposure (sum of |position sizes|)
        n_active_positions : int
            Number of currently open positions
        n_valid_pairs : int
            Number of valid pairs available for trading
        pending_entry_count : int
            Number of pending entry signals

        Returns
        -------
        float
            Position size multiplier (0.3 to 1.5 typically)
        """
        # 1. Under-diversified check: scale down if few valid pairs
        if n_valid_pairs < self.min_pairs_for_full_size:
            diversification_scale = self.under_diversified_scale
            logger.debug(
                "Under-diversified: %d pairs < %d minimum, scaling to %.2f",
                n_valid_pairs, self.min_pairs_for_full_size, diversification_scale
            )
        else:
            diversification_scale = 1.0

        # 2. Exposure gap check: scale up if below target and signals available
        exposure_gap = self.target_gross_exposure - current_exposure
        if exposure_gap > 0 and pending_entry_count > 0:
            # Scale up proportionally to gap (more gap = more scale up)
            gap_ratio = exposure_gap / self.target_gross_exposure
            scale_up = 1.0 + gap_ratio * 0.5  # Conservative scale-up
            scale_up = min(scale_up, self.max_scale_up)
            logger.debug(
                "Below target exposure: %.1f%% vs %.1f%% target, scaling up by %.2f",
                current_exposure * 100, self.target_gross_exposure * 100, scale_up
            )
        else:
            scale_up = 1.0

        # Combine factors
        final_scale = diversification_scale * scale_up

        # Apply bounds
        final_scale = max(self.min_scale, min(final_scale, self.max_scale_up))

        return final_scale

    def compute_position_scales_for_entries(
        self,
        entries: pd.DataFrame,
        current_positions: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute per-bar, per-pair position scale factors.

        Parameters
        ----------
        entries : pd.DataFrame
            Entry signals (bool, T × P)
        current_positions : pd.DataFrame, optional
            Current position sizes (T × P)

        Returns
        -------
        pd.DataFrame
            Position scale multipliers (T × P)
        """
        T, P = entries.shape
        n_valid_pairs = P

        # Initialize scales to 1.0
        scales = pd.DataFrame(1.0, index=entries.index, columns=entries.columns)

        # If we have current positions, compute exposure-based scaling
        if current_positions is not None:
            for t in range(T):
                # Current exposure at this bar
                current_exposure = float(current_positions.iloc[t].abs().sum())

                # Number of active positions
                n_active = int((current_positions.iloc[t].abs() > 0.01).sum())

                # Pending entries at this bar
                pending_entries = int(entries.iloc[t].sum())

                if pending_entries > 0:
                    scale = self.compute_position_scale(
                        current_exposure=current_exposure,
                        n_active_positions=n_active,
                        n_valid_pairs=n_valid_pairs,
                        pending_entry_count=pending_entries,
                    )
                    scales.iloc[t, :] = scale
        else:
            # Without current positions, just apply diversification scaling
            if n_valid_pairs < self.min_pairs_for_full_size:
                scales.iloc[:, :] = self.under_diversified_scale

        return scales

    def get_exposure_summary(
        self,
        position_sizes: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Get summary statistics of exposure.

        Parameters
        ----------
        position_sizes : pd.DataFrame
            Position sizes over time

        Returns
        -------
        Dict[str, float]
            Exposure statistics
        """
        gross_exposure = position_sizes.abs().sum(axis=1)

        return {
            "mean_gross_exposure": float(gross_exposure.mean()),
            "max_gross_exposure": float(gross_exposure.max()),
            "min_gross_exposure": float(gross_exposure.min()),
            "target_exposure": self.target_gross_exposure,
            "exposure_utilization_pct": float(gross_exposure.mean() / self.target_gross_exposure * 100),
        }


def create_exposure_controller() -> ExposureController:
    """Factory function to create an ExposureController from config."""
    return ExposureController.from_config()
