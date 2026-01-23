"""
src/models/pair_scorer.py

Multi-Factor Tradability Scoring - Replace binary cointegration with composite scores.

This module implements a multi-factor scoring system for pair selection that goes
beyond simple p-value thresholds. It evaluates pairs on multiple tradability factors
to select pairs that are not just statistically cointegrated but actually tradable.

Key factors:
1. Cointegration p-value (lower is better)
2. Recent mean-reversion strength (higher is better)
3. Beta stability (lower is better)
4. Spread variance stability (lower is better)
5. Training stop-loss rate (lower is better)
6. Training net expectancy (higher is better)
7. Historical OOS performance (higher is better)

Example usage:
    scorer = PairScorer.from_config()
    scores = scorer.score_pairs(train_df, candidate_pairs, historical_pnl)
    valid_pairs = [s.pair for s in scores if s.composite_score > 0.4]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from src.backtest import config_backtest as cfg

logger = logging.getLogger("models.pair_scorer")


@dataclass
class PairScore:
    """
    Multi-factor tradability score for a single pair.

    All factor scores are normalized to 0-1 range where higher is better.
    """

    pair: str
    cointegration_pvalue: float
    half_life_bars: float

    # Tradability factors (raw values)
    recent_mean_reversion_strength: float = 0.0  # ADF stat (more negative = better)
    beta_stability: float = 1.0                   # CV of rolling beta (lower = better)
    spread_variance_stability: float = 1.0        # CV of rolling var (lower = better)
    training_stop_rate: float = 0.5              # Stop-loss rate in training
    training_net_expectancy: float = 0.0         # Simulated P&L on training
    historical_oos_performance: float = 0.0      # OOS P&L from prior windows

    # Computed scores (0-1, higher = better)
    coint_score: float = field(init=False, default=0.0)
    mr_score: float = field(init=False, default=0.0)
    beta_score: float = field(init=False, default=0.0)
    var_score: float = field(init=False, default=0.0)
    stop_score: float = field(init=False, default=0.0)
    expectancy_score: float = field(init=False, default=0.0)
    oos_score: float = field(init=False, default=0.0)
    composite_score: float = field(init=False, default=0.0)

    def __post_init__(self):
        """Compute normalized scores after initialization."""
        self.compute_scores()

    def compute_scores(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute weighted tradability score.

        Parameters
        ----------
        weights : Dict[str, float], optional
            Custom weights for each factor. Uses defaults if not provided.

        Returns
        -------
        float
            Composite score in range [0, 1]
        """
        weights = weights or {
            "coint": 0.15,
            "mr": 0.20,
            "beta": 0.15,
            "var": 0.10,
            "stop": 0.15,
            "expectancy": 0.15,
            "oos": 0.10,
        }

        # Normalize each factor to 0-1 scale
        # Cointegration: lower p-value = better
        self.coint_score = 1 - min(self.cointegration_pvalue / 0.05, 1)

        # Mean reversion strength: more negative ADF stat = better
        self.mr_score = min(max(-self.recent_mean_reversion_strength / 4, 0), 1)

        # Beta stability: lower CV = better
        self.beta_score = 1 - min(self.beta_stability, 1)

        # Variance stability: lower CV = better
        self.var_score = 1 - min(self.spread_variance_stability, 1)

        # Stop rate: lower = better
        self.stop_score = 1 - min(self.training_stop_rate / 0.5, 1)

        # Expectancy: higher = better (normalized around -2% to +2%)
        self.expectancy_score = min(max(self.training_net_expectancy + 0.02, 0) / 0.04, 1)

        # Historical OOS: higher = better (normalized around -1% to +1%)
        self.oos_score = min(max(self.historical_oos_performance + 0.01, 0) / 0.02, 1)

        # Compute weighted composite
        self.composite_score = (
            weights["coint"] * self.coint_score +
            weights["mr"] * self.mr_score +
            weights["beta"] * self.beta_score +
            weights["var"] * self.var_score +
            weights["stop"] * self.stop_score +
            weights["expectancy"] * self.expectancy_score +
            weights["oos"] * self.oos_score
        )

        return self.composite_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pair": self.pair,
            "cointegration_pvalue": self.cointegration_pvalue,
            "half_life_bars": self.half_life_bars,
            "recent_mr_strength": self.recent_mean_reversion_strength,
            "beta_stability": self.beta_stability,
            "spread_var_stability": self.spread_variance_stability,
            "training_stop_rate": self.training_stop_rate,
            "training_expectancy": self.training_net_expectancy,
            "historical_oos": self.historical_oos_performance,
            "composite_score": self.composite_score,
            "factor_scores": {
                "coint": self.coint_score,
                "mr": self.mr_score,
                "beta": self.beta_score,
                "var": self.var_score,
                "stop": self.stop_score,
                "expectancy": self.expectancy_score,
                "oos": self.oos_score,
            },
        }


@dataclass
class PairScorer:
    """
    Multi-factor pair scoring system.

    Replaces binary cointegration filtering with a comprehensive
    tradability assessment.
    """

    # Configuration
    min_composite_score: float = 0.4
    recent_lookback_days: int = 30
    rolling_window_bars: int = 480  # 5 days at 15-min
    pair_separator: str = "-"

    # Weights for composite score
    weights: Dict[str, float] = field(default_factory=lambda: {
        "coint": 0.15,
        "mr": 0.20,
        "beta": 0.15,
        "var": 0.10,
        "stop": 0.15,
        "expectancy": 0.15,
        "oos": 0.10,
    })

    @classmethod
    def from_config(cls) -> "PairScorer":
        """Create a PairScorer using configuration parameters."""
        return cls(
            min_composite_score=getattr(cfg, "PAIR_SCORE_MIN_COMPOSITE", 0.4),
            recent_lookback_days=getattr(cfg, "PAIR_SCORE_RECENT_LOOKBACK_DAYS", 30),
            rolling_window_bars=getattr(cfg, "PAIR_SCORE_ROLLING_WINDOW_BARS", 480),
            pair_separator=getattr(cfg, "PAIR_ID_SEPARATOR", "-"),
        )

    def score_pairs(
        self,
        train_df: pd.DataFrame,
        pairs: List[str],
        historical_pnl: Optional[Dict[str, Dict]] = None,
        bars_per_day: int = 96,
    ) -> List[PairScore]:
        """
        Compute tradability scores for all candidate pairs.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with price columns for each coin
        pairs : List[str]
            List of pair identifiers (e.g., ["ETH-BTC", "SOL-ETH"])
        historical_pnl : Dict[str, Dict], optional
            Historical OOS performance per pair
        bars_per_day : int
            Number of bars per day (default 96 for 15-min)

        Returns
        -------
        List[PairScore]
            Sorted list of pair scores (highest first)
        """
        historical_pnl = historical_pnl or {}
        scores = []

        for pair in pairs:
            try:
                score = self._score_pair(
                    train_df, pair, historical_pnl, bars_per_day
                )
                if score is not None:
                    scores.append(score)
            except Exception as e:
                logger.warning("Failed to score pair %s: %s", pair, e)
                continue

        # Sort by composite score descending
        scores.sort(key=lambda s: s.composite_score, reverse=True)

        logger.info(
            "Scored %d pairs: top=%s (%.3f), bottom=%s (%.3f)",
            len(scores),
            scores[0].pair if scores else "N/A",
            scores[0].composite_score if scores else 0,
            scores[-1].pair if scores else "N/A",
            scores[-1].composite_score if scores else 0,
        )

        return scores

    def _score_pair(
        self,
        train_df: pd.DataFrame,
        pair: str,
        historical_pnl: Dict[str, Dict],
        bars_per_day: int,
    ) -> Optional[PairScore]:
        """Score a single pair."""
        try:
            coin1, coin2 = pair.split(self.pair_separator)
        except ValueError:
            return None

        if coin1 not in train_df.columns or coin2 not in train_df.columns:
            return None

        y = train_df[coin1].dropna()
        x = train_df[coin2].dropna()

        # Align series
        common_idx = y.index.intersection(x.index)
        if len(common_idx) < self.rolling_window_bars * 2:
            return None

        y = y.loc[common_idx]
        x = x.loc[common_idx]

        # 1. Cointegration on full training
        try:
            _, coint_pval, _ = coint(y, x)
        except Exception:
            coint_pval = 1.0

        # 2. Compute OLS beta
        beta = self._compute_beta(y, x)
        spread = y - beta * x

        # 3. Half-life
        half_life = self._compute_half_life(spread)

        # 4. Recent mean-reversion strength (last N days)
        recent_rows = self.recent_lookback_days * bars_per_day
        if len(spread) > recent_rows:
            recent_spread = spread.iloc[-recent_rows:]
            try:
                recent_adf = adfuller(recent_spread.dropna(), maxlag=20)[0]
            except Exception:
                recent_adf = 0.0
        else:
            recent_adf = 0.0

        # 5. Beta stability (rolling beta CV)
        rolling_beta = self._compute_rolling_beta(y, x, self.rolling_window_bars)
        beta_mean = rolling_beta.mean()
        beta_std = rolling_beta.std()
        beta_stability = beta_std / abs(beta_mean) if abs(beta_mean) > 1e-6 else 1.0

        # 6. Spread variance stability
        rolling_var = spread.rolling(self.rolling_window_bars).var()
        var_mean = rolling_var.mean()
        var_std = rolling_var.std()
        var_stability = var_std / var_mean if var_mean > 1e-10 else 1.0

        # 7. Training stop rate (simulate with simple rules)
        training_stop_rate = self._simulate_training_stop_rate(spread, half_life)

        # 8. Training expectancy
        training_expectancy = self._simulate_training_expectancy(spread, half_life)

        # 9. Historical OOS performance
        hist_data = historical_pnl.get(pair, {})
        hist_oos = hist_data.get("net_pnl", 0.0)

        return PairScore(
            pair=pair,
            cointegration_pvalue=coint_pval,
            half_life_bars=half_life,
            recent_mean_reversion_strength=recent_adf,
            beta_stability=beta_stability,
            spread_variance_stability=var_stability,
            training_stop_rate=training_stop_rate,
            training_net_expectancy=training_expectancy,
            historical_oos_performance=hist_oos,
        )

    def _compute_beta(self, y: pd.Series, x: pd.Series) -> float:
        """Compute OLS beta."""
        x_arr = x.values
        y_arr = y.values
        cov = np.cov(y_arr, x_arr)[0, 1]
        var_x = np.var(x_arr)
        return cov / var_x if var_x > 1e-10 else 1.0

    def _compute_rolling_beta(
        self,
        y: pd.Series,
        x: pd.Series,
        window: int,
    ) -> pd.Series:
        """Compute rolling beta."""
        cov = y.rolling(window).cov(x)
        var_x = x.rolling(window).var()
        return cov / var_x.replace(0, np.nan)

    def _compute_half_life(self, spread: pd.Series) -> float:
        """Compute half-life from spread series."""
        spread_clean = spread.dropna()
        if len(spread_clean) < 50:
            return 200.0  # Default

        spread_lag = spread_clean.shift(1).dropna()
        spread_diff = spread_clean.diff().dropna()

        # Align
        common = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[common]
        spread_diff = spread_diff.loc[common]

        if len(spread_lag) < 10:
            return 200.0

        # Simple OLS: delta_spread = phi * spread_lag + epsilon
        cov = np.cov(spread_diff.values, spread_lag.values)[0, 1]
        var_lag = np.var(spread_lag.values)
        phi = cov / var_lag if var_lag > 1e-10 else 0.0

        if phi >= 0:
            return 500.0  # Not mean-reverting

        half_life = -np.log(2) / phi
        return max(10, min(half_life, 1000))

    def _simulate_training_stop_rate(
        self,
        spread: pd.Series,
        half_life: float,
    ) -> float:
        """
        Simulate stop-loss rate on training data.

        Uses a simplified simulation to estimate how often positions
        would hit stop-loss thresholds.
        """
        spread_clean = spread.dropna()
        if len(spread_clean) < 100:
            return 0.5  # Default

        # Compute z-score
        spread_mean = spread_clean.rolling(int(half_life)).mean()
        spread_std = spread_clean.rolling(int(half_life)).std()
        z_score = (spread_clean - spread_mean) / spread_std.replace(0, np.nan)
        z_score = z_score.dropna()

        if len(z_score) < 50:
            return 0.5

        # Simple simulation: count when z exceeds 3 (potential stop)
        entry_z = getattr(cfg, "ENTRY_Z", 2.0)
        stop_z = getattr(cfg, "STOP_LOSS_Z", 3.0)

        # Find entry points
        entry_mask = z_score.abs() > entry_z
        stop_mask = z_score.abs() > stop_z

        entries = entry_mask.sum()
        stops = stop_mask.sum()

        if entries == 0:
            return 0.5

        return min(stops / entries, 1.0)

    def _simulate_training_expectancy(
        self,
        spread: pd.Series,
        half_life: float,
    ) -> float:
        """
        Simulate expected P&L on training data.

        Uses a simplified simulation to estimate trading expectancy.
        """
        spread_clean = spread.dropna()
        if len(spread_clean) < 100:
            return 0.0

        # Compute z-score
        window = int(min(max(half_life, 60), 500))
        spread_mean = spread_clean.rolling(window).mean()
        spread_std = spread_clean.rolling(window).std()
        z_score = (spread_clean - spread_mean) / spread_std.replace(0, np.nan)
        z_score = z_score.dropna()

        if len(z_score) < 50:
            return 0.0

        # Track simulated P&L
        entry_z = getattr(cfg, "ENTRY_Z", 2.0)
        exit_z = getattr(cfg, "EXIT_Z", 0.5)

        total_pnl = 0.0
        n_trades = 0
        in_trade = False
        entry_spread = 0.0

        spread_vals = spread_clean.loc[z_score.index]
        z_vals = z_score.values
        spread_arr = spread_vals.values

        for i in range(len(z_vals)):
            z = z_vals[i]
            s = spread_arr[i]

            if not in_trade:
                if abs(z) > entry_z:
                    in_trade = True
                    entry_spread = s
            else:
                if abs(z) < exit_z:
                    # Exit trade
                    pnl = (entry_spread - s) if entry_spread > 0 else (s - entry_spread)
                    total_pnl += pnl / abs(entry_spread) if abs(entry_spread) > 1e-10 else 0
                    n_trades += 1
                    in_trade = False

        if n_trades == 0:
            return 0.0

        return total_pnl / n_trades

    def filter_by_score(
        self,
        scores: List[PairScore],
        min_score: Optional[float] = None,
        max_pairs: Optional[int] = None,
    ) -> List[str]:
        """
        Filter pairs by composite score.

        Parameters
        ----------
        scores : List[PairScore]
            List of scored pairs
        min_score : float, optional
            Minimum composite score (default from config)
        max_pairs : int, optional
            Maximum number of pairs to return

        Returns
        -------
        List[str]
            Filtered pair identifiers
        """
        min_score = min_score or self.min_composite_score

        filtered = [s.pair for s in scores if s.composite_score >= min_score]

        if max_pairs is not None and len(filtered) > max_pairs:
            filtered = filtered[:max_pairs]

        return filtered


def compute_pair_tradability_scores(
    train_df: pd.DataFrame,
    pairs: List[str],
    historical_pnl: Optional[Dict[str, Dict]] = None,
    **kwargs,
) -> List[PairScore]:
    """
    Convenience function to compute tradability scores.

    This is the main entry point for pair scoring.
    """
    scorer = PairScorer.from_config()
    return scorer.score_pairs(train_df, pairs, historical_pnl, **kwargs)
