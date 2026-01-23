"""
src/adaptive/sector_diversification.py

Category/sector diversification for pairs trading.

Purpose:
- Use clustering to limit exposure per sector (e.g., max 2 pairs from "MEME" cluster)
- Prevent concentration risk from correlated pairs in the same market segment

Key functions:
- get_pair_cluster_assignments: Map pairs to their cluster IDs
- filter_pairs_by_cluster_limits: Enforce max positions per cluster
- compute_cluster_score_penalty: Penalize signals from overrepresented clusters
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("SectorDiversification")


@dataclass
class ClusterStats:
    """Statistics for cluster diversification."""
    total_pairs: int
    clusters_used: int
    max_per_cluster: int
    pairs_removed: int
    cluster_distribution: Dict[int, int]


# Default configuration
DEFAULT_CLUSTER_CONFIG = {
    "max_positions_per_cluster": 2,    # Max pairs from same cluster
    "concentration_penalty": 0.1,      # Score penalty per additional pair in cluster
    "cluster_file": "data/cluster_assignments.json",  # Optional pre-computed clusters
}


def parse_pair_id(pair_id: str, separator: str = "-") -> Tuple[str, str]:
    """
    Parse pair ID into constituent coins.

    Args:
        pair_id: Pair identifier like "ETH-BTC" or "SOL-ETH"
        separator: Character separating coins

    Returns:
        (coin_y, coin_x) tuple
    """
    parts = pair_id.split(separator)
    if len(parts) != 2:
        logger.warning(f"Invalid pair ID format: {pair_id}")
        return (pair_id, pair_id)
    return (parts[0], parts[1])


def get_coin_cluster(
    coin: str,
    cluster_map: Dict[int, List[str]],
) -> int:
    """
    Get cluster ID for a coin.

    Args:
        coin: Coin symbol
        cluster_map: {cluster_id: [coin_list]} mapping

    Returns:
        Cluster ID, or -1 if not found
    """
    for cluster_id, coins in cluster_map.items():
        if coin in coins:
            return cluster_id
    return -1


def get_pair_cluster_assignments(
    pairs: List[str],
    cluster_map: Dict[int, List[str]],
    separator: str = "-",
) -> Dict[str, Tuple[int, int]]:
    """
    Get cluster assignments for each pair's constituent coins.

    For pairs trading, a pair like "ETH-BTC" has two coins, each potentially
    in different clusters. We track both for diversification.

    Args:
        pairs: List of pair IDs
        cluster_map: {cluster_id: [coin_list]} mapping
        separator: Pair ID separator

    Returns:
        {pair_id: (cluster_y, cluster_x)} mapping
    """
    assignments = {}
    for pair_id in pairs:
        coin_y, coin_x = parse_pair_id(pair_id, separator)
        cluster_y = get_coin_cluster(coin_y, cluster_map)
        cluster_x = get_coin_cluster(coin_x, cluster_map)
        assignments[pair_id] = (cluster_y, cluster_x)

    return assignments


def get_pair_primary_cluster(
    pair_id: str,
    cluster_map: Dict[int, List[str]],
    separator: str = "-",
) -> int:
    """
    Get primary cluster for a pair.

    For diversification purposes, we use the Y-leg's cluster as primary
    (the dependent asset being traded).

    Args:
        pair_id: Pair identifier
        cluster_map: Cluster mapping
        separator: Pair ID separator

    Returns:
        Primary cluster ID
    """
    coin_y, _ = parse_pair_id(pair_id, separator)
    return get_coin_cluster(coin_y, cluster_map)


def count_cluster_exposure(
    active_pairs: List[str],
    cluster_map: Dict[int, List[str]],
    separator: str = "-",
) -> Dict[int, int]:
    """
    Count exposure to each cluster from active positions.

    Counts both Y and X legs since both contribute to cluster risk.

    Args:
        active_pairs: List of currently active pair IDs
        cluster_map: Cluster mapping
        separator: Pair ID separator

    Returns:
        {cluster_id: count} of exposure per cluster
    """
    exposure = Counter()

    for pair_id in active_pairs:
        coin_y, coin_x = parse_pair_id(pair_id, separator)
        cluster_y = get_coin_cluster(coin_y, cluster_map)
        cluster_x = get_coin_cluster(coin_x, cluster_map)

        # Count each coin's cluster (0.5 each to avoid double-counting full pairs)
        # Or count Y-leg only for simpler logic
        if cluster_y >= 0:
            exposure[cluster_y] += 1
        if cluster_x >= 0:
            exposure[cluster_x] += 1

    return dict(exposure)


def filter_pairs_by_cluster_limits(
    pairs: List[str],
    cluster_map: Dict[int, List[str]],
    max_per_cluster: int = 2,
    existing_positions: Optional[List[str]] = None,
    separator: str = "-",
    priority_scores: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], ClusterStats]:
    """
    Filter pairs to enforce cluster diversification limits.

    Args:
        pairs: List of candidate pair IDs (in priority order)
        cluster_map: {cluster_id: [coin_list]} mapping
        max_per_cluster: Maximum pairs per cluster
        existing_positions: Currently held positions (already count toward limits)
        separator: Pair ID separator
        priority_scores: Optional {pair_id: score} for sorting (higher = better)

    Returns:
        (filtered_pairs, stats)
    """
    existing_positions = existing_positions or []

    # Start with existing position cluster counts
    cluster_counts = count_cluster_exposure(existing_positions, cluster_map, separator)

    # Sort pairs by priority score if provided
    if priority_scores:
        pairs = sorted(pairs, key=lambda p: priority_scores.get(p, 0), reverse=True)

    selected_pairs = []
    pairs_removed = 0

    for pair_id in pairs:
        # Get clusters for both legs
        coin_y, coin_x = parse_pair_id(pair_id, separator)
        cluster_y = get_coin_cluster(coin_y, cluster_map)
        cluster_x = get_coin_cluster(coin_x, cluster_map)

        # Check if either leg exceeds cluster limit
        y_count = cluster_counts.get(cluster_y, 0) if cluster_y >= 0 else 0
        x_count = cluster_counts.get(cluster_x, 0) if cluster_x >= 0 else 0

        # Allow if both clusters are below limit
        if y_count < max_per_cluster and x_count < max_per_cluster:
            selected_pairs.append(pair_id)
            # Update counts
            if cluster_y >= 0:
                cluster_counts[cluster_y] = cluster_counts.get(cluster_y, 0) + 1
            if cluster_x >= 0:
                cluster_counts[cluster_x] = cluster_counts.get(cluster_x, 0) + 1
        else:
            pairs_removed += 1
            logger.debug(
                f"Pair {pair_id} excluded: cluster {cluster_y}={y_count}, "
                f"cluster {cluster_x}={x_count} (limit={max_per_cluster})"
            )

    stats = ClusterStats(
        total_pairs=len(pairs),
        clusters_used=len([c for c in cluster_counts.values() if c > 0]),
        max_per_cluster=max_per_cluster,
        pairs_removed=pairs_removed,
        cluster_distribution=cluster_counts,
    )

    return selected_pairs, stats


def compute_cluster_score_penalty(
    pair_id: str,
    active_positions: List[str],
    cluster_map: Dict[int, List[str]],
    base_penalty: float = 0.1,
    separator: str = "-",
) -> float:
    """
    Compute score penalty based on cluster concentration.

    Penalizes signals from clusters that are already well-represented
    in the portfolio.

    Args:
        pair_id: Candidate pair
        active_positions: Currently active positions
        cluster_map: Cluster mapping
        base_penalty: Penalty per existing position in same cluster
        separator: Pair ID separator

    Returns:
        Score multiplier (1.0 = no penalty, <1.0 = penalized)
    """
    if not cluster_map:
        return 1.0

    # Get candidate pair's clusters
    coin_y, coin_x = parse_pair_id(pair_id, separator)
    cluster_y = get_coin_cluster(coin_y, cluster_map)
    cluster_x = get_coin_cluster(coin_x, cluster_map)

    # Count existing exposure
    exposure = count_cluster_exposure(active_positions, cluster_map, separator)

    # Penalty based on highest exposure
    y_exposure = exposure.get(cluster_y, 0) if cluster_y >= 0 else 0
    x_exposure = exposure.get(cluster_x, 0) if cluster_x >= 0 else 0
    max_exposure = max(y_exposure, x_exposure)

    # Linear penalty: 1.0 for 0 exposure, decreasing by base_penalty per position
    penalty = max_exposure * base_penalty
    multiplier = max(0.3, 1.0 - penalty)  # Floor at 0.3 to not completely kill signals

    return multiplier


def compute_cluster_penalties_matrix(
    candidate_pairs: List[str],
    active_positions: List[str],
    cluster_map: Dict[int, List[str]],
    base_penalty: float = 0.1,
    separator: str = "-",
) -> np.ndarray:
    """
    Compute cluster penalties for multiple pairs.

    Args:
        candidate_pairs: List of candidate pair IDs
        active_positions: Currently active positions
        cluster_map: Cluster mapping
        base_penalty: Penalty per existing position
        separator: Pair ID separator

    Returns:
        1D array of penalty multipliers
    """
    penalties = np.ones(len(candidate_pairs))

    for i, pair_id in enumerate(candidate_pairs):
        penalties[i] = compute_cluster_score_penalty(
            pair_id, active_positions, cluster_map, base_penalty, separator
        )

    return penalties


def load_cluster_map_from_file(
    filepath: str,
) -> Dict[int, List[str]]:
    """
    Load pre-computed cluster map from JSON file.

    Expected format:
    {
        "0": ["BTC", "ETH"],
        "1": ["SOL", "AVAX", "MATIC"],
        ...
    }

    Args:
        filepath: Path to JSON file

    Returns:
        {cluster_id: [coin_list]} mapping
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Cluster file not found: {filepath}")
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    # Convert string keys to int
    cluster_map = {}
    for key, coins in data.items():
        try:
            cluster_id = int(key)
            cluster_map[cluster_id] = coins
        except ValueError:
            logger.warning(f"Invalid cluster key: {key}")

    logger.info(f"Loaded cluster map with {len(cluster_map)} clusters from {filepath}")
    return cluster_map


def save_cluster_map_to_file(
    cluster_map: Dict[int, List[str]],
    filepath: str,
) -> None:
    """
    Save cluster map to JSON file.

    Args:
        cluster_map: {cluster_id: [coin_list]} mapping
        filepath: Output path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert int keys to strings for JSON
    data = {str(k): v for k, v in cluster_map.items()}

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved cluster map to {filepath}")


def invert_cluster_map(
    cluster_map: Dict[int, List[str]],
) -> Dict[str, int]:
    """
    Invert cluster map: {cluster_id: [coins]} -> {coin: cluster_id}

    Args:
        cluster_map: Original cluster mapping

    Returns:
        {coin_symbol: cluster_id} mapping
    """
    coin_to_cluster = {}
    for cluster_id, coins in cluster_map.items():
        for coin in coins:
            coin_to_cluster[coin] = cluster_id
    return coin_to_cluster


def get_cluster_diversity_score(
    pairs: List[str],
    cluster_map: Dict[int, List[str]],
    separator: str = "-",
) -> float:
    """
    Compute diversity score for a set of pairs.

    Higher score = more diverse (pairs spread across clusters)
    Score = 1 - HHI (Herfindahl-Hirschman Index)

    Args:
        pairs: List of pair IDs
        cluster_map: Cluster mapping
        separator: Pair ID separator

    Returns:
        Diversity score [0, 1]
    """
    if not pairs or not cluster_map:
        return 0.0

    # Count cluster exposure
    exposure = count_cluster_exposure(pairs, cluster_map, separator)

    if sum(exposure.values()) == 0:
        return 0.0

    total = sum(exposure.values())
    shares = [count / total for count in exposure.values()]

    # HHI = sum of squared market shares
    hhi = sum(s ** 2 for s in shares)

    # Diversity score = 1 - HHI (higher = more diverse)
    return 1.0 - hhi


if __name__ == "__main__":
    # Test the sector diversification module
    print("Testing sector diversification...")

    # Mock cluster map
    cluster_map = {
        0: ["BTC", "ETH"],           # Blue chips
        1: ["SOL", "AVAX", "MATIC"], # L1s
        2: ["DOGE", "SHIB", "PEPE"], # Memes
        3: ["LINK", "UNI", "AAVE"],  # DeFi
    }

    # Test pairs
    pairs = [
        "SOL-ETH",    # L1-Blue chip
        "AVAX-ETH",   # L1-Blue chip (same cluster as SOL)
        "MATIC-ETH",  # L1-Blue chip (same cluster as SOL, AVAX)
        "DOGE-BTC",   # Meme-Blue chip
        "LINK-ETH",   # DeFi-Blue chip
        "SHIB-BTC",   # Meme-Blue chip (same cluster as DOGE)
    ]

    # Test filtering
    print("\n1. Testing filter_pairs_by_cluster_limits:")
    filtered, stats = filter_pairs_by_cluster_limits(
        pairs, cluster_map, max_per_cluster=2
    )
    print(f"   Input pairs: {pairs}")
    print(f"   Filtered pairs: {filtered}")
    print(f"   Stats: {stats}")

    # Test penalty calculation
    print("\n2. Testing compute_cluster_score_penalty:")
    active = ["SOL-ETH", "DOGE-BTC"]
    candidate = "AVAX-ETH"
    penalty = compute_cluster_score_penalty(
        candidate, active, cluster_map, base_penalty=0.1
    )
    print(f"   Active positions: {active}")
    print(f"   Candidate: {candidate}")
    print(f"   Penalty multiplier: {penalty:.2f}")

    # Test diversity score
    print("\n3. Testing get_cluster_diversity_score:")
    diverse_pairs = ["SOL-ETH", "DOGE-BTC", "LINK-ETH"]
    concentrated_pairs = ["SOL-ETH", "AVAX-ETH", "MATIC-ETH"]
    print(f"   Diverse pairs {diverse_pairs}: {get_cluster_diversity_score(diverse_pairs, cluster_map):.3f}")
    print(f"   Concentrated pairs {concentrated_pairs}: {get_cluster_diversity_score(concentrated_pairs, cluster_map):.3f}")
