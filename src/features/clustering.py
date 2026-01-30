import pandas as pd
import numpy as np
import logging
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from src.features.engineering import FeatureEngineer
from src.models.autoencoder import train_autoencoder
from src.utils.universe import get_liquid_universe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Clustering")

# Configurable parameters
DEFAULT_N_CLUSTERS = int(os.getenv('CLUSTER_COUNT', 7))
DEFAULT_TOP_N_COINS = int(os.getenv('CLUSTER_TOP_N', 50))
DEFAULT_LOOKBACK_DAYS = int(os.getenv('CLUSTER_LOOKBACK', 50))


def _generate_clustering_results(
    lookback_days: int = None,
    n_clusters: int = None,
    top_n_coins: int = None
):
    """
    Generate clustering results for the coin universe.

    Args:
        lookback_days: Days of historical data to use
        n_clusters: Number of clusters to create
        top_n_coins: Number of top liquid coins to analyze

    Returns:
        DataFrame with features and cluster assignments
    """
    lookback_days = lookback_days or DEFAULT_LOOKBACK_DAYS
    n_clusters = n_clusters or DEFAULT_N_CLUSTERS
    top_n_coins = top_n_coins or DEFAULT_TOP_N_COINS

    logger.info("=" * 50)
    logger.info(f"1. Data Extraction (Lookback: {lookback_days}d, Top {top_n_coins} coins)")

    universe = get_liquid_universe(top_n_coins, use_buffer=False)
    engine = FeatureEngineer(universe)
    engine.load_data(lookback_days=lookback_days)

    logger.info("2. Strategic Feature Engineering")
    raw_features = engine.calculate_features()

    if raw_features.empty:
        logger.error("No features generated.")
        return pd.DataFrame()

    logger.info("3. Training AI Market Mapper")
    model, latent_space = train_autoencoder(raw_features)
    logger.info(f"   Latent Space Dimension: {latent_space.shape[1]}")

    latent_scaler = RobustScaler()
    latent_scaled = latent_scaler.fit_transform(latent_space)

    logger.info(f"4. Clustering into {n_clusters} Regimes")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(latent_scaled)

    results = raw_features.copy()
    results['cluster'] = clusters

    # Sort clusters by beta (risk level)
    cluster_perf = results.groupby('cluster')['beta'].mean()
    sorted_map = {old: new for new, old in enumerate(cluster_perf.sort_values().index)}
    results['cluster'] = results['cluster'].map(sorted_map)

    return results

def get_cluster_map(lookback_days: int = 60, n_clusters: int = None, top_n_coins: int = None):
    """
    Get cluster map for cointegration scanner.

    Args:
        lookback_days: Days of historical data
        n_clusters: Number of clusters (default from env or 7)
        top_n_coins: Number of coins (default from env or 50)

    Returns:
        Dictionary mapping cluster_id -> list of coin symbols
    """
    results = _generate_clustering_results(
        lookback_days=lookback_days,
        n_clusters=n_clusters,
        top_n_coins=top_n_coins
    )

    if results.empty:
        return {}

    cluster_map = {}
    for c in sorted(results['cluster'].unique()):
        coins = results[results['cluster'] == c].index.tolist()
        cluster_map[c] = coins

    logger.info(f"Created cluster map with {len(cluster_map)} clusters")
    return cluster_map

def run_clustering_pipeline(lookback_days: int = 50):
    """
    Run the full clustering pipeline with detailed reporting.

    Args:
        lookback_days: Days of historical data to use
    """
    results = _generate_clustering_results(lookback_days=lookback_days)

    if results.empty:
        logger.error("Pipeline failed - no results")
        return

    print("\n" + "=" * 80)
    print("      AI GENERATED MARKET MAP (Sorted by Beta Risk)")
    print("=" * 80)

    for c in sorted(results['cluster'].unique()):
        cluster_df = results[results['cluster'] == c]
        coins = cluster_df.index.tolist()
        avg = cluster_df.mean(numeric_only=True)

        # Hierarchical Labeling
        label = "UNKNOWN"
        desc = "General Market Movement"

        if avg['beta'] < 0.5:
            label = "DEFENSIVE / HEDGE"
            desc = "Uncorrelated or Safe Haven"
        elif (avg['volatility_z'] > 1.2 or avg['beta'] > 1.3) and avg['alpha'] < 0:
            label = "TOXIC VOLATILITY / DOWNTREND"
            desc = "High Risk but Negative Alpha (Avoid/Short)"
        elif avg['alpha'] > 0.5:
            label = "IDIOSYNCRATIC MOONSHOT"
            desc = "Massive Alpha decoupling from Market Beta"
        elif (avg['volatility_z'] > 1.2 or avg['beta'] > 1.3) and avg['alpha'] > 0:
            label = "HIGH OCTANE / MEMES"
            desc = "Aggressive Growth Leaders (High Beta Winners)"
        elif avg['alpha'] > 0.15:
            label = "TRUE MARKET LEADERS"
            desc = "Strongest Trenders beating the Index"
        elif avg['alpha'] < -0.05:
            label = "LAGGARDS / WEAK"
            desc = "Underperforming the benchmark"
        elif abs(avg['rsi_rel']) > 15 or abs(avg['volume_flow']) > 1.0:
            label = "MOMENTUM PLAY"
            desc = "High relative volume or RSI divergence"
        elif avg['volatility_z'] > 0.5:
            label = "AGGRESSIVE MARKET FOLLOWERS"
            desc = "Correlated but Higher Volatility (Mid-Caps)"
        else:
            label = "BROAD MARKET / SYSTEMATIC"
            desc = "Moving with Index (Low Volatility Core)"

        print(f"\nCLUSTER {c} : {label} ({len(coins)} coins)")
        print(f"   {desc}")
        print(f"Coins: {coins}")
        print(f"   Beta: {avg['beta']:.2f} | Alpha: {avg['alpha']:.4f} | Vol Z: {avg['volatility_z']:.2f}")
        print(f"   Flow: {avg['volume_flow']:.2f} | RSI Rel: {avg['rsi_rel']:.1f} | Funding: {avg['funding_sentiment']:.4f}")
        print("-" * 80)


if __name__ == "__main__":
    run_clustering_pipeline()