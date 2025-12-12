import pandas as pd
import numpy as np
import logging
from src.utils.loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureEngineer")


class FeatureEngineer:
    """
    Feature engineering for coin clustering and analysis.

    Generates 7 strategic features:
    1. Beta - Market correlation
    2. Alpha - Excess return
    3. Volatility Z-Score - Relative volatility
    4. RSI Relative - Momentum indicator
    5. Volume Flow - Liquidity changes
    6. Funding Sentiment - Futures premium
    7. Mean Reversion - Price deviation from MA
    """

    def __init__(self, universe):
        self.universe = list(universe) if not isinstance(universe, list) else universe
        # Ensure BTC is included as a sanity check anchor
        if 'BTC' not in self.universe:
            self.universe.append('BTC')
        self.market_data = pd.DataFrame()

    def load_data(self, lookback_days=40):
        """Load market data with buffer for rolling windows."""
        buffer = 10
        logger.info(f"Loading Data for {lookback_days + buffer} days...")

        loader = DataLoader(self.universe)
        self.market_data = loader.fetch_data(lookback_days=lookback_days + buffer)

        if self.market_data.empty:
            logger.error("DataLoader returned empty DataFrame.")
        else:
            logger.info(f"Loaded matrix: {self.market_data.shape}")

    def calculate_features(self):
        """Calculate 7 strategic features for all coins."""
        if self.market_data.empty:
            logger.error("Market Data is empty. Skipping features.")
            return pd.DataFrame()

        logger.info("Calculating 7 Strategic Features...")

        try:
            # Extract Data Slices
            try:
                closes = self.market_data.xs('close', axis=1, level=1).ffill()
                volumes = self.market_data.xs('turnover', axis=1, level=1).fillna(0)
                funding = self.market_data.xs('funding_rate', axis=1, level=1).fillna(0)
            except KeyError as e:
                logger.error(f"Key Error (Check QuestDB columns): {e}")
                return pd.DataFrame()

            # Validate prices before log transform
            if (closes <= 0).any().any():
                logger.warning("Found non-positive prices, replacing with NaN")
                closes = closes.where(closes > 0)
                closes = closes.ffill().bfill()

            # Calculate Log Returns with safety
            returns = np.log(closes / closes.shift(1).replace(0, np.nan))

            # Construct Robust Market Benchmark (trimmed mean)
            def trimmed_mean(row):
                if row.isnull().all():
                    return np.nan
                valid = row.dropna()
                if len(valid) < 3:
                    return valid.mean() if len(valid) > 0 else np.nan
                low = valid.quantile(0.1)
                high = valid.quantile(0.9)
                return valid[(valid >= low) & (valid <= high)].mean()

            market_ret = returns.apply(trimmed_mean, axis=1)
            market_ret = market_ret.fillna(0)

            features = pd.DataFrame(index=closes.columns)

            WINDOW_30D = 24 * 30

            # FEATURE 1: BETA (Structural Correlation)
            cov = returns.rolling(WINDOW_30D).cov(market_ret)
            var = market_ret.rolling(WINDOW_30D).var()
            # Safe division
            var_safe = var.replace(0, np.nan)
            beta = cov.div(var_safe, axis=0).iloc[-1]
            features['beta'] = beta.fillna(1.0)

            # FEATURE 2: ALPHA (Excess Return)
            coin_ret_sum = returns.tail(WINDOW_30D).sum()
            mkt_ret_sum = market_ret.tail(WINDOW_30D).sum()
            features['alpha'] = coin_ret_sum - (features['beta'] * mkt_ret_sum)

            # FEATURE 3: VOLATILITY Z-SCORE
            vol_30d = returns.tail(WINDOW_30D).std()
            vol_std_of_universe = vol_30d.std()
            if vol_std_of_universe > 1e-10:
                features['volatility_z'] = (vol_30d - vol_30d.median()) / vol_std_of_universe
            else:
                features['volatility_z'] = 0

            # FEATURE 4: RELATIVE RSI
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14 * 24).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14 * 24).mean()
            # Safe division - replace 0 with small value
            loss_safe = loss.replace(0, 1e-10)
            rs = gain / loss_safe
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            rsi_last = rsi.iloc[-1]
            features['rsi_rel'] = rsi_last - rsi_last.median()

            # FEATURE 5: VOLUME FLOW
            vol_3d = volumes.tail(24 * 3).mean()
            vol_30d_vol = volumes.tail(24 * 30).mean()
            # Safe division
            vol_30d_safe = vol_30d_vol.replace(0, 1)
            features['volume_flow'] = (vol_3d / vol_30d_safe) - 1.0

            # FEATURE 6: FUNDING SENTIMENT
            avg_funding = funding.tail(24 * 3).mean() * 100
            features['funding_sentiment'] = avg_funding

            # FEATURE 7: MEAN REVERSION
            ma_30d = closes.rolling(24 * 30).mean().iloc[-1]
            price_now = closes.iloc[-1]
            std_30d = closes.rolling(24 * 30).std().iloc[-1]
            # Safe division
            std_30d_safe = std_30d.replace(0, np.nan)
            features['mean_reversion'] = (price_now - ma_30d) / std_30d_safe

            # Final Cleanup
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

            logger.info(f"Generated {len(features.columns)} features for {len(features)} coins.")
            return features

        except Exception as e:
            logger.error(f"CRITICAL ERROR in feature calculation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()