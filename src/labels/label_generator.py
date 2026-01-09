"""Label generation for bubble detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate labels for bubble detection using various methods."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize label generator.
        
        Args:
            config: Configuration dictionary with labeling parameters.
        """
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "bubble_threshold": 0.2,  # 20% deviation from trend
            "crash_threshold": -0.15,  # 15% crash threshold
            "lookforward_window": 20,  # Days to look forward for crashes
            "min_bubble_duration": 5,  # Minimum bubble duration
            "min_crash_duration": 3,  # Minimum crash duration
            "trend_window": 200,  # Window for trend calculation
            "volatility_window": 20,  # Window for volatility calculation
            "regime_threshold": 0.1,  # Threshold for regime change
        }

    def generate_bubble_labels(
        self, 
        data: pd.DataFrame, 
        method: str = "deviation"
    ) -> pd.DataFrame:
        """Generate bubble detection labels.
        
        Args:
            data: DataFrame with OHLCV data and features.
            method: Labeling method ('deviation', 'momentum', 'volatility', 'combined').
            
        Returns:
            DataFrame with bubble labels added.
        """
        df = data.copy()
        
        if method == "deviation":
            df = self._generate_deviation_labels(df)
        elif method == "momentum":
            df = self._generate_momentum_labels(df)
        elif method == "volatility":
            df = self._generate_volatility_labels(df)
        elif method == "combined":
            df = self._generate_combined_labels(df)
        else:
            raise ValueError(f"Unknown labeling method: {method}")
            
        return df

    def _generate_deviation_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate labels based on price deviation from trend."""
        df = data.copy()
        
        # Calculate trend using moving average
        trend = df['close'].rolling(window=self.config["trend_window"]).mean()
        
        # Calculate deviation from trend
        deviation = (df['close'] - trend) / trend
        
        # Identify bubble periods
        bubble_mask = deviation > self.config["bubble_threshold"]
        
        # Look forward for crashes
        crash_labels = self._look_forward_crashes(
            df['close'], 
            bubble_mask, 
            self.config["lookforward_window"]
        )
        
        df['bubble_label'] = bubble_mask.astype(int)
        df['crash_label'] = crash_labels
        df['deviation'] = deviation
        
        return df

    def _generate_momentum_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate labels based on momentum indicators."""
        df = data.copy()
        
        # Calculate momentum
        momentum_short = df['close'].pct_change(5)
        momentum_long = df['close'].pct_change(20)
        
        # Identify extreme momentum
        momentum_threshold = df['momentum_20'].rolling(100).quantile(0.95)
        bubble_mask = momentum_short > momentum_threshold
        
        # Look forward for crashes
        crash_labels = self._look_forward_crashes(
            df['close'], 
            bubble_mask, 
            self.config["lookforward_window"]
        )
        
        df['bubble_label'] = bubble_mask.astype(int)
        df['crash_label'] = crash_labels
        
        return df

    def _generate_volatility_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate labels based on volatility patterns."""
        df = data.copy()
        
        # Calculate volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=self.config["volatility_window"]).std()
        
        # Identify low volatility periods (potential bubble buildup)
        vol_threshold = volatility.rolling(100).quantile(0.2)
        low_vol_mask = volatility < vol_threshold
        
        # Look forward for crashes
        crash_labels = self._look_forward_crashes(
            df['close'], 
            low_vol_mask, 
            self.config["lookforward_window"]
        )
        
        df['bubble_label'] = low_vol_mask.astype(int)
        df['crash_label'] = crash_labels
        
        return df

    def _generate_combined_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate labels using combined indicators."""
        df = data.copy()
        
        # Calculate multiple indicators
        trend = df['close'].rolling(window=self.config["trend_window"]).mean()
        deviation = (df['close'] - trend) / trend
        
        momentum = df['close'].pct_change(20)
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=self.config["volatility_window"]).std()
        
        # Combine indicators with weights
        deviation_score = np.clip(deviation / self.config["bubble_threshold"], 0, 2)
        momentum_score = np.clip(momentum / momentum.rolling(100).std(), -2, 2)
        volatility_score = np.clip(
            (volatility.rolling(100).mean() - volatility) / volatility.rolling(100).std(),
            -2, 2
        )
        
        # Combined score
        combined_score = (
            0.5 * deviation_score + 
            0.3 * momentum_score + 
            0.2 * volatility_score
        )
        
        # Identify bubble periods
        bubble_threshold = combined_score.rolling(100).quantile(0.9)
        bubble_mask = combined_score > bubble_threshold
        
        # Look forward for crashes
        crash_labels = self._look_forward_crashes(
            df['close'], 
            bubble_mask, 
            self.config["lookforward_window"]
        )
        
        df['bubble_label'] = bubble_mask.astype(int)
        df['crash_label'] = crash_labels
        df['combined_score'] = combined_score
        
        return df

    def _look_forward_crashes(
        self, 
        prices: pd.Series, 
        bubble_mask: pd.Series, 
        lookforward_window: int
    ) -> pd.Series:
        """Look forward to identify crashes after bubble periods."""
        crash_labels = pd.Series(0, index=prices.index)
        
        bubble_periods = self._get_continuous_periods(bubble_mask)
        
        for start, end in bubble_periods:
            # Look forward from the end of bubble period
            lookforward_end = min(end + lookforward_window, len(prices) - 1)
            
            if lookforward_end > end:
                future_prices = prices.iloc[end:lookforward_end + 1]
                peak_price = prices.iloc[end]
                
                # Check for crash (significant price drop)
                min_price = future_prices.min()
                crash_magnitude = (min_price - peak_price) / peak_price
                
                if crash_magnitude < self.config["crash_threshold"]:
                    crash_idx = future_prices.idxmin()
                    crash_labels.loc[crash_idx] = 1
                    
        return crash_labels

    def _get_continuous_periods(self, mask: pd.Series) -> List[Tuple[int, int]]:
        """Get continuous periods where mask is True."""
        periods = []
        in_period = False
        start = None
        
        for i, value in enumerate(mask):
            if value and not in_period:
                start = i
                in_period = True
            elif not value and in_period:
                periods.append((start, i - 1))
                in_period = False
                
        if in_period:
            periods.append((start, len(mask) - 1))
            
        return periods

    def generate_regime_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime change labels."""
        df = data.copy()
        
        # Calculate regime indicators
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        
        # Regime 1: Bull market (high returns, low volatility)
        bull_mask = (
            (returns.rolling(50).mean() > 0) & 
            (volatility < volatility.rolling(100).quantile(0.5))
        )
        
        # Regime 2: Bear market (negative returns)
        bear_mask = returns.rolling(50).mean() < -self.config["regime_threshold"]
        
        # Regime 3: High volatility (crisis)
        crisis_mask = volatility > volatility.rolling(100).quantile(0.8)
        
        # Assign regime labels
        df['regime'] = 0  # Normal
        df.loc[bull_mask, 'regime'] = 1  # Bull
        df.loc[bear_mask, 'regime'] = 2  # Bear
        df.loc[crisis_mask, 'regime'] = 3  # Crisis
        
        return df

    def generate_triple_barrier_labels(
        self, 
        data: pd.DataFrame, 
        upper_barrier: float = 0.1,
        lower_barrier: float = -0.1,
        time_barrier: int = 20
    ) -> pd.DataFrame:
        """Generate triple barrier labels for bubble detection.
        
        Args:
            data: DataFrame with price data.
            upper_barrier: Upper price barrier (e.g., 0.1 for 10%).
            lower_barrier: Lower price barrier (e.g., -0.1 for -10%).
            time_barrier: Maximum time to hold position.
            
        Returns:
            DataFrame with triple barrier labels.
        """
        df = data.copy()
        
        labels = []
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            
            # Look forward
            future_prices = df['close'].iloc[i + 1:i + time_barrier + 1]
            
            if len(future_prices) == 0:
                labels.append(0)  # No label
                continue
                
            # Check barriers
            upper_hit = (future_prices / current_price - 1) >= upper_barrier
            lower_hit = (future_prices / current_price - 1) <= lower_barrier
            
            if upper_hit.any():
                labels.append(1)  # Upper barrier hit
            elif lower_hit.any():
                labels.append(-1)  # Lower barrier hit
            else:
                labels.append(0)  # Time barrier hit
                
        df['triple_barrier_label'] = labels
        
        return df

    def generate_volatility_regime_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate labels based on volatility regime changes."""
        df = data.copy()
        
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        
        # Calculate volatility percentiles
        vol_percentiles = volatility.rolling(100).quantile([0.2, 0.8])
        
        # Low volatility regime
        low_vol_mask = volatility < vol_percentiles.iloc[:, 0]
        
        # High volatility regime
        high_vol_mask = volatility > vol_percentiles.iloc[:, 1]
        
        # Transition from low to high volatility (potential bubble burst)
        vol_regime_change = (
            low_vol_mask.shift(1) & high_vol_mask
        ).astype(int)
        
        df['vol_regime_change'] = vol_regime_change
        
        return df

    def validate_labels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Validate generated labels.
        
        Args:
            data: DataFrame with labels.
            
        Returns:
            Dictionary with validation metrics.
        """
        metrics = {}
        
        if 'bubble_label' in data.columns:
            bubble_rate = data['bubble_label'].mean()
            metrics['bubble_rate'] = bubble_rate
            
            if bubble_rate < 0.01:
                logger.warning("Very low bubble rate detected")
            elif bubble_rate > 0.3:
                logger.warning("Very high bubble rate detected")
                
        if 'crash_label' in data.columns:
            crash_rate = data['crash_label'].mean()
            metrics['crash_rate'] = crash_rate
            
        if 'bubble_label' in data.columns and 'crash_label' in data.columns:
            # Check correlation between bubbles and crashes
            correlation = data['bubble_label'].corr(data['crash_label'])
            metrics['bubble_crash_correlation'] = correlation
            
        return metrics
