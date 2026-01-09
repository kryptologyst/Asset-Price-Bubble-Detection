"""Feature engineering module for bubble detection."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for bubble detection."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary with feature engineering parameters.
        """
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        self._fitted = False

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "moving_averages": [5, 10, 20, 50, 100, 200],
            "volatility_windows": [5, 10, 20, 50],
            "momentum_windows": [5, 10, 20, 50],
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_period": 20,
            "bollinger_std": 2,
            "price_windows": [5, 10, 20, 50, 100],
            "volume_windows": [5, 10, 20, 50]
        }

    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data and datetime index.
            
        Returns:
            DataFrame with technical indicators added.
        """
        df = data.copy()
        
        # Moving averages
        for window in self.config["moving_averages"]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
        # Price ratios
        for window in self.config["moving_averages"]:
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
            
        # Volatility indicators
        for window in self.config["volatility_windows"]:
            df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
            df[f'volatility_{window}_pct'] = (
                df['close'].rolling(window=window).std() / 
                df['close'].rolling(window=window).mean()
            )
            
        # Momentum indicators
        for window in self.config["momentum_windows"]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'roc_{window}'] = df['close'].pct_change(window)
            
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.config["rsi_period"])
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(
            df['close'], 
            self.config["macd_fast"], 
            self.config["macd_slow"], 
            self.config["macd_signal"]
        )
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            df['close'], 
            self.config["bollinger_period"], 
            self.config["bollinger_std"]
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume indicators
        for window in self.config["volume_windows"]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
            
        # Price-volume indicators
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        return df

    def create_bubble_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create specific features for bubble detection.
        
        Args:
            data: DataFrame with OHLCV and technical indicators.
            
        Returns:
            DataFrame with bubble-specific features added.
        """
        df = data.copy()
        
        # Price acceleration (second derivative)
        df['price_acceleration'] = df['close'].diff().diff()
        
        # Exponential growth indicators
        for window in [5, 10, 20, 50]:
            df[f'exponential_growth_{window}'] = self._calculate_exponential_growth(
                df['close'], window
            )
            
        # Bubble intensity (price deviation from trend)
        for window in [20, 50, 100, 200]:
            trend = df['close'].rolling(window=window).mean()
            df[f'bubble_intensity_{window}'] = (df['close'] - trend) / trend
            
        # Volatility clustering
        df['volatility_clustering'] = self._calculate_volatility_clustering(df['close'])
        
        # Market regime indicators
        df['regime_bull'] = (df['close'] > df['close'].rolling(200).mean()).astype(int)
        df['regime_volatile'] = (df['volatility_20'] > df['volatility_20'].rolling(50).quantile(0.8)).astype(int)
        
        # Price extremes
        for window in [20, 50, 100]:
            df[f'price_extreme_{window}'] = (
                df['close'].rolling(window=window).rank(pct=True) > 0.95
            ).astype(int)
            
        # Volume extremes
        for window in [20, 50, 100]:
            df[f'volume_extreme_{window}'] = (
                df['volume'].rolling(window=window).rank(pct=True) > 0.95
            ).astype(int)
            
        return df

    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features for bubble detection.
        
        Args:
            data: DataFrame with price and technical data.
            
        Returns:
            DataFrame with statistical features added.
        """
        df = data.copy()
        
        # Skewness and kurtosis
        for window in [20, 50, 100]:
            returns = df['close'].pct_change()
            df[f'skewness_{window}'] = returns.rolling(window=window).skew()
            df[f'kurtosis_{window}'] = returns.rolling(window=window).kurt()
            
        # Autocorrelation
        for window in [20, 50]:
            returns = df['close'].pct_change()
            df[f'autocorr_{window}'] = returns.rolling(window=window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
            )
            
        # Hurst exponent (long memory)
        df['hurst_exponent'] = self._calculate_hurst_exponent(df['close'])
        
        # Fractal dimension
        df['fractal_dimension'] = self._calculate_fractal_dimension(df['close'])
        
        # Detrended fluctuation analysis
        df['dfa_scaling'] = self._calculate_dfa_scaling(df['close'])
        
        return df

    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different indicators.
        
        Args:
            data: DataFrame with technical indicators.
            
        Returns:
            DataFrame with interaction features added.
        """
        df = data.copy()
        
        # Price-volume interactions
        df['price_volume_trend'] = df['close'].pct_change() * df['volume'].pct_change()
        
        # Momentum-volatility interactions
        df['momentum_volatility'] = df['momentum_20'] * df['volatility_20']
        
        # RSI-MACD interactions
        df['rsi_macd_interaction'] = df['rsi'] * df['macd']
        
        # Bollinger Band position with momentum
        df['bb_momentum_interaction'] = df['bb_position'] * df['momentum_20']
        
        # Volume-weighted momentum
        df['volume_weighted_momentum'] = (
            df['momentum_20'] * df['volume_ratio_20']
        )
        
        return df

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform features.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Transformed DataFrame with scaled features.
        """
        # Select numeric columns for scaling
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Fit scaler
        self.scaler.fit(data[numeric_columns].fillna(0))
        self._fitted = True
        
        # Transform data
        data_scaled = data.copy()
        data_scaled[numeric_columns] = self.scaler.transform(
            data[numeric_columns].fillna(0)
        )
        
        return data_scaled

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Transformed DataFrame.
        """
        if not self._fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_scaled = data.copy()
        data_scaled[numeric_columns] = self.scaler.transform(
            data[numeric_columns].fillna(0)
        )
        
        return data_scaled

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int, 
        slow: int, 
        signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int, 
        std_dev: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _calculate_exponential_growth(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate exponential growth rate."""
        log_prices = np.log(prices)
        growth_rate = log_prices.rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
        )
        return growth_rate

    def _calculate_volatility_clustering(self, prices: pd.Series) -> pd.Series:
        """Calculate volatility clustering indicator."""
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()
        clustering = volatility.rolling(50).apply(
            lambda x: stats.skew(x) if len(x) > 1 else np.nan
        )
        return clustering

    def _calculate_hurst_exponent(self, prices: pd.Series, max_lag: int = 100) -> pd.Series:
        """Calculate Hurst exponent for long memory detection."""
        def hurst_calc(ts):
            if len(ts) < 10:
                return np.nan
            lags = range(2, min(max_lag, len(ts) // 2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
            
        return prices.rolling(200).apply(hurst_calc)

    def _calculate_fractal_dimension(self, prices: pd.Series) -> pd.Series:
        """Calculate fractal dimension using box-counting method."""
        def fractal_calc(ts):
            if len(ts) < 20:
                return np.nan
            # Simplified fractal dimension calculation
            n = len(ts)
            scales = np.logspace(0.5, np.log10(n/4), 10).astype(int)
            counts = []
            for scale in scales:
                boxes = np.ceil(n / scale)
                counts.append(boxes)
            if len(counts) > 1:
                poly = np.polyfit(np.log(scales), np.log(counts), 1)
                return -poly[0]
            return np.nan
            
        return prices.rolling(200).apply(fractal_calc)

    def _calculate_dfa_scaling(self, prices: pd.Series) -> pd.Series:
        """Calculate Detrended Fluctuation Analysis scaling exponent."""
        def dfa_calc(ts):
            if len(ts) < 50:
                return np.nan
            # Simplified DFA calculation
            n = len(ts)
            scales = np.logspace(1, np.log10(n/4), 10).astype(int)
            fluctuations = []
            for scale in scales:
                # Detrend and calculate fluctuation
                segments = n // scale
                if segments < 2:
                    continue
                fluctuation = np.std(ts[:segments * scale].values.reshape(segments, scale), axis=1).mean()
                fluctuations.append(fluctuation)
            if len(fluctuations) > 1:
                poly = np.polyfit(np.log(scales[:len(fluctuations)]), np.log(fluctuations), 1)
                return poly[0]
            return np.nan
            
        return prices.rolling(200).apply(dfa_calc)
