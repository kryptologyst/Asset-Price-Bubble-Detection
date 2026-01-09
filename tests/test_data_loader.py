"""Tests for data loading module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.config is not None
        assert loader.config.random_seed == 42

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        loader = DataLoader()
        data = loader.generate_synthetic_data(n_days=100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert data.index.name is None or isinstance(data.index, pd.DatetimeIndex)

    def test_validate_data(self):
        """Test data validation."""
        loader = DataLoader()
        
        # Valid data
        valid_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        assert loader.validate_data(valid_data) == True
        
        # Invalid data (negative prices)
        invalid_data = pd.DataFrame({
            'open': [100, -101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        assert loader.validate_data(invalid_data) == False

    def test_resample_data(self):
        """Test data resampling."""
        loader = DataLoader()
        
        # Create daily data
        dates = pd.date_range('2020-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 30),
            'high': np.random.uniform(110, 120, 30),
            'low': np.random.uniform(90, 100, 30),
            'close': np.random.uniform(100, 110, 30),
            'volume': np.random.uniform(1000, 2000, 30)
        }, index=dates)
        
        # Resample to weekly
        resampled = loader.resample_data(data, 'W')
        
        assert isinstance(resampled, pd.DataFrame)
        assert len(resampled) < len(data)  # Should have fewer observations
        assert all(col in resampled.columns for col in ['open', 'high', 'low', 'close', 'volume'])
