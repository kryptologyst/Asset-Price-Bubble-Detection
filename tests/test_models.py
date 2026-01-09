"""Tests for bubble detection models."""

import pytest
import pandas as pd
import numpy as np

from src.models.bubble_detector import (
    MovingAverageDetector,
    TechnicalIndicatorDetector,
    LogisticRegressionDetector,
    RandomForestDetector
)


class TestBubbleDetectors:
    """Test cases for bubble detection models."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 2000, 100),
            'rsi': np.random.uniform(30, 70, 100),
            'macd': np.random.uniform(-1, 1, 100),
            'bb_position': np.random.uniform(0, 1, 100),
            'volatility_20': np.random.uniform(0.01, 0.05, 100)
        }, index=dates)
        return data

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return pd.Series(np.random.randint(0, 2, 100))

    def test_moving_average_detector(self, sample_data):
        """Test MovingAverageDetector."""
        detector = MovingAverageDetector()
        
        # Test fit
        detector.fit(sample_data, pd.Series([0, 1] * 50))
        assert detector.is_fitted == True
        
        # Test predict
        predictions = detector.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test predict_proba
        probabilities = detector.predict_proba(sample_data)
        assert probabilities.shape == (len(sample_data), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_technical_indicator_detector(self, sample_data):
        """Test TechnicalIndicatorDetector."""
        detector = TechnicalIndicatorDetector()
        
        # Test fit
        detector.fit(sample_data, pd.Series([0, 1] * 50))
        assert detector.is_fitted == True
        
        # Test predict
        predictions = detector.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert all(pred in [0, 1] for pred in predictions)

    def test_logistic_regression_detector(self, sample_data, sample_labels):
        """Test LogisticRegressionDetector."""
        detector = LogisticRegressionDetector()
        
        # Test fit
        detector.fit(sample_data, sample_labels)
        assert detector.is_fitted == True
        
        # Test predict
        predictions = detector.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test predict_proba
        probabilities = detector.predict_proba(sample_data)
        assert probabilities.shape == (len(sample_data), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Test feature importance
        importance = detector.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0

    def test_random_forest_detector(self, sample_data, sample_labels):
        """Test RandomForestDetector."""
        detector = RandomForestDetector()
        
        # Test fit
        detector.fit(sample_data, sample_labels)
        assert detector.is_fitted == True
        
        # Test predict
        predictions = detector.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test predict_proba
        probabilities = detector.predict_proba(sample_data)
        assert probabilities.shape == (len(sample_data), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        # Test feature importance
        importance = detector.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0

    def test_detector_not_fitted_error(self, sample_data):
        """Test that detectors raise error when not fitted."""
        detector = MovingAverageDetector()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(sample_data)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict_proba(sample_data)
