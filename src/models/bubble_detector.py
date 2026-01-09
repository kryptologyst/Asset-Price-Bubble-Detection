"""Bubble detection models."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class BaseBubbleDetector(ABC):
    """Base class for bubble detection models."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the detector.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseBubbleDetector":
        """Fit the model.
        
        Args:
            X: Feature matrix.
            y: Target labels.
            
        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted probabilities.
        """
        pass

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        return None


class MovingAverageDetector(BaseBubbleDetector):
    """Simple moving average-based bubble detector."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the detector."""
        super().__init__(config)
        self.config = config or {
            "short_window": 50,
            "long_window": 200,
            "threshold": 1.2
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MovingAverageDetector":
        """Fit the detector (no training needed for this simple model)."""
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions based on moving average deviation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if 'close' not in X.columns:
            raise ValueError("Close price column not found")
            
        # Calculate moving averages
        short_ma = X['close'].rolling(window=self.config["short_window"]).mean()
        long_ma = X['close'].rolling(window=self.config["long_window"]).mean()
        
        # Calculate deviation
        deviation = short_ma / long_ma
        
        # Predict bubbles
        predictions = (deviation > self.config["threshold"]).astype(int)
        
        return predictions.fillna(0).values

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (simplified)."""
        predictions = self.predict(X)
        # Convert to probabilities (simplified)
        proba = np.column_stack([1 - predictions, predictions])
        return proba


class TechnicalIndicatorDetector(BaseBubbleDetector):
    """Bubble detector using technical indicators."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the detector."""
        super().__init__(config)
        self.config = config or {
            "rsi_threshold": 70,
            "macd_threshold": 0,
            "bb_threshold": 0.95,
            "volatility_threshold": 1.5
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TechnicalIndicatorDetector":
        """Fit the detector."""
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions based on technical indicators."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        predictions = np.zeros(len(X))
        
        # RSI-based signals
        if 'rsi' in X.columns:
            rsi_signal = X['rsi'] > self.config["rsi_threshold"]
            predictions += rsi_signal.astype(int)
            
        # MACD-based signals
        if 'macd' in X.columns:
            macd_signal = X['macd'] > self.config["macd_threshold"]
            predictions += macd_signal.astype(int)
            
        # Bollinger Bands position
        if 'bb_position' in X.columns:
            bb_signal = X['bb_position'] > self.config["bb_threshold"]
            predictions += bb_signal.astype(int)
            
        # Volatility signals
        if 'volatility_20' in X.columns:
            vol_threshold = X['volatility_20'].rolling(100).quantile(0.8)
            vol_signal = X['volatility_20'] > vol_threshold * self.config["volatility_threshold"]
            predictions += vol_signal.astype(int)
            
        # Convert to binary predictions
        predictions = (predictions >= 2).astype(int)
        
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        predictions = self.predict(X)
        proba = np.column_stack([1 - predictions, predictions])
        return proba


class LogisticRegressionDetector(BaseBubbleDetector):
    """Logistic regression-based bubble detector."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the detector."""
        super().__init__(config)
        self.config = config or {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42
        }
        self.model = LogisticRegression(
            C=self.config["C"],
            max_iter=self.config["max_iter"],
            random_state=self.config["random_state"]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionDetector":
        """Fit the logistic regression model."""
        # Select numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        self.feature_columns = numeric_features.tolist()
        
        # Prepare features
        X_processed = X[self.feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = X[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X_processed)
        
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = X[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X_processed)
        
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None
            
        importance = np.abs(self.model.coef_[0])
        return pd.Series(importance, index=self.feature_columns).sort_values(ascending=False)


class RandomForestDetector(BaseBubbleDetector):
    """Random forest-based bubble detector."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the detector."""
        super().__init__(config)
        self.config = config or {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42
        }
        self.model = RandomForestClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            min_samples_split=self.config["min_samples_split"],
            min_samples_leaf=self.config["min_samples_leaf"],
            random_state=self.config["random_state"]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestDetector":
        """Fit the random forest model."""
        # Select numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        self.feature_columns = numeric_features.tolist()
        
        # Prepare features
        X_processed = X[self.feature_columns].fillna(0)
        
        # Fit model
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = X[self.feature_columns].fillna(0)
        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = X[self.feature_columns].fillna(0)
        return self.model.predict_proba(X_processed)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None
            
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_columns).sort_values(ascending=False)


class XGBoostDetector(BaseBubbleDetector):
    """XGBoost-based bubble detector."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the detector."""
        super().__init__(config)
        self.config = config or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        self.model = xgb.XGBClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            random_state=self.config["random_state"],
            eval_metric='logloss'
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostDetector":
        """Fit the XGBoost model."""
        # Select numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        self.feature_columns = numeric_features.tolist()
        
        # Prepare features
        X_processed = X[self.feature_columns].fillna(0)
        
        # Fit model
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = X[self.feature_columns].fillna(0)
        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = X[self.feature_columns].fillna(0)
        return self.model.predict_proba(X_processed)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None
            
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_columns).sort_values(ascending=False)


class LightGBMDetector(BaseBubbleDetector):
    """LightGBM-based bubble detector."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the detector."""
        super().__init__(config)
        self.config = config or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        self.model = lgb.LGBMClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            random_state=self.config["random_state"],
            verbose=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMDetector":
        """Fit the LightGBM model."""
        # Select numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        self.feature_columns = numeric_features.tolist()
        
        # Prepare features
        X_processed = X[self.feature_columns].fillna(0)
        
        # Fit model
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = X[self.feature_columns].fillna(0)
        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_processed = X[self.feature_columns].fillna(0)
        return self.model.predict_proba(X_processed)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if not self.is_fitted:
            return None
            
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_columns).sort_values(ascending=False)


class EnsembleDetector(BaseBubbleDetector):
    """Ensemble bubble detector combining multiple models."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the ensemble detector."""
        super().__init__(config)
        self.config = config or {
            "models": ["logistic", "random_forest", "xgboost"],
            "weights": None,  # Equal weights if None
            "voting": "soft"  # "soft" or "hard"
        }
        
        self.models = {}
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize individual models."""
        model_configs = {
            "logistic": LogisticRegressionDetector(),
            "random_forest": RandomForestDetector(),
            "xgboost": XGBoostDetector(),
            "lightgbm": LightGBMDetector(),
            "technical": TechnicalIndicatorDetector(),
            "moving_average": MovingAverageDetector()
        }
        
        for model_name in self.config["models"]:
            if model_name in model_configs:
                self.models[model_name] = model_configs[model_name]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleDetector":
        """Fit all models in the ensemble."""
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model")
            model.fit(X, y)
            
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
            
        predictions = []
        weights = self.config["weights"]
        
        if weights is None:
            weights = [1.0] * len(self.models)
            
        for i, (model_name, model) in enumerate(self.models.items()):
            pred = model.predict(X)
            predictions.append(pred * weights[i])
            
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Convert to binary predictions
        return (ensemble_pred > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble probability predictions."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
            
        probabilities = []
        weights = self.config["weights"]
        
        if weights is None:
            weights = [1.0] * len(self.models)
            
        for i, (model_name, model) in enumerate(self.models.items()):
            proba = model.predict_proba(X)
            probabilities.append(proba * weights[i])
            
        # Average probabilities
        ensemble_proba = np.mean(probabilities, axis=0)
        
        return ensemble_proba

    def get_feature_importance(self) -> Dict[str, pd.Series]:
        """Get feature importance from all models."""
        if not self.is_fitted:
            return {}
            
        importance_dict = {}
        for model_name, model in self.models.items():
            importance = model.get_feature_importance()
            if importance is not None:
                importance_dict[model_name] = importance
                
        return importance_dict
