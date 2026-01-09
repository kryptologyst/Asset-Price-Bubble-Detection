"""Evaluation metrics for bubble detection models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive evaluation for bubble detection models."""

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize the evaluator.
        
        Args:
            config: Configuration dictionary with evaluation parameters.
        """
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "time_series_split": True,
            "n_splits": 5,
            "test_size": 0.2,
            "random_state": 42,
            "scoring_metrics": [
                "accuracy", "precision", "recall", "f1", "auc", "ap"
            ]
        }

    def evaluate_model(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            sample_weight: Sample weights (optional).
            
        Returns:
            Dictionary with evaluation metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        metrics['precision'] = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
        
        # Probability-based metrics
        if y_proba is not None:
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba.flatten()
                
            metrics['auc'] = roc_auc_score(y_true, y_proba_pos, sample_weight=sample_weight)
            metrics['average_precision'] = average_precision_score(
                y_true, y_proba_pos, sample_weight=sample_weight
            )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Additional metrics
        if tp + fp > 0:
            metrics['specificity'] = tn / (tn + fp)
        else:
            metrics['specificity'] = 0.0
            
        if tp + fn > 0:
            metrics['sensitivity'] = tp / (tp + fn)
        else:
            metrics['sensitivity'] = 0.0
        
        return metrics

    def evaluate_time_series_model(
        self,
        data: pd.DataFrame,
        detector,
        target_column: str = 'bubble_label',
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Union[float, Dict]]:
        """Evaluate model with time series cross-validation.
        
        Args:
            data: DataFrame with features and targets.
            detector: Trained detector model.
            target_column: Name of target column.
            feature_columns: List of feature columns to use.
            
        Returns:
            Dictionary with evaluation results.
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col != target_column]
        
        X = data[feature_columns]
        y = data[target_column]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config["n_splits"])
        
        cv_scores = []
        cv_metrics = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Skip if test set has only one class
            if len(y_test.unique()) < 2:
                continue
                
            # Train detector
            detector.fit(X_train, y_train)
            
            # Make predictions
            y_pred = detector.predict(X_test)
            y_proba = detector.predict_proba(X_test) if hasattr(detector, 'predict_proba') else None
            
            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred, y_proba)
            cv_metrics.append(metrics)
            
            # Calculate scores
            scores = {}
            for metric_name in self.config["scoring_metrics"]:
                if metric_name in metrics:
                    scores[metric_name] = metrics[metric_name]
            cv_scores.append(scores)
        
        # Aggregate results
        results = {}
        
        # Mean and std of scores
        for metric_name in self.config["scoring_metrics"]:
            metric_scores = [score.get(metric_name, np.nan) for score in cv_scores]
            metric_scores = [score for score in metric_scores if not np.isnan(score)]
            
            if metric_scores:
                results[f'{metric_name}_mean'] = np.mean(metric_scores)
                results[f'{metric_name}_std'] = np.std(metric_scores)
        
        # Detailed metrics for each fold
        results['cv_metrics'] = cv_metrics
        
        return results

    def evaluate_bubble_detection_performance(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        actual_bubbles: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Evaluate bubble detection performance with financial context.
        
        Args:
            data: DataFrame with price data.
            predictions: Series with bubble predictions.
            actual_bubbles: Series with actual bubble labels (optional).
            
        Returns:
            Dictionary with bubble-specific metrics.
        """
        metrics = {}
        
        # Align data
        common_index = data.index.intersection(predictions.index)
        aligned_data = data.loc[common_index]
        aligned_predictions = predictions.loc[common_index]
        
        if actual_bubbles is not None:
            aligned_actual = actual_bubbles.loc[common_index]
            
            # Standard classification metrics
            classification_metrics = self.evaluate_model(
                aligned_actual, 
                aligned_predictions.values
            )
            metrics.update(classification_metrics)
        
        # Financial performance metrics
        financial_metrics = self._calculate_financial_metrics(
            aligned_data, aligned_predictions
        )
        metrics.update(financial_metrics)
        
        # Bubble-specific metrics
        bubble_metrics = self._calculate_bubble_metrics(
            aligned_data, aligned_predictions
        )
        metrics.update(bubble_metrics)
        
        return metrics

    def _calculate_financial_metrics(
        self,
        data: pd.DataFrame,
        predictions: pd.Series
    ) -> Dict[str, float]:
        """Calculate financial performance metrics."""
        metrics = {}
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Strategy returns (short when bubble predicted)
        strategy_returns = -predictions.shift(1) * returns
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return metrics
        
        # Basic return metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        
        metrics['strategy_total_return'] = total_return
        metrics['strategy_annualized_return'] = annualized_return
        
        # Risk metrics
        volatility = strategy_returns.std() * np.sqrt(252)
        metrics['strategy_volatility'] = volatility
        
        # Risk-adjusted returns
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
            metrics['strategy_sharpe_ratio'] = sharpe_ratio
        
        # Hit rate
        positive_returns = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns[strategy_returns != 0])
        if total_trades > 0:
            hit_rate = positive_returns / total_trades
            metrics['strategy_hit_rate'] = hit_rate
        
        return metrics

    def _calculate_bubble_metrics(
        self,
        data: pd.DataFrame,
        predictions: pd.Series
    ) -> Dict[str, float]:
        """Calculate bubble-specific metrics."""
        metrics = {}
        
        # Bubble detection frequency
        bubble_frequency = predictions.mean()
        metrics['bubble_detection_frequency'] = bubble_frequency
        
        # Average bubble duration
        bubble_periods = self._get_continuous_periods(predictions)
        if bubble_periods:
            avg_duration = np.mean([end - start + 1 for start, end in bubble_periods])
            metrics['avg_bubble_duration'] = avg_duration
        else:
            metrics['avg_bubble_duration'] = 0
        
        # Price impact analysis
        price_impact = self._calculate_price_impact(data, predictions)
        metrics.update(price_impact)
        
        return metrics

    def _get_continuous_periods(self, series: pd.Series) -> List[Tuple[int, int]]:
        """Get continuous periods where series is True."""
        periods = []
        in_period = False
        start = None
        
        for i, value in enumerate(series):
            if value and not in_period:
                start = i
                in_period = True
            elif not value and in_period:
                periods.append((start, i - 1))
                in_period = False
                
        if in_period:
            periods.append((start, len(series) - 1))
            
        return periods

    def _calculate_price_impact(
        self,
        data: pd.DataFrame,
        predictions: pd.Series
    ) -> Dict[str, float]:
        """Calculate price impact of bubble predictions."""
        metrics = {}
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Get bubble periods
        bubble_periods = self._get_continuous_periods(predictions)
        
        if not bubble_periods:
            metrics['avg_bubble_return'] = 0
            metrics['max_bubble_return'] = 0
            return metrics
        
        # Calculate returns during bubble periods
        bubble_returns = []
        for start, end in bubble_periods:
            period_returns = returns.iloc[start:end+1]
            if len(period_returns) > 0:
                bubble_returns.append(period_returns.sum())
        
        if bubble_returns:
            metrics['avg_bubble_return'] = np.mean(bubble_returns)
            metrics['max_bubble_return'] = np.max(bubble_returns)
            metrics['min_bubble_return'] = np.min(bubble_returns)
        
        return metrics

    def compare_models(
        self,
        models: Dict[str, any],
        data: pd.DataFrame,
        target_column: str = 'bubble_label'
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dictionary mapping model names to model objects.
            data: DataFrame with features and targets.
            target_column: Name of target column.
            
        Returns:
            DataFrame with comparison results.
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")
            
            try:
                # Get evaluation results
                eval_results = self.evaluate_time_series_model(
                    data, model, target_column
                )
                
                # Extract mean scores
                model_scores = {}
                for key, value in eval_results.items():
                    if key.endswith('_mean'):
                        metric_name = key.replace('_mean', '')
                        model_scores[metric_name] = value
                
                model_scores['model'] = model_name
                results.append(model_scores)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
                # Add default scores for failed models
                model_scores = {
                    'model': model_name,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'auc': 0.0,
                    'average_precision': 0.0
                }
                results.append(model_scores)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('model')
        
        return comparison_df

    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
        model_name: str = "Model"
    ) -> str:
        """Generate a formatted evaluation report.
        
        Args:
            metrics: Dictionary with evaluation metrics.
            model_name: Name of the model.
            
        Returns:
            Formatted evaluation report.
        """
        report = f"\n{'='*50}\n"
        report += f"EVALUATION REPORT: {model_name}\n"
        report += f"{'='*50}\n\n"
        
        # Classification Metrics
        report += "CLASSIFICATION METRICS:\n"
        report += "-" * 30 + "\n"
        
        classification_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision'
        ]
        
        for metric in classification_metrics:
            if metric in metrics:
                report += f"{metric.upper():<20}: {metrics[metric]:.4f}\n"
        
        # Financial Metrics
        report += "\nFINANCIAL METRICS:\n"
        report += "-" * 30 + "\n"
        
        financial_metrics = [
            'strategy_total_return', 'strategy_annualized_return',
            'strategy_volatility', 'strategy_sharpe_ratio', 'strategy_hit_rate'
        ]
        
        for metric in financial_metrics:
            if metric in metrics:
                report += f"{metric.upper():<20}: {metrics[metric]:.4f}\n"
        
        # Bubble-Specific Metrics
        report += "\nBUBBLE-SPECIFIC METRICS:\n"
        report += "-" * 30 + "\n"
        
        bubble_metrics = [
            'bubble_detection_frequency', 'avg_bubble_duration',
            'avg_bubble_return', 'max_bubble_return'
        ]
        
        for metric in bubble_metrics:
            if metric in metrics:
                report += f"{metric.upper():<20}: {metrics[metric]:.4f}\n"
        
        report += f"\n{'='*50}\n"
        
        return report
