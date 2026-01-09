#!/usr/bin/env python3
"""Main training script for bubble detection models."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from labels.label_generator import LabelGenerator
from models.bubble_detector import (
    MovingAverageDetector, TechnicalIndicatorDetector,
    LogisticRegressionDetector, RandomForestDetector,
    XGBoostDetector, LightGBMDetector, EnsembleDetector
)
from backtest.backtester import Backtester, BacktestConfig
from utils.evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def setup_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def load_and_prepare_data(config: Dict) -> pd.DataFrame:
    """Load and prepare data for training."""
    logger.info("Loading and preparing data")
    
    # Initialize data loader
    data_config = config.get("data", {})
    loader = DataLoader(data_config)
    
    # Load data
    if data_config.get("sources", {}).get("synthetic", {}).get("enabled", False):
        logger.info("Generating synthetic data")
        data = loader.generate_synthetic_data(
            n_days=data_config["sources"]["synthetic"]["n_days"],
            initial_price=data_config["sources"]["synthetic"]["initial_price"],
            drift=data_config["sources"]["synthetic"]["drift"],
            volatility=data_config["sources"]["synthetic"]["volatility"],
            bubble_probability=data_config["sources"]["synthetic"]["bubble_probability"],
            bubble_magnitude=data_config["sources"]["synthetic"]["bubble_magnitude"]
        )
    else:
        # Load real data
        symbols = data_config.get("sources", {}).get("yahoo_finance", {}).get("symbols", ["AAPL"])
        symbol = symbols[0]  # Use first symbol for now
        
        logger.info(f"Loading data for {symbol}")
        data = loader.load_stock_data(
            symbol=symbol,
            period=data_config.get("sources", {}).get("yahoo_finance", {}).get("period", "5y")
        )
    
    # Validate data
    if not loader.validate_data(data):
        raise ValueError("Data validation failed")
    
    logger.info(f"Loaded data: {len(data)} observations")
    return data


def engineer_features(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Engineer features for bubble detection."""
    logger.info("Engineering features")
    
    # Initialize feature engineer
    feature_config = config.get("models", {}).get("features", {})
    engineer = FeatureEngineer(feature_config)
    
    # Create technical indicators
    data_with_indicators = engineer.create_technical_indicators(data)
    
    # Create bubble-specific features
    data_with_bubble_features = engineer.create_bubble_features(data_with_indicators)
    
    # Create statistical features
    data_with_stats = engineer.create_statistical_features(data_with_bubble_features)
    
    # Create interaction features
    data_with_interactions = engineer.create_interaction_features(data_with_stats)
    
    logger.info(f"Created {len(data_with_interactions.columns)} features")
    return data_with_interactions


def generate_labels(data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Generate labels for bubble detection."""
    logger.info("Generating labels")
    
    # Initialize label generator
    label_config = config.get("models", {}).get("labels", {})
    generator = LabelGenerator(label_config)
    
    # Generate bubble labels using different methods
    data_with_labels = generator.generate_bubble_labels(data, method="combined")
    
    # Generate regime labels
    data_with_regimes = generator.generate_regime_labels(data_with_labels)
    
    # Generate triple barrier labels
    data_with_triple_barrier = generator.generate_triple_barrier_labels(data_with_regimes)
    
    # Validate labels
    validation_metrics = generator.validate_labels(data_with_triple_barrier)
    logger.info(f"Label validation metrics: {validation_metrics}")
    
    return data_with_triple_barrier


def train_models(data: pd.DataFrame, config: Dict) -> Dict[str, any]:
    """Train multiple bubble detection models."""
    logger.info("Training models")
    
    # Prepare features and targets
    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in ['bubble_label', 'crash_label', 'regime', 'triple_barrier_label']]
    
    X = data[feature_columns].fillna(0)
    y = data['bubble_label']
    
    # Initialize models
    models = {}
    model_configs = config.get("models", {}).get("detectors", {})
    
    # Moving Average Detector
    models['moving_average'] = MovingAverageDetector(model_configs.get("moving_average", {}))
    
    # Technical Indicator Detector
    models['technical_indicator'] = TechnicalIndicatorDetector(model_configs.get("technical_indicator", {}))
    
    # Machine Learning Models
    models['logistic_regression'] = LogisticRegressionDetector(model_configs.get("logistic_regression", {}))
    models['random_forest'] = RandomForestDetector(model_configs.get("random_forest", {}))
    models['xgboost'] = XGBoostDetector(model_configs.get("xgboost", {}))
    models['lightgbm'] = LightGBMDetector(model_configs.get("lightgbm", {}))
    
    # Ensemble Model
    models['ensemble'] = EnsembleDetector(model_configs.get("ensemble", {}))
    
    # Train models
    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}")
        try:
            model.fit(X, y)
            trained_models[name] = model
            logger.info(f"Successfully trained {name}")
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
    
    return trained_models


def evaluate_models(models: Dict[str, any], data: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Evaluate trained models."""
    logger.info("Evaluating models")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare models
    comparison_results = evaluator.compare_models(models, data)
    
    # Save results
    results_path = Path("assets") / "model_comparison.csv"
    results_path.parent.mkdir(exist_ok=True)
    comparison_results.to_csv(results_path)
    
    logger.info(f"Model comparison saved to {results_path}")
    return comparison_results


def run_backtests(models: Dict[str, any], data: pd.DataFrame, config: Dict) -> Dict[str, any]:
    """Run backtests for all models."""
    logger.info("Running backtests")
    
    # Initialize backtester
    backtest_config = BacktestConfig(
        initial_capital=config.get("backtest", {}).get("portfolio", {}).get("initial_capital", 100000),
        transaction_cost=config.get("backtest", {}).get("costs", {}).get("transaction_cost", 0.001),
        slippage=config.get("backtest", {}).get("costs", {}).get("slippage", 0.0005)
    )
    
    backtester = Backtester(backtest_config)
    
    # Prepare data
    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in ['bubble_label', 'crash_label', 'regime', 'triple_barrier_label']]
    
    X = data[feature_columns].fillna(0)
    
    # Run backtests
    backtest_results = {}
    
    for name, model in models.items():
        logger.info(f"Running backtest for {name}")
        try:
            # Generate predictions
            predictions = model.predict(X)
            prediction_series = pd.Series(predictions, index=data.index)
            
            # Run backtest
            result = backtester.run_backtest(data, prediction_series)
            backtest_results[name] = result
            
            # Save equity curve
            equity_path = Path("assets") / f"{name}_equity_curve.csv"
            result.equity_curve.to_csv(equity_path)
            
            logger.info(f"Backtest completed for {name}")
            
        except Exception as e:
            logger.error(f"Backtest failed for {name}: {e}")
    
    return backtest_results


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train bubble detection models")
    parser.add_argument("--config", type=str, default="configs/models.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="assets", help="Output directory")
    
    args = parser.parse_args()
    
    # Set random seed
    setup_random_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load and prepare data
        data = load_and_prepare_data(config)
        
        # Engineer features
        data_with_features = engineer_features(data, config)
        
        # Generate labels
        data_with_labels = generate_labels(data_with_features, config)
        
        # Train models
        models = train_models(data_with_labels, config)
        
        # Evaluate models
        evaluation_results = evaluate_models(models, data_with_labels, config)
        
        # Run backtests
        backtest_results = run_backtests(models, data_with_labels, config)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Models trained: {list(models.keys())}")
        print(f"Data points: {len(data_with_labels)}")
        print(f"Features: {len(data_with_labels.select_dtypes(include=[np.number]).columns)}")
        print(f"Results saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
