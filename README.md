# Asset Price Bubble Detection

**RESEARCH DEMO - NOT INVESTMENT ADVICE**

This project is for educational and research purposes only. It is not intended as investment advice and may contain inaccuracies. Backtests are hypothetical and past performance does not guarantee future results.

## Overview

This project implements advanced asset price bubble detection using multiple machine learning approaches and financial indicators. It focuses on early-warning systems for market bubbles across different asset classes.

## Features

- **Multiple Detection Methods**: Moving averages, technical indicators, machine learning models
- **Comprehensive Feature Engineering**: Technical indicators, volatility measures, momentum signals
- **Realistic Backtesting**: Transaction costs, slippage, and proper time-based validation
- **Interactive Demo**: Streamlit-based visualization and analysis
- **Research-Grade Evaluation**: Both ML metrics and financial performance metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Asset-Price-Bubble-Detection.git
cd Asset-Price-Bubble-Detection

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.bubble_detector import BubbleDetector
from src.data.data_loader import DataLoader

# Load data
loader = DataLoader()
data = loader.load_stock_data("AAPL", start="2020-01-01", end="2024-01-01")

# Initialize detector
detector = BubbleDetector()

# Train and predict
detector.fit(data)
predictions = detector.predict(data)

# Run backtest
from src.backtest.backtester import Backtester
backtester = Backtester()
results = backtester.run_backtest(data, predictions)
```

### Interactive Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── labels/            # Label generation
│   ├── models/            # ML models
│   ├── backtest/          # Backtesting framework
│   ├── risk/              # Risk management
│   └── utils/             # Utilities
├── data/                  # Data storage
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── demo/                  # Streamlit demo
└── docs/                  # Documentation
```

## Configuration

The project uses OmegaConf for configuration management. Key configuration files:

- `configs/data.yaml`: Data loading parameters
- `configs/models.yaml`: Model hyperparameters
- `configs/backtest.yaml`: Backtesting parameters

## Models

### Baseline Models
- **Moving Average Deviation**: Simple price deviation from long-term moving average
- **Technical Indicators**: RSI, MACD, Bollinger Bands

### Advanced Models
- **XGBoost**: Gradient boosting with financial features
- **LightGBM**: Fast gradient boosting
- **Time Series Models**: ARIMA, GARCH for volatility modeling

## Evaluation Metrics

### Machine Learning Metrics
- **Classification**: AUC, Precision, Recall, F1-Score
- **Regression**: RMSE, MAE, R²

### Financial Metrics
- **Returns**: CAGR, Sharpe Ratio, Sortino Ratio
- **Risk**: Maximum Drawdown, Volatility, VaR
- **Trading**: Hit Rate, Average Trade P&L, Turnover

## Data Sources

The project supports multiple data sources:
- **Yahoo Finance**: Free stock data via yfinance
- **Custom CSV**: Upload your own data
- **Synthetic Data**: Generated for testing

## Risk Management

- **Time-based Splits**: Prevents data leakage
- **Walk-forward Validation**: Realistic out-of-sample testing
- **Transaction Costs**: Realistic trading costs
- **Position Sizing**: Risk-adjusted position sizing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. It is not intended as investment advice and should not be used for actual trading decisions. The authors are not responsible for any financial losses incurred through the use of this software.

- Past performance does not guarantee future results
- Backtests are hypothetical and may not reflect real trading conditions
- Market conditions can change rapidly and unpredictably
- Always consult with qualified financial advisors before making investment decisions
# Asset-Price-Bubble-Detection
