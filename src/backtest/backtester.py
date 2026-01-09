"""Backtesting framework for bubble detection strategies."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 1.0  # Maximum position size
    rebalance_frequency: str = "D"  # Rebalancing frequency
    benchmark_symbol: Optional[str] = None


@dataclass
class BacktestResults:
    """Results from backtesting."""
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    benchmark_returns: Optional[pd.Series] = None


class Backtester:
    """Backtesting engine for bubble detection strategies."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """Initialize the backtester.
        
        Args:
            config: Backtesting configuration.
        """
        self.config = config or BacktestConfig()
        self.results = None

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> BacktestResults:
        """Run backtest on bubble detection signals.
        
        Args:
            data: DataFrame with OHLCV data.
            signals: Series with bubble detection signals (0/1).
            benchmark_data: Optional benchmark data for comparison.
            
        Returns:
            BacktestResults object with performance metrics.
        """
        logger.info("Starting backtest")
        
        # Align data and signals
        aligned_data, aligned_signals = self._align_data(data, signals)
        
        # Generate positions based on signals
        positions = self._generate_positions(aligned_signals)
        
        # Calculate returns
        returns = self._calculate_returns(aligned_data, positions)
        
        # Generate equity curve
        equity_curve = self._calculate_equity_curve(returns)
        
        # Generate trade log
        trades = self._generate_trade_log(aligned_data, positions)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(returns, equity_curve)
        
        # Benchmark comparison
        benchmark_returns = None
        if benchmark_data is not None:
            benchmark_returns = self._calculate_benchmark_returns(benchmark_data)
            benchmark_metrics = self._calculate_metrics(benchmark_returns, None)
            metrics.update({f"benchmark_{k}": v for k, v in benchmark_metrics.items()})
        
        self.results = BacktestResults(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
            benchmark_returns=benchmark_returns
        )
        
        logger.info("Backtest completed")
        return self.results

    def _align_data(
        self, 
        data: pd.DataFrame, 
        signals: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align data and signals by index."""
        # Find common index
        common_index = data.index.intersection(signals.index)
        
        if len(common_index) == 0:
            raise ValueError("No common dates between data and signals")
            
        aligned_data = data.loc[common_index]
        aligned_signals = signals.loc[common_index]
        
        logger.info(f"Aligned data: {len(aligned_data)} observations")
        return aligned_data, aligned_signals

    def _generate_positions(self, signals: pd.Series) -> pd.Series:
        """Generate position sizes based on signals."""
        positions = pd.Series(0.0, index=signals.index)
        
        # Simple strategy: go short when bubble detected
        positions[signals == 1] = -self.config.max_position_size
        
        return positions

    def _calculate_returns(
        self, 
        data: pd.DataFrame, 
        positions: pd.Series
    ) -> pd.Series:
        """Calculate strategy returns."""
        # Calculate price returns
        price_returns = data['close'].pct_change()
        
        # Calculate strategy returns (short position)
        strategy_returns = -positions.shift(1) * price_returns
        
        # Apply transaction costs
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * self.config.transaction_cost
        
        # Apply slippage
        slippage_costs = position_changes * self.config.slippage
        
        # Net returns
        net_returns = strategy_returns - transaction_costs - slippage_costs
        
        return net_returns.fillna(0)

    def _calculate_equity_curve(self, returns: pd.Series) -> pd.Series:
        """Calculate equity curve."""
        equity_curve = (1 + returns).cumprod() * self.config.initial_capital
        return equity_curve

    def _generate_trade_log(
        self, 
        data: pd.DataFrame, 
        positions: pd.Series
    ) -> pd.DataFrame:
        """Generate detailed trade log."""
        trades = []
        
        position_changes = positions.diff()
        trade_dates = position_changes[position_changes != 0].index
        
        for date in trade_dates:
            trade = {
                'date': date,
                'price': data.loc[date, 'close'],
                'position': positions.loc[date],
                'position_change': position_changes.loc[date],
                'volume': data.loc[date, 'volume'] if 'volume' in data.columns else 0
            }
            trades.append(trade)
            
        return pd.DataFrame(trades)

    def _calculate_metrics(
        self, 
        returns: pd.Series, 
        equity_curve: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        metrics['total_return'] = total_return
        metrics['annualized_return'] = annualized_return
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        metrics['volatility'] = volatility
        
        # Risk-adjusted returns
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
            metrics['sharpe_ratio'] = sharpe_ratio
        
        # Drawdown metrics
        if equity_curve is not None:
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min()
            metrics['max_drawdown'] = max_drawdown
            
            # Calmar ratio
            if max_drawdown < 0:
                calmar_ratio = annualized_return / abs(max_drawdown)
                metrics['calmar_ratio'] = calmar_ratio
        
        # Hit rate
        positive_returns = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        if total_trades > 0:
            hit_rate = positive_returns / total_trades
            metrics['hit_rate'] = hit_rate
        
        # Average trade return
        non_zero_returns = returns[returns != 0]
        if len(non_zero_returns) > 0:
            avg_trade_return = non_zero_returns.mean()
            metrics['avg_trade_return'] = avg_trade_return
        
        # Skewness and Kurtosis
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        
        return metrics

    def _calculate_benchmark_returns(self, benchmark_data: pd.DataFrame) -> pd.Series:
        """Calculate benchmark returns."""
        if 'close' not in benchmark_data.columns:
            raise ValueError("Benchmark data must contain 'close' column")
            
        benchmark_returns = benchmark_data['close'].pct_change()
        return benchmark_returns.fillna(0)

    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary as DataFrame."""
        if self.results is None:
            raise ValueError("No backtest results available")
            
        metrics_df = pd.DataFrame([self.results.metrics]).T
        metrics_df.columns = ['Value']
        
        return metrics_df

    def plot_equity_curve(self, ax=None) -> None:
        """Plot equity curve."""
        if self.results is None:
            raise ValueError("No backtest results available")
            
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        ax.plot(self.results.equity_curve.index, self.results.equity_curve.values)
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True)
        
        if ax is None:
            plt.show()

    def plot_drawdown(self, ax=None) -> None:
        """Plot drawdown."""
        if self.results is None:
            raise ValueError("No backtest results available")
            
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        peak = self.results.equity_curve.expanding().max()
        drawdown = (self.results.equity_curve - peak) / peak
        
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, color='red')
        ax.set_title('Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True)
        
        if ax is None:
            plt.show()


class WalkForwardBacktester:
    """Walk-forward backtesting for time series validation."""

    def __init__(
        self,
        train_window: int = 252,  # 1 year
        test_window: int = 63,    # 3 months
        step_size: int = 21       # 1 month
    ) -> None:
        """Initialize walk-forward backtester.
        
        Args:
            train_window: Training window size in days.
            test_window: Test window size in days.
            step_size: Step size for rolling window.
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.results = []

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        detector_class,
        detector_config: Optional[dict] = None
    ) -> List[BacktestResults]:
        """Run walk-forward backtesting.
        
        Args:
            data: DataFrame with OHLCV data.
            signals: Series with bubble detection signals.
            detector_class: Detector class to use.
            detector_config: Configuration for detector.
            
        Returns:
            List of BacktestResults for each test period.
        """
        logger.info("Starting walk-forward backtesting")
        
        results = []
        backtester = Backtester()
        
        # Generate train/test splits
        splits = self._generate_splits(data)
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}")
            
            # Get train/test data
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]
            train_signals = signals.loc[train_start:train_end]
            test_signals = signals.loc[test_start:test_end]
            
            # Train detector
            detector = detector_class(detector_config)
            detector.fit(train_data, train_signals)
            
            # Generate predictions
            predictions = detector.predict(test_data)
            prediction_series = pd.Series(predictions, index=test_data.index)
            
            # Run backtest
            result = backtester.run_backtest(test_data, prediction_series)
            results.append(result)
            
        self.results = results
        logger.info("Walk-forward backtesting completed")
        
        return results

    def _generate_splits(self, data: pd.DataFrame) -> List[Tuple]:
        """Generate train/test splits for walk-forward validation."""
        splits = []
        
        start_idx = 0
        while start_idx + self.train_window + self.test_window < len(data):
            train_start = data.index[start_idx]
            train_end = data.index[start_idx + self.train_window - 1]
            test_start = data.index[start_idx + self.train_window]
            test_end = data.index[start_idx + self.train_window + self.test_window - 1]
            
            splits.append((train_start, train_end, test_start, test_end))
            
            start_idx += self.step_size
            
        return splits

    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all walk-forward periods."""
        if not self.results:
            raise ValueError("No walk-forward results available")
            
        # Aggregate metrics
        all_returns = []
        all_equity_curves = []
        
        for result in self.results:
            all_returns.append(result.returns)
            all_equity_curves.append(result.equity_curve)
            
        # Combine all returns
        combined_returns = pd.concat(all_returns)
        
        # Calculate overall metrics
        backtester = Backtester()
        metrics = backtester._calculate_metrics(combined_returns)
        
        return metrics
