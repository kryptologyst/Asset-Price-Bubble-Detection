"""Streamlit demo for bubble detection."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from labels.label_generator import LabelGenerator
from models.bubble_detector import (
    MovingAverageDetector, TechnicalIndicatorDetector,
    LogisticRegressionDetector, RandomForestDetector,
    XGBoostDetector, EnsembleDetector
)
from backtest.backtester import Backtester, BacktestConfig
from utils.evaluator import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="Asset Price Bubble Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Asset Price Bubble Detection</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research demonstration tool for educational purposes only.</strong></p>
    <ul>
        <li>This tool is NOT investment advice</li>
        <li>Past performance does not guarantee future results</li>
        <li>Backtests are hypothetical and may not reflect real trading conditions</li>
        <li>Always consult with qualified financial advisors before making investment decisions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Synthetic Data", "Yahoo Finance"],
    help="Choose between synthetic data or real market data"
)

# Symbol selection for Yahoo Finance
symbol = "AAPL"
if data_source == "Yahoo Finance":
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)"
    )

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Moving Average", "Technical Indicators", "Logistic Regression", 
     "Random Forest", "XGBoost", "Ensemble"],
    help="Choose the bubble detection model"
)

# Parameters
st.sidebar.subheader("Parameters")
bubble_threshold = st.sidebar.slider(
    "Bubble Threshold",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Threshold for bubble detection"
)

lookforward_window = st.sidebar.slider(
    "Lookforward Window",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="Days to look forward for crash detection"
)

# Backtest parameters
st.sidebar.subheader("Backtest Parameters")
initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000
)

transaction_cost = st.sidebar.slider(
    "Transaction Cost (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05
) / 100

# Main content
@st.cache_data
def load_data(data_source: str, symbol: str) -> pd.DataFrame:
    """Load and prepare data."""
    loader = DataLoader()
    
    if data_source == "Synthetic Data":
        data = loader.generate_synthetic_data(
            n_days=1000,
            initial_price=100.0,
            drift=0.0005,
            volatility=0.02,
            bubble_probability=0.1,
            bubble_magnitude=0.3
        )
    else:
        try:
            data = loader.load_stock_data(symbol=symbol, period="5y")
        except Exception as e:
            st.error(f"Failed to load data for {symbol}: {e}")
            return None
    
    return data

@st.cache_data
def process_data(data: pd.DataFrame, bubble_threshold: float, lookforward_window: int) -> pd.DataFrame:
    """Process data with features and labels."""
    # Engineer features
    engineer = FeatureEngineer()
    data_with_features = engineer.create_technical_indicators(data)
    data_with_features = engineer.create_bubble_features(data_with_features)
    data_with_features = engineer.create_statistical_features(data_with_features)
    data_with_features = engineer.create_interaction_features(data_with_features)
    
    # Generate labels
    generator = LabelGenerator({
        "bubble_threshold": bubble_threshold,
        "lookforward_window": lookforward_window
    })
    data_with_labels = generator.generate_bubble_labels(data_with_features, method="combined")
    
    return data_with_labels

@st.cache_data
def train_model(data: pd.DataFrame, model_type: str):
    """Train the selected model."""
    # Prepare features and targets
    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in ['bubble_label', 'crash_label']]
    
    X = data[feature_columns].fillna(0)
    y = data['bubble_label']
    
    # Initialize model
    if model_type == "Moving Average":
        model = MovingAverageDetector()
    elif model_type == "Technical Indicators":
        model = TechnicalIndicatorDetector()
    elif model_type == "Logistic Regression":
        model = LogisticRegressionDetector()
    elif model_type == "Random Forest":
        model = RandomForestDetector()
    elif model_type == "XGBoost":
        model = XGBoostDetector()
    elif model_type == "Ensemble":
        model = EnsembleDetector()
    
    # Train model
    model.fit(X, y)
    
    return model, X, y

def create_price_chart(data: pd.DataFrame, predictions: pd.Series) -> go.Figure:
    """Create interactive price chart with bubble predictions."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price Chart with Bubble Detection", "Volume"),
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    if 'sma_200' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['sma_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='green', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Bubble predictions
    bubble_dates = data.index[predictions == 1]
    bubble_prices = data.loc[bubble_dates, 'close']
    
    fig.add_trace(
        go.Scatter(
            x=bubble_dates,
            y=bubble_prices,
            mode='markers',
            name='Bubble Detected',
            marker=dict(color='red', size=8, symbol='triangle-up')
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"Asset Price Analysis - {data_source}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True
    )
    
    return fig

def create_performance_chart(equity_curve: pd.Series) -> go.Figure:
    """Create performance chart."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        )
    )
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400
    )
    
    return fig

def main():
    """Main demo function."""
    # Load data
    with st.spinner("Loading data..."):
        data = load_data(data_source, symbol)
        if data is None:
            return
    
    # Process data
    with st.spinner("Processing data..."):
        processed_data = process_data(data, bubble_threshold, lookforward_window)
    
    # Train model
    with st.spinner("Training model..."):
        model, X, y = train_model(processed_data, model_type)
    
    # Make predictions
    predictions = model.predict(X)
    prediction_series = pd.Series(predictions, index=processed_data.index)
    
    # Calculate probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            bubble_probabilities = probabilities[:, 1]
        else:
            bubble_probabilities = probabilities.flatten()
    else:
        bubble_probabilities = predictions.astype(float)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "📊 Model Performance", "💰 Backtest Results", "🔍 Feature Analysis"])
    
    with tab1:
        st.subheader("Price Chart with Bubble Detection")
        
        # Create price chart
        price_chart = create_price_chart(processed_data, prediction_series)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Observations", len(processed_data))
        
        with col2:
            bubble_count = predictions.sum()
            st.metric("Bubbles Detected", bubble_count)
        
        with col3:
            bubble_rate = predictions.mean() * 100
            st.metric("Bubble Rate (%)", f"{bubble_rate:.2f}")
        
        with col4:
            avg_probability = bubble_probabilities.mean() * 100
            st.metric("Avg Bubble Probability (%)", f"{avg_probability:.2f}")
    
    with tab2:
        st.subheader("Model Performance Metrics")
        
        # Calculate evaluation metrics
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(y, predictions, bubble_probabilities)
        
        # Display metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Classification Metrics")
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        with col2:
            st.markdown("### Advanced Metrics")
            if 'auc' in metrics:
                st.metric("AUC", f"{metrics['auc']:.4f}")
            if 'average_precision' in metrics:
                st.metric("Average Precision", f"{metrics['average_precision']:.4f}")
            st.metric("Specificity", f"{metrics['specificity']:.4f}")
            st.metric("Sensitivity", f"{metrics['sensitivity']:.4f}")
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        cm_data = {
            'Predicted': ['No Bubble', 'Bubble'],
            'Actual No Bubble': [metrics['true_negatives'], metrics['false_positives']],
            'Actual Bubble': [metrics['false_negatives'], metrics['true_positives']]
        }
        cm_df = pd.DataFrame(cm_data)
        st.dataframe(cm_df, use_container_width=True)
    
    with tab3:
        st.subheader("Backtest Results")
        
        # Run backtest
        with st.spinner("Running backtest..."):
            backtest_config = BacktestConfig(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost
            )
            backtester = Backtester(backtest_config)
            result = backtester.run_backtest(processed_data, prediction_series)
        
        # Performance chart
        performance_chart = create_performance_chart(result.equity_curve)
        st.plotly_chart(performance_chart, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{result.metrics['total_return']:.2%}")
        
        with col2:
            st.metric("Annualized Return", f"{result.metrics['annualized_return']:.2%}")
        
        with col3:
            st.metric("Volatility", f"{result.metrics['volatility']:.2%}")
        
        with col4:
            st.metric("Sharpe Ratio", f"{result.metrics['sharpe_ratio']:.2f}")
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Max Drawdown", f"{result.metrics['max_drawdown']:.2%}")
        
        with col6:
            st.metric("Hit Rate", f"{result.metrics['hit_rate']:.2%}")
        
        with col7:
            st.metric("Avg Trade Return", f"{result.metrics['avg_trade_return']:.2%}")
        
        with col8:
            st.metric("Calmar Ratio", f"{result.metrics['calmar_ratio']:.2f}")
    
    with tab4:
        st.subheader("Feature Analysis")
        
        # Feature importance
        if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
            feature_importance = model.get_feature_importance()
            
            # Top 20 features
            top_features = feature_importance.head(20)
            
            # Create bar chart
            fig = px.bar(
                x=top_features.values,
                y=top_features.index,
                orientation='h',
                title="Top 20 Feature Importance",
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.markdown("### Feature Importance Table")
            st.dataframe(feature_importance.head(20), use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
        
        # Feature correlation heatmap
        st.markdown("### Feature Correlation Heatmap")
        numeric_features = processed_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = processed_data[numeric_features].corr()
        
        # Select only bubble-related features for heatmap
        bubble_features = [col for col in correlation_matrix.columns if 'bubble' in col.lower() or 'volatility' in col.lower() or 'momentum' in col.lower()]
        if bubble_features:
            bubble_corr = correlation_matrix.loc[bubble_features, bubble_features]
            
            fig = px.imshow(
                bubble_corr,
                title="Bubble-Related Features Correlation",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
