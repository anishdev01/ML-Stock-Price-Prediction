# src/Indian_Stock_Price_predictor/plots.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional


def _validate_cols(df: pd.DataFrame, cols: List[str], name: str):
    if df is None or df.empty:
        raise ValueError(f"{name}: DataFrame is empty or None")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: Missing required columns: {missing}")


# 1) Original stock price with important features (volume, MA, BB, RSI areas)
def plot_price_with_indicators(
    df: pd.DataFrame,
    title: str = "Price with Key Indicators"
) -> go.Figure:
    """
    Interactive multi-panel chart:
    - Panel 1: Close price + SMA(20), SMA(50) + Bollinger Bands (20, 2)
    - Panel 2: Volume + Volume SMA(10)
    - Panel 3: RSI(14) with 30/70 lines
    """
    _validate_cols(df, ['date', 'close', 'volume'], "plot_price_with_indicators")

    # Compute missing but useful indicators on-the-fly (non-destructive)
    work = df.copy()
    if 'sma_20' not in work.columns:
        work['sma_20'] = work['close'].rolling(20, min_periods=1).mean()
    if 'sma_50' not in work.columns:
        work['sma_50'] = work['close'].rolling(50, min_periods=1).mean()
    if 'bb_upper' not in work.columns or 'bb_lower' not in work.columns:
        bb_mid = work['close'].rolling(20, min_periods=1).mean()
        bb_std = work['close'].rolling(20, min_periods=1).std()
        work['bb_upper'] = bb_mid + 2 * bb_std
        work['bb_lower'] = bb_mid - 2 * bb_std
    if 'volume_sma_10' not in work.columns and 'volume' in work.columns:
        work['volume_sma_10'] = work['volume'].rolling(10, min_periods=1).mean()
    if 'rsi' not in work.columns:
        delta = work['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        work['rsi'] = 100 - (100 / (1 + rs))
        work['rsi'] = work['rsi'].fillna(50)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=("Price + Moving Averages + Bollinger Bands", "Volume", "RSI(14)"),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Panel 1: Price and indicators
    fig.add_trace(go.Scatter(x=work['date'], y=work['close'], name='Close', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=work['date'], y=work['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=work['date'], y=work['sma_50'], name='SMA 50', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=work['date'], y=work['bb_upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=work['date'], y=work['bb_lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)

    # Panel 2: Volume
    fig.add_trace(go.Bar(x=work['date'], y=work['volume'], name='Volume', marker_color='rgba(0,123,255,0.4)'), row=2, col=1)
    if 'volume_sma_10' in work.columns:
        fig.add_trace(go.Scatter(x=work['date'], y=work['volume_sma_10'], name='Vol SMA 10', line=dict(color='red')), row=2, col=1)

    # Panel 3: RSI
    fig.add_trace(go.Scatter(x=work['date'], y=work['rsi'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    fig.update_layout(
        title=title, height=850, hovermode='x unified', showlegend=True
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig


# 2) Technical analysis dashboard (returns, MACD, ATR, correlation heatmap)
def plot_technical_dashboard(
    df: pd.DataFrame,
    title: str = "Technical Analysis Dashboard"
) -> go.Figure:
    """
    4-panel technical dashboard:
    - Panel 1: Daily returns (%)
    - Panel 2: MACD + Signal + Histogram
    - Panel 3: ATR(14) proxy: true range avg
    - Panel 4: Correlation heatmap of numeric features
    """
    _validate_cols(df, ['date', 'close'], "plot_technical_dashboard")
    work = df.copy()

    # Returns
    if 'daily_return' not in work.columns:
        work['daily_return'] = work['close'].pct_change()

    # MACD (12, 26, 9) if not present
    if 'macd' not in work.columns or 'macd_signal' not in work.columns:
        ema12 = work['close'].ewm(span=12, adjust=False).mean()
        ema26 = work['close'].ewm(span=26, adjust=False).mean()
        work['macd'] = ema12 - ema26
        work['macd_signal'] = work['macd'].ewm(span=9, adjust=False).mean()
    if 'macd_histogram' not in work.columns:
        work['macd_histogram'] = work['macd'] - work['macd_signal']

    # ATR(14) proxy
    if all(c in work.columns for c in ['high', 'low', 'close']):
        tr1 = work['high'] - work['low']
        tr2 = (work['high'] - work['close'].shift(1)).abs()
        tr3 = (work['low'] - work['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        work['atr_14'] = tr.rolling(14, min_periods=1).mean()
    else:
        work['atr_14'] = np.nan

    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("Daily Returns (%)", "MACD", "ATR(14)", "Correlation Heatmap"),
        vertical_spacing=0.12, horizontal_spacing=0.1
    )

    # Panel 1: Returns
    fig.add_trace(
        go.Bar(x=work['date'], y=(work['daily_return'] * 100), name='Daily Return %', marker_color='teal'),
        row=1, col=1
    )

    # Panel 2: MACD
    fig.add_trace(go.Scatter(x=work['date'], y=work['macd'], name='MACD', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=work['date'], y=work['macd_signal'], name='Signal', line=dict(color='orange')), row=1, col=2)
    fig.add_trace(go.Bar(x=work['date'], y=work['macd_histogram'], name='Histogram', marker_color='gray'), row=1, col=2)

    # Panel 3: ATR
    fig.add_trace(go.Scatter(x=work['date'], y=work['atr_14'], name='ATR(14)', line=dict(color='purple')), row=2, col=1)

    # Panel 4: Correlation heatmap for numeric cols
    num_df = work.select_dtypes(include=[np.number]).copy()
    if not num_df.empty:
        corr = num_df.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0,
                text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size": 9}
            ),
            row=2, col=2
        )

    fig.update_layout(title=title, height=900, showlegend=False)
    return fig


# 3) Plots and tables for model results
def plot_model_performance(performance_df: pd.DataFrame) -> go.Figure:
    """
    Bar charts for RMSE, MAE, R2, MSE across models.
    performance_df must contain columns: ['RMSE','MAE','R2','MSE'] and be indexed by model name.
    """
    if performance_df is None or performance_df.empty:
        raise ValueError("plot_model_performance: Results DataFrame is empty or None")
    required = ['RMSE', 'MAE', 'R2', 'MSE']
    missing = [c for c in required if c not in performance_df.columns]
    if missing:
        raise ValueError(f"plot_model_performance: Missing required metrics: {missing}")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RMSE (Lower Better)', 'MAE (Lower Better)', 'R² (Higher Better)', 'MSE (Lower Better)'),
        vertical_spacing=0.15
    )
    models = performance_df.index.astype(str)

    fig.add_trace(go.Bar(x=models, y=performance_df['RMSE'], name='RMSE', marker_color='indianred',
                         text=performance_df['RMSE'].round(3), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=performance_df['MAE'], name='MAE', marker_color='steelblue',
                         text=performance_df['MAE'].round(3), textposition='auto'), row=1, col=2)
    fig.add_trace(go.Bar(x=models, y=performance_df['R2'], name='R²', marker_color='seagreen',
                         text=performance_df['R2'].round(3), textposition='auto'), row=2, col=1)
    fig.add_trace(go.Bar(x=models, y=performance_df['MSE'], name='MSE', marker_color='goldenrod',
                         text=performance_df['MSE'].round(3), textposition='auto'), row=2, col=2)

    fig.update_layout(title="Model Performance Comparison", height=650, showlegend=False)
    fig.update_xaxes(tickangle=45)
    return fig


def create_performance_table(performance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a neatly rounded table for Streamlit display.
    """
    if performance_df is None or performance_df.empty:
        raise ValueError("create_performance_table: Results DataFrame is empty or None")
    table = performance_df.copy()
    for c in ['RMSE', 'MAE', 'R2', 'MSE']:
        if c in table.columns:
            table[c] = table[c].astype(float).round(4)
    if 'overall_rank' in table.columns:
        table['overall_rank'] = table['overall_rank'].astype(float).round(2)
    return table


def plot_feature_importance(
    importance_by_model: Dict[str, pd.DataFrame],
    top_n: int = 20,
    title: str = "Feature Importance (Tree-based Models)"
) -> go.Figure:
    """
    importance_by_model: dict of model_name -> DataFrame with columns ['feature','importance']
    Produces a horizontal bar chart for top features (aggregated or first available model).
    """
    if not importance_by_model:
        raise ValueError("plot_feature_importance: No importance data provided")

    # If multiple models, average importance across them (by feature)
    merged = None
    for model, df_imp in importance_by_model.items():
        if df_imp is None or df_imp.empty or 'feature' not in df_imp.columns or 'importance' not in df_imp.columns:
            continue
        temp = df_imp[['feature', 'importance']].copy()
        temp = temp.groupby('feature', as_index=False)['importance'].mean()
        merged = temp if merged is None else merged.merge(temp, on='feature', how='outer', suffixes=('', f'_{model}'))

    if merged is None or merged.empty:
        raise ValueError("plot_feature_importance: Could not aggregate importance")

    # Average across columns
    imp_cols = [c for c in merged.columns if c != 'feature']
    merged['avg_importance'] = merged[imp_cols].mean(axis=1, skipna=True)
    merged = merged.sort_values('avg_importance', ascending=False).head(top_n)

    fig = go.Figure(go.Bar(
        x=merged['avg_importance'][::-1],
        y=merged['feature'][::-1],
        orientation='h',
        marker_color='teal',
        text=merged['avg_importance'].round(3)[::-1],
        textposition='auto',
        name='Avg Importance'
    ))
    fig.update_layout(title=title, height=min(600, 30 * len(merged) + 150))
    return fig


# 4) Overlay model predictions on original stock graph
def plot_predictions_overlay(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    y_test: Optional[np.ndarray] = None,
    test_start_index: Optional[int] = None,
    title: str = "Predictions Overlay on Price"
) -> go.Figure:
    """
    Overlay each model’s test predictions (aligned to test segment) on top of the close price.
    - df: must have ['date','close']
    - predictions: dict model_name -> y_pred array (length == len(y_test))
    - y_test: actual test targets to plot alongside
    - test_start_index: starting row index in df where test set begins (for alignment)
    """
    _validate_cols(df, ['date', 'close'], "plot_predictions_overlay")
    if y_test is not None and test_start_index is None:
        raise ValueError("plot_predictions_overlay: Provide test_start_index to align y_test/predictions on the timeline")

    fig = go.Figure()
    # Full close price baseline
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Close', line=dict(color='black')))

    if y_test is not None:
        # Align y_test to dates
        dates_test = df['date'].iloc[test_start_index:test_start_index + len(y_test)]
        fig.add_trace(go.Scatter(x=dates_test, y=y_test, name='Actual (Test Target)', line=dict(color='blue')))

    # Add each model prediction aligned to test dates
    if predictions:
        colors = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
        for i, (model, preds) in enumerate(predictions.items()):
            if preds is None:
                continue
            if y_test is not None and len(preds) != len(y_test):
                continue
            if y_test is not None:
                dates_pred = dates_test
            else:
                # fallback: align to last len(preds) points
                dates_pred = df['date'].iloc[-len(preds):]
            fig.add_trace(go.Scatter(
                x=dates_pred, y=preds, name=f'{model} (Pred)', line=dict(color=colors[i % len(colors)], dash='dash')
            ))

    fig.update_layout(
        title=title, height=600, hovermode='x unified', showlegend=True,
        xaxis_title="Date", yaxis_title="Price"
    )
    return fig


# Other highly useful frontend plots

def plot_residuals(
    y_true: np.ndarray,
    y_pred: Dict[str, np.ndarray],
    title: str = "Residual Analysis (Test Set)"
) -> go.Figure:
    """
    Residual plot per model (bars): residual = y_true - y_pred
    """
    if y_true is None or len(y_true) == 0:
        raise ValueError("plot_residuals: y_true is empty")
    if not y_pred:
        raise ValueError("plot_residuals: y_pred is empty")

    fig = make_subplots(rows=1, cols=1, subplot_titles=("Residuals per Model",))
    x_vals = list(range(len(y_true)))
    colors = px.colors.qualitative.Safe + px.colors.qualitative.D3

    for i, (model, preds) in enumerate(y_pred.items()):
        if preds is None or len(preds) != len(y_true):
            continue
        residuals = y_true - preds
        fig.add_trace(go.Bar(x=x_vals, y=residuals, name=model, marker_color=colors[i % len(colors)]))

    fig.update_layout(title=title, height=450, barmode='group', hovermode='x unified', showlegend=True)
    fig.update_xaxes(title_text="Test Index")
    fig.update_yaxes(title_text="Residual (Actual - Predicted)")
    return fig


def plot_prediction_distribution(
    predictions: Dict[str, float],
    current_price: Optional[float] = None,
    title: str = "Next-Step Prediction Distribution"
) -> go.Figure:
    """
    Shows a distribution of single-step predictions across models (as a strip/box).
    """
    vals = [v for v in predictions.values() if v is not None and np.isfinite(v)]
    models = [k for k, v in predictions.items() if v is not None and np.isfinite(v)]
    if not vals:
        raise ValueError("plot_prediction_distribution: No valid predictions")

    dfp = pd.DataFrame({"model": models, "prediction": vals})
    fig = px.strip(dfp, x="model", y="prediction", title=title)
    fig.update_traces(jitter=0.3, marker=dict(size=10, color='teal'))
    if current_price is not None:
        fig.add_hline(y=current_price, line_dash="dash", line_color="red", annotation_text="Current Price", annotation_position="top left")
    fig.update_layout(height=450)
    return fig


def create_predictions_table(
    predictions: Dict[str, float],
    current_price: float
) -> pd.DataFrame:
    """
    Returns a summary table for Streamlit display (per-model next prediction with deltas).
    """
    if not predictions:
        raise ValueError("create_predictions_table: predictions empty")
    if current_price is None or not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("create_predictions_table: invalid current_price")

    rows = []
    for model, pred in predictions.items():
        if pred is None or not np.isfinite(pred):
            continue
        change = pred - current_price
        change_pct = (change / current_price) * 100
        rows.append({
            "Model": model,
            "Predicted": round(float(pred), 3),
            "Δ": round(float(change), 3),
            "Δ%": f"{change_pct:.2f}%"
        })
    return pd.DataFrame(rows).sort_values("Predicted")