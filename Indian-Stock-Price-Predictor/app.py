import streamlit as st
import numpy as np
from datetime import date, datetime, timedelta
import sys
import traceback

# Correct place for set_page_config
st.set_page_config(page_title="Indian Stock Price Predictor", page_icon="📈", layout="wide")

# Now add src folder to sys.path and import your own modules
import pathlib
src_path = pathlib.Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from src.Indian_Stock_Price_Prediction.components.data_ingestion import DataIngestion
from src.Indian_Stock_Price_Prediction.pipelines.prediction_pipeline import predict_pipeline_using_training
from src.Indian_Stock_Price_Prediction.plots import (
    plot_price_with_indicators,
    plot_technical_dashboard,
    plot_model_performance,
    plot_predictions_overlay,
    create_performance_table,
    plot_prediction_distribution,
    create_predictions_table,
)
# Header
st.markdown('<h1 style="text-align:center;">📈 Indian Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("Predict future stock prices using supervised machine learning models and interactive technical analysis.")

# Sidebar – Configuration
st.sidebar.header("🔧 Configuration")

indian_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Larsen & Toubro": "LT.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Titan Company": "TITAN.NS",
    "Wipro": "WIPRO.NS",
}
selected_stock_name = st.sidebar.selectbox("🏢 Select Indian Stock", list(indian_stocks.keys()))
selected_stock = indian_stocks[selected_stock_name]

custom_stock = st.sidebar.text_input("Or enter custom NSE symbol (e.g., ADANIPORTS.NS)", placeholder="STOCKNAME.NS")
if custom_stock:
    selected_stock = custom_stock.upper()
    selected_stock_name = custom_stock

max_date = date.today() + timedelta(days=30)
prediction_date = st.sidebar.date_input(
    "📅 Select prediction date",
    min_value=date.today(),
    max_value=max_date,
    value=date.today() + timedelta(days=1)
)

st.sidebar.subheader("🤖 Options")
show_tech = st.sidebar.checkbox("Technical Analysis", value=True)
show_supervised = st.sidebar.checkbox("Supervised ML", value=True)

with st.sidebar.expander("⚙️ Advanced Settings"):
    lookback_days = st.slider("Historical data days", 60, 2000, 365)
    test_size = st.slider("Test size fraction", 0.1, 0.4, 0.2, step=0.05)
    prediction_days = st.number_input("Prediction horizon (days ahead)", min_value=1, max_value=5, value=1)
    scale_features = st.checkbox("Scale features", value=True)
    use_ensemble = st.checkbox("Show ensemble prediction", value=True)

run_button = st.sidebar.button("🚀 Run Analysis", type="primary")

if run_button:
    try:
        # 1) Fetch data
        with st.spinner(f"Fetching data for {selected_stock_name}..."):
            ingestion = DataIngestion()
            df = ingestion.fetch_stock_data(selected_stock, str(prediction_date), lookback_days)
            if df is None or df.empty:
                st.error("No data found. Try another symbol or increase lookback days.")
                st.stop()
            price_col = "adj_close" if "adj_close" in df.columns else "close"
            current_price = float(df[price_col].iloc[-1]) if price_col in df.columns else None

        st.success(f"✅ Data fetched. Current price: ₹{current_price:.2f}" if current_price else "✅ Data fetched.")

        # 2) Tabs
        tab_names = []
        if show_tech:
            tab_names.append("📊 Technical Analysis")
        if show_supervised:
            tab_names.append("🤖 Supervised ML")
        tab_names.append("📈 Summary")
        tabs = st.tabs(tab_names)

        # 3) Technical Analysis
        if show_tech:
            with tabs[tab_names.index("📊 Technical Analysis")]:
                st.header("📊 Technical Analysis")

                # Price + indicators
                fig_price = plot_price_with_indicators(df, title=f"{selected_stock_name} — Price & Key Indicators")
                st.plotly_chart(fig_price, use_container_width=True)

                # Technical dashboard
                fig_dash = plot_technical_dashboard(df, title=f"{selected_stock_name} — Technical Dashboard")
                st.plotly_chart(fig_dash, use_container_width=True)

                # Recent data preview
                st.subheader("📋 Recent Data (last 15 rows)")
                preview_cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'] if c in df.columns]
                st.dataframe(df[preview_cols].tail(15), use_container_width=True)

        # 4) Supervised ML (training + predictions)
        performance_df = None
        trained_models = None
        test_predictions = None
        feature_names = None
        test_start_index = None
        y_test = None
        predictions_next = None
        ensemble_pred = None

        if show_supervised:
            with tabs[tab_names.index("🤖 Supervised ML")]:
                st.header("🤖 Supervised Machine Learning Models")
                with st.spinner("Training models and generating predictions..."):
                    # Use the pipeline that reuses training steps and adds next-step predictions
                    out = predict_pipeline_using_training(
                        ticker=selected_stock,
                        prediction_date=str(prediction_date),
                        lookback_days=lookback_days,
                        test_size=test_size,
                        prediction_days=prediction_days,
                        scale=scale_features,
                        ensemble=use_ensemble
                    )

                    performance_df = out["performance"]
                    trained_models = out["trained_models"]
                    test_predictions = out["test_predictions"]
                    feature_names = out["feature_names"]
                    ensemble_pred = out.get("ensemble_prediction", None)
                    predictions_next = out.get("predictions", None)

                # Performance table
                if performance_df is not None and not performance_df.empty:
                    st.subheader("📊 Model Performance (Test)")
                    st.dataframe(create_performance_table(performance_df), use_container_width=True)
                    fig_perf = plot_model_performance(performance_df)
                    st.plotly_chart(fig_perf, use_container_width=True)
                else:
                    st.warning("No performance results to display.")

                # Overlay predictions on price (test segment)
                split_idx = int(len(df) * (1 - test_size))
                test_start_index = split_idx

                if test_predictions:
                    any_preds = next(iter(test_predictions.values()))
                    if any_preds is not None:
                        test_len = len(any_preds)
                        if test_len > 0 and test_start_index + test_len <= len(df):
                            st.subheader("🧩 Predictions Overlay on Price (Test Window)")
                            fig_overlay = plot_predictions_overlay(
                                df=df,
                                predictions=test_predictions,
                                y_test=None,  # Optional if you have ground truth
                                test_start_index=test_start_index,
                                title=f"{selected_stock_name} — Test Predictions Overlay"
                            )
                            st.plotly_chart(fig_overlay, use_container_width=True)

                # Next-step predictions summary (ensemble + table)
                if predictions_next:
                    st.subheader("🔮 Next-Step Predictions")
                    if use_ensemble and ensemble_pred is not None and np.isfinite(ensemble_pred):
                        st.metric("Ensemble Prediction", f"₹{ensemble_pred:.2f}")
                    if current_price is not None:
                        try:
                            st.dataframe(create_predictions_table(predictions_next, current_price), use_container_width=True)
                        except Exception:
                            pass
                        try:
                            fig_dist = plot_prediction_distribution(predictions_next, current_price=current_price)
                            st.plotly_chart(fig_dist, use_container_width=True)
                        except Exception:
                            pass

        # 5) Summary
        with tabs[-1]:
            st.header("📈 Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Current Status")
                st.write(f"**Stock:** {selected_stock_name}")
                st.write(f"**Symbol:** {selected_stock}")
                if 'current_price' in locals() and current_price:
                    st.write(f"**Current Price:** ₹{current_price:.2f}")
                st.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Prediction Target:** {prediction_date}")
                st.write(f"**Data Points:** {len(df)} days")
            with col2:
                if show_supervised and performance_df is not None and not performance_df.empty:
                    st.subheader("🎯 ML Summary")
                    best_model = performance_df['RMSE'].idxmin()
                    st.write(f"**Best Model (by RMSE):** {best_model}")
                    st.write(f"**Best RMSE:** {performance_df.loc[best_model, 'RMSE']:.3f}")
                    st.write(f"**Models Trained:** {len(performance_df)}")
                    if use_ensemble and ensemble_pred is not None and np.isfinite(ensemble_pred) and current_price is not None:
                        diff = ensemble_pred - current_price
                        diff_pct = diff / current_price * 100
                        reco = "BUY 📈" if diff > 0 else "SELL 📉" if diff < 0 else "HOLD ➖"
                        st.write(f"**Ensemble:** ₹{ensemble_pred:.2f} ({diff:+.2f}, {diff_pct:+.2f}%)")
                        st.write(f"**Recommendation:** {reco}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        with st.expander("Details"):
            st.code(traceback.format_exc())

else:
    st.info("Select a stock, adjust options if needed, and click '🚀 Run Analysis'.")

