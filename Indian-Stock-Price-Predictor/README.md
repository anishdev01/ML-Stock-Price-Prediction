# 📈 Indian Stock Price Predictor

<div align="center">

**A production-ready Streamlit application that predicts Indian stock prices using supervised machine learning — powered by real-time NSE data from Yahoo Finance.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189AB4)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://indianstockpriceprediction.streamlit.app/)

</div>

---

## 🚀 Live Demo

🔗 **https://indian-stock-price-predictor.streamlit.app/**

---

## 📸 Screenshot

<img width="1853" height="805" alt="image" src="https://github.com/user-attachments/assets/98d7c142-af76-4ec9-8a19-867087664f6a" />


---

## 🌟 Overview

**Indian Stock Price Predictor** is an end-to-end machine learning application that fetches live historical data from the NSE (National Stock Exchange of India) via Yahoo Finance, engineers technical indicators, trains and benchmarks multiple regression models, and produces next-day price forecasts — all through an interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Technical Analysis** | Moving averages (SMA/EMA), RSI, MACD, Bollinger Bands, and more — rendered with Plotly |
| 🤖 **Multi-Model ML** | Trains 7 regression models simultaneously with hyperparameter tuning via `GridSearchCV` |
| 🔮 **Price Forecasting** | Next-step price prediction with ensemble voting across all trained models |
| 📉 **Backtesting View** | Actual vs. predicted overlay on the test window for visual validation |
| 📋 **Performance Table** | Ranks models by RMSE, MAE, and R² so you can instantly identify the best performer |
| ⚙️ **Configurable Pipeline** | Adjustable lookback window (60–2000 days), test split, prediction horizon, and feature scaling |
| 🏢 **15 Pre-loaded Stocks** | Includes Reliance, TCS, Infosys, HDFC Bank, ICICI, SBI, Airtel, ITC, and more |
| 🔡 **Custom NSE Symbol** | Enter any NSE-listed symbol (e.g., `ADANIPORTS.NS`) for instant analysis |
| 📝 **Structured Logging** | Full logging and custom exception handling throughout every pipeline step |

---

## 🤖 Machine Learning Models

The following models are trained, cross-validated with `TimeSeriesSplit`, and tuned using `GridSearchCV` on every run:

| # | Model | Hyperparameter Tuning |
|---|---|---|
| 1 | Linear Regression | — |
| 2 | Ridge Regression | `alpha` |
| 3 | Decision Tree Regressor | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| 4 | Random Forest Regressor | `n_estimators`, `max_depth`, `min_samples_split` |
| 5 | Gradient Boosting Regressor | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| 6 | AdaBoost Regressor | `n_estimators`, `learning_rate` |
| 7 | XGBoost Regressor | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` |

> Ensemble prediction averages the output of all trained models to reduce individual model variance.

---

## 🗂️ Project Structure

```
Indian_Stock_Price_Prediction/
│
├── app.py                              # 🎨 Streamlit dashboard entry point
├── main.py                             # Orchestration script
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
├── environment.yml                     # Conda environment config
├── Dockerfile                          # Container definition
│
└── src/
    └── Indian_Stock_Price_Prediction/
        │
        ├── components/
        │   ├── data_ingestion.py       # Fetches & cleans NSE stock data via yfinance
        │   ├── data_transformation.py  # Feature engineering & scaling pipeline
        │   └── model_trainer.py        # Model training, tuning & evaluation
        │
        ├── pipelines/
        │   ├── training_pipeline.py    # End-to-end training orchestration
        │   └── prediction_pipeline.py  # Reuses training steps + generates forecasts
        │
        ├── plots.py                    # Plotly chart builders (indicators, overlays, distributions)
        ├── utils.py                    # Shared helper utilities
        ├── logger.py                   # Centralised logging configuration
        └── exception.py               # Custom exception classes
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io/) |
| **Data Source** | [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance / NSE) |
| **ML Framework** | [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [CatBoost](https://catboost.ai/) |
| **Visualisation** | [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Environment** | Python 3.9+, Conda / venv |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` or `conda`
- Internet connection (live data is pulled from Yahoo Finance)

### 1. Clone the Repository

```bash
git clone https://github.com/PiyushAgarwalcs/Indian_Stock_Price_Prediction.git
cd Indian_Stock_Price_Prediction
```

### 2. Create a Virtual Environment

```bash
# Using pip + venv
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Or using Conda
conda env create -f environment.yml
conda activate stock_prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 How to Use

1. **Select a stock** from the sidebar dropdown (15 pre-loaded NSE stocks) or type any NSE ticker (e.g., `ADANIPORTS.NS`).
2. **Choose a prediction date** — up to 30 days ahead from today.
3. **Adjust advanced settings** (optional):
   - Historical lookback window (60–2000 days)
   - Train/test split ratio
   - Prediction horizon (1–5 days)
   - Enable/disable feature scaling and ensemble forecasting
4. Click **🚀 Run Analysis** to kick off the pipeline.
5. Explore results across three tabs:
   - **📊 Technical Analysis** — price charts + indicator dashboard
   - **🤖 Supervised ML** — model performance table, test overlay, and next-step forecasts
   - **📈 Summary** — quick-glance stats and a BUY / SELL / HOLD signal based on ensemble output

---

## 📉 Example Outputs

- Candlestick price chart with SMA, EMA, RSI, MACD, Bollinger Bands
- Model performance comparison table (RMSE, MAE, R², overall rank)
- Actual vs. predicted price overlay on the test window
- Next-day price forecast distribution across all models
- Ensemble prediction with directional recommendation (📈 BUY / 📉 SELL / ➖ HOLD)

---

## 🔮 Roadmap

- [ ] LSTM / Transformer-based deep learning models
- [ ] Sentiment analysis from financial news
- [ ] Multi-stock comparison dashboard
- [ ] Automated model retraining & caching
- [ ] Full Docker deployment + CI/CD pipeline
- [ ] Cloud hosting on Streamlit Community Cloud / AWS / GCP

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Kamalesh Sarkar**
📌 *Learning • Building • Growing*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/kamalesh-sarkar-341a48299)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/Kamaleshsarkar89189)

---

<div align="center">
⭐ If you found this project helpful, give it a star — it means a lot!
</div>
