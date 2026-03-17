import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import sys
import os

# Solution: Add the project root to Python path
# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Find the project root by looking for the 'src' directory
project_root = current_dir
while project_root and 'src' not in os.listdir(project_root):
    parent = os.path.dirname(project_root)
    if parent == project_root:  # reached filesystem root
        break
    project_root = parent

# Add project root to sys.path
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added to Python path: {project_root}")

# Debug: Print current working directory and sys.path
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {current_dir}")
print(f"Project root: {project_root}")
print(f"Python path includes: {[p for p in sys.path if 'Indian_Stock_Price_Prediction' in p]}")

# Now the imports should work
from src.Indian_Stock_Price_Prediction.components.data_ingestion import DataIngestion
from src.Indian_Stock_Price_Prediction.components.data_transformation import transform_data
from src.Indian_Stock_Price_Prediction.pipelines.training_pipeline import train_pipeline

def predict_pipeline_using_training(
    ticker: str,
    prediction_date: str,
    lookback_days: int = 365,
    test_size: float = 0.2,
    prediction_days: int = 1,
    scale: bool = True,
    ensemble: bool = True
) -> Dict[str, Any]:
    """
    Build predictions by leveraging train_pipeline for training/evaluation,
    then recomputing the latest feature row for inference.
    """
    # Train and get trained models + performance
    train_out = train_pipeline(
        ticker=ticker,
        prediction_date=prediction_date,
        lookback_days=lookback_days,
        test_size=test_size,
        prediction_days=prediction_days,
        # scale=scale  # make sure your train_pipeline accepts this parameter
    )
    
    # Extract results from training pipeline
    trained_models = train_out["trained_models"]
    performance_df = train_out["performance"]
    feature_names = train_out["feature_names"]
    
    # Get additional training results if available
    preprocessor = train_out.get("preprocessor", None)
    best_params = train_out.get("best_params", {})
    
    # Fetch fresh data
    ingestion = DataIngestion()
    df = ingestion.fetch_stock_data(ticker=ticker, prediction_date=prediction_date, lookback_days=lookback_days)
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    current_price: Optional[float] = float(df[price_col].iloc[-1]) if price_col in df.columns else None
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    X_train, X_test, y_train, y_test, scaler, feat_names_2 = transform_data(
        train_df=train_df,
        test_df=test_df,
        target_col=None,
        prediction_days=prediction_days,
        # scale=scale
    )
    
    # Align with training feature order if needed
    if feature_names and list(feature_names) != list(feat_names_2):
        try:
            idx = [feat_names_2.index(f) for f in feature_names]
            X_test_aligned = X_test[:, idx]
        except Exception:
            X_test_aligned = X_test
    else:
        X_test_aligned = X_test
    
    # Generate test predictions for visualization
    test_predictions: Dict[str, np.ndarray] = {}
    for name, model in trained_models.items():
        try:
            test_preds = model.predict(X_test_aligned)
            test_predictions[name] = test_preds
        except Exception as e:
            print(f"Error generating test predictions for {name}: {str(e)}")
            test_predictions[name] = np.full(len(X_test_aligned), np.nan)
    
    # Predict next target using last available feature row
    last_feature_row = X_test_aligned[-1].reshape(1, -1)
    predictions: Dict[str, float] = {}
    
    for name, model in trained_models.items():
        try:
            predictions[name] = float(model.predict(last_feature_row)[0])
        except Exception:
            predictions[name] = np.nan
    
    ensemble_pred: Optional[float] = None
    if ensemble:
        valid = [v for v in predictions.values() if v is not None and np.isfinite(v)]
        ensemble_pred = float(np.mean(valid)) if valid else None
    
    # Return all expected keys
    return {
        # Next-step predictions
        "predictions": predictions,
        "ensemble_prediction": ensemble_pred,
        "current_price": current_price,
        
        # Training results (required by Streamlit app)
        "performance": performance_df,
        "trained_models": trained_models,  # ✅ Added this key
        "test_predictions": test_predictions,  # ✅ Added this key
        "feature_names": feature_names,
        "preprocessor": preprocessor if preprocessor is not None else scaler,  # ✅ Added this key
        "best_params": best_params,  # ✅ Added this key
        
        # Metadata
        "metadata": {
            "ticker": ticker,
            "prediction_date": prediction_date,
            "rows": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "prediction_days": int(prediction_days),
            "test_size": test_size,
            "lookback_days": lookback_days,
            "scaling_enabled": scale,
            "ensemble_enabled": ensemble
        }
    }

if __name__ == "__main__":
    # Example:
    out = predict_pipeline_using_training("RELIANCE.NS", "2025-08-01", 365, 0.2, prediction_days=1)
    print("Keys returned:", list(out.keys()))
    print("Ensemble:", out["ensemble_prediction"])
    print(pd.DataFrame.from_dict(out["predictions"], orient="index", columns=["next_target"]))