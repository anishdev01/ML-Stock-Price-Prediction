# training_pipeline.py

import numpy as np
import pandas as pd
from typing import Dict, Any
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


from src.Indian_Stock_Price_Prediction.components.data_ingestion import DataIngestion  
from src.Indian_Stock_Price_Prediction.components.data_transformation import transform_data
from src.Indian_Stock_Price_Prediction.components.model_trainer import ModelTrainer

def train_pipeline(
	ticker: str,
	prediction_date: str,
	lookback_days: int = 365,
	test_size: float = 0.2,
	prediction_days: int = 1
) -> Dict[str, Any]:
	"""
	Full training pipeline:
	1) Fetch historical data
	2) Time-based train/test split
	3) Feature engineering + scaling
	4) Train all models and evaluate

	Returns:
	- {
		"performance": pd.DataFrame,
		"trained_models": Dict[str, estimator],
		"test_predictions": Dict[str, np.ndarray],
		"feature_names": list,
		"preprocessor": fitted ColumnTransformer,
		"metadata": { "rows": int, "train_rows": int, "test_rows": int }
	  }
	"""
	# 1) Fetch data using DataIngestion
	ingestion = DataIngestion()
	df = ingestion.fetch_stock_data(
		ticker=ticker,
		prediction_date=prediction_date,
		lookback_days=lookback_days
	)

	if df is None or df.empty:
		raise ValueError("No data returned from fetch_stock_data")

	# 2) Time-based split (no shuffle)
	if not 0 < test_size < 1:
		raise ValueError("test_size must be between 0 and 1")
	split_idx = int(len(df) * (1 - test_size))
	if split_idx <= 0 or split_idx >= len(df):
		raise ValueError("Invalid split index; adjust lookback_days or test_size")

	train_df = df.iloc[:split_idx].reset_index(drop=True)
	test_df = df.iloc[split_idx:].reset_index(drop=True)

	# 3) Transform (feature engineering + scaling)
	X_train, X_test, y_train, y_test, preprocessor, feature_names = transform_data(
		train_df=train_df,
		test_df=test_df,
		target_col=None,
		prediction_days=prediction_days
	)

	# 4) Train and evaluate all models
	trainer = ModelTrainer()
	performance_df, trained_models, test_predictions, best_params = trainer.train_with_splits(
		X_train, y_train, X_test, y_test
	)

	return {
		"performance": performance_df,
		"trained_models": trained_models,
		"test_predictions": test_predictions,
		"feature_names": feature_names,
		"preprocessor": preprocessor,
		"best_params": best_params,
		"metadata": {
			"rows": int(len(df)),
			"train_rows": int(len(train_df)),
			"test_rows": int(len(test_df))
		}
	}

if __name__ == "__main__":
	# Example usage:
	result = train_pipeline("RELIANCE.NS", "2025-08-01", 365, 0.2, prediction_days=1)
	print(result["performance"].head())
	train_out = train_pipeline("RELIANCE.NS", "2025-08-01", 365, 0.2, prediction_days=1)
	print("Keys returned:", list(train_out.keys()))
	print("Values types:", {k: type(v) for k, v in train_out.items()})
	pass
