# data_transformation.py

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def _ensure_price_column(df: pd.DataFrame) -> str:
	"""
	Ensure we have a price column to derive features from.
	Returns the chosen price column name ('adj_close' preferred).
	"""
	if 'adj_close' in df.columns:
		return 'adj_close'
	if 'close' in df.columns:
		return 'close'
	# Fallback: try to find anything that looks like close/price
	candidates = [c for c in df.columns if any(k in c.lower() for k in ['adj close', 'adj_close', 'close', 'price', 'last'])]
	if not candidates:
		raise ValueError("No price-like column found. Expected one of ['adj_close','close','price'].")
	return candidates[0]

def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Add common technical indicators (SMA/EMA/MACD/RSI/Bollinger/returns/volatility/volume features).
	"""
	df = df.copy()
	price_col = _ensure_price_column(df)
	if price_col == 'close' and 'adj_close' not in df.columns:
		df['adj_close'] = df['close']
		price_col = 'adj_close'

	# Daily returns
	df['daily_return'] = df[price_col].pct_change()

	# SMA
	df['sma_5'] = df[price_col].rolling(window=5, min_periods=1).mean()
	df['sma_10'] = df[price_col].rolling(window=10, min_periods=1).mean()
	df['sma_20'] = df[price_col].rolling(window=20, min_periods=1).mean()

	# EMA
	df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
	df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()

	# MACD
	df['macd'] = df['ema_12'] - df['ema_26']
	df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
	df['macd_histogram'] = df['macd'] - df['macd_signal']

	# RSI
	delta = df[price_col].diff()
	gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
	loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
	rs = gain / loss.replace(0, np.nan)
	df['rsi'] = 100 - (100 / (1 + rs))
	df['rsi'] = df['rsi'].fillna(50)

	# Bollinger Bands
	df['bb_middle'] = df[price_col].rolling(window=20, min_periods=1).mean()
	bb_std = df[price_col].rolling(window=20, min_periods=1).std()
	df['bb_upper'] = df['bb_middle'] + 2 * bb_std
	df['bb_lower'] = df['bb_middle'] - 2 * bb_std
	df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

	# Momentum and price changes
	df['momentum_10'] = df[price_col] / df[price_col].shift(10) - 1
	df['price_change_1d'] = df[price_col].pct_change(1)
	df['price_change_5d'] = df[price_col].pct_change(5)

	# Volume features
	if 'volume' in df.columns:
		df['volume_sma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
		df['volume_ratio'] = df['volume'] / df['volume_sma_10']
		# OBV
		df['obv'] = 0.0
		for i in range(1, len(df)):
			if df[price_col].iloc[i] > df[price_col].iloc[i - 1]:
				df.loc[df.index[i], 'obv'] = df['obv'].iloc[i - 1] + df['volume'].iloc[i]
			elif df[price_col].iloc[i] < df[price_col].iloc[i - 1]:
				df.loc[df.index[i], 'obv'] = df['obv'].iloc[i - 1] - df['volume'].iloc[i]
			else:
				df.loc[df.index[i], 'obv'] = df['obv'].iloc[i - 1]

	# Volatility and range
	df['volatility_20'] = df['daily_return'].rolling(window=20, min_periods=1).std()
	if {'high', 'low'}.issubset(df.columns):
		df['price_range'] = df['high'] - df['low']

	# Support/Resistance (rolling)
	df['support_20'] = df[price_col].rolling(window=20, min_periods=1).min()
	df['resistance_20'] = df[price_col].rolling(window=20, min_periods=1).max()

	return df

def _add_lag_trend_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
	"""
	Add lag features and trend labels based on the target_col.
	"""
	df = df.copy()
	for lag in [1, 2, 3, 5, 10]:
		df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
		if 'volume' in df.columns:
			df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

	if 'sma_5' in df.columns and 'sma_20' in df.columns:
		df['ma_ratio_5_20'] = df['sma_5'] / df['sma_20']
	if 'sma_20' in df.columns:
		df['price_vs_sma20'] = df[target_col] / df['sma_20'] - 1

	df['trend_5d'] = np.where(df[target_col] > df[target_col].shift(5), 1, 0)
	df['trend_10d'] = np.where(df[target_col] > df[target_col].shift(10), 1, 0)
	return df

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Basic cleaning: remove fully empty rows and clip extreme outliers.
	"""
	df = df.copy()
	df = df.dropna(how='all')
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	if len(numeric_cols) > 0:
		df[numeric_cols] = df[numeric_cols].ffill().bfill()
	for col in [c for c in ['daily_return', 'volume_ratio'] if c in df.columns]:
		mean_val = df[col].mean()
		std_val = df[col].std()
		df[col] = df[col].clip(lower=mean_val - 3 * std_val, upper=mean_val + 3 * std_val)
	return df

def _build_supervised_xy(df: pd.DataFrame, target_col: str, prediction_days: int) -> Tuple[pd.DataFrame, pd.Series]:
	"""
	Build supervised matrices (features X and target y).
	"""
	df = df.copy()
	df['target'] = df[target_col].shift(-prediction_days)
	df = df.dropna(subset=['target'])

	feature_cols = [c for c in df.columns if c not in ['target', 'date', target_col]]
	X = df[feature_cols].copy()
	y = df['target'].copy()

	valid_mask = ~(X.isna().any(axis=1) | y.isna())
	X = X[valid_mask]
	y = y[valid_mask]
	return X, y

def _build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
	"""
	Build a ColumnTransformer with:
	- Numeric: SimpleImputer(median) + StandardScaler()
	- Categorical (if any): SimpleImputer(most_frequent) + OneHotEncoder + StandardScaler(with_mean=False)
	"""
	categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
	numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

	num_pipeline = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler())
	])
	cat_pipeline = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
		("scaler", StandardScaler(with_mean=False))
	])

	transformers = []
	if numeric_cols:
		transformers.append(("num_pipeline", num_pipeline, numeric_cols))
	if categorical_cols:
		transformers.append(("cat_pipeline", cat_pipeline, categorical_cols))

	if not transformers:
		# Edge case: no valid columns; create a passthrough to avoid errors
		return ColumnTransformer([], remainder='drop')

	return ColumnTransformer(transformers, remainder='drop')

def transform_data(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
	target_col: Optional[str] = None,
	prediction_days: int = 1
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, Any, List[str]]:
	"""
	Complete data transformation:
	- Feature engineering on concatenated (train+test) for consistent rolling/lag context
	- Supervised target creation with prediction_days horizon
	- Scaling via ColumnTransformer (fit on train, transform test) with imputation
	- Returns transformed arrays and y splits. No saving.

	Returns:
	- X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, feature_names_out
	"""
	if not isinstance(train_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
		raise ValueError("train_df and test_df must be pandas DataFrames")

	# Concatenate to compute rolling/lags consistently
	train_len = len(train_df)
	df_all = pd.concat([train_df.copy(), test_df.copy()], axis=0, ignore_index=True)

	# Ensure proper time order if date exists
	if 'date' in df_all.columns:
		df_all['date'] = pd.to_datetime(df_all['date'])
		df_all = df_all.sort_values('date').reset_index(drop=True)

	# Determine target column
	if target_col is None:
		target_col = 'adj_close' if 'adj_close' in df_all.columns else 'close'
		if target_col not in df_all.columns:
			target_col = _ensure_price_column(df_all)

	# Feature engineering
	df_all = _add_technical_indicators(df_all)
	df_all = _add_lag_trend_features(df_all, target_col=target_col)
	df_all = _clean_data(df_all)

	# Split back
	df_train = df_all.iloc[:train_len].copy()
	df_test = df_all.iloc[train_len:].copy()

	# Supervised matrices
	X_train_raw, y_train = _build_supervised_xy(df_train, target_col=target_col, prediction_days=prediction_days)
	X_test_raw, y_test = _build_supervised_xy(df_test, target_col=target_col, prediction_days=prediction_days)

	# Align feature columns
	common_cols = sorted(list(set(X_train_raw.columns) & set(X_test_raw.columns)))
	X_train_raw = X_train_raw[common_cols].copy()
	X_test_raw = X_test_raw[common_cols].copy()

	# Build and fit preprocessor (imputation + scaling; plus encoding if categoricals exist)
	preprocessor = _build_preprocessor(X_train_raw)
	X_train_transformed = preprocessor.fit_transform(X_train_raw)
	X_test_transformed = preprocessor.transform(X_test_raw)

	# Try to extract feature names after transformation
	feature_names_out: List[str] = []
	try:
		# sklearn >= 1.0
		feature_names_out = preprocessor.get_feature_names_out().tolist()  # type: ignore[attr-defined]
	except Exception:
		# Fallback: use raw column names; encoded columns will be expanded unnamed
		feature_names_out = common_cols

	return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, feature_names_out

# if __name__ == "__main__":
# 	# Example (requires train_df, test_df from your ingestion step)
# 	# Xtr, Xte, ytr, yte, preproc, feat_names = transform_data(train_df, test_df, prediction_days=1)
# 	pass



if __name__ == "__main__":
	from data_ingestion import DataIngestion

	ingestion = DataIngestion()
	result = ingestion.get_data_with_split(
		ticker='TCS.NS',
		prediction_date='2024-01-20',
		lookback_days=365,
		test_size=0.2
	)

	X_train_df = result['X_train']
	X_test_df = result['X_test']
	y_train = result['y_train']
	y_test = result['y_test']

	X_train_transformed, X_test_transformed, y_train_out, y_test_out, preprocessor, feat_names = transform_data(
		X_train_df, X_test_df, target_col='adj_close', prediction_days=1
	)

	print("Feature matrix shapes:")
	print("Train:", X_train_transformed.shape)
	print("Test:", X_test_transformed.shape)
