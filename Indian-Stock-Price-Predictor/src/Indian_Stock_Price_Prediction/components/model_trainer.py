# model_trainer.py

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
	RandomForestRegressor,
	GradientBoostingRegressor,
	AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Optional XGBoost (present in your original reference)
try:
	from xgboost import XGBRegressor
	_HAS_XGB = True
except Exception:
	_HAS_XGB = False


@dataclass
class ModelTrainerConfig:
	random_state: int = 42
	cv_splits: int = 5
	n_jobs: int = -1
	scoring: str = "r2"  # maximize R2


class ModelTrainer:
	def __init__(self):
		self.config = ModelTrainerConfig()

	def eval_metrics(self, actual, pred):
		rmse = np.sqrt(mean_squared_error(actual, pred))
		mae = mean_absolute_error(actual, pred)
		r2 = r2_score(actual, pred)
		mse = rmse**2
		return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2), "MSE": float(mse)}

	def _get_models(self):
		models = {
			"Linear Regression": LinearRegression(),
			"Ridge Regression": Ridge(random_state=self.config.random_state),
			"Decision Tree": DecisionTreeRegressor(random_state=self.config.random_state),
			"Random Forest": RandomForestRegressor(
				random_state=self.config.random_state, n_jobs=self.config.n_jobs
			),
			"Gradient Boosting": GradientBoostingRegressor(
				random_state=self.config.random_state
			),
			"AdaBoost Regressor": AdaBoostRegressor(
				random_state=self.config.random_state
			),
		}
		if _HAS_XGB:
			models["XGBRegressor"] = XGBRegressor(
				random_state=self.config.random_state,
				n_jobs=self.config.n_jobs,
				verbosity=0,
			)
		return models

	def _get_param_spaces(self):
		"""Optimized parameter grids for GridSearchCV - balanced between thoroughness and speed"""
		params = {
			"Decision Tree": {
				"max_depth": [None, 5, 10],
				"min_samples_split": [2, 5, 10],
				"min_samples_leaf": [1, 2, 4],
			},
			"Random Forest": {
				"n_estimators": [100, 200, 300],
				"max_depth": [None, 10, 15],
				"min_samples_split": [2, 5],
				"min_samples_leaf": [1, 2],
			},
			"Gradient Boosting": {
				"n_estimators": [200],
				"learning_rate": [0.01],
				"max_depth": [3, 5], 
				"subsample": [0.8, 0.9],
			},
			"Linear Regression": {},
			"Ridge Regression": {
				"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
			},
			"AdaBoost Regressor": {
				"n_estimators": [50, 100, 200],
				"learning_rate": [0.01, 0.1],
			},
		}
		if _HAS_XGB:
			params["XGBRegressor"] = {
				"n_estimators": [100, 200],
				"learning_rate": [0.1, 0.2],
				"max_depth": [3, 4, 6],
				"subsample": [0.8, 1.0],
				"colsample_bytree": [0.8, 1.0],
			}
		return params

	def _fit_all_and_evaluate(self, X_train, y_train, X_test, y_test):
		models = self._get_models()
		param_spaces = self._get_param_spaces()
		tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)

		perf_rows = []
		trained_models = {}
		test_predictions = {}
		best_params = {}

		print("Using GridSearchCV for hyperparameter tuning...")

		for name, model in models.items():
			print(f"Training {name}...")
			search_space = param_spaces.get(name, {})
			
			if search_space:
				# Use GridSearchCV with param_grid (not param_distributions)
				search = GridSearchCV(
					estimator=model,
					param_grid=search_space,  # ✅ Correct parameter for GridSearchCV
					cv=tscv,
					scoring=self.config.scoring,
					n_jobs=1,  # Set to 1 to avoid potential issues
					verbose=0,
				)
				search.fit(X_train, y_train)
				best_model = search.best_estimator_
				best_params[name] = search.best_params_
				print(f"  Best params: {search.best_params_}")
			else:
				# No hyperparameters to tune (e.g., Linear Regression)
				best_model = model.fit(X_train, y_train)
				best_params[name] = {}
				print(f"  No hyperparameters to tune")

			y_pred = best_model.predict(X_test)
			metrics = self.eval_metrics(y_test, y_pred)

			row = {"Model": name}
			row.update(metrics)
			perf_rows.append(row)

			trained_models[name] = best_model
			test_predictions[name] = np.asarray(y_pred)
			print(f"✅ {name} completed - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}")

		performance_df = pd.DataFrame(perf_rows).set_index("Model")
		performance_df = performance_df.sort_values("R2", ascending=False)

		# Optional: add ranks for quick comparison
		performance_df["rank_rmse"] = performance_df["RMSE"].rank(ascending=True, method="min")
		performance_df["rank_mae"] = performance_df["MAE"].rank(ascending=True, method="min")
		performance_df["rank_r2"] = performance_df["R2"].rank(ascending=False, method="min")
		performance_df["overall_rank"] = (
			performance_df["rank_rmse"] + performance_df["rank_mae"] + performance_df["rank_r2"]
		)

		return performance_df, trained_models, test_predictions, best_params

	def initiate_model_trainer(self, train_array, test_array):
		"""
		Original-compatible API:
		- Accepts train_array/test_array with last column as target
		- Trains/tunes all models and returns all metrics (no logging, no saving)
		Returns:
		- performance_df, trained_models, test_predictions, best_params
		"""
		try:
			X_train, y_train, X_test, y_test = (
				train_array[:, :-1],
				train_array[:, -1],
				test_array[:, :-1],
				test_array[:, -1],
			)
			return self._fit_all_and_evaluate(X_train, y_train, X_test, y_test)
		except Exception as e:
			raise RuntimeError(f"Model training failed: {e}") from e

	def train_with_splits(self, X_train, y_train, X_test, y_test):
		"""
		Convenience API for transformed data already split as arrays.
		Returns:
		- performance_df, trained_models, test_predictions, best_params
		"""
		try:
			return self._fit_all_and_evaluate(X_train, y_train, X_test, y_test)
		except Exception as e:
			raise RuntimeError(f"Model training failed: {e}") from e
