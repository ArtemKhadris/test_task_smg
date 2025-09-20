"""
Training script for Madrid Housing Price Prediction models.

This script performs cross-validation training of baseline and LightGBM models,
selects the best hyperparameters, and saves the final model with evaluation metrics.
"""

import json
import math
from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

# Configuration constants
DATA_PATH = Path("data/processed/processed.csv")
SPLITS_DIR = Path("data/splits")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
OUTPUT_MODEL_PATH = Path("models/final_model.joblib")
METRICS_PATH = Path("models/metrics.json")

RANDOM_STATE = 42  # Random seed for reproducibility

# Hyperparameter grid for LightGBM (small for quick execution)
LGB_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [6, 8],
}

INNER_CV = 3  # Number of cross-validation folds for GridSearchCV


def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true (array-like): Ground truth values
        y_pred (array-like): Predicted values

    Returns:
        float: Root Mean Squared Error
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    """
    Calculate comprehensive regression evaluation metrics.

    Args:
        y_true (array-like): Ground truth values
        y_pred (array-like): Predicted values

    Returns:
        dict: Dictionary containing MAE, RMSE, and R² metrics
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    """
    Main training pipeline execution.

    Performs cross-validation training, hyperparameter optimization,
    and final model training with comprehensive evaluation.

    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If target column is not found in dataset
    """
    # Validate required files exist
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {DATA_PATH}")
    if not SPLITS_DIR.exists():
        raise FileNotFoundError(f"Splits directory not found: {SPLITS_DIR}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)
    if "buy_price" not in df.columns:
        raise ValueError("Target column 'buy_price' not found in processed dataset.")

    # Load data splits
    with open(SPLITS_DIR / "test_indices.json", "r") as f:
        test_indices = json.load(f)

    with open(SPLITS_DIR / "cv_folds.json", "r") as f:
        cv_folds = json.load(f)

    # Load preprocessor for feature transformation
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Initialize containers for cross-validation results
    baseline_fold_metrics = []
    lgb_fold_metrics = []
    lgb_best_params_per_fold = []

    # Perform cross-validation across all folds
    for fold_key, fold_data in cv_folds.items():
        print(f"\n=== Fold {fold_key} ===")
        train_idx = fold_data["train_idx"]
        val_idx = fold_data["val_idx"]

        # Prepare training and validation data
        X_train = df.loc[train_idx].drop(columns=["buy_price"])
        y_train = df.loc[train_idx]["buy_price"]

        X_val = df.loc[val_idx].drop(columns=["buy_price"])
        y_val = df.loc[val_idx]["buy_price"]

        # ---------- Baseline: Linear Regression ----------
        baseline_pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        )

        # Train baseline model
        baseline_pipe.fit(X_train, y_train)
        y_pred_baseline = baseline_pipe.predict(X_val)
        metrics_baseline = evaluate(y_val, y_pred_baseline)
        baseline_fold_metrics.append(metrics_baseline)
        print(f"Baseline metrics (fold {fold_key}): {metrics_baseline}")

        # ---------- LightGBM with GridSearchCV ----------
        lgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LGBMRegressor(random_state=RANDOM_STATE)),
            ]
        )

        # Perform hyperparameter tuning with grid search
        gs = GridSearchCV(
            estimator=lgb_pipeline,
            param_grid=LGB_PARAM_GRID,
            cv=INNER_CV,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        gs.fit(X_train, y_train)
        print(f"  Best params (fold {fold_key}): {gs.best_params_}")

        # Evaluate best model from grid search
        best_est = gs.best_estimator_
        y_pred_lgb = best_est.predict(X_val)
        metrics_lgb = evaluate(y_val, y_pred_lgb)
        lgb_fold_metrics.append(metrics_lgb)
        lgb_best_params_per_fold.append(gs.best_params_)
        print(f"LightGBM metrics (fold {fold_key}): {metrics_lgb}")

    # Aggregate metrics across all folds
    def aggregate(metrics_list):
        """
        Aggregate metrics across cross-validation folds.

        Args:
            metrics_list (list): List of metric dictionaries

        Returns:
            dict: Averaged metrics across all folds
        """
        dfm = pd.DataFrame(metrics_list)
        return dfm.mean().to_dict()

    baseline_cv_metrics = aggregate(baseline_fold_metrics)
    lgb_cv_metrics = aggregate(lgb_fold_metrics)
    print("\n=== CV Summary ===")
    print(f"Baseline CV mean metrics: {baseline_cv_metrics}")
    print(f"LightGBM CV mean metrics: {lgb_cv_metrics}")

    # Final training on complete training set (excluding test data)
    print("\n=== Final training on full train set (all data except test) ===")
    all_indices = df.index.tolist()
    train_indices_full = [i for i in all_indices if i not in test_indices]

    X_train_full = df.loc[train_indices_full].drop(columns=["buy_price"])
    y_train_full = df.loc[train_indices_full]["buy_price"]

    X_test = df.loc[test_indices].drop(columns=["buy_price"])
    y_test = df.loc[test_indices]["buy_price"]

    # Perform final hyperparameter optimization on full training set
    print("Running GridSearchCV on the full train set to find final hyperparameters...")
    lgb_pipeline_full = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMRegressor(random_state=RANDOM_STATE)),
        ]
    )

    gs_full = GridSearchCV(
        estimator=lgb_pipeline_full,
        param_grid=LGB_PARAM_GRID,
        cv=INNER_CV,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )

    gs_full.fit(X_train_full, y_train_full)
    print(f"Best params on full train: {gs_full.best_params_}")

    # Save final trained model
    final_model = gs_full.best_estimator_
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, OUTPUT_MODEL_PATH)
    print(f"Final model saved to {OUTPUT_MODEL_PATH}")

    # Evaluate final model on test set
    y_test_pred = final_model.predict(X_test)
    test_metrics = evaluate(y_test, y_test_pred)
    print(f"Test metrics: {test_metrics}")

    # Save comprehensive metrics and metadata
    metrics_dict = {
        "baseline_cv_metrics": baseline_cv_metrics,
        "lgb_cv_metrics": lgb_cv_metrics,
        "lgb_best_params_per_fold": lgb_best_params_per_fold,
        "final_best_params": gs_full.best_params_,
        "test_metrics": test_metrics,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"All metrics saved to {METRICS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
