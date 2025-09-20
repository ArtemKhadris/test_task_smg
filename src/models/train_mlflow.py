"""
ML training script with comprehensive MLflow logging for Madrid Housing Price Prediction.

This script performs end-to-end machine learning training with cross-validation,
hyperparameter tuning, and extensive MLflow tracking for experiment management.
"""

import argparse
import json
import math
import hashlib
import datetime
from pathlib import Path
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import mlflow
import mlflow.sklearn


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
    Calculate standard regression evaluation metrics.

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


def file_sha256(path: Path) -> str:
    """
    Calculate SHA256 hash of a file for data versioning.

    Args:
        path (Path): Path to the file

    Returns:
        str: SHA256 hash of the file content
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_training(config_path: Path):
    """
    Execute complete training pipeline with MLflow logging.

    Args:
        config_path (Path): Path to YAML configuration file

    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If target column is not found in dataset
    """
    # Load experiment configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Configure paths from config with defaults
    DATA_PATH = Path(cfg.get("data_path", "data/processed/processed.csv"))
    SPLITS_DIR = Path(cfg.get("splits_dir", "data/splits"))
    PREPROCESSOR_PATH = Path(cfg.get("preprocessor_path", "models/preprocessor.joblib"))
    OUT_DIR = Path(cfg.get("out_dir", "models"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate required files exist
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {DATA_PATH}")
    if not SPLITS_DIR.exists():
        raise FileNotFoundError(f"Splits directory not found: {SPLITS_DIR}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    # Load data and splits
    df = pd.read_csv(DATA_PATH)
    with open(SPLITS_DIR / "test_indices.json", "r") as f:
        test_indices = json.load(f)
    with open(SPLITS_DIR / "cv_folds.json", "r") as f:
        cv_folds = json.load(f)

    # Validate target column
    target_col = cfg.get("target_col", "buy_price")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataset.")

    # Load preprocessor for feature transformation
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Compute data hash for version tracking
    data_hash = file_sha256(DATA_PATH)

    # Configure MLflow experiment
    mlflow.set_experiment(cfg.get("mlflow_experiment", "madrid_housing_experiments"))

    # Start MLflow run with timestamp-based name
    run_name = cfg.get(
        "run_name", f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        # Log experiment metadata
        mlflow.log_param("config_file", str(config_path))
        mlflow.log_param("data_path", str(DATA_PATH))
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("use_log_target", bool(cfg.get("use_log_target", False)))

        # Save configuration as artifact
        mlflow.log_artifact(local_path=str(config_path), artifact_path="config")

        # Initialize containers for cross-validation results
        baseline_fold_metrics = []
        lgb_fold_metrics = []
        lgb_best_params_per_fold = []

        # Perform cross-validation across all folds
        for fold_key, fold_data in cv_folds.items():
            train_idx = fold_data["train_idx"]
            val_idx = fold_data["val_idx"]

            # Prepare training and validation data
            X_train = df.loc[train_idx].drop(columns=[target_col])
            y_train = df.loc[train_idx][target_col].copy()

            X_val = df.loc[val_idx].drop(columns=[target_col])
            y_val = df.loc[val_idx][target_col].copy()

            # Apply log transformation if configured
            if cfg.get("use_log_target", False):
                y_train = np.log1p(y_train)
                y_val = np.log1p(y_val)

            # Train and evaluate baseline linear regression
            baseline_pipe = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
            )
            baseline_pipe.fit(X_train, y_train)
            y_pred_baseline = baseline_pipe.predict(X_val)

            # Reverse log transformation for evaluation if applied
            if cfg.get("use_log_target", False):
                y_pred_baseline = np.expm1(y_pred_baseline)
                y_val_eval = np.expm1(y_val)
            else:
                y_val_eval = y_val

            metrics_baseline = evaluate(y_val_eval, y_pred_baseline)
            baseline_fold_metrics.append(metrics_baseline)

            # Train and evaluate LightGBM with grid search
            lgb_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", LGBMRegressor(random_state=cfg.get("random_state", 42))),
                ]
            )

            # Configure parameter grid from config or use defaults
            param_grid = cfg.get(
                "lgb_param_grid",
                {
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [6, 8],
                },
            )

            # Perform grid search with cross-validation
            gs = GridSearchCV(
                estimator=lgb_pipeline,
                param_grid=param_grid,
                cv=cfg.get("inner_cv", 3),
                scoring="neg_mean_squared_error",
                n_jobs=cfg.get("n_jobs", -1),
                verbose=0,
            )
            gs.fit(X_train, y_train)
            best_est = gs.best_estimator_
            y_pred_lgb = best_est.predict(X_val)

            # Reverse log transformation if applied
            if cfg.get("use_log_target", False):
                y_pred_lgb = np.expm1(y_pred_lgb)

            metrics_lgb = evaluate(y_val_eval, y_pred_lgb)
            lgb_fold_metrics.append(metrics_lgb)
            lgb_best_params_per_fold.append(gs.best_params_)

            # Log per-fold metrics to MLflow
            for mname, mval in metrics_baseline.items():
                mlflow.log_metric(f"baseline_{mname}_fold{fold_key}", mval)
            for mname, mval in metrics_lgb.items():
                mlflow.log_metric(f"lgb_{mname}_fold{fold_key}", mval)

        # Aggregate cross-validation metrics
        def agg(list_of_dicts):
            """Helper function to aggregate metrics across folds."""
            dfm = pd.DataFrame(list_of_dicts)
            return dfm.mean().to_dict()

        baseline_cv_metrics = agg(baseline_fold_metrics)
        lgb_cv_metrics = agg(lgb_fold_metrics)

        # Log aggregated cross-validation metrics
        for k, v in baseline_cv_metrics.items():
            mlflow.log_metric(f"baseline_cv_{k}", v)
        for k, v in lgb_cv_metrics.items():
            mlflow.log_metric(f"lgb_cv_{k}", v)

        # Log best parameters from each fold
        mlflow.log_param(
            "lgb_best_params_per_fold", json.dumps(lgb_best_params_per_fold)
        )

        # Train final model on full training set (excluding test data)
        all_indices = df.index.tolist()
        train_indices_full = [i for i in all_indices if i not in test_indices]
        X_train_full = df.loc[train_indices_full].drop(columns=[target_col])
        y_train_full = df.loc[train_indices_full][target_col].copy()
        X_test = df.loc[test_indices].drop(columns=[target_col])
        y_test = df.loc[test_indices][target_col].copy()

        # Apply log transformation if configured
        if cfg.get("use_log_target", False):
            y_train_full = np.log1p(y_train_full)

        # Perform final grid search on full training data
        lgb_pipeline_full = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LGBMRegressor(random_state=cfg.get("random_state", 42))),
            ]
        )
        gs_full = GridSearchCV(
            estimator=lgb_pipeline_full,
            param_grid=cfg.get("lgb_param_grid", {}),
            cv=cfg.get("inner_cv", 3),
            scoring="neg_mean_squared_error",
            n_jobs=cfg.get("n_jobs", -1),
        )
        gs_full.fit(X_train_full, y_train_full)

        # Log final best parameters
        mlflow.log_param("final_best_params", json.dumps(gs_full.best_params_))

        final_model = gs_full.best_estimator_

        # Save final model locally and log to MLflow
        final_model_path = OUT_DIR / f"final_model_{run.info.run_id}.joblib"
        joblib.dump(final_model, final_model_path)
        mlflow.log_artifact(str(final_model_path), artifact_path="models_local")

        # Log model to MLflow model registry
        mlflow.sklearn.log_model(final_model, artifact_path="model")

        # Evaluate final model on test set
        y_test_pred = final_model.predict(X_test)
        if cfg.get("use_log_target", False):
            y_test_pred = np.expm1(y_test_pred)
        test_metrics = evaluate(y_test, y_test_pred)

        # Log test metrics
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Save comprehensive metrics locally
        metrics_out = {
            "baseline_cv_metrics": baseline_cv_metrics,
            "lgb_cv_metrics": lgb_cv_metrics,
            "lgb_best_params_per_fold": lgb_best_params_per_fold,
            "final_best_params": gs_full.best_params_,
            "test_metrics": test_metrics,
            "data_sha256": data_hash,
            "run_id": run.info.run_id,
        }
        (OUT_DIR / f"metrics_{run.info.run_id}.json").write_text(
            json.dumps(metrics_out, indent=2)
        )

        print("Run complete. Run ID:", run.info.run_id)
        print("Test metrics:", test_metrics)


if __name__ == "__main__":
    # Configure command-line interface
    parser = argparse.ArgumentParser(description="Train ML models with MLflow logging")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML experiment config"
    )
    args = parser.parse_args()

    # Execute training pipeline
    run_training(Path(args.config))
