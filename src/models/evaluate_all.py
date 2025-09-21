"""
Comprehensive model evaluation script for Madrid Housing Price Prediction.

This script evaluates all trained models in the models/ directory, generating
performance metrics, visualizations, and feature importance analysis.
"""

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.preprocessing.preprocessing import FeatureEngineer

# Configurable paths
DATA_PATH = Path("data/processed/processed.csv")
SPLITS_PATH = Path("data/splits/test_indices.json")
MODELS_DIR = Path("models")
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
REPORTS_DIR = Path("reports")

# Create reports directory if it doesn't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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


def log_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error on log-transformed values.

    Useful for positive-valued targets like prices to reduce scale sensitivity.

    Args:
        y_true (array-like): Ground truth values
        y_pred (array-like): Predicted values

    Returns:
        float: Mean Absolute Error of log-transformed values
    """
    # assume positive-valued target (prices)
    return mean_absolute_error(np.log1p(y_true), np.log1p(y_pred))


def evaluate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics for regression.

    Args:
        y_true (array-like): Ground truth values
        y_pred (array-like): Predicted values

    Returns:
        dict: Dictionary containing MAE, RMSE, R², and log-MAE metrics
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "log_mae": float(log_mae(y_true, y_pred)),
    }


def get_transformed_feature_names(transformer, X_sample):
    """
    Extract human-readable feature names after transformation.

    Attempts multiple strategies to get feature names from transformers,
    with fallbacks to generic names if specific names are unavailable.

    Args:
        transformer: Fitted transformer object
        X_sample: Sample input data for feature name inference

    Returns:
        list: List of feature names or generic names if unavailable
    """
    try:
        # preferred: give X so ColumnTransformer/OneHotEncoder can infer names
        names = transformer.get_feature_names_out(X_sample)
        return list(names)
    except Exception:
        try:
            names = transformer.get_feature_names_out()
            return list(names)
        except Exception:
            # fallback: transform and create generic names
            try:
                X_tr = transformer.transform(X_sample)
                n = X_tr.shape[1]
                return [f"f{i}" for i in range(n)]
            except Exception:
                return None


def evaluate_model_file(
    model_path: Path, df: pd.DataFrame, test_indices: list, preprocessor
):
    """
    Evaluate a single model file and generate comprehensive reports.

    Args:
        model_path (Path): Path to the model file
        df (pd.DataFrame): Processed dataset
        test_indices (list): Indices of test samples
        preprocessor: Optional preprocessor for non-pipeline models

    Returns:
        dict: Model evaluation results with metrics or None if evaluation fails
    """
    model_name = model_path.stem
    out_dir = REPORTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Evaluating model: {model_name} ===")

    # Load model from file
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"  ERROR: cannot load model {model_path}: {e}")
        return None

    # Prepare test set
    if "buy_price" not in df.columns:
        print("  ERROR: target 'buy_price' not in dataset")
        return None

    X_test = df.drop(columns=["buy_price"]).loc[test_indices]
    y_test = df.loc[test_indices, "buy_price"].reset_index(drop=True)

    # Check if model is a pipeline
    pipeline = model if isinstance(model, Pipeline) else None

    # Generate predictions with appropriate preprocessing
    try:
        y_pred = model.predict(X_test)
        used_transformer = pipeline[:-1] if pipeline is not None else None
        estimator = pipeline[-1] if pipeline is not None else model
    except Exception as e:
        # fallback: if we have a preprocessor, transform X_test then predict with estimator
        if preprocessor is not None:
            try:
                X_test_tr = preprocessor.transform(X_test)
                y_pred = model.predict(X_test_tr)
                used_transformer = preprocessor
                estimator = model
            except Exception as e2:
                print(
                    f"  ERROR: prediction failed even after preprocessor transform: {e2}"
                )
                return None
        else:
            print(f"  ERROR: prediction failed and no preprocessor available: {e}")
            return None

    y_pred = np.array(y_pred).ravel()
    y_true = np.array(y_test).ravel()

    # Calculate and save metrics
    metrics = evaluate_metrics(y_true, y_pred)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Create Predicted vs Actual scatter plot
    try:
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, alpha=0.6, s=18)
        minv = min(y_true.min(), y_pred.min())
        maxv = max(y_true.max(), y_pred.max())
        plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
        plt.xlabel("Actual buy_price")
        plt.ylabel("Predicted buy_price")
        plt.title(f"{model_name} - Predicted vs Actual")
        plt.tight_layout()
        ppath = out_dir / "pred_vs_actual.png"
        plt.savefig(ppath, dpi=150)
        plt.close()
        print(f"  Saved plot: {ppath}")
    except Exception as e:
        print("  Could not save Pred vs Actual plot:", e)

    # Create residuals histogram
    try:
        residuals = y_true - y_pred
        plt.figure(figsize=(7, 4))
        plt.hist(residuals, bins=40, edgecolor="k", alpha=0.7)
        plt.xlabel("Residual (Actual - Predicted)")
        plt.ylabel("Frequency")
        plt.title(f"{model_name} - Residuals")
        plt.tight_layout()
        rpath = out_dir / "residuals.png"
        plt.savefig(rpath, dpi=150)
        plt.close()
        print(f"  Saved plot: {rpath}")
    except Exception as e:
        print("  Could not save residuals plot:", e)

    # Extract and visualize feature importances
    fi_saved = False
    try:
        if used_transformer is None:
            raise RuntimeError(
                "No transformer available to produce feature names for importances"
            )

        # Try to get feature names
        feature_names = get_transformed_feature_names(used_transformer, X_test)

        # Transform test data for analysis
        X_test_tr = used_transformer.transform(X_test)
        try:
            X_test_tr_arr = (
                X_test_tr.toarray()
                if hasattr(X_test_tr, "toarray")
                else np.asarray(X_test_tr)
            )
        except Exception:
            X_test_tr_arr = np.asarray(X_test_tr)

        # Extract feature importances from estimator
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_)
        else:
            raise RuntimeError("Estimator has no feature_importances_ or coef_")

        # Align feature names with importances
        if feature_names is None or len(feature_names) != len(importances):
            # fallback: create generic names
            n_feats = len(importances)
            feature_names = [f"f{i}" for i in range(n_feats)]

        # Create and save feature importance data
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
        ficsv = out_dir / "feature_importances.csv"
        fi_df.to_csv(ficsv, index=False)

        # Create feature importance visualization
        topk = min(50, len(fi_df))
        top_df = fi_df.head(topk).sort_values("importance")
        plt.figure(figsize=(8, max(3, topk * 0.12)))
        plt.barh(top_df["feature"], top_df["importance"])
        plt.xlabel("Importance")
        plt.title(f"{model_name} - Feature importances (top {topk})")
        plt.tight_layout()
        fipng = out_dir / "feature_importances.png"
        plt.savefig(fipng, dpi=150)
        plt.close()
        print(f"  Saved feature importances to {ficsv} and {fipng}")
        fi_saved = True
    except Exception as e:
        print("  Could not extract/save feature importances:", e)

    # Generate SHAP summary plot if possible
    try:
        import shap

        if used_transformer is None:
            raise RuntimeError("No transformer available for SHAP")

        estimator_for_shap = estimator
        # Transform data for SHAP analysis
        X_test_tr = used_transformer.transform(X_test)
        try:
            X_test_tr_arr = (
                X_test_tr.toarray()
                if hasattr(X_test_tr, "toarray")
                else np.asarray(X_test_tr)
            )
        except Exception:
            X_test_tr_arr = np.asarray(X_test_tr)

        # Build DataFrame with feature names if available
        if (
            "feature_names" in locals()
            and feature_names is not None
            and len(feature_names) == X_test_tr_arr.shape[1]
        ):
            X_shap_df = pd.DataFrame(X_test_tr_arr, columns=feature_names)
        else:
            X_shap_df = pd.DataFrame(
                X_test_tr_arr, columns=[f"f{i}" for i in range(X_test_tr_arr.shape[1])]
            )

        # Sample data for SHAP analysis (faster computation)
        n_samples = min(200, X_shap_df.shape[0])
        X_shap_sample = X_shap_df.sample(n=n_samples, random_state=42)

        # Compute and plot SHAP values
        explainer = shap.TreeExplainer(estimator_for_shap)
        shap_values = explainer.shap_values(X_shap_sample)
        shap.summary_plot(shap_values, X_shap_sample, show=False)
        shap_png = out_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(shap_png, dpi=150)
        plt.close()
        print(f"  Saved SHAP summary to {shap_png}")
    except Exception as e:
        print("  Could not compute SHAP summary (skipping):", e)

    return {"model": model_name, **metrics}


def main():
    """
    Main function to execute comprehensive model evaluation.

    Loads data, test splits, and evaluates all models in the models directory,
    generating comprehensive reports and summary metrics.
    """
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings(
        "ignore",
        message=".*X does not have valid feature names.*",
        category=UserWarning,
    )

    # Basic data validation
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {DATA_PATH}")
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"Test indices not found: {SPLITS_PATH}")
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    # Load data and test splits
    df = pd.read_csv(DATA_PATH)
    with open(SPLITS_PATH, "r") as f:
        test_indices = json.load(f)

    # Load optional preprocessor for non-pipeline models
    preprocessor = None
    if PREPROCESSOR_PATH.exists():
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"Loaded preprocessor from {PREPROCESSOR_PATH}")
        except Exception as e:
            print("Could not load preprocessor:", e)
            preprocessor = None
    else:
        print(
            "No preprocessor.joblib found; raw estimators will be called directly (may fail)."
        )

    # Find all model files in models directory
    model_files = sorted([p for p in MODELS_DIR.glob("*.joblib") if p.is_file()])
    if not model_files:
        print("No .joblib models found in models/ directory.")
        return

    # Evaluate each model and collect results
    summary = []
    for mpath in model_files:
        res = evaluate_model_file(mpath, df, test_indices, preprocessor)
        if res is not None:
            summary.append(res)

    # Save comprehensive summary report
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_csv = REPORTS_DIR / "summary_metrics.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSaved summary metrics to {summary_csv}")
    else:
        print("No model evaluations completed.")


if __name__ == "__main__":
    main()
