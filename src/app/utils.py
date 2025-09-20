"""
Utility functions for Madrid Housing Price Prediction API.
This module provides helper functions for model loading and management,
supporting both MLflow Model Registry and local model files.
"""

import os
import joblib
from pathlib import Path
from typing import Any, Tuple
import mlflow


def load_model() -> Tuple[Any, dict]:
    """
    Load machine learning model with fallback strategy.
    Attempts to load model from MLflow Model Registry first. If MLflow is not
    available or fails, falls back to loading from local joblib file.
    Returns:
        Tuple[Any, dict]: Loaded model object and model metadata dictionary
    Raises:
        FileNotFoundError: If neither MLflow nor local model file is available
    Notes:
        Environment variables used:
        - MLFLOW_TRACKING_URI: MLflow tracking server URI
        - MODEL_NAME: Name of the model in MLflow registry
        - MODEL_VERSION: Version of the model in MLflow registry
    """
    # Initialize model metadata with default values
    model_info = {"name": "madrid_housing_model", "version": "unknown", "source": ""}

    # Get environment variables for MLflow configuration
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME", "madrid_housing_model")
    model_version = os.getenv("MODEL_VERSION")

    # Attempt to load model from MLflow Model Registry if configured
    if mlflow_uri:
        try:
            # Construct model URI for MLflow
            model_uri = f"models:/{model_name}/{model_version or 'latest'}"
            model = mlflow.sklearn.load_model(model_uri)

            # Update model metadata with MLflow information
            model_info["version"] = model_version or "latest"
            model_info["source"] = f"MLflow: {model_uri}"
            return model, model_info

        except Exception as e:
            # Log fallback to local model and continue
            print("Could not load from MLflow, fallback to joblib:", e)

    # Fallback: load model from local joblib file
    model_path = Path("models/final_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model from local file
    model = joblib.load(model_path)

    # Update model metadata with local file information
    model_info["version"] = "local"
    model_info["source"] = str(model_path.resolve())

    return model, model_info
