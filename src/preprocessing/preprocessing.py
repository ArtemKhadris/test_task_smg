"""
Preprocessing pipeline for Madrid Housing Price Prediction.

This module defines custom transformers and builds a comprehensive preprocessing
pipeline including feature engineering, imputation, and scaling.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering of housing data.

    Creates derived features from existing numeric columns to capture
    additional relationships in the housing data.
    """

    def fit(self, X, y=None):
        """
        Fit the transformer (no operation needed for feature engineering).

        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target values

        Returns:
            self: Returns the instance itself
        """
        return self

    def transform(self, X):
        """
        Apply feature engineering transformations.

        Args:
            X (pd.DataFrame): Input features to transform

        Returns:
            pd.DataFrame: Transformed features with new engineered columns
        """
        X = X.copy()

        # Create rooms per square meter feature
        if "sq_mt_useful" in X.columns and "n_rooms" in X.columns:
            X["rooms_per_m2"] = X["sq_mt_useful"] / (X["n_rooms"] + 1e-3)

        # Create bathrooms per room ratio
        if "n_bathrooms" in X.columns and "n_rooms" in X.columns:
            X["bathrooms_per_room"] = X["n_bathrooms"] / (X["n_rooms"] + 1e-3)

        # Create efficiency ratio (useful area vs built area)
        if "sq_mt_built" in X.columns and "sq_mt_useful" in X.columns:
            X["efficiency_ratio"] = X["sq_mt_useful"] / (X["sq_mt_built"] + 1e-3)

        return X


def build_preprocessing_pipeline():
    """
    Build a comprehensive preprocessing pipeline for housing data.

    Returns:
        Pipeline: Scikit-learn pipeline with feature engineering and preprocessing steps

    The pipeline includes:
    - Feature engineering (custom transformations)
    - Numeric feature imputation and scaling
    """
    # Selected numeric features for preprocessing
    numeric_features = ["sq_mt_built", "sq_mt_useful", "n_rooms", "n_bathrooms"]

    # Numeric pipeline: impute missing values and scale features
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Impute missing values with median
            (
                "scaler",
                StandardScaler(),
            ),  # Standardize features to zero mean and unit variance
        ]
    )

    # Column transformer for numeric features only
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ]
    )

    # Full pipeline: feature engineering followed by preprocessing
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),  # Custom feature engineering
            ("preprocessor", preprocessor),  # Standard preprocessing
        ]
    )

    return pipeline


def save_pipeline(pipeline, path: str = "models/preprocessor.joblib"):
    """
    Save the preprocessing pipeline to disk.

    Args:
        pipeline (Pipeline): Fitted preprocessing pipeline to save
        path (str): Path where to save the pipeline

    Returns:
        None
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Preprocessing pipeline saved to {path}")


def load_pipeline(path: str = "models/preprocessor.joblib"):
    """
    Load a preprocessing pipeline from disk.

    Args:
        path (str): Path to the saved pipeline file

    Returns:
        Pipeline: Loaded preprocessing pipeline
    """
    return joblib.load(path)


if __name__ == "__main__":
    """
    Example usage and testing of the preprocessing pipeline.

    This section demonstrates how to use the pipeline with processed data,
    fit it on training data, and save the fitted pipeline.
    """
    # Load processed data
    data_path = Path("data/processed/processed.csv")
    df = pd.read_csv(data_path)

    # Select features and target
    features = ["sq_mt_built", "sq_mt_useful", "n_rooms", "n_bathrooms", "has_parking"]
    target = "buy_price"

    X = df[features]
    y = df[target]

    # Build and fit the preprocessing pipeline
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X, y)

    # Save the fitted pipeline for later use
    save_pipeline(pipeline)
