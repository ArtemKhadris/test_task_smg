"""
Pytest configuration and fixtures for Madrid Housing Price Prediction tests.

This module provides shared fixtures and configuration for unit tests
across the project. It is automatically discovered by pytest.
"""

import pandas as pd
import pytest
from src.preprocessing.preprocessing import build_preprocessing_pipeline, save_pipeline


@pytest.fixture
def sample_df():
    """
    Create a small but sufficient DataFrame for testing purposes.

    Returns:
        pd.DataFrame: A DataFrame with 6 rows containing synthetic housing data
                     with various feature combinations for comprehensive testing.

    The DataFrame includes:
    - buy_price: Target variable with decreasing values
    - sq_mt_built: Built area in square meters
    - sq_mt_useful: Useful area in square meters
    - n_rooms: Number of rooms (alternating 2 and 3)
    - n_bathrooms: Number of bathrooms (mostly 1, some 2)
    - has_parking: Boolean indicating parking availability
    - neighborhood_id: Categorical neighborhood identifiers
    - house_type_id: Categorical house type identifiers
    - year_built: Year built with decreasing values
    """
    rows = []
    for i in range(6):
        rows.append(
            {
                "buy_price": 200000 - i * 10000,  # Decreasing price from 200k to 150k
                "sq_mt_built": 100 - i * 5,  # Decreasing built area from 100 to 75
                "sq_mt_useful": 90 - i * 4,  # Decreasing useful area from 90 to 66
                "n_rooms": 3 if i % 2 == 0 else 2,  # Alternating 3 and 2 rooms
                "n_bathrooms": 2 if i % 3 == 0 else 1,  # Mostly 1 bathroom, some 2
                "has_parking": bool(i % 2),  # Alternating parking availability
                "neighborhood_id": f"N{i%3}",  # Neighborhoods N0, N1, N2
                "house_type_id": f"H{i%2}",  # House types H0, H1
                "year_built": 2000 - i,  # Decreasing year from 2000 to 1995
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_preprocessor(tmp_path, sample_df):
    """
    Build and persist a preprocessor fitted on sample data for testing.

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_df: Sample DataFrame fixture

    Returns:
        Path: Path to the saved preprocessor joblib file

    This fixture:
    1. Builds a preprocessing pipeline
    2. Fits it on the sample DataFrame (excluding target)
    3. Saves it to a temporary location
    4. Returns the path for use in tests
    """
    # Build preprocessing pipeline
    pipeline = build_preprocessing_pipeline()

    # Prepare features (exclude target column)
    X = sample_df.drop(columns=["buy_price"])

    # Fit pipeline on sample data
    pipeline.fit(X)

    # Save pipeline to temporary location
    path = tmp_path / "preprocessor.joblib"
    save_pipeline(pipeline, path=str(path))

    return path
