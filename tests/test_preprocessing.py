"""
Tests for preprocessing pipeline functionality.

This module contains tests for loading and transforming data using
the preprocessing pipeline to ensure proper functionality.
"""

import joblib
import numpy as np


def test_preprocessor_loads(tmp_preprocessor):
    """
    Test that the preprocessor can be successfully loaded from disk.

    Args:
        tmp_preprocessor (Path): Pytest fixture providing path to saved preprocessor

    Verifies:
    - Preprocessor file can be loaded without errors
    - Loaded preprocessor object is not None
    """
    preprocessor = joblib.load(str(tmp_preprocessor))
    assert preprocessor is not None


def test_preprocessor_transform(tmp_preprocessor, sample_df):
    """
    Test that the preprocessor can transform sample data correctly.

    Args:
        tmp_preprocessor (Path): Pytest fixture providing path to saved preprocessor
        sample_df (pd.DataFrame): Pytest fixture providing sample housing data

    Verifies:
    - Preprocessor can transform input data without errors
    - Transformed data has the same number of rows as input
    - Transformed data contains no NaN values
    - Both sparse and dense matrix formats are handled correctly
    """
    # Load preprocessor from temporary file
    preproc = joblib.load(str(tmp_preprocessor))

    # Prepare input data (exclude target column)
    X = sample_df.drop(columns=["buy_price"])

    # Apply transformation
    X_tr = preproc.transform(X)
    assert X_tr is not None

    # Handle both sparse and dense matrix formats
    arr = X_tr.toarray() if hasattr(X_tr, "toarray") else np.asarray(X_tr)

    # Verify transformation results
    assert arr.shape[0] == X.shape[0]  # Same number of rows
    assert not np.isnan(arr).any()  # No missing values in output
