"""
Tests for data splitting functionality.

This module contains tests for the create_splits function that generates
train/test splits and cross-validation folds for machine learning experiments.
"""

import json
from src.utils.split_data import create_splits


def test_create_splits(tmp_path, sample_df):
    """
    Test the create_splits function with sample data.

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_df: Pytest fixture providing sample housing data

    Verifies:
    - Split files are created in the specified output directory
    - Test indices file exists and contains a list
    - Cross-validation folds file exists and contains a dictionary
    - Correct number of folds are created based on n_splits parameter
    """
    # Save sample data to temporary CSV file
    data_csv = tmp_path / "processed.csv"
    sample_df.to_csv(data_csv, index=False)

    # Create output directory for splits
    out_dir = tmp_path / "splits"

    # Call the function under test
    create_splits(
        data_path=str(data_csv),
        target_col="buy_price",
        test_size=0.25,  # 25% test, 75% train
        n_splits=2,  # 2-fold cross-validation
        random_state=0,  # Fixed random seed for reproducibility
        output_dir=str(out_dir),
    )

    # Verify output files exist
    test_idx_file = out_dir / "test_indices.json"
    cv_file = out_dir / "cv_folds.json"
    assert test_idx_file.exists()
    assert cv_file.exists()

    # Load and verify the contents of the split files
    with open(test_idx_file, "r") as f:
        ti = json.load(f)
    with open(cv_file, "r") as f:
        folds = json.load(f)

    # Verify data types and structure
    assert isinstance(ti, list)  # Test indices should be a list
    assert isinstance(folds, dict)  # Folds should be a dictionary
    assert len(folds) == 2  # Should have exactly 2 folds as requested
