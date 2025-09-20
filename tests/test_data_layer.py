"""
Data layer tests for Madrid Housing Price Prediction project.

This module contains tests for data persistence functionality,
specifically focusing on preprocessor serialization and deserialization.
"""

import joblib
from pathlib import Path


def test_save_and_load_preprocessor(tmp_preprocessor):
    """
    Test preprocessor serialization and deserialization functionality.

    Args:
        tmp_preprocessor (Path): Pytest fixture providing path to saved preprocessor

    Verifies:
    - Preprocessor file exists at the specified path
    - Preprocessor can be successfully loaded from disk
    - Loaded preprocessor object is not None
    - Preprocessor can be re-serialized to a different location
    - New serialized file exists at the specified location
    """
    # Verify the preprocessor file exists (from fixture)
    assert Path(tmp_preprocessor).exists()

    # Load preprocessor from disk
    p = joblib.load(str(tmp_preprocessor))
    assert p is not None

    # Test re-serialization to a different location
    new_path = Path(tmp_preprocessor).parent / "preproc_copy.joblib"
    joblib.dump(p, str(new_path))

    # Verify the new file was created successfully
    assert new_path.exists()
