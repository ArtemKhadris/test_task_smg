"""
Tests for trained model functionality.

This module contains tests that verify the trained model can load
and make predictions with expected output formats.
"""

import joblib
import pandas as pd


def test_model_predicts():
    """
    Test that the trained model can load and make predictions.

    Verifies:
    - Model file can be loaded successfully
    - Model can make predictions on sample input data
    - Prediction output has correct shape (1 prediction)
    - Prediction value is a valid float number

    Raises:
        FileNotFoundError: If the model file does not exist
        Exception: If model prediction fails for any reason
    """
    # Load the trained model from disk
    model = joblib.load("models/final_model.joblib")

    # Create sample input data for prediction
    df = pd.DataFrame(
        [
            {
                "sq_mt_built": 120,  # Square meters built
                "sq_mt_useful": 100,  # Square meters useful area
                "n_rooms": 3,  # Number of rooms
                "n_bathrooms": 2,  # Number of bathrooms
                "has_parking": True,  # Parking availability
            }
        ]
    )

    # Make prediction using the loaded model
    y_pred = model.predict(df)

    # Verify prediction output format and type
    assert y_pred.shape == (1,)  # Should return exactly one prediction
    assert isinstance(
        float(y_pred[0]), float
    )  # Prediction should be convertible to float
