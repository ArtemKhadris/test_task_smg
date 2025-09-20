"""
Mock tests for model loading functionality.

This module contains tests that use mocking to simulate MLflow model loading
without requiring actual MLflow server connectivity.
"""

from unittest import mock
import src.app.utils as app_utils


class DummyModel:
    """
    Dummy model class for testing prediction functionality.

    This mock model always returns the same prediction value (42)
    regardless of input to simulate a trained model's predict method.
    """

    def predict(self, X):
        """
        Mock prediction method that returns constant values.

        Args:
            X: Input data (ignored in this mock implementation)

        Returns:
            list: List of constant predictions (42) with same length as input
        """
        return [42] * len(X)


def test_load_model_from_mlflow(monkeypatch):
    """
    Test MLflow model loading with mocked MLflow client.

    This test simulates loading a model from MLflow registry by:
    1. Setting fake MLflow environment variables
    2. Mocking the mlflow.sklearn.load_model function
    3. Verifying the model loading process returns expected results

    Args:
        monkeypatch: Pytest monkeypatch fixture for environment variables

    Verifies:
    - Loaded model is the mocked fake model
    - Model info contains correct MLflow source indication
    - Model version defaults to 'latest' when not specified
    """
    # Set fake MLflow environment variables
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake-mlflow:5000")
    monkeypatch.setenv("MODEL_NAME", "dummy_model")

    fake_model = DummyModel()

    # Mock the mlflow module to avoid actual server calls
    with mock.patch("src.app.utils.mlflow") as mock_mlflow:
        # Configure mock to return our dummy model
        mock_mlflow.sklearn.load_model.return_value = fake_model

        # Call the function under test
        model, info = app_utils.load_model()

    # Verify the results
    assert model is fake_model
    assert info["source"].startswith("MLflow:")
    assert info["version"] == "latest"
