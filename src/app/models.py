"""
Pydantic models for Madrid Housing Price Prediction API.
This module defines the request and response models for the API endpoints,
including data validation, serialization, and documentation.
"""

from typing import List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request model for single housing price prediction.
    Attributes:
        sq_mt_built (float): Total built area in square meters
        sq_mt_useful (float): Usable area in square meters
        n_rooms (int): Number of rooms
        n_bathrooms (int): Number of bathrooms
        has_parking (bool): Availability of a parking space
    """

    sq_mt_built: float = Field(..., description="Total built area in square meters")
    sq_mt_useful: float = Field(..., description="Usable area in square meters")
    n_rooms: int = Field(..., description="Number of rooms")
    n_bathrooms: int = Field(..., description="Number of bathrooms")
    has_parking: bool = Field(..., description="Availability of a parking space")


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch housing price predictions.

    Attributes:
        inputs (List[PredictionRequest]): List of prediction requests for batch processing
    """

    inputs: List[PredictionRequest]


class PredictionResponse(BaseModel):
    """
    Response model for single housing price prediction.

    Attributes:
        prediction (float): Predicted housing price
    """

    prediction: float


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch housing price predictions.

    Attributes:
        predictions (List[float]): List of predicted housing prices
    """

    predictions: List[float]


class ModelInfoResponse(BaseModel):
    """
    Response model for ML model metadata information.
    Attributes:
        name (str): Name of the ML model
        version (str): Version of the ML model
        source (str): Source or origin of the ML model
    """

    name: str
    version: str
    source: str
