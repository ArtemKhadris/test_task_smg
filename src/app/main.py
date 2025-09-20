"""
Main FastAPI application module for Madrid Housing Price Prediction API.

This module creates and configures the FastAPI application, loads the ML model,
and defines the API endpoints for housing price predictions.
"""

import logging
from fastapi import FastAPI, HTTPException
from src.app.models import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
)
from src.app.utils import load_model
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("app")

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Madrid Housing Price API",
    description="FastAPI service for predicting housing prices in Madrid",
    version="1.0.0",
)

# Load ML model during application startup
try:
    model, model_info = load_model()
    logger.info(f"Model loaded successfully: {model_info}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    # Set model to None and create error info if loading fails
    model, model_info = None, {
        "name": "madrid_housing_model",
        "version": "unavailable",
        "source": "error",
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API service status.
    Returns:
        dict: Status of the API service and model availability
    """
    status = model is not None
    return {"status": "ok" if status else "error"}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info_endpoint():
    """
    Retrieve information about the loaded ML model.
    Returns:
        ModelInfoResponse: Model metadata including name, version, and source
    Raises:
        HTTPException: If model information is not available
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")
    return ModelInfoResponse(**model_info)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict housing price for a single input instance.
    Args:
        request (PredictionRequest): Input features for prediction
    Returns:
        PredictionResponse: Predicted housing price
    Raises:
        HTTPException: 500 if model not loaded, 400 for prediction errors
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    try:
        # Convert request to DataFrame for model prediction
        df = pd.DataFrame([request.model_dump()])
        y_pred = model.predict(df)[0]
        return PredictionResponse(prediction=float(y_pred))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    """
    Predict housing prices for multiple input instances in batch.
    Args:
        request (BatchPredictionRequest): List of input features for batch prediction
    Returns:
        BatchPredictionResponse: List of predicted housing prices
    Raises:
        HTTPException: 500 if model not loaded, 400 for prediction errors
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    try:
        # Convert batch requests to DataFrame for model prediction
        df = pd.DataFrame([r.model_dump() for r in request.inputs])
        preds = model.predict(df).tolist()
        return BatchPredictionResponse(predictions=[float(p) for p in preds])
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
