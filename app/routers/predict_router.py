"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: predict_router.py

Description: Define routes for predictions.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.diabetes_schema import DiabetesData
from app.services.prediction_service import predict_diabetes_progression

router = APIRouter(
    prefix="/predict"
)


@router.post("/{model_name}")
def predict(model_name: str, data: DiabetesData):
    try:
        prediction = predict_diabetes_progression(model_name, data)
        return {"model": model_name, "prediction": prediction}
    except:
        raise HTTPException(status_code=404, detail="Model not found")
