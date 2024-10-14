"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: predict_router.py

Description: Define routers for predictions.
"""

from fastapi import APIRouter, HTTPException

from app.ml.regression_model import get_model
from app.ml.transformations import normalize_sample
from app.schemas.diabetes_schema import DiabetesData
from app.services.prediction_service import predict_diabetes_progression

router = APIRouter(
    prefix="/predict"
)


@router.post("/{model_name}")
def predict(model_name: str, data: DiabetesData):
    try:
        print(model_name)
        model = get_model(model_name)
        print(model)
        rmse = model.train()
        normalized = normalize_sample(data)
        print(normalized)
        print("******************************")
        prediction = predict_diabetes_progression(model, normalized)
        print(prediction)
        return {"model": model_name,
                "rmse": rmse,
                "prediction": prediction}
    except:
        raise HTTPException(status_code=404, detail="Model not found")
