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
from app.schemas.model_schema import RegressionModelType, ModelData
from app.services.prediction_service import predict_diabetes_progression

router = APIRouter(
    prefix="/predict"
)


@router.post("", response_model=ModelData,
            name='Sample diabetes prediction',
            summary='Predicts diabetes progression from a sample',
            description='Predict diabetes progression specifying a model')
def predict(request_model: RegressionModelType, data: DiabetesData):
    try:
        model = get_model(request_model)
        train_rmse, test_rmse = model.train()
        normalized = normalize_sample(data)
        prediction = predict_diabetes_progression(model, normalized)
        return {"model": request_model,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "prediction": prediction}
    except:
        raise HTTPException(status_code=404, detail="Model not found")
