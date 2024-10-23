"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: predict_router.py

Description: Define routers for predictions.
"""

from fastapi import APIRouter, HTTPException

from app.database import DB_DEPENDENCY
from app.ml.regression_model import get_model
from app.ml.transformations import normalize_sample
from app.models.prediction_model import PredictionModel
from app.schemas.diabetes_schema import DiabetesData
from app.schemas.model_schema import RegressionModelType
from app.schemas.prediction_schema import PredictionSchema
from app.services.prediction_service import predict_diabetes_progression

router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)


@router.post("", response_model=PredictionSchema,
             name='Sample diabetes prediction',
             summary='Predicts diabetes progression from a sample',
             description='Predict diabetes progression specifying a model')
def predict(db: DB_DEPENDENCY, request_model: RegressionModelType, data: DiabetesData):
    try:
        model = get_model(request_model)
        train_rmse, test_rmse = model.train()
        normalized = normalize_sample(data)
        prediction = predict_diabetes_progression(model, normalized)
        model_data = {"model_name": request_model,
                      "train_error": train_rmse,
                      "test_error": test_rmse,
                      "prediction": prediction}

        predict_model = PredictionModel(**model_data)
        db.add(predict_model)
        db.commit()
        db.refresh(predict_model)

        return predict_model



    except:
        raise HTTPException(status_code=404, detail="Model not found")
