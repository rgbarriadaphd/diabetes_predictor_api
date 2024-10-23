"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: train_router.py

Description: Define routers for train
"""
from fastapi import APIRouter, HTTPException

from app.database import DB_DEPENDENCY
from app.ml.regression_model import get_model
from app.models.train_model import TrainModel
from app.schemas.model_schema import RegressionModelType
from app.schemas.train_schema import TrainSchema

router = APIRouter(
    prefix="/train",
    tags=["train"]
)


@router.post("", response_model=TrainSchema,
            name='Train a model',
            summary='Train a linear model for regression task',
            description='Train a linear model for regression task')
def train(db: DB_DEPENDENCY, request_model: RegressionModelType, alpha: float = 1.0):
    try:
        model = get_model(request_model, alpha=alpha)
        train_rmse, test_rmse = model.train()
        model_data = {"model_name": request_model,
                      "train_error": train_rmse,
                      "test_error": test_rmse}
        train_model = TrainModel(**model_data)

        db.add(train_model)
        db.commit()
        db.refresh(train_model)

        return train_model
    except:
        raise HTTPException(status_code=404, detail=f"Model {request_model} not found")
