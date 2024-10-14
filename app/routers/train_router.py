"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: train_router.py

Description: Define routers for train
"""
from fastapi import APIRouter, HTTPException

from app.ml.regression_model import get_model

router = APIRouter(
    prefix="/train"
)


@router.post("/{model_name}")
def train(model_name: str, alpha: float = 1.0):
    try:
        model = get_model(model_name, alpha=alpha)
        rmse = model.train()
        return {
            "message": f"Model {model_name} trained successfully.",
            "rmse": rmse
        }
    except:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
