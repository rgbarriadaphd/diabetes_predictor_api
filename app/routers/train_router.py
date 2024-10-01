"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: train_router.py

Description: Define routes for train
"""
from fastapi import APIRouter, HTTPException

from app.ml.regression_model import get_model

router = APIRouter(
    prefix="/train"
)


@router.post("/{model_name}")
def predict(model_name: str, alpha: float = 1.0):
    try:
        model = get_model(model_name, alpha=alpha)
        metrics = model.train()
        return {
            "message": f"Model {model_name} trained successfully.",
            "metrics": metrics
        }

    except:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
