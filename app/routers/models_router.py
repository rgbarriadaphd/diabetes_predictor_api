"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: models_router.py

Description: Define routers for models
"""
from typing import Union

from fastapi import APIRouter

MODEL_LIST = [
    {"id": 1, "name": "LinearRegression",
     "description": "Ordinary least squares Linear Regression",
     "trained": True},
    {"id": 2, "name": "RidgeRegression",
     "description": "Ridge regression imposes a penalty on the size of the coefficients",
     "trained": False},
    {"id": 3, "name": "LassoRegression",
     "description": "The Lasso is a linear model that estimates sparse coefficients",
     "trained": False},
]

router = APIRouter(
    prefix="/models"
)


@router.get("")
async def get_models(trained: Union[bool, None] = None):
    if trained is not None:
        filtered_models = list(filter(lambda model: model["trained"] == trained, MODEL_LIST))
        return filtered_models
    return MODEL_LIST
