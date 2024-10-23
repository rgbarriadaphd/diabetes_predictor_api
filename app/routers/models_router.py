"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: models_router.py

Description: Define routers for models
"""
from typing import Union
from app.database import DB_DEPENDENCY
from fastapi import APIRouter, status
from app.models.regression_model import RegressionModel
from app.schemas.model_schema import RegressionModelSchema

router = APIRouter(
    prefix="/models",
    tags=["models"]
)


@router.get("", status_code=status.HTTP_200_OK,
            name='Get model list by trained',
            summary='Get model list by trained')
async def get_models(db: DB_DEPENDENCY, trained: Union[bool, None] = None):
    if trained is None:
        return []
    trained_models = db.query(RegressionModel).filter(RegressionModel.trained == trained).all()
    if trained_models:
        return trained_models
    else:
        return []


@router.get("/all", status_code=status.HTTP_200_OK,
            name='Get all models',
            summary='Get all models')
async def get_all_models(db: DB_DEPENDENCY):
    return db.query(RegressionModel).all()


@router.post("/creation", status_code=status.HTTP_201_CREATED,
             name='Create new model',
             summary='Create new model')
async def create_model(db: DB_DEPENDENCY, model_request: RegressionModelSchema):
    regression_model = RegressionModel(**model_request.model_dump())

    db.add(regression_model)
    db.commit()
    db.refresh(regression_model)

    return regression_model
