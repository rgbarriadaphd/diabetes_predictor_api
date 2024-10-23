"""
# Author = ruben
# Date: 15/10/24
# Project: diabetes_predictor_api
# File: model_schema.py

Description: Schema definition for model data
"""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class RegressionModelType(str, Enum):
    linear = "linear"
    ridge = "ridge"
    lasso = "lasso"

class RegressionModelSchema(BaseModel):
    id: Optional[int] = None
    name: str = Field(title="Name of the model", max_length=30)
    description: str = Field(title="Description of the model", min_length=5, max_length=300)
    model_type: RegressionModelType
    trained: bool = Field(title="Whether the model is trained or not")
    train_error: Optional[float] = None
    test_error: Optional[float] = None
