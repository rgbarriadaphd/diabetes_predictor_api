"""
# Author = ruben
# Date: 15/10/24
# Project: diabetes_predictor_api
# File: model_schema.py

Description: Schema definition for train data
"""
from pydantic import BaseModel, Field
from typing import Optional


class TrainSchema(BaseModel):
    id: Optional[int] = None
    model_name: str = Field(title="Name of the model", max_length=30)
    train_error: float = None
    test_error: float = None

