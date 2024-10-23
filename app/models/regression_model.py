"""
# Author = ruben
# Date: 23/10/24
# Project: diabetes_predictor_api
# File: regression_model.py

Description: regression model for database
"""
from sqlalchemy import Boolean, Column, Integer, String, Float

from app.database import Base


class RegressionModel(Base):
    __tablename__ = "regression_model"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    model_type = Column(String)
    trained = Column(Boolean, default=False)
    train_error = Column(Float)
    test_error = Column(Float)
