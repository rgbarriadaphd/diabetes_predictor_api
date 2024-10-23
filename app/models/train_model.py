"""
# Author = ruben
# Date: 23/10/24
# Project: diabetes_predictor_api
# File: prediction_model.py

Description: train model for database
"""
from sqlalchemy import Column, Integer, String, Float

from app.database import Base

class TrainModel(Base):
    __tablename__ = "train_model"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    train_error = Column(Float)
    test_error = Column(Float)

