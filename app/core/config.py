"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: config.py

Description: General project settings
"""

class Settings:
    PROJECT_NAME: str = "Diabetes Progression Predictor API"
    MODEL_PATH: str = "app/ml/trained_models/diabetes_model.pkl"

settings = Settings()

