"""
# Author = ruben
# Date: 14/10/24
# Project: diabetes_predictor_api
# File: transformations.py

Description: Data transformations utility
"""
import numpy as np

from app.ml.dataset import DiabetesDataset
from app.schemas.diabetes_schema import DiabetesData

# Dataset instance
diabetes_dataset = DiabetesDataset()

def normalize_sample(data: DiabetesData) -> DiabetesData:
    return diabetes_dataset.normalize(data)


