"""
# Author = ruben
# Date: 14/10/24
# Project: diabetes_predictor_api
# File: transformations.py

Description: Data transformations utility
"""
import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd

from app.schemas.diabetes_schema import DiabetesData


def get_diabetes_stats():
    diabetes_data = load_diabetes()
    df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)

    means = df.mean().to_dict()
    std_devs = df.std().to_dict()
    n_samples = len(df)

    return means, std_devs, n_samples


def normalize_sample(data: DiabetesData) -> DiabetesData:
    normalized_values = {}
    feature_names = data.model_fields.keys()
    means, std_devs, n_samples = get_diabetes_stats()
    for i, field in enumerate(feature_names):
        mean = means[field]
        std_dev = std_devs[field]
        normalized_value = (getattr(data, field) - mean) / (std_dev * np.sqrt(n_samples))
        normalized_values[field] = normalized_value
    return DiabetesData(**normalized_values)

