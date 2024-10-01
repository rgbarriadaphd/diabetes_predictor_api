"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: prediction_service.py

Description:  Prediction service logic
"""

import numpy as np

from app.ml.regression_model import get_model


def predict_diabetes_progression(model_name, data):
    model = get_model(model_name)

    # Convertir los datos en un array numpy
    features = np.array([[data.age, data.sex, data.bmi, data.bp,
                          data.s1, data.s2, data.s3, data.s4,
                          data.s5, data.s6]])

    # Hacer la predicción
    prediction = model.predict(features)
    return prediction[0]
