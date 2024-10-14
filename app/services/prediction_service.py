"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: prediction_service.py

Description:  Prediction service logic
"""
import numpy as np

def predict_diabetes_progression(model, data):
    features = np.array([[data.age, data.sex, data.bmi, data.bp,
                          data.s1, data.s2, data.s3, data.s4,
                          data.s5, data.s6]])
    prediction = model.predict(features)
    return prediction[0]
