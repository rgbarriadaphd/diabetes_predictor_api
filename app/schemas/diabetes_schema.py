"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: diabetes_schema.py

Description: "Enter description here"
"""
from pydantic import BaseModel

class DiabetesData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
