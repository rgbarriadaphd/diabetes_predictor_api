"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: regression_model.py

Description:  Train and load the model.
"""
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from app.core.config import settings
import numpy as np
from abc import ABC
from typing import Any


class BaseLinearRegressionModel(ABC):
    def __init__(self, model: Any):
        self.model = model

    def train(self) -> float:
        diabetes = load_diabetes()
        x = diabetes.data
        y = diabetes.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.model.fit(x_train, y_train)
        return self.model.score(x_test, y_test)

    def predict(self, x: np.ndarray) -> Any:
        return self.model.predict(x)


class LinearRegressionModel(BaseLinearRegressionModel):
    def __init__(self):
        super().__init__(LinearRegression())


class RidgeRegressionModel(BaseLinearRegressionModel):
    def __init__(self, alpha: float = 1.0):
        super().__init__(Ridge(alpha=alpha))


class LassoRegressionModel(BaseLinearRegressionModel):
    def __init__(self, alpha: float = 1.0):
        super().__init__(Lasso(alpha=alpha))


# Factory to instance the model based on name
def get_model(model_name: str, **kwargs) -> BaseLinearRegressionModel:
    print(f"** Model name: {model_name}")
    if model_name == "LinearRegression":
        return LinearRegressionModel()
    elif model_name == "Ridge":
        return RidgeRegressionModel(**kwargs)
    elif model_name == "Lasso":
        return LassoRegressionModel(**kwargs)
    else:
        raise ValueError(f"Model '{model_name}' not supported")

