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
from sklearn.metrics import mean_squared_error

from math import sqrt
import numpy as np
from abc import ABC
from typing import Any

from app.ml.dataset import DiabetesDataset
from app.schemas.model_schema import RegressionModelType


class BaseLinearRegressionModel(ABC):
    def __init__(self, model: Any):
        self.dataset = DiabetesDataset()
        self.model = model

    def train(self) -> (float, float):
        x = self.dataset.data
        y = self.dataset.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        self.model.fit(x_train, y_train)

        # Get prediction of training data
        y_pred_train = self.model.predict(x_train)
        rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))

        # Get prediction of test data
        y_pred_test = self.model.predict(x_test)
        rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))

        return rmse_train, rmse_test

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
def get_model(request_model: RegressionModelType, **kwargs) -> BaseLinearRegressionModel:
    if request_model == "linear":
        return LinearRegressionModel()
    elif request_model == "ridge":
        return RidgeRegressionModel(**kwargs)
    elif request_model == "lasso":
        return LassoRegressionModel(**kwargs)
    else:
        raise ValueError(f"Model '{request_model}' not supported")

