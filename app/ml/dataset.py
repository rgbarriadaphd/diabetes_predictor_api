"""
# Author = ruben
# Date: 15/10/24
# Project: diabetes_predictor_api
# File: dataset.py

Description: Implementation of dataset class to load input data
"""
from sklearn.datasets import load_diabetes
import numpy as np

from app.schemas.diabetes_schema import DiabetesData

"""
From original source: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.read.tab.out.txt

This means and std deviation of the original dataset will be usedfor sample normalization

The MEANS Procedure
-----------------------------------------------
Variable      N            Mean         Std Dev
-----------------------------------------------
age         442      48.5180995      13.1090278
sex         442       1.4683258       0.4995612
bmi         442      26.3757919       4.4181216
bp          442      94.6470136      13.8312834
s1          442     189.1402715      34.6080517
s2          442     115.4391403      30.4130810
s3          442      49.7884615      12.9342022
s4          442       4.0702489       1.2904499
s5          442       4.6414109       0.5223906
s6          442      91.2601810      11.4963347
y           442     152.1334842      77.0930045
-----------------------------------------------
"""


def from_numpy(array: np.ndarray) -> DiabetesData:
    field_names = list(DiabetesData.model_fields.keys())

    if len(array) != len(field_names):
        raise ValueError("Array size does not match number of fields of DiabetesData")

    data_dict = {field_names[i]: array[i] for i in range(len(field_names))}
    return DiabetesData(**data_dict)


class DiabetesDataset:
    # Unique dataset instance
    _instance = None

    # Constant class attributes for sample normalization.
    # Means and std dev of each feature from original dataset. They won't change after initialization
    N_SAMPLES = 442
    MEANS = np.array([48.5180995, 1.4683258, 26.3757919, 94.6470136, 189.1402715,
                      115.4391403, 49.7884615, 4.0702489, 4.6414109, 91.2601810])
    STD_DEVS = np.array([13.1090278, 0.4995612, 4.4181216, 13.8312834, 34.6080517,
                         30.4130810, 12.9342022, 1.2904499, 0.5223906, 11.4963347])

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiabetesDataset, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):
        dataset = load_diabetes()
        self.data = dataset.data
        self.target = dataset.target
        self.feature_names = dataset.feature_names
        self.DESCR = dataset.DESCR

    def normalize(self, sample: DiabetesData) -> DiabetesData:
        np_sample = np.array(list(sample.model_dump().values()))
        normalized_sample = (np_sample - self.MEANS) / (self.STD_DEVS * np.sqrt(self.N_SAMPLES))
        return from_numpy(normalized_sample)

    def get_feature_names(self):
        return self.feature_names
