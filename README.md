# Diabetes Progression Predictor API

## Overview

This project uses a machine learning model to predict diabetes progression based on patient medical data. It uses the
scikit-learn diabetes dataset (https://scikit-learn.org/1.5/datasets/toy_dataset.html#diabetes-dataset) and creates an
API with FastAPI to serve predictions.

## Project Description

The Diabetes Progression Predictor API is a web-based API that takes in patient medical data and returns a prediction of
the patient's diabetes progression. The API uses a trained machine learning model to make predictions based on the input
data.

## Data Set Characteristics:

Ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements were
obtained for each of n = 442 diabetes patients, as well as the response of interest, a quantitative measure of disease
progression one year after baseline.

Number of Instances: 442

Number of Attributes: First 10 columns are numeric predictive values

Target: Column 11 is a quantitative measure of disease progression one year after baseline

Attribute Information:

- age
- sex
- bmi body mass index
- bp average blood pressure
- s1 tc, total serum cholesterol
- s2 ldl, low-density lipoproteins
- s3 hdl, high-density lipoproteins
- s4 tch, total cholesterol / HDL
- s5 ltg, possibly log of serum triglycerides level
- s6 glu, blood sugar level

Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times the square
root of n_samples (i.e. the sum of squares of each column totals 1).

Source URL: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

For more information see: Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) “Least Angle
Regression,” Annals of Statistics (with discussion),
407-499. (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)


## API Endpoints
The API has several endpoints that allow users to interact with the model:

/predict: Takes in patient medical data and returns a prediction of the patient's diabetes progression.
/train: Trains the machine learning model using the scikit-learn diabetes dataset.
/models: Returns a list of available machine learning models.

## Requirements
- Python 3.7+
- FastAPI
- scikit-learn
- numpy
- pandas

## Installation
To install the API, run the following command:

```bash
pip install -r requirements.txt
```
## Running the API

To run the API at localhost, execute the following command:
```bash
uvicorn main:app --reload
```

## API Documentation
The API documentation can be found at /docs.