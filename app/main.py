"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: main.py

Description: Application entry point.
"""
from fastapi import FastAPI

from app.database import engine
from app.ml.dataset import DiabetesDataset
from app.models import regression_model, prediction_model, train_model
from app.routers import predict_router, models_router, train_router

app = FastAPI()
app.title = "Diabetes Progression Predictor API"

app.include_router(predict_router.router)
app.include_router(models_router.router)
app.include_router(train_router.router)

# dataset instance (Singleton)
diabetes_dataset = DiabetesDataset()

# database creation
regression_model.Base.metadata.create_all(bind=engine)
prediction_model.Base.metadata.create_all(bind=engine)
train_model.Base.metadata.create_all(bind=engine)

@app.get("/")
async def home():
    return {"name": "Diabetes Progression Predictor API",
            "version": "1.0.1"}
