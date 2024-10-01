"""
# Author = ruben
# Date: 1/10/24
# Project: diabetes_predictor_api
# File: main.py

Description: Application entry point.
"""
from fastapi import FastAPI

from app.routers import predict_router, models_router, train_router

app = FastAPI()
app.title = "Diabetes Progression Predictor API"


app.include_router(predict_router.router)
app.include_router(models_router.router)
app.include_router(train_router.router)

@app.get("/")
async def home():
    return {"name": "Diabetes Progression Predictor API",
            "version": "1.0.0"}
