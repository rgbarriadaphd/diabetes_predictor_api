"""
# Author = ruben
# Date: 23/10/24
# Project: diabetes_predictor_api
# File: database.py

Description: Manages a database connection
"""
from typing import Annotated

from fastapi import Depends
from requests import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

SQL_ALCHEMY_DATABASE_URL = "sqlite:///./regression.db"

engine = create_engine(SQL_ALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

DB_DEPENDENCY = Annotated[Session, Depends(get_db)]