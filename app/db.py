from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, Resume
from dotenv import load_dotenv
from fastapi import FastAPI
import os
load_dotenv()
app=FastAPI()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        


