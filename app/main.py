from fastapi import FastAPI
from app.init_db import init_db
from app.routes.papers import router as papers_router
from dotenv import load_dotenv
import os

load_dotenv()   # 👈 this is important

DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

app = FastAPI(title="Scientific Literature Explorer API")

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(papers_router)

@app.get("/")
def root():
    return {"status": "Scientific Explorer API running"}
