from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.init_db import init_db
from app.routes.papers import router as papers_router
from dotenv import load_dotenv
import os

# Load .env only for local dev (Render/Railway already provide env vars)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-prod")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

app = FastAPI(title="Scientific Literature Explorer API")

# ✅ CORS: allow your Streamlit UI to call this API
# For production you can replace "*" with your UI domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(papers_router)

@app.get("/")
def root():
    return {"status": "Scientific Explorer API running"}
