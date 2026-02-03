from fastapi import FastAPI
from app.init_db import init_db
from app.routes.papers import router as papers_router

app = FastAPI(title="Scientific Literature Explorer API")

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(papers_router)

@app.get("/")
def root():
    return {"status": "Scientific Explorer API running"}
