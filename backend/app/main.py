from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(
    title="EarthVision AI",
    description="Satellite Image Analysis & Change Detection",
    version="0.1.0",
)


# Allow React frontend to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve satellite images
app.mount(
    "/data",
    StaticFiles(directory="/mnt/d/pocs/EarthVisionAI/data"),
    name="data",
)

# Serve generated and reference result images
app.mount(
    "/results",
    StaticFiles(
        directory="/mnt/d/pocs/EarthVisionAI/results"
    ),
    name="results",
)

app.include_router(router, prefix="/api/v1")


@app.get("/api/v1/health")
def health():
    return {
        "status": "UP",
        "service": "EarthVision AI",
        "version": "0.1.0",
    }