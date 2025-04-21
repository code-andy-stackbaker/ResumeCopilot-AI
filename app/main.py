from fastapi import FastAPI
from app.api.v1.routes import router as v1_router

app = FastAPI(title="Resume Keyword Generator")

# Include all versioned routes
app.include_router(v1_router, prefix="/api/v1")