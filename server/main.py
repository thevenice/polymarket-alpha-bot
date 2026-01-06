"""FastAPI application for Alphapoly."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server import __version__
from server.routers import data, pipeline, prices

app = FastAPI(
    title="Alphapoly API",
    description="API for Polymarket alpha detection pipeline",
    version=__version__,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/data", tags=["data"])
app.include_router(prices.router, prefix="/prices", tags=["prices"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Alphapoly API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "data": "/data",
            "prices": "/prices",
            "pipeline": "/pipeline",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
