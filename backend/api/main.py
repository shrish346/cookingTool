"""
FastAPI entry point for Chef's Loop API.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routes import router
from .deps import get_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes and cleans up resources.
    """
    # Startup: Initialize connections
    settings = get_settings()
    print(f"Starting {settings.app_name}...")
    
    # Pre-warm Redis connection
    redis_client = await get_redis()
    await redis_client.ping()
    print("Redis connection established")
    
    yield
    
    # Shutdown: Cleanup
    redis_client = await get_redis()
    await redis_client.close()
    print("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="Transform cooking videos into step-by-step recipes with looping video clips",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api", tags=["recipes"])
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "app": settings.app_name}
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)

