from contextlib import asynccontextmanager

from backend.app.api.v1.api_router import api_router
from backend.app.core.config import get_settings
from backend.app.core.errors import register_exception_handlers
from backend.app.core.logging import setup_logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Initializing AERIS API service...")
    logger.info(f"Project: {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT} | Debug: {settings.DEBUG}")
    logger.info(f"Allowed CORS Origins: {settings.cors_origins_list}")
    yield
    logger.info("Shutting down AERIS API service...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_FULL_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan,
    )

    # Configure CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Global Exception Handlers (RFC 7807)
    register_exception_handlers(app)

    # Include API Routers
    app.include_router(api_router, prefix=settings.API_V1_STR)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
    )
