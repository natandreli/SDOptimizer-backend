from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info("Starting up SDOptimizer Backend...")

    yield

    logger.info("Shutting down...")
