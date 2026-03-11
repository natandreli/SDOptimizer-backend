from fastapi import APIRouter

router = APIRouter(tags=["Healthcheck"])


@router.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "SDOptimizer Backend"}
