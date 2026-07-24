"""Route modules for the frozen v1 API."""

from fastapi import APIRouter, Depends

from app.api.deps import require_api_key_contract
from app.api.routes import (
    analyses,
    boxes,
    conversations,
    exports,
    files,
    health,
    knowledge,
    models,
    queries,
    runs,
)

api_router = APIRouter(dependencies=[Depends(require_api_key_contract)])
api_router.include_router(health.router)
api_router.include_router(models.router)
api_router.include_router(analyses.router)
api_router.include_router(boxes.router)
api_router.include_router(runs.router)
api_router.include_router(queries.router)
api_router.include_router(conversations.router)
api_router.include_router(knowledge.router)
api_router.include_router(exports.router)
api_router.include_router(files.router)

__all__ = ["api_router"]
