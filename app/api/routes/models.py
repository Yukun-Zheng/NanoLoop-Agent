"""Model registry listing and explicit recommendation endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool

from app.analysis.authorization import require_read
from app.api.deps import (
    get_file_store,
    get_inference_gateway,
    get_repositories,
    require_api_key_contract,
)
from app.api.interop import invoke
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.common import ApiResponse
from app.contracts.enums import ModelFamily, ModelStatus, ModelVariant, QualityTier
from app.contracts.identity import PrincipalContext
from app.contracts.models import (
    ModelCandidate,
    ModelListData,
    ModelMetadata,
    ModelRecommendationData,
    ModelRecommendationRequest,
)
from app.db.repositories import SqlAlchemyRepositorySet
from app.inference.profiling import profile_image_for_model
from app.storage import LocalFileStore
from app.storage.paths import StoragePathError

router = APIRouter(prefix="/models", tags=["models"], responses=COMMON_ERROR_RESPONSES)


@router.get("", response_model=ApiResponse[ModelListData], operation_id="listModels")
async def list_models(
    request: Request,
    gateway: Annotated[Any, Depends(get_inference_gateway)],
    status: Annotated[ModelStatus | None, Query()] = None,
    family: Annotated[ModelFamily | None, Query()] = None,
    variant: Annotated[ModelVariant | None, Query()] = None,
    quality_tier: Annotated[QualityTier | None, Query()] = None,
    material: Annotated[str | None, Query(min_length=1, max_length=255)] = None,
) -> ApiResponse[ModelListData]:
    raw = await invoke(gateway, "list_models", only_ready=status == ModelStatus.READY)
    models = _model_records(raw)
    if status is not None:
        models = [model for model in models if model.status == status]
    if family is not None:
        models = [model for model in models if model.family == family]
    if variant is not None:
        models = [model for model in models if model.variant == variant]
    if quality_tier is not None:
        models = [model for model in models if model.quality_tier == quality_tier]
    if material is not None:
        expected = material.casefold()
        models = [
            model
            for model in models
            if any(candidate.casefold() == expected for candidate in model.applicable_materials)
        ]
    return success_response(ModelListData(models=models), request=request)


@router.post(
    "/recommend",
    response_model=ApiResponse[ModelRecommendationData],
    operation_id="recommendModels",
)
async def recommend_models(
    payload: ModelRecommendationRequest,
    request: Request,
    gateway: Annotated[Any, Depends(get_inference_gateway)],
    repositories: Annotated[SqlAlchemyRepositorySet, Depends(get_repositories)],
    file_store: Annotated[LocalFileStore, Depends(get_file_store)],
    principal: Annotated[PrincipalContext, Depends(require_api_key_contract)],
) -> ApiResponse[ModelRecommendationData]:
    profile_reason: str | None = None
    recommendation_request = payload
    if payload.auto_profile:
        tenant_id = principal.tenant_id
        if tenant_id is None:
            raise ValueError("principal must carry a tenant ID")
        job_id = payload.job_id
        if job_id is None:
            raise ValueError("auto_profile requires job_id")
        scope = repositories.jobs.get_scope(job_id, tenant_id=tenant_id)
        require_read(principal, scope)
        image = repositories.images.get_scoped(
            job_id,
            payload.image_id,
            tenant_id=tenant_id,
        )
        storage_path = repositories.images.get_storage_path_scoped(
            job_id,
            payload.image_id,
            tenant_id=tenant_id,
        )
        try:
            managed_path = file_store.paths.require_managed(
                storage_path,
                must_exist=True,
            )
            profile = await run_in_threadpool(
                profile_image_for_model,
                managed_path,
                image.analysis_roi.valid_rect,
            )
        except (OSError, StoragePathError, ValueError):
            profile_reason = "图像预检不可用，已使用通用模型配置"
        else:
            recommendation_request = payload.model_copy(
                update={"target_profile": profile.variant}
            )
            profile_reason = profile.reason

    raw = await invoke(gateway, "recommend", recommendation_request)
    if isinstance(raw, ModelRecommendationData):
        data = raw
    elif isinstance(raw, dict) and "candidates" in raw:
        data = ModelRecommendationData.model_validate(raw)
    else:
        data = ModelRecommendationData(
            candidates=[ModelCandidate.model_validate(item) for item in raw]
        )
    if profile_reason:
        data = data.model_copy(
            update={
                "candidates": [
                    candidate.model_copy(
                        update={"reasons": [profile_reason, *candidate.reasons]}
                    )
                    for candidate in data.candidates
                ]
            }
        )
    return success_response(data, request=request)


def _model_records(value: Any) -> list[ModelMetadata]:
    if isinstance(value, ModelListData):
        return value.models
    if isinstance(value, dict) and "models" in value:
        value = value["models"]
    if not isinstance(value, list):
        raise TypeError("InferenceGateway.list_models must return a list or ModelListData")
    return [ModelMetadata.model_validate(item) for item in value]
