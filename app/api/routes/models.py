"""Model registry listing and explicit recommendation endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import get_inference_gateway
from app.api.interop import invoke
from app.api.responses import success_response
from app.api.routing import COMMON_ERROR_RESPONSES
from app.contracts.common import ApiResponse
from app.contracts.enums import ModelFamily, ModelStatus, ModelVariant, QualityTier
from app.contracts.models import (
    ModelCandidate,
    ModelListData,
    ModelMetadata,
    ModelRecommendationData,
    ModelRecommendationRequest,
)

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
) -> ApiResponse[ModelRecommendationData]:
    raw = await invoke(gateway, "recommend", payload)
    if isinstance(raw, ModelRecommendationData):
        data = raw
    elif isinstance(raw, dict) and "candidates" in raw:
        data = ModelRecommendationData.model_validate(raw)
    else:
        data = ModelRecommendationData(
            candidates=[ModelCandidate.model_validate(item) for item in raw]
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
