"""Public inference service surface."""

from app.inference.adapters.base import BaseSegmentationAdapter, SegmentationAdapter
from app.inference.cache import AdapterCache
from app.inference.gateway import InferenceGateway
from app.inference.registry import (
    ModelArtifactProvenance,
    ModelRegistration,
    ModelRegistryService,
    ValidatedModelBundle,
)
from app.inference.snapshots import ModelArtifactSnapshotError, ModelArtifactSnapshotStore

__all__ = [
    "AdapterCache",
    "BaseSegmentationAdapter",
    "InferenceGateway",
    "ModelArtifactProvenance",
    "ModelArtifactSnapshotError",
    "ModelArtifactSnapshotStore",
    "ModelRegistration",
    "ModelRegistryService",
    "SegmentationAdapter",
    "ValidatedModelBundle",
]
