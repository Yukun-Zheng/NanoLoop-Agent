"""Model-family adapters with lazy family-module imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.inference.adapters.base import BaseSegmentationAdapter, SegmentationAdapter

if TYPE_CHECKING:
    from app.inference.adapters.sam2 import SAM2Adapter
    from app.inference.adapters.unet import UNetAdapter
    from app.inference.adapters.yolo_seg import YOLOSegAdapter

__all__ = [
    "BaseSegmentationAdapter",
    "SAM2Adapter",
    "SegmentationAdapter",
    "UNetAdapter",
    "YOLOSegAdapter",
]


def __getattr__(name: str) -> Any:
    """Load a concrete adapter only when explicitly requested."""

    if name == "UNetAdapter":
        from app.inference.adapters.unet import UNetAdapter

        return UNetAdapter
    if name == "YOLOSegAdapter":
        from app.inference.adapters.yolo_seg import YOLOSegAdapter

        return YOLOSegAdapter
    if name == "SAM2Adapter":
        from app.inference.adapters.sam2 import SAM2Adapter

        return SAM2Adapter
    raise AttributeError(name)
