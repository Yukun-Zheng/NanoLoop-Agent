"""Persistable mappings between original pixels and an analysis crop."""

from typing import Literal

from pydantic import Field

from app.contracts.analyses import PixelRect, ROIBox
from app.contracts.common import ContractModel


class TransformRecord(ContractModel):
    schema_version: Literal[1] = 1
    original_width: int = Field(gt=0)
    original_height: int = Field(gt=0)
    crop: PixelRect
    model_width: int | None = Field(default=None, gt=0)
    model_height: int | None = Field(default=None, gt=0)
    pad_left: int = Field(default=0, ge=0)
    pad_top: int = Field(default=0, ge=0)

    @property
    def analysis_width(self) -> int:
        return self.crop.x2 - self.crop.x1

    @property
    def analysis_height(self) -> int:
        return self.crop.y2 - self.crop.y1

    def original_to_analysis(self, box: ROIBox) -> ROIBox:
        return box.model_copy(
            update={
                "x1": box.x1 - self.crop.x1,
                "x2": box.x2 - self.crop.x1,
                "y1": box.y1 - self.crop.y1,
                "y2": box.y2 - self.crop.y1,
            }
        )

    def analysis_to_original(self, box: ROIBox) -> ROIBox:
        return box.model_copy(
            update={
                "x1": box.x1 + self.crop.x1,
                "x2": box.x2 + self.crop.x1,
                "y1": box.y1 + self.crop.y1,
                "y2": box.y2 + self.crop.y1,
            }
        )
