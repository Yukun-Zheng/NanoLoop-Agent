"""Deterministic particle measurements and image-level aggregation."""

from dataclasses import dataclass
from math import pi, sqrt
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from skimage.measure import perimeter

from app.analysis.config import MorphometryConfig
from app.analysis.postprocessing import NormalizedInstance
from app.contracts.analyses import ImageSummaryDTO, ParticleRecordDTO
from app.contracts.enums import QualityStatus
from app.core.errors import InvalidImageError


@dataclass(frozen=True, slots=True)
class MorphometryResult:
    particles: list[ParticleRecordDTO]
    image_summary: ImageSummaryDTO
    warnings: list[str]


def measure(
    *,
    run_id: str,
    instances: list[NormalizedInstance],
    roi_mask: NDArray[np.bool_],
    scale_nm_per_pixel: float | None,
    config: MorphometryConfig,
) -> MorphometryResult:
    roi_area_px = int(np.count_nonzero(roi_mask))
    if roi_area_px == 0:
        raise InvalidImageError(details={"reason": "empty_analysis_roi"})
    if scale_nm_per_pixel is not None and scale_nm_per_pixel <= 0:
        raise ValueError("scale_nm_per_pixel must be positive")

    particles: list[ParticleRecordDTO] = []
    union = np.zeros(roi_mask.shape, dtype=np.bool_)
    perimeter_total_px = 0.0
    for instance in instances:
        area = float(np.count_nonzero(instance.mask))
        particle_perimeter = float(
            perimeter(instance.mask, neighborhood=config.perimeter_neighborhood)
        )
        perimeter_total_px += particle_perimeter
        union |= instance.mask
        equivalent_px = 2.0 * sqrt(area / pi)
        circularity = 4.0 * pi * area / (particle_perimeter**2) if particle_perimeter else None
        if circularity is not None:
            circularity = min(1.0, circularity)
        particles.append(
            ParticleRecordDTO(
                particle_id=f"particle_{uuid4().hex}",
                run_id=run_id,
                instance_index=instance.instance_index,
                area_px=area,
                perimeter_px=particle_perimeter,
                equivalent_diameter_px=equivalent_px,
                equivalent_diameter_nm=(
                    equivalent_px * scale_nm_per_pixel
                    if scale_nm_per_pixel is not None
                    else None
                ),
                circularity=circularity,
                bbox=instance.bbox,
                confidence=instance.confidence,
            )
        )

    count = len(particles)
    coverage_ratio = float(np.count_nonzero(union)) / roi_area_px
    mean_px = (
        float(np.mean([particle.equivalent_diameter_px for particle in particles]))
        if particles
        else None
    )
    number_density_px2 = count / roi_area_px
    perimeter_density_px = perimeter_total_px / roi_area_px
    number_density_um2: float | None = None
    perimeter_density_um: float | None = None
    mean_nm: float | None = None
    warnings: list[str] = []
    if scale_nm_per_pixel is not None:
        roi_area_um2 = roi_area_px * (scale_nm_per_pixel / 1000.0) ** 2
        number_density_um2 = count / roi_area_um2
        perimeter_total_um = perimeter_total_px * scale_nm_per_pixel / 1000.0
        perimeter_density_um = perimeter_total_um / roi_area_um2
        mean_nm = mean_px * scale_nm_per_pixel if mean_px is not None else None
    else:
        warnings.append("physical_scale_missing_pixel_metrics_only")

    summary = ImageSummaryDTO(
        run_id=run_id,
        particle_count=count,
        roi_area_px=roi_area_px,
        number_density_px2=number_density_px2,
        number_density_um2=number_density_um2,
        mean_equivalent_diameter_px=mean_px,
        mean_equivalent_diameter_nm=mean_nm,
        coverage_ratio=coverage_ratio,
        perimeter_density_px=perimeter_density_px,
        perimeter_density_um=perimeter_density_um,
        quality_status=QualityStatus.PASS,
    )
    return MorphometryResult(particles=particles, image_summary=summary, warnings=warnings)
