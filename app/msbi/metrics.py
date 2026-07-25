"""Pixel, boundary, pseudo-instance, count, and morphology metrics for MSBI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance


@dataclass(frozen=True, slots=True)
class MatchCounts:
    true_positive: int
    false_positive: int
    false_negative: int
    matched_ious: tuple[float, ...]


def _labels(binary: NDArray[np.generic], minimum_area: int = 0) -> NDArray[np.int32]:
    labels, count = ndi.label(
        np.asarray(binary, dtype=bool),
        structure=np.ones((3, 3), dtype=bool),
    )
    if minimum_area > 1 and count:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= minimum_area
        keep[0] = False
        labels, _ = ndi.label(
            keep[labels],
            structure=np.ones((3, 3), dtype=bool),
        )
    return np.asarray(labels, dtype=np.int32)


def _instance_iou_matrix(
    truth: NDArray[np.int32],
    prediction: NDArray[np.int32],
) -> NDArray[np.float64]:
    truth_count = int(truth.max())
    prediction_count = int(prediction.max())
    matrix = np.zeros((truth_count, prediction_count), dtype=np.float64)
    truth_sizes = np.bincount(truth.ravel(), minlength=truth_count + 1)
    prediction_sizes = np.bincount(prediction.ravel(), minlength=prediction_count + 1)
    paired = np.stack((truth.ravel(), prediction.ravel()), axis=1)
    foreground_pairs = paired[(paired[:, 0] > 0) & (paired[:, 1] > 0)]
    if foreground_pairs.size == 0:
        return matrix
    pairs, intersections = np.unique(foreground_pairs, axis=0, return_counts=True)
    for (truth_id, prediction_id), intersection in zip(
        pairs,
        intersections,
        strict=True,
    ):
        union = (
            truth_sizes[truth_id]
            + prediction_sizes[prediction_id]
            - intersection
        )
        matrix[truth_id - 1, prediction_id - 1] = intersection / union
    return matrix


def _match(
    truth: NDArray[np.int32],
    prediction: NDArray[np.int32],
    threshold: float,
) -> MatchCounts:
    matrix = _instance_iou_matrix(truth, prediction)
    if matrix.size:
        truth_indices, prediction_indices = linear_sum_assignment(-matrix)
        matched = tuple(
            float(matrix[truth_id, prediction_id])
            for truth_id, prediction_id in zip(
                truth_indices,
                prediction_indices,
                strict=True,
            )
            if matrix[truth_id, prediction_id] >= threshold
        )
    else:
        matched = ()
    true_positive = len(matched)
    return MatchCounts(
        true_positive=true_positive,
        false_positive=int(prediction.max()) - true_positive,
        false_negative=int(truth.max()) - true_positive,
        matched_ious=matched,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _boundary(binary: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return np.asarray(binary & ~ndi.binary_erosion(binary), dtype=bool)


def _boundary_f1(
    truth: NDArray[np.bool_],
    prediction: NDArray[np.bool_],
    tolerance_px: int,
) -> float:
    truth_boundary = _boundary(truth)
    prediction_boundary = _boundary(prediction)
    if not truth_boundary.any() and not prediction_boundary.any():
        return 1.0
    truth_dilated = ndi.binary_dilation(truth_boundary, iterations=tolerance_px)
    prediction_dilated = ndi.binary_dilation(prediction_boundary, iterations=tolerance_px)
    precision = _safe_divide(
        float(np.count_nonzero(prediction_boundary & truth_dilated)),
        float(np.count_nonzero(prediction_boundary)),
    )
    recall = _safe_divide(
        float(np.count_nonzero(truth_boundary & prediction_dilated)),
        float(np.count_nonzero(truth_boundary)),
    )
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _hd95(truth: NDArray[np.bool_], prediction: NDArray[np.bool_]) -> float | None:
    truth_boundary = _boundary(truth)
    prediction_boundary = _boundary(prediction)
    if not truth_boundary.any() or not prediction_boundary.any():
        return None
    distance_to_truth = ndi.distance_transform_edt(~truth_boundary)
    distance_to_prediction = ndi.distance_transform_edt(~prediction_boundary)
    distances = np.concatenate(
        (
            distance_to_truth[prediction_boundary],
            distance_to_prediction[truth_boundary],
        )
    )
    return float(np.percentile(distances, 95))


def _equivalent_diameters(labels: NDArray[np.int32]) -> NDArray[np.float64]:
    sizes = np.bincount(labels.ravel())[1:]
    return 2.0 * np.sqrt(sizes.astype(np.float64) / np.pi)


def evaluate_calibration_metrics(
    truth_binary: NDArray[np.generic],
    prediction_labels: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.generic],
    boundary_tolerance_px: int = 2,
) -> dict[str, float]:
    """Compute the four gate-facing metrics used by bounded decoder search."""

    valid = np.asarray(valid_mask, dtype=bool)
    truth = np.asarray(truth_binary, dtype=bool) & valid
    labels = np.where(
        valid,
        np.asarray(prediction_labels, dtype=np.int32),
        0,
    ).astype(np.int32)
    if labels.shape != truth.shape:
        raise ValueError("prediction_labels shape must match truth")
    prediction = labels > 0
    true_positive_px = int(np.count_nonzero(truth & prediction))
    false_positive_px = int(np.count_nonzero(~truth & prediction & valid))
    false_negative_px = int(np.count_nonzero(truth & ~prediction))
    dice = _safe_divide(
        2.0 * true_positive_px,
        2.0 * true_positive_px + false_positive_px + false_negative_px,
    )
    truth_labels = _labels(truth)
    match = _match(truth_labels, labels, 0.5)
    instance_precision = _safe_divide(
        match.true_positive,
        match.true_positive + match.false_positive,
    )
    instance_recall = _safe_divide(
        match.true_positive,
        match.true_positive + match.false_negative,
    )
    return {
        "pixel_dice": dice,
        "boundary_f1": _boundary_f1(
            truth,
            prediction,
            boundary_tolerance_px,
        ),
        "instance_f1_50": _safe_divide(
            2.0 * instance_precision * instance_recall,
            instance_precision + instance_recall,
        ),
        "count_absolute_error": float(
            abs(int(labels.max()) - int(truth_labels.max()))
        ),
    }


def evaluate_binary_instance_prediction(
    truth_binary: NDArray[np.generic],
    prediction_binary: NDArray[np.generic],
    *,
    valid_mask: NDArray[np.generic],
    prediction_min_area_px: int = 0,
    boundary_tolerance_px: int = 2,
    prediction_labels: NDArray[np.generic] | None = None,
) -> dict[str, float | int | None]:
    valid = np.asarray(valid_mask, dtype=bool)
    truth = np.asarray(truth_binary, dtype=bool) & valid
    prediction = np.asarray(prediction_binary, dtype=bool) & valid
    true_positive_px = int(np.count_nonzero(truth & prediction))
    false_positive_px = int(np.count_nonzero(~truth & prediction & valid))
    false_negative_px = int(np.count_nonzero(truth & ~prediction))
    precision_px = _safe_divide(
        true_positive_px,
        true_positive_px + false_positive_px,
    )
    recall_px = _safe_divide(
        true_positive_px,
        true_positive_px + false_negative_px,
    )
    dice = _safe_divide(
        2.0 * true_positive_px,
        2.0 * true_positive_px + false_positive_px + false_negative_px,
    )
    iou = _safe_divide(
        true_positive_px,
        true_positive_px + false_positive_px + false_negative_px,
    )
    truth_labels = _labels(truth)
    if prediction_labels is None:
        resolved_prediction_labels = _labels(prediction, prediction_min_area_px)
    else:
        resolved_prediction_labels = np.asarray(prediction_labels, dtype=np.int32)
        if resolved_prediction_labels.shape != truth.shape:
            raise ValueError("prediction_labels shape must match truth")
        resolved_prediction_labels = np.where(
            valid,
            resolved_prediction_labels,
            0,
        ).astype(np.int32)
    match50 = _match(truth_labels, resolved_prediction_labels, 0.5)
    match75 = _match(truth_labels, resolved_prediction_labels, 0.75)
    instance_precision = _safe_divide(
        match50.true_positive,
        match50.true_positive + match50.false_positive,
    )
    instance_recall = _safe_divide(
        match50.true_positive,
        match50.true_positive + match50.false_negative,
    )
    instance_f1 = _safe_divide(
        2.0 * instance_precision * instance_recall,
        instance_precision + instance_recall,
    )
    panoptic_quality = _safe_divide(
        sum(match50.matched_ious),
        match50.true_positive
        + 0.5 * match50.false_positive
        + 0.5 * match50.false_negative,
    )
    truth_diameters = _equivalent_diameters(truth_labels)
    prediction_diameters = _equivalent_diameters(resolved_prediction_labels)
    quantiles = (0.1, 0.5, 0.9)
    truth_quantiles = (
        np.quantile(truth_diameters, quantiles)
        if len(truth_diameters)
        else np.zeros(3)
    )
    prediction_quantiles = (
        np.quantile(prediction_diameters, quantiles)
        if len(prediction_diameters)
        else np.zeros(3)
    )
    truth_mean = float(truth_diameters.mean()) if len(truth_diameters) else 0.0
    prediction_mean = (
        float(prediction_diameters.mean()) if len(prediction_diameters) else 0.0
    )
    truth_median = float(np.median(truth_diameters)) if len(truth_diameters) else 0.0
    prediction_median = (
        float(np.median(prediction_diameters)) if len(prediction_diameters) else 0.0
    )
    return {
        "pixel_dice": dice,
        "pixel_iou": iou,
        "pixel_precision": precision_px,
        "pixel_recall": recall_px,
        "boundary_f1": _boundary_f1(truth, prediction, boundary_tolerance_px),
        "hd95_px": _hd95(truth, prediction),
        "truth_instance_count": int(truth_labels.max()),
        "prediction_instance_count": int(resolved_prediction_labels.max()),
        "instance_precision_50": instance_precision,
        "instance_recall_50": instance_recall,
        "instance_f1_50": instance_f1,
        "ap50": None,
        "ap75": None,
        "matched_recall_50": instance_recall,
        "matched_recall_75": _safe_divide(
            match75.true_positive,
            int(truth_labels.max()),
        ),
        "panoptic_quality": panoptic_quality,
        "count_absolute_error": abs(
            int(resolved_prediction_labels.max()) - int(truth_labels.max())
        ),
        "count_relative_error": _safe_divide(
            abs(int(resolved_prediction_labels.max()) - int(truth_labels.max())),
            int(truth_labels.max()),
        ),
        "coverage_error": abs(float(prediction.mean()) - float(truth.mean())),
        "mean_diameter_error_px": abs(prediction_mean - truth_mean),
        "median_diameter_error_px": abs(prediction_median - truth_median),
        "d10_error_px": abs(float(prediction_quantiles[0] - truth_quantiles[0])),
        "d50_error_px": abs(float(prediction_quantiles[1] - truth_quantiles[1])),
        "d90_error_px": abs(float(prediction_quantiles[2] - truth_quantiles[2])),
        "diameter_wasserstein_px": (
            float(wasserstein_distance(truth_diameters, prediction_diameters))
            if len(truth_diameters) and len(prediction_diameters)
            else None
        ),
    }
