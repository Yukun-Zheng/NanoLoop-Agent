from __future__ import annotations

import numpy as np

from app.msbi.metrics import (
    evaluate_binary_instance_prediction,
    evaluate_calibration_metrics,
)


def test_calibration_metrics_match_full_evaluation_subset() -> None:
    truth = np.zeros((32, 32), dtype=bool)
    truth[3:10, 4:12] = True
    truth[18:27, 17:28] = True
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[4:11, 4:12] = 1
    labels[18:26, 18:28] = 2
    valid = np.ones((32, 32), dtype=bool)
    valid[-2:] = False

    focused = evaluate_calibration_metrics(
        truth,
        labels,
        valid_mask=valid,
    )
    full = evaluate_binary_instance_prediction(
        truth,
        labels > 0,
        valid_mask=valid,
        prediction_labels=labels,
    )

    assert focused == {
        name: float(full[name])
        for name in (
            "pixel_dice",
            "boundary_f1",
            "instance_f1_50",
            "count_absolute_error",
        )
    }
