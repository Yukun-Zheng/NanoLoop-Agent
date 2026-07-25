from __future__ import annotations

import numpy as np

from app.msbi.targets import TargetConfig, generate_instance_targets, semantic_to_pseudo_instances


def test_target_generation_uses_instance_boundaries_and_valid_mask() -> None:
    labels = np.zeros((48, 64), dtype=np.int32)
    labels[8:20, 8:20] = 1
    labels[18:38, 30:52] = 2
    valid = np.ones_like(labels, dtype=bool)
    valid[-6:] = False
    targets = generate_instance_targets(
        labels,
        valid_mask=valid,
        config=TargetConfig(
            small_area_max_px=150,
            large_area_min_px=300,
            center_sigma_max=4,
        ),
    )

    assert targets["foreground"].shape == labels.shape
    assert np.max(targets["center"]) > 0.95
    assert np.count_nonzero(targets["boundary"]) > 0
    assert np.min(targets["distance"]) >= 0
    assert np.max(targets["distance"]) <= 1
    assert np.all(targets["valid"][-6:] == 0)
    assert np.all(targets["foreground"][-6:] == 0)
    assert np.all(targets["scale"][labels == 1] == 0)
    assert np.all(targets["scale"][labels == 2] == 1)


def test_semantic_conversion_is_explicitly_connected_component_based() -> None:
    semantic = np.zeros((16, 16), dtype=bool)
    semantic[2:5, 2:5] = True
    semantic[10:14, 10:14] = True
    labels = semantic_to_pseudo_instances(semantic)

    assert labels.dtype == np.int32
    assert set(np.unique(labels)) == {0, 1, 2}
