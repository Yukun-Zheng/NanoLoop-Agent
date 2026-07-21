# unet-small-balanced-v1

Status: **unavailable**. No checkpoint is distributed with this repository.

This declaration reserves the semantic-segmentation adapter for a TorchScript U-Net supplied by a
model developer. Before enabling it, provide an immutable checkpoint, record its lowercase SHA-256
in `registry.yaml`, document training data and licensing here, and add fixture-backed acceptance
metrics. The adapter emits a float probability array and a thresholded binary PNG.

The fixed 130 px bottom crop has no delivered source-image dimension evidence. A future ready
registration must freeze `expected_image_size=[height, width]`; until then the registry must fail
closed even if a checkpoint is supplied.

No accuracy, material applicability, or hardware claim has been established yet.
