# unet-small-balanced-v1

Status: **runtime ready; scientific acceptance pending Small-B**. The deployable TorchScript is
tracked with this repository at the project owner's request. Runtime readiness does not establish
Small-particle segmentation accuracy or scientific fitness.

## Ownership and permitted use

- Model asset owner: 郭境濠
- Checkpoint origin: trained by 郭境濠 from NanoLoop-Agent project training data
- Training data origin: internal experimental data from the NanoLoop-Agent project research group
- Permitted use: internal NanoLoop-Agent development, testing, validation, deployment, and
  demonstration
- Training-data permission: internal model training, engineering validation, and testing for the
  NanoLoop-Agent project
- Repository inclusion: the project owner explicitly requested integration of the compatible
  deployable TorchScript after receiving the bundle from the model developer
- Distribution restriction: repository inclusion does not grant third-party redistribution,
  commercial-use, sublicensing, training-data, or source-checkpoint rights
- Asset custody: the source checkpoint, training data, original ZIP and future scientific evidence
  remain in controlled external storage

## Frozen engineering identity

- Model ID: `unet-small-balanced-v1`
- Source checkpoint: `best_unet_small.pth`
- Source checkpoint SHA-256:
  `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008`
- Source format: PyTorch `state_dict`
- Architecture profile: `small_batchnorm`
- Expected state-dict contract: 128 keys
- State values: 3,355,667
- Input/output channels: one grayscale input channel and one foreground-logit output channel
- Patch size: `256 x 256`
- Sliding-window stride: `128 x 128`
- Upsampling: bilinear with `align_corners=True`
- Export format: scripted TorchScript
- Exported filename: `unet-small-balanced-v1.pt`
- Repository path: `model_artifacts/weights/unet-small-balanced-v1.pt`
- Repository TorchScript size: `13,560,272` bytes
- Repository TorchScript SHA-256:
  `09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d`

The delivered TorchScript SHA-256 was
`e31bd7100d410fe3af93041ccf6956e27d562214d9ddcb40ac76b905840d6d28`.
It produced the same output as the source checkpoint under PyTorch 2.13, but it serialized the
newer `aten::_upsample_lanczos2d_aa` operator and could not load under the repository's supported
PyTorch 2.6 lower bound. The tracked runtime asset was therefore re-exported from the exact source
checkpoint with the current repository exporter under PyTorch 2.6.0. The compatible export also
loads under PyTorch 2.13.0 and is numerically identical to the delivered export on the frozen test
tensor.

## Input and preprocessing

The Small training and test assets contain 47 declared Small images, all exactly `2048 x 1536 px`.
Production inference therefore freezes `expected_image_size=[1536, 2048]` in `[height, width]`
order.

Preprocessing is:

1. convert the source image to grayscale;
2. exclude the bottom 130 rows before model inference;
3. convert pixels to float32 in `[0, 1]`, equivalent to division by `255`;
4. extract overlapping `256 x 256` patches with stride `128`;
5. use reflect padding where a dimension is smaller than one patch;
6. combine overlapping patch probabilities with uniform averaging.

The model emits single-channel logits. The Adapter applies sigmoid once; the config must therefore
remain `output_activation: logits`.

## Engineering threshold

The engineering default is strict `probability > 0.30`. This value comes from the controlled Small
test workflow and is frozen only to make engineering inference reproducible.

It is **not** a scientifically calibrated threshold. Threshold calibration, independent scientific
evaluation, pixel/instance/count metrics, morphometry acceptance, tolerance policy, and scientific
PASS/FAIL belong to Small-B and have not been delivered or independently verified.

## Runtime integration

The deployable artifact is loaded only through:

```text
InferenceGateway
-> app.inference.adapters.unet:UNetAdapter
-> SegmentationOutput
```

The ready bundle contains the exact TorchScript, this config, this model card, and a registry entry
with the verified TorchScript SHA-256. `InferenceGateway` freezes a
content-addressed manifest containing weight, config, model-card, and Adapter identities before
execution.

Independent integration verification used CPU, Python 3.12.13 and PyTorch 2.6.0 for strict
checkpoint loading and the compatible export, then PyTorch 2.13.0 for forward-compatibility
loading and cross-export comparison. The checkpoint had exactly 128 matching keys, no
missing/unexpected keys or shape mismatches, and no non-finite floating values. Eager versus
compatible TorchScript logits, repeated compatible TorchScript logits, and delivered versus
compatible TorchScript logits all had maximum absolute error `0.0` on the frozen
`[1, 1, 256, 256]` test tensor.

The exact checked-in artifact was also mounted read-only into a one-time Debian 12 Linux ARM64
container with Python 3.12.13 and `torch 2.6.0+cpu`. Its repository SHA matched, it loaded on CPU,
produced finite `[1, 1, 256, 256]` float32 logits, and repeated inference had maximum absolute
error `0.0`. This is an artifact load/forward check, not the complete target deployment or a
scientific evaluation.

## Validation status and limitations

The following runtime checks are complete:

- safe `weights_only=True` checkpoint inspection and strict 128-key load;
- PyTorch 2.6.0 export, reload, finite output and exact eager/TorchScript comparison;
- PyTorch 2.13.0 compatible-export load and exact comparison with the delivered export;
- Debian 12 Linux ARM64 artifact load and deterministic forward under `torch 2.6.0+cpu`;
- deterministic repeated inference;
- registry health, content-addressed bundle freeze, full-image and BOXES prediction, 130-row bottom
  exclusion, ROI exterior zeroing and unload lifecycle using a deterministic synthetic full-size
  image.

The delivery contains no real SEM smoke image/output, source/sample split manifest, calibrated
Small-B threshold/minimum-area evidence, independent human GT, pixel/instance/count/morphometry
metrics, target-environment performance record or approved tolerance policy. Scientific
acceptance therefore remains pending Small-B. The authored registry status `ready` means only that
the verified runtime bundle is selectable when the optional model dependencies are installed.
