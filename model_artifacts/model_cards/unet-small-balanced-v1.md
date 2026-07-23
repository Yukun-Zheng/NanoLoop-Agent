# unet-small-balanced-v1

Status: **unavailable — pending cloud validation**. The checkpoint and deployable TorchScript are
controlled private assets and are not distributed with this repository.

## Ownership and permitted use

- Model asset owner: 郭境濠
- Checkpoint origin: trained by 郭境濠 from NanoLoop-Agent project training data
- Training data origin: internal experimental data from the NanoLoop-Agent project research group
- Permitted use: internal NanoLoop-Agent development, testing, validation, deployment, and
  demonstration
- Training-data permission: internal model training, engineering validation, and testing for the
  NanoLoop-Agent project
- Distribution restriction: the checkpoint, TorchScript, private bundle, and training data must
  not be distributed publicly without permission from both the model asset owner and the project
  owner
- Asset custody: controlled private storage only; no private model or data asset is committed to Git

## Frozen engineering identity

- Model ID: `unet-small-balanced-v1`
- Source checkpoint: `best_unet_small.pth`
- Source checkpoint SHA-256:
  `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008`
- Source format: PyTorch `state_dict`
- Architecture profile: `small_batchnorm`
- Expected state-dict contract: 128 keys
- Input/output channels: one grayscale input channel and one foreground-logit output channel
- Patch size: `256 x 256`
- Sliding-window stride: `128 x 128`
- Upsampling: bilinear with `align_corners=True`
- Export format: scripted TorchScript
- Exported filename: `unet-small-balanced-v1.pt`
- TorchScript SHA-256: **pending cloud validation**
- Strict checkpoint load: **pending cloud validation**
- Eager/TorchScript numerical comparison: **pending cloud validation**
- Reloaded TorchScript inference: **pending cloud validation**
- Deterministic repeated inference: **pending cloud validation**

The architecture declaration follows the controlled Small training and test notebooks. Cloud
validation must still prove that the exact checkpoint strictly loads into this declaration before
the private registry may declare the model ready.

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
PASS/FAIL belong to Small-B and have not started.

## Runtime integration

The deployable artifact is loaded only through:

```text
InferenceGateway
-> app.inference.adapters.unet:UNetAdapter
-> SegmentationOutput
```

The private ready bundle must contain the exact TorchScript, this config, this model card, and a
registry entry with the verified TorchScript SHA-256. `InferenceGateway` freezes a
content-addressed manifest containing weight, config, model-card, and Adapter identities before
execution.

Cloud validation is fixed to the current repository runtime contract: Linux, Python 3.11,
PyTorch 2.6.0, and torchvision 0.21.0. The historical Large training environment recorded Python
3.10 and PyTorch 2.2.2; that training record is not presented as the Small export runtime.

## Validation status and limitations

The following remain **pending cloud validation**:

- strict 128-key checkpoint loading;
- TorchScript export, reload, and SHA-256;
- eager/TorchScript shape, finite-value, and numerical comparison;
- deterministic repeated inference;
- private ready registry and content-addressed manifest;
- Gateway health, predict, and unload lifecycle;
- full-image and BOXES ROI inference;
- real engineering smoke and cloud test results.

Until all items pass and the resulting environment, commands, and reports are preserved in the
controlled run record, the public registry must remain `unavailable`.
