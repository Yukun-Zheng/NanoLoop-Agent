# unet-small-balanced-v1

Status: **unavailable**. No checkpoint is distributed with this repository.

This declaration is for the confirmed single-channel, BatchNorm U-Net supplied by the model
developer. Its source checkpoint is a PyTorch `state_dict` with SHA-256
`915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008`; the production artifact is
an externally stored TorchScript export and is not distributed with this repository.

## Model assets

- Source checkpoint: `best_unet_small.pth`
- Source format: PyTorch `state_dict`
- Source SHA-256: `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008`
- Exported model: `unet-small-balanced-v1.pt`
- Export format: TorchScript
- External bundle relative path: `weights/unet-small-balanced-v1.pt`
- Export SHA-256: `dac85662061b12f1d8e8d583500558b787dd2c98795bf2ef816ddc95c3491446`

Neither model asset is distributed with this repository.

## Export verification

- Export device: CPU
- Validation input: `[1, 1, 256, 256]` `float32`
- Checkpoint/model keys: `128/128`
- Missing keys: none
- Unexpected keys: none
- Shape mismatches: none
- Eager/TorchScript logits maximum absolute error: `0.0`
- Eager/TorchScript probability maximum absolute error: `0.0`

## Independent test-set evidence

The independent test set contains three small test images: `SrNi-1`, `SrNi-2`, and `SrNi-3`.
Evaluation used the TorchScript artifact with SHA-256
`dac85662061b12f1d8e8d583500558b787dd2c98795bf2ef816ddc95c3491446`, strict threshold
`probability > 0.30`, `invalid_bottom_px=130`, seed `2026`, and CPU execution. Ground truth
consisted of full-size binary masks; the bottom 130 px were excluded from evaluation.

| Image | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| SrNi-1 | 0.7605629505 | 0.6136358041 | 0.6946071909 | 0.8403583740 |
| SrNi-2 | 0.8285408068 | 0.7072724441 | 0.7720875732 | 0.8939007331 |
| SrNi-3 | 0.7953094716 | 0.6601774089 | 0.7252144363 | 0.8804042580 |
| Micro | 0.7966136751 | 0.6619766726 | 0.7325296066 | 0.8729852604 |
| Macro | 0.7948044096 | 0.6603618857 | 0.7306364001 | 0.8715544550 |

This evidence covers only these three small test images and must not be extrapolated to other
materials or models. Formal acceptance thresholds have not yet been frozen. Precision is lower
than recall in this test set, indicating a risk of overpredicted area and false positives. The
registry entry must remain `unavailable` while the remaining acceptance evidence is incomplete.

## B-board postprocessing calibration evidence

The `small` subset in `train_groups.csv` contains 44 images. An sklearn `ShuffleSplit`-equivalent
split with `test_size=0.20` and `random_state=2026` produced 35 training images and 9 validation
images. The validation images were `BiCr-1`, `BiNi-3`, `GdNi-2`, `LaCo-3`, `NdNi-3`, `SmNi-3`,
`SrCr-2`, `SrIn-2`, and `YNi-1`.

Calibration held the following postprocessing rules fixed:

- Connected components
- `watershed_enabled=false`
- `fill_holes=true`
- `exclude_border=true`
- `connectivity=2`
- `perimeter_neighborhood=8`
- `scale_nm_per_pixel=100/184`

The compared `min_area_px` values were `0`, `4`, `8`, `16`, `32`, and `64`. The prespecified
selection rule minimized the equally weighted mean of particle-count MAPE, mean-equivalent-
diameter MAPE, and perimeter-density MAPE.

The selected value was `min_area_px=64`, with the following validation results:

- Macro Dice: `0.8932496745`
- Count MAPE: `0.0880211601`
- Diameter MAPE: `0.0345797221`
- Perimeter-density MAPE: `0.0624093440`
- Composite MAPE: `0.0616700753`
- GT baseline count: `647`
- GT count at `64`: `634`
- GT retention: `97.9907%`
- GT count at each of `4`, `8`, `16`, `32`, and `64`: `634`
- At this scale, 64 px corresponds to approximately `18.90 nm²` and an equivalent diameter of
  approximately `4.91 nm`

This is a configuration for the current small-data domain and must not be extrapolated to other
scales or materials. A complete run using this calibration must explicitly freeze
`min_area_px=64`; the global default must not be changed. The registry entry remains
`unavailable`.

Inference converts the input to grayscale, divides pixels by 255, removes the fixed bottom 130 px
information bar, then uses 256 px patches with 128 px stride, reflect padding, and equal overlap
averaging. The one-channel logits receive sigmoid and use the strict rule `probability > 0.30`.
The returned mask preserves original dimensions with the bottom bar set to zero; that area is also
frozen as invalid and excluded from downstream scientific ROI statistics.

Before enabling this registry entry, document the external artifact's immutable location and
retention policy, training-data and license evidence, fixture-backed acceptance metrics, and real
full-image smoke-test evidence.

The training notebook did not apply the same bottom crop that its inference notebook applies. This
known training/inference discrepancy must remain documented and must not be described as resolved
without new scientific evidence.
