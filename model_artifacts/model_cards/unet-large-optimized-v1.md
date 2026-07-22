# unet-large-optimized-v1

Status: **unavailable**. The external model artifact is not distributed with this repository, and
the available evidence is not sufficient to claim scientific readiness.

This is the optimized, single-channel large-particle U-Net. Each `DoubleConv` uses two
`Conv2d(bias=False) + GroupNorm(8) + ReLU` blocks. Decoder upsampling is bilinear with
`align_corners=False`, followed by skip concatenation and padded spatial alignment. The final
`Conv2d(32, 1, 1)` retains its bias.

## Model assets

- Source checkpoint: `best_unet_large_optimized.pth`
- Source format: PyTorch `state_dict`
- Source size: `13,421,286` bytes
- Source SHA-256: `5c5dbcae61f40f8eb1fef27c7b69592a727260898330abc546f7e7a6833035bd`
- Exported model: `unet-large-optimized-v1.pt`
- Export format: TorchScript
- External bundle relative path: `weights/unet-large-optimized-v1.pt`
- Approximate exported size: `13 MiB`
- Export SHA-256: `007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05`

Neither model asset is distributed with this repository.

## Export verification

- Architecture profile: `large_groupnorm_optimized`
- Export device: CPU
- Validation input: `[1, 1, 512, 512]` `float32`
- State-dict loading: strict
- Checkpoint/model keys: `56/56`
- Missing keys: none
- Unexpected keys: none
- Shape mismatches: none
- Eager/TorchScript logits maximum absolute error: `0.0`
- Eager/TorchScript probability maximum absolute error: `0.0`
- Reloaded TorchScript output: finite

## Training and production inference contract

Training converted images to grayscale, divided pixels by 255, used 512 px patches, and cropped
130 px from the bottom. Confirmed production inference instead crops 180 px from the bottom. This
training/production discrepancy is a known unresolved risk and must remain visible in acceptance
evidence.

Production inference uses 512 px patches with 256 px stride, right/bottom reflect padding, and
uniform averaging in overlap regions. The one-channel logits receive sigmoid, followed by the
calibrated strict rule `probability > 0.50`.

## Threshold calibration evidence

Threshold calibration used only these six large validation fields of view:

- `NdZn-2.tif`
- `LaMn-3.tif`
- `LaMn-1.tif`
- `BaCo-3.tif`
- `BaCu-1.tif`
- `BaCr-3.tif`

Each image was inferred exactly once through the production `InferenceGateway` using the verified
TorchScript artifact. The resulting raw probability array was cached, and thresholds were scanned
offline without repeating model inference. Evaluation excluded the bottom 180 px from all TP, FP,
FN, and TN counts and used the strict comparison `probability > threshold`.

The prespecified candidates were `0.20`, `0.25`, `0.30`, `0.35`, `0.40`, `0.45`, `0.50`, `0.55`,
`0.60`, `0.65`, `0.70`, `0.75`, and `0.80`. The selection rule first maximized Macro Dice, then
Macro IoU, and then preferred the candidate closest to the previous experimental threshold of
`0.60`. This rule selected the strict threshold `probability > 0.50`.

| Aggregate | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Macro | 0.7086369503967919 | 0.6023708166699938 | 0.7066930166008482 | 0.8030260085603212 |
| Micro | 0.722377672920259 | 0.5654078342317297 | 0.63788002425226 | 0.8326797307831924 |

Manual review of all six GT/prediction comparisons found:

- `NdZn-2`, `LaMn-1`, and `LaMn-3` were comparatively consistent.
- `BaCr-3` contained false positives and shape discrepancies.
- `BaCo-3` showed substantial under-segmentation.
- `BaCu-1` showed substantial false positives.

These differences demonstrate material-domain sensitivity. The calibrated threshold must not be
described as stable for every large-particle material. This is field-of-view-level validation and
does not establish sample-level independence or cross-material generalization.

## Large-specific postprocessing calibration

The large `min_area_px` calibration reused the six cached probability arrays from threshold
calibration and did not repeat model inference. It kept the selected strict threshold
`probability > 0.50`, bottom exclusion of 180 px, and scale `100/184 nm_per_pixel` fixed. GT used
the unfiltered `min_area_px=0` baseline; predictions were evaluated at candidate values `0`, `16`,
`32`, `64`, `128`, `256`, `512`, and `1024`.

All candidates used the project's production connected-component splitting, morphometry, and unit
conversion with these frozen rules:

- `watershed_enabled=false`
- `fill_holes=true`
- `exclude_border=true`
- `connectivity=2`
- `perimeter_neighborhood=8`

The prespecified selection rule minimized the six-field Macro Composite MAPE, defined as the equal
mean of particle-count MAPE, mean-equivalent-diameter MAPE, and perimeter-density MAPE. Ties were
resolved by higher Macro Dice and then smaller `min_area_px`; an unresolved tie had to fail rather
than select an arbitrary value.

Final overlay review accepted `min_area_px=512`. At the frozen scale this is
`151.22873345935727 nm²`, corresponding to an equivalent-circle diameter of
`13.87625323135418 nm`. GT retention at this candidate was `100%`.

This result is limited to the six listed validation fields of view and is not sample-level
independent. The final review does not remove the known material-domain limitations: `BaCo-3`
shows under-segmentation and `BaCu-1` shows false-positive segmentation. The frozen threshold and
postprocessing parameters therefore must not be presented as stable across all large-particle
materials.

## Frozen scientific parameters

A complete Large analysis must explicitly freeze the following model-specific values; these do
not replace or modify global analysis defaults:

- Threshold: strict `probability > 0.50`
- `min_area_px=512`
- Minimum physical area: `151.22873345935727 nm²`
- Minimum-area equivalent-circle diameter: `13.87625323135418 nm`
- `watershed_enabled=false`
- `fill_holes=true`
- `exclude_border=true`
- `connectivity=2`
- `perimeter_neighborhood=8`
- `bottom_crop_px=180`
- `scale_nm_per_pixel=100/184` (`0.5434782608695652`)

## Frozen independent test-set evidence

The formal full Analysis workflow completed for the three fixed test fields `SrZr-3.tif`,
`BaCu-2.tif`, and `PrCu-3.tif`. Independent evaluation then read only each completed Analysis
`pred_mask.png` and the corresponding human `test_mask_human`; it did not repeat inference, read
training or validation masks, or use any test result to change a parameter.

Both prediction and GT used nonzero pixels as foreground. TP, FP, FN, and TN were calculated only
over the top `2048 x 1356 px`. The bottom 180 px (`y=1356..1536`) were excluded completely. The
evaluated runs froze strict `probability > 0.50`, `min_area_px=512`,
`watershed_enabled=false`, `fill_holes=true`, `exclude_border=true`, `connectivity=2`,
`perimeter_neighborhood=8`, and `scale_nm_per_pixel=100/184`.

| Test field | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| `SrZr-3` | 0.9392828149931417 | 0.8855167317639607 | 0.923919927306771 | 0.9551652480856273 |
| `BaCu-2` | 0.724665460199322 | 0.5682159759529292 | 0.8119095758655747 | 0.654351788772072 |
| `PrCu-3` | 0.7520219431319688 | 0.602592280363702 | 0.7653143163046803 | 0.7391834247410116 |

| Aggregate | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Macro | 0.8053234061081441 | 0.6854416626935307 | 0.8337146064923419 | 0.7829001538662369 |
| Micro | 0.7734422618347466 | 0.630579578741805 | 0.8211666401761994 | 0.7309604656803299 |

The Micro confusion counts were `TP=144660`, `FP=31504`, `FN=53244`, and `TN=8101856`.
`BaCu-2` recall was `0.654351788772072` (approximately `0.6544`), which is an explicit
under-detection limitation.

These are three independent, non-overlapping fields of view, not three sample-level independent
observations. The results evaluate only the already frozen model and must not be used to tune the
threshold, `min_area_px`, or any other scientific parameter. They do not establish cross-material
stability or scientific readiness.

## Data split and current evidence limits

The large training pool contained 28 image files. With seed `2026`, Python
`random.Random(2026).shuffle` produced 22 training files and 6 validation files. The recorded best
patch-validation Dice was `0.7909`; this patch metric must not be represented as an independent
full-image result.

The three test files are `SrZr-3.tif`, `BaCu-2.tif`, and `PrCu-3.tif`. They are different,
non-overlapping fields of view from samples also represented by related training views. They are
therefore file-level and field-of-view-level independent, but not sample-level independent. This
limits the strength and scope of any later test-set claim.

The large-specific threshold and postprocessing parameters are calibrated and frozen above. The
small-model value of 64 px is not a large-model default and must not be reused. Independent
sample-level evidence remains incomplete even though the real full-Analysis smoke and the frozen
three-field independent evaluation are complete. This model therefore cannot be described as scientifically ready.
Its registry status must remain `unavailable`.
