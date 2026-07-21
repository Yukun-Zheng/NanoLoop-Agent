# unet-agglomerated-specialized-v1

Status: **unavailable**. The TorchScript identity and validation-only parameters are frozen at
`probability >= 0.25` and `min_area_px=1024`. The public registry must remain unavailable, and a
private registry must not mark this model ready before the frozen configuration completes a full
Analysis smoke run.

## Delivery audit boundary

The 2026-07-21 handoff ZIP did **not** include the TorchScript/checkpoint, machine-readable
calibration or independent-test JSON, a source/sample-level split manifest, or a license and
custody ledger for the images, masks, and model assets. Therefore every numerical result below is
recorded as a developer-reported result, not as a result independently reproduced from this Git
repository. The missing original training metadata also means training/test independence cannot be
verified. These gaps block readiness and scientific-MVP acceptance even though the code seam and
synthetic tests can be reviewed.

## Scientific target definition

The GT foreground represents each whole agglomerate, not the individual primary particles inside
an agglomerate. Consequently, `watershed_enabled=false` is the default scientific interpretation.
Later particle count, equivalent diameter, density, coverage, and perimeter statistics describe
whole agglomerates. They must not be presented as internal single-particle statistics.

## Frozen source asset and architecture

- Source checkpoint: `best_unet_agglomerated_specialized.pth`
- Source format: raw `OrderedDict` state_dict
- Source checkpoint SHA-256:
  `e2be19c6fe1e843856fb339d13de8baed8d748f88558ba7bd3eaaa20b90ede21`
- State-dict keys: `63`
- Confirmed CPU strict load: `missing_keys=[]`, `unexpected_keys=[]`
- Network: `AgglomeratedUNet`, one input channel and one logits output channel
- Encoder channels: `1 -> 32 -> 64 -> 128 -> 256`, with three downsamplings
- Context: two 256-channel residual blocks with dilation 2 and 4
- Decoder channels: `256 -> 128 -> 64 -> 32`
- Normalization layers: GroupNorm with the largest compatible group count from `8, 4, 2, 1`
- Upsampling: bilinear interpolation with `align_corners=false`

All residual, skip, and downsampling convolutions use `bias=false`. The verified source definition
and 63-key contract retain the final `output = Conv2d(32, 1, 1)` bias; removing `output.bias`
would produce a 62-key model and would not strict-load the confirmed checkpoint.

The external TorchScript asset is not distributed by this repository. Its frozen SHA-256 is
`d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9`. The dedicated cloud export
reported eager/TorchScript logits and probability maximum absolute error `0.0`, output shape
`(1, 1, 384, 384)`, and finite outputs after reload.

## Frozen preprocessing and inference seam

The fixed production seam is:

- Exclude the bottom 130 px before normalization or inference.
- Convert the cropped image to one-channel grayscale.
- Compute P1 and P99 once over that complete cropped image.
- Clip and scale P1--P99 to `[0, 1]`; a constant image maps to all zeros.
- Do not apply fixed mean/std normalization.
- Use `384 x 384` patches with overlap 96 px and stride 288 px.
- Do not pad ordinary large images to a tile-grid multiple; anchor a final tile at each uncovered
  right/bottom edge.
- Only when an image dimension is smaller than 384 px, reflect-pad it to the minimum patch size
  and crop the fused probability back to the original inference extent.
- Fuse overlaps with a Hann window whose weights have a lower bound of `0.05`.
- Apply sigmoid to logits.
- Compare probability with the frozen validation threshold `0.25` using `>=` (`gte`).
- Restore a full-size output with the bottom 130 px probability and mask fixed to zero.

The bottom 130 px must also be excluded from the Analysis ROI, morphometry, density denominators,
and all later validation/test metrics. `100/184 nm_per_pixel`
(`0.5434782608695652 nm_per_pixel`) is developer-reported for the evaluated fields; physical-unit
outputs are valid only when each uploaded image independently carries the confirmed scale.

## Validation and independent-test boundary

The agglomerated training pool contains 21 fields of view, all recorded as `2048 x 1536 px`.
Only these four fields may be used for threshold, `min_area_px`, or any other parameter selection:

- `BiCu-3.tif`
- `BaNi-3.tif`
- `BaNi-1.tif`
- `BaNi-2.tif`

The independent test fields `YCu-1.tif`, `YCu-2.tif`, and `YCu-3.tif` must never be read during
calibration or used for parameter selection. They remain reserved for the later frozen test.

The original training metadata JSON is missing. The frozen threshold was not recovered from the
checkpoint: it comes from new, traceable validation evidence using the four fields above. The
prespecified score selected `0.25`, combining Micro Dice and equal-weight recall over available GT
size buckets. The evidence JSON SHA-256 is
`9c76289a61ab870b59cda079eb732222a1267d3ecf47636244967872c4130a02`. At `0.25`, selection score
was `0.7554787140413239`, Macro Dice was `0.7399114940759038`, Micro Dice was
`0.7351176348009941`, Micro IoU was `0.5811747044835565`, Micro Precision was
`0.60035015714734`, Micro Recall was `0.9479046515529643`, and whole-agglomerate detection was
`347/357` (`0.9719887955182073`). Tiny GT contained only 12 components and small GT only one, so
the equal-weight bucket rule is sensitive to a small number of small objects. Visual review passed;
`BaNi-3` and especially `BaNi-2` retain visible false-positive regions. A separate read-only
validation task must calibrate `min_area_px`; it may clean small fragments but cannot be claimed to
repair larger structural false positives.

## Frozen validation min-area evidence

The read-only min-area scan reused the four cached validation probability arrays produced by the
threshold calibration. It did not load the model, repeat inference, read original images, or access
YCu test data. The min-area evidence JSON SHA-256 is
`0e3d4c94aa4ff532e65d2803ed5259583eabbc4dcd8ba889595329823f9399c5`.

The prespecified rule uniquely selected `min_area_px=1024`, corresponding to
`302.45746691871454 nm²` and an equivalent circular diameter of `19.623985514704565 nm` at
`100/184 nm_per_pixel`. Its four-field validation results were:

- Macro Composite MAPE: `0.16633235387069195`
- Macro count MAPE: `0.03748548761609907`
- Macro mean-diameter MAPE: `0.17562337812385911`
- Macro perimeter-density MAPE: `0.2858881958721178`
- Macro Dice: `0.7471895809869473`
- GT retention: `307/350` (`0.8771428571428571`)

The frozen Analysis postprocessing contract is `watershed_enabled=false`, `fill_holes=true`,
`exclude_border=true`, `connectivity=2`, and `perimeter_neighborhood=8`. The threshold comparison
remains `gte`, the bottom 130 px remain outside inference and measurement, and the fixed scale is
`100/184 nm_per_pixel`.

This min-area threshold removed 43 GT agglomerates under the diagnostic retention calculation,
with the largest retention limitations visible for BiCu-3 and BaNi-2. It is the optimum under the
prespecified aggregate-statistics rule; it does **not** mean that agglomerates smaller than
`19.623985514704565 nm` do not exist. BaNi-2 also retains false-positive and statistical bias that
small-fragment filtering cannot resolve. Neither threshold nor min-area may be changed using the
reserved YCu independent test.

## Frozen calibration and readiness limits

Current readiness limits are:

- `default_threshold=0.25` is frozen from the new four-field validation evidence.
- `expected_image_size=[1536, 2048]` (`height, width`) is frozen from the 21-field training pool;
  other dimensions are out of scope and must fail closed before the fixed bottom crop is applied.
- `min_area_px=1024` is frozen from the read-only four-field validation evidence.
- The YCu independent test was read only after both parameters were frozen; its results must never
  be used to retune either parameter.
- No statement of scientific readiness, cross-material stability, or sample-level independence is
  supported.

The public registry status must remain `unavailable`. Recording the frozen default threshold is
scientific metadata, not authorization to run the model; the missing asset and evidence package
still blocks public or private readiness.

## Full Analysis smoke acceptance gate

The dedicated smoke entry point is `scripts/models/smoke_unet_agglomerated_analysis.py`. Its cloud
execution is deliberately restricted to the calibrated validation field `BiCu-3.tif`; it has no
mask input and must not read YCu data. The required production path is
`AnalysisCreationService -> create_runs(FULL_IMAGE) -> InferenceGateway -> execute_run`.

The smoke is accepted only when the exact external TorchScript SHA-256 is verified, the resolved
inference/postprocessing/morphometry configuration equals the frozen contract, the prediction mask
contains zero foreground pixels in the bottom 130 rows, the effective ROI and density calculations
exclude those rows, and all Analysis metadata and review artifacts exist. Until a real cloud run
passes these checks, this model remains unavailable. A successful report permits the exact asset to
be declared ready in a repository-external private registry only; it does not change the public
registry and does not authorize YCu execution.

The CLI private-registry input must itself remain `unavailable`. The script validates that
preflight declaration, constructs a temporary smoke-only Gateway manifest, and writes
`private-registry-ready.yaml` to the external result directory only after the full run and every
acceptance check succeed.

## Frozen YCu independent-test evidence

The frozen independent test comprises `YCu-1.tif`, `YCu-2.tif`, and `YCu-3.tif`: three
independent fields of view (FOV), **not** three independent materials or a sample-level independent
test. It used only the existing formal Analysis `pred_mask.png`, `run_config.json`, and image
metadata with the three human masks; it did not load the model, invoke the Gateway, rerun
inference, or change any parameter.

The exact evaluated private TorchScript asset SHA-256 was
`d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9`. The frozen contract was
`probability >= 0.25`, `min_area_px=1024`, bottom `130 px` excluded from inference and metrics,
and whole-agglomerate semantics (not primary particles within an agglomerate).

Pixel-level independent-test results were Macro Dice `0.8678420653`, Micro Dice `0.8575525540`,
and Macro IoU `0.7707467511`. Aggregate-statistics errors were markedly less uniform: coverage
Macro MAPE was `3.7712858326%`; agglomerate-count Macro MAPE was `51.2323232323%`; mean equivalent
diameter Macro MAPE was `82.3173359269%`; number-density Macro MAPE was `51.2323232323%`; and
perimeter-density Macro MAPE was `16.8649123230%`.

`YCu-2` has substantial missed small targets. The developer-reported results suggest possible
coverage/semantic-mask use under the frozen contract, but that use is not repository-verified. It
must **not** be claimed as a high-precision tool
for agglomerate count, number density, or equivalent-diameter statistics. These three FOV results
do not establish independent-material performance. The public registry remains `unavailable`; this
evidence does not authorize public readiness.
