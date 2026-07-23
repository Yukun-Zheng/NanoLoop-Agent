# U-Net A+B handoff

## Current status

- Repository: `NanoLoop-Agent`
- Original PR branch: `feat/a-real-unet-large-v1` (historical, already integrated)
- Original PR base: `yukun` (historical); every follow-up branch must start from and target `main`
- `model_artifacts/registry.yaml` now authors Large U-Net as `ready` because the 2026-07-23
  delivery supplied the actual TorchScript and its matching SHA-256; runtime health still fails
  closed when the optional Torch dependency or any bundle byte is missing.
- The same registry now authors Small U-Net as `ready` after strict checkpoint loading and a
  compatibility re-export produced a TorchScript that was separately verified on the supported
  PyTorch 2.6.0 lower bound and PyTorch 2.13.0. Its Small-B scientific acceptance remains pending.
- The later `ModelAssets-large-a.zip` and `ModelAssets-large-b.zip` deliveries describe one Large
  model, not two model IDs. Their TorchScript bytes are identical to the checked-in runtime asset.
- Historical three-field prediction/GT pixel counts and metrics have now been independently
  recalculated and recorded in the
  [A/B asset acceptance audit](model-assets-large-a-b-acceptance-2026-07-23.md).
- Runtime readiness and historical pixel verification do not upgrade the current bundle to
  scientific acceptance. License/custody, source/sample split, explicit tolerance policy and a
  current Adapter/config/card rerun remain required.

The deployable Large and Small TorchScripts are tracked in this repository at the project owner's
request. Their redundant source checkpoints are not tracked, but their immutable SHA-256 values
remain recorded. These explicit integrations do not authorize committing other weights or
establish third-party redistribution rights.

## External model identities

| Model | Source checkpoint SHA-256 | TorchScript SHA-256 |
| --- | --- | --- |
| `unet-small-balanced-v1` | `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008` | `09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d` |
| `unet-large-optimized-v1` | `5c5dbcae61f40f8eb1fef27c7b69592a727260898330abc546f7e7a6833035bd` | `007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05` |

The runtime TorchScripts are distributed at
`model_artifacts/weights/unet-large-optimized-v1.pt` and
`model_artifacts/weights/unet-small-balanced-v1.pt`; source checkpoints remain external.

The historical Small handoff mentioned
`dac85662061b12f1d8e8d583500558b787dd2c98795bf2ef816ddc95c3491446`; it is not an accepted
Small-A artifact identity. The later delivered TorchScript
`e31bd7100d410fe3af93041ccf6956e27d562214d9ddcb40ac76b905840d6d28` was numerically correct
under PyTorch 2.13.0 but could not load under the supported PyTorch 2.6.0 lower bound because it
serialized `aten::_upsample_lanczos2d_aa`. The checked-in `09d1818...03171d` artifact was
re-exported from the strictly verified checkpoint with the repository exporter and matches the
delivered artifact exactly on the validation input. See the
[Small-A acceptance audit](model-assets-small-a-acceptance-2026-07-23.md).

## Frozen scientific contracts

- Small: grayscale `/255`, patch/stride `256/128`, reflect padding, uniform fusion, strict
  `probability > 0.30`, bottom exclusion 130 px, `min_area_px=64` when explicitly configured.
- Large: grayscale `/255`, patch/stride `512/256`, reflect padding, uniform fusion, strict
  `probability > 0.50`, bottom exclusion 180 px.
- Large `calibrated_analysis`: `min_area_px=512`, `151.22873345935727 nm²`, equivalent-circle
  diameter `13.87625323135418 nm`, `watershed=false`, `fill_holes=true`, `exclude_border=true`,
  connectivity 2, perimeter neighborhood 8, scale `100/184 nm_per_pixel`.
- Large validation evidence is field-of-view-level, not sample-level independent. Known material
  sensitivity remains; do not claim cross-material stability or full scientific readiness.

## What the repository verifies

- `UNetAdapter` emits the thresholded semantic mask without applying B-module minimum-area
  filtering. Canonical postprocessing and morphometry remain in `app/analysis/`.
- A private bundle resolves weight/config/model-card paths relative to its own registry.
- The registry fails closed when Torch is missing, the weight is absent, or its SHA-256 differs.
- The checked-in registry exposes Large and Small as ready only when their runtime bundle and Torch
  dependency validate; missing or changed assets remain fail closed.

Repository tests verify both runtime asset identities and integration contracts. For Large they
also recompute the delivered historical pixel metrics from their recorded confusion counts; the
one-time delivery audit recalculated those counts directly from the external prediction and
human-GT bytes. For Small the real Gateway test covers full-image and BOXES inference, deterministic
repeatability, bottom-row exclusion, snapshot freezing and unload, but the delivery contained no
SEM/GT or Small-B evidence. Neither model is thereby granted current-bundle scientific acceptance.

## Required private acceptance package

Large's runtime weight, historical outputs, three fixed SEM/GT pairs, per-image pixel metrics and a
non-degraded historical Analysis run have been delivered externally. Small's runtime weight and
source checkpoint have been delivered, but its Small-B evidence has not. The remaining acceptance
package must separately provide:

1. Model/data license or written authorization plus a custody ledger.
2. Source-image or sample-level split manifest.
3. Target Python/Torch/CUDA or CPU environment, exact command and identifiable Git commit.
4. Project-owner-approved tolerance policy.
5. Complete machine-readable threshold and minimum-area evidence.
6. A current-bundle, non-degraded Gateway-to-Analysis rerun and generated artifact manifest.
7. Current-bundle instance, morphometry, quality and export acceptance results.
8. For Small, the authorized Small-B test set, masks, split, threshold/min-area calibration and
   independently reproducible pixel/instance metrics.

The duplicate TorchScript, source checkpoint, later unapproved weights, SEM images, GT masks,
probabilities, predictions, SQLite files and run outputs remain external and must not enter Git.

## Reproduction on the controlled Linux host

First run repository checks:

```bash
python -m pytest -q \
  tests/unit/inference/test_unet_large_assets.py \
  tests/unit/inference/test_unet_small_assets.py \
  tests/unit/inference/test_unet_tiling.py \
  tests/unit/scripts/test_smoke_unet_large_analysis.py
python -m ruff check .
git diff --check
```

Then run the real three-image Analysis smoke with the private ready registry:

```bash
python scripts/models/smoke_unet_large_analysis.py \
  --image-dir /external/path/to/test_images/images \
  --registry /external/path/to/private-ready-registry.yaml \
  --output-root /external/new/nonexistent/large-analysis-smoke
```

The fixed developer-reported images are `SrZr-3.tif`, `BaCu-2.tif` and `PrCu-3.tif`. Acceptance must
check terminal states, status histories, bundle hashes, frozen configuration, bottom-180 ROI
evidence, density consistency, quality warnings, masks, canonical instances, particle CSVs,
visualizations, reports and execution provenance. Preserve the resulting evidence outside Git and
hand its manifest to the project owner.
