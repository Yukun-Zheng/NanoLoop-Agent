# Large U-Net A+B handoff

## Current status

- Repository: `NanoLoop-Agent`
- Original PR branch: `feat/a-real-unet-large-v1` (historical, already integrated)
- Original PR base: `yukun` (historical); every follow-up branch must start from and target `main`
- Public `model_artifacts/registry.yaml` intentionally keeps Large U-Net `unavailable` because the
  repository does not contain its checkpoint or complete acceptance bundle.
- A controlled external registry may declare Large U-Net `ready` only when the actual TorchScript,
  matching SHA-256, config, model card, runtime dependency and license evidence are present.

The checkpoint is a mandatory project deliverable, but it must be transferred separately through
private storage or the target server. “Do not commit weights” means “do not put them in public Git”;
it does not mean that the model owner may omit the checkpoint from the handoff.

## External model identities

| Model | Source checkpoint SHA-256 | TorchScript SHA-256 |
| --- | --- | --- |
| `unet-small-balanced-v1` | `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008` | `dac85662061b12f1d8e8d583500558b787dd2c98795bf2ef816ddc95c3491446` |
| `unet-large-optimized-v1` | `5c5dbcae61f40f8eb1fef27c7b69592a727260898330abc546f7e7a6833035bd` | `007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05` |

Both TorchScript files belong under `weights/` in a private external bundle. Neither is distributed
by this repository.

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

## What this PR verifies

- `UNetAdapter` emits the thresholded semantic mask without applying B-module minimum-area
  filtering. Canonical postprocessing and morphometry remain in `app/analysis/`.
- A private bundle resolves weight/config/model-card paths relative to its own registry.
- The registry fails closed when Torch is missing, the weight is absent, or its SHA-256 differs.
- The checked-in public registry remains honest when private assets are unavailable.

These repository tests verify integration contracts. They do not independently reproduce the
developer-reported scientific metrics or prove that the private checkpoint has been accepted by the
project owner.

## Required private acceptance package

The model owner must separately provide:

1. The deployable Large TorchScript file and, when required for reproducibility, the source training
   checkpoint.
2. A private `registry.yaml` entry declaring `ready`, with the exact TorchScript SHA-256.
3. Matching config and model card, plus Python/Torch/CUDA or CPU runtime details.
4. Model/data license or authorization records.
5. Source-image or sample-level split manifest, fixed SEM images and human GT masks.
6. Machine-readable per-image metrics, threshold/min-area evidence and known failure cases.
7. A real, non-degraded Gateway-to-Analysis smoke record and generated artifact manifest.

Weights, SEM images, GT masks, probabilities, predictions, SQLite files and run outputs remain
external and must not enter Git.

## Reproduction on the controlled Linux host

First run repository checks:

```bash
python -m pytest -q \
  tests/unit/inference/test_unet_large_assets.py \
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
