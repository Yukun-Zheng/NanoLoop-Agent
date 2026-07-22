# U-Net A+B handoff

## Current workspace

- Repository: `NanoLoop-Agent`
- Branch: `feat/ab-unet-v1`
- The worktree contains intentional tracked and untracked changes. Do not reset, delete, stage,
  commit, push, or create a PR until the remaining Linux and real-image checks are complete.
- Public `model_artifacts/registry.yaml` must keep both U-Net models `unavailable`.
- Model weights, SEM images, GT masks, probabilities, predictions, SQLite files, and run outputs
  remain external and must not enter Git.

## External model identities

| Model | Source checkpoint SHA-256 | TorchScript SHA-256 |
| --- | --- | --- |
| `unet-small-balanced-v1` | `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008` | `dac85662061b12f1d8e8d583500558b787dd2c98795bf2ef816ddc95c3491446` |
| `unet-large-optimized-v1` | `5c5dbcae61f40f8eb1fef27c7b69592a727260898330abc546f7e7a6833035bd` | `007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05` |

Both TorchScript files are external bundle assets under `weights/`; neither is distributed by the
repository.

## Frozen scientific contracts

- Small: grayscale `/255`, patch/stride `256/128`, reflect padding, uniform fusion, strict
  `probability > 0.30`, bottom exclusion 130 px, `min_area_px=64` when explicitly configured.
- Large: grayscale `/255`, patch/stride `512/256`, reflect padding, uniform fusion, strict
  `probability > 0.50`, bottom exclusion 180 px.
- Large `calibrated_analysis`: `min_area_px=512`, `151.22873345935727 nm²`, equivalent-circle
  diameter `13.87625323135418 nm`, `watershed=false`, `fill_holes=true`, `exclude_border=true`,
  connectivity 2, perimeter neighborhood 8, scale `100/184 nm_per_pixel`.
- Large validation is field-of-view-level, not sample-level independent. `BaCo-3` has known
  under-segmentation and `BaCu-1` known false positives. Do not claim cross-material stability or
  scientific readiness.

## Implemented seams

- U-Net adapter supports grayscale input, uniform tiled fusion, strict thresholding, configurable
  bottom exclusion, and full-size invalid-bottom output.
- TorchScript exporter supports `small_batchnorm` and `large_groupnorm_optimized` profiles with
  strict state-dict diagnostics and eager/TorchScript equivalence checks.
- Small full Analysis smoke script exists at `scripts/models/smoke_unet_small_analysis.py`.
- Large threshold and min-area calibration scripts exist under `scripts/models/`.
- Large three-image full Analysis smoke script exists at
  `scripts/models/smoke_unet_large_analysis.py`. It only accepts an image directory, private
  registry, and new external output root; it has no test-mask input.

## Immediate next step on Linux cloud

Apply the latest hotfix, then run:

```bash
python -m pytest -q tests/unit/scripts/test_smoke_unet_large_analysis.py
python -m ruff check .
git diff --check
```

If those pass, run the real held-out Analysis smoke without any GT directory:

```bash
python scripts/models/smoke_unet_large_analysis.py \
  --image-dir /external/path/to/test_images/images \
  --registry /external/path/to/private-ready-registry.yaml \
  --output-root /external/new/nonexistent/large-analysis-smoke
```

The fixed images are `SrZr-3.tif`, `BaCu-2.tif`, and `PrCu-3.tif`. Check all three final states,
status histories, model/config/card/adapter hashes, frozen configuration, bottom-180 ROI evidence,
density consistency checks, quality warnings, masks, instances, particle CSVs, overlays, labeled
particles, reports, and execution provenance.

## Latest verification and transfer files

- Windows: Ruff passed; relevant non-Analysis tests `25 passed`; `git diff --check` passed with only
  existing LF/CRLF warnings.
- The new Large Analysis test cannot collect on Windows because existing
  `app/storage/pinned_file.py` requires POSIX `os.O_DIRECTORY`. Do not patch around it; run on Linux.
- Latest Analysis hotfix:
  `NanoLoop-Agent-large-analysis-smoke-hotfix.tar.gz`, SHA-256
  `d89a097993053a3c637a92877c602a94b606e18e516a6eb7a99c6139dca5fdbc`.
- Large science-freeze hotfix SHA-256:
  `34283144af6a7a7a87f21b6399439228f6438ed40106b470ab0f32b2fd54995f`.
- Transfer directory: `C:\Users\22290\Desktop\AI4S\nanoloop-smoke-transfer`.
- `NanoLoop-Agent-current-worktree.tar.gz` predates the latest Large changes; do not use it as the
  final source snapshot without regenerating it.

## New-task guardrails

Continue from the existing worktree. First inspect `git status --short` and this handoff. Do not
change frozen scientific values, global defaults, Adapter, Gateway, public readiness status, or any
other model family while validating the Large three-image full Analysis chain.
