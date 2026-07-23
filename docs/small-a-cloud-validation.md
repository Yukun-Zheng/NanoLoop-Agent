# Small-A controlled cloud validation

> Status on 2026-07-23: repository intake is complete. The source checkpoint was strictly loaded,
> and the checked-in runtime was re-exported with PyTorch 2.6.0, verified again with PyTorch 2.13.0,
> and exercised through the real registry/snapshot/Gateway lifecycle on CPU. The public registry
> now truthfully reports runtime `ready`. The exact artifact has also passed a minimal
> Debian 12 Linux ARM64 load/forward check with `torch 2.6.0+cpu`; the checklist below remains for
> the complete target-host Gateway/Analysis rerun. See the
> [Small-A acceptance audit](model-assets-small-a-acceptance-2026-07-23.md). This checklist remains
> the target-Linux reproduction procedure and does not represent Small-B scientific acceptance.

This checklist reproduces the accepted repository runtime on the frozen target-Linux baseline.
It does not re-export the model and does not perform Small-B calibration or scientific evaluation.

## Frozen environment

- OS: Debian 12 (bookworm), matching the repository container base
- Python: 3.12
- PyTorch: 2.13.0
- torchvision: 0.28.0
- Device for acceptance: CPU
- Model ID: `unet-small-balanced-v1`
- Expected repository TorchScript SHA-256:
  `09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d`
- Expected image size: `2048 x 1536 px`

PyTorch 2.13.0 and torchvision 0.28.0 are the exact pair used by the current unified application
runtime. The checked-in Small-A compatibility export remains independently verified under
PyTorch 2.6.0, but Large serializes a reference to `aten::_upsample_lanczos2d_aa`; therefore 2.6 is
not a valid version for the complete application. The historical Large training notebook recorded
Python 3.10 and PyTorch 2.2.2; that older training environment is not used as the current
application validation runtime.

## Private paths

Use a private filesystem unavailable to public web serving:

```bash
export SMALL_ROOT=/srv/nanoloop-private/unet-small-balanced-v1
export SMALL_RECORD="$SMALL_ROOT/run-record"
export SMALL_TORCHSCRIPT="$PWD/model_artifacts/weights/unet-small-balanced-v1.pt"
export SMALL_REGISTRY="$PWD/model_artifacts/registry.yaml"
export SMALL_IMAGE=/srv/nanoloop-private/authorized-images/SrNi-1.tif
```

## Environment and source record

Run from the checked-out NanoLoop-Agent repository:

```bash
set -euo pipefail
test -z "$(git status --porcelain)"
git rev-parse HEAD
git merge-base --is-ancestor origin/main HEAD

python3.11 -m venv .venv-small-a-cloud
source .venv-small-a-cloud/bin/activate
python -m pip install --upgrade "pip<26"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.13.0" "torchvision==0.28.0"
python -m pip install -e ".[analysis,dev]"
python -m pip check

install -d -m 0750 \
  "$SMALL_RECORD"

{
  cat /etc/os-release
  uname -a
  lscpu
  python --version
  python -m pip --version
  python -c 'import torch, torchvision; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("cuda", torch.version.cuda); print("device", "cpu")'
  git rev-parse HEAD
  git status --short --branch
  python -m pip freeze
} > "$SMALL_RECORD/environment.txt"
```

## Repository artifact and registry identity

The deployment validation uses the exact checked-in runtime, not a newly exported candidate.
Verify both the artifact and registry identity before any load:

```bash
test -f "$SMALL_TORCHSCRIPT"
printf '%s  %s\n' \
  '09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d' \
  "$SMALL_TORCHSCRIPT" | sha256sum --check -
grep -F 'model_id: unet-small-balanced-v1' "$SMALL_REGISTRY"
grep -F \
  'weight_sha256: 09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d' \
  "$SMALL_REGISTRY"
sha256sum "$SMALL_TORCHSCRIPT" > "$SMALL_RECORD/torchscript.sha256"
```

The checkpoint identity, strict 128-key load, compatibility re-export and eager/TorchScript
comparison are frozen in the
[machine-readable delivery audit](../model_artifacts/evidence/unet-small-balanced-v1/delivery-audit-2026-07-23.json).
Reproducing the export is a separate provenance exercise; it must not replace the checked-in
runtime during deployment validation.

## Gateway lifecycle, full image, BOXES ROI, and determinism

The controlled integration test uses the real private bundle and real TorchScript:

```bash
export NANOLOOP_SMALL_PRIVATE_REGISTRY="$SMALL_REGISTRY"
export NANOLOOP_SMALL_SMOKE_IMAGE="$SMALL_IMAGE"

python -m pytest -q -s \
  tests/integration/test_unet_small_private_bundle.py \
  | tee "$SMALL_RECORD/gateway-validation.txt"
```

It verifies:

- registry and pre-load health;
- frozen schema-v1 content-addressed model bundle;
- two identical full-image predictions with the same seed;
- exact `probability.npy` and binary-mask SHA equality;
- bottom 130 rows are zero;
- BOXES ROI exterior is zero;
- deterministic Torch controls are active;
- loaded health reports CPU;
- explicit unload empties the Adapter cache.

## Analysis smoke

Use pixel-only mode unless an independently verified physical scale is supplied:

```bash
python scripts/models/smoke_unet_small_analysis.py \
  --image "$SMALL_IMAGE" \
  --registry "$SMALL_REGISTRY" \
  --output-root "$SMALL_ROOT/smoke-run-1" \
  --sample-id SrNi-1 \
  --pixel-only \
  --device cpu \
  --seed 2026 \
  | tee "$SMALL_RECORD/gateway-smoke.json"
```

The output must report `engineering_acceptance`, schema-v3 bundle provenance, ready health before
and after prediction, an empty cache after unload, and
`scientific_acceptance_eligible=false`.

## Repository gates

```bash
python -m pytest -q \
  tests/unit/inference \
  tests/unit/scripts/test_export_unet_small_torchscript.py \
  tests/unit/scripts/test_smoke_unet_small_analysis.py

python -m pytest -q -s tests/integration/test_unet_small_private_bundle.py

python -m ruff check \
  scripts/models/export_unet_small_torchscript.py \
  scripts/models/smoke_unet_small_analysis.py \
  tests/unit/inference/test_unet_small_assets.py \
  tests/unit/scripts/test_export_unet_small_torchscript.py \
  tests/unit/scripts/test_smoke_unet_small_analysis.py \
  tests/integration/test_unet_small_private_bundle.py

git diff --check
git status --short --branch
git diff --name-status main...HEAD
```

Save the exact commands actually executed, without secrets:

```bash
cp docs/small-a-cloud-validation.md "$SMALL_RECORD/commands.txt"
```

## Finalization rule

Repository runtime acceptance is already recorded against TorchScript SHA-256
`09d1818c72652179e2590897cf409f7691e18e5e1a0f55476f90f7369a03171d`. A target-Linux rerun must
use that exact repository artifact and preserve its output manifest outside Git. A failed or
different target deployment must be reported as a deployment-specific failure and must not mutate
the recorded asset identity. Small-B scientific acceptance remains pending until an authorized
independent test set, masks, split, calibrated settings and reproducible metrics are delivered.
