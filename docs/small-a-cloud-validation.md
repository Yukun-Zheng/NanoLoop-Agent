# Small-A controlled cloud validation

This checklist completes the runtime evidence that cannot be produced in the local Windows
environment. It does not perform Small-B calibration or scientific evaluation.

## Frozen environment

- OS: Debian 12 (bookworm), matching the repository container base
- Python: 3.11
- PyTorch: 2.6.0
- torchvision: 0.21.0
- Device for acceptance: CPU
- Model ID: `unet-small-balanced-v1`
- Expected checkpoint SHA-256:
  `915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008`
- Expected image size: `2048 x 1536 px`

PyTorch 2.6.0 and torchvision 0.21.0 are the lowest exact pair allowed by the current
`pyproject.toml` model-runtime contract. The historical Large training notebook recorded Python
3.10 and PyTorch 2.2.2; that older training environment is not used as the current application
validation runtime.

## Private paths

Use a private filesystem unavailable to public web serving:

```bash
export SMALL_ROOT=/srv/nanoloop-private/unet-small-balanced-v1
export SMALL_SOURCE="$SMALL_ROOT/source"
export SMALL_BUNDLE="$SMALL_ROOT/private-model-bundle"
export SMALL_RECORD="$SMALL_ROOT/run-record"
export SMALL_CHECKPOINT="$SMALL_SOURCE/best_unet_small.pth"
export SMALL_TORCHSCRIPT="$SMALL_BUNDLE/weights/unet-small-balanced-v1.pt"
export SMALL_REGISTRY="$SMALL_BUNDLE/registry.yaml"
export SMALL_IMAGE=/srv/nanoloop-private/authorized-images/SrNi-1.tif
```

The corresponding Windows custody root is
`C:\Users\22290\Desktop\AI4S\NanoLoop-ModelAssets\unet-small-balanced-v1`. Creating that directory
or copying the checkpoint requires explicit operator authorization and is not performed by the
repository changes.

## Environment and source record

Run from the checked-out NanoLoop-Agent repository:

```bash
set -euo pipefail
test "$(git branch --show-current)" = "feat/a-small-unet-v1"
test -z "$(git status --porcelain)"
git rev-parse HEAD
git rev-parse main

python3.11 -m venv .venv-small-a-cloud
source .venv-small-a-cloud/bin/activate
python -m pip install --upgrade "pip<26"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.6.0" "torchvision==0.21.0"
python -m pip install -e ".[analysis,dev]"
python -m pip check

install -d -m 0750 \
  "$SMALL_SOURCE" \
  "$SMALL_BUNDLE/configs" \
  "$SMALL_BUNDLE/model_cards" \
  "$SMALL_BUNDLE/weights" \
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

Upload `best_unet_small.pth` to `$SMALL_CHECKPOINT` through the approved private channel, then
verify custody before any load:

```bash
test -f "$SMALL_CHECKPOINT"
printf '%s  %s\n' \
  '915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008' \
  "$SMALL_CHECKPOINT" | sha256sum --check -
```

## Strict load, export, reload, and repeatability

The export command performs the 128-key/shape diagnosis, strict load, eager/TorchScript comparison,
save/reload, exact repeated logits/probability inference, and both SHA calculations:

```bash
test ! -e "$SMALL_TORCHSCRIPT"
python scripts/models/export_unet_small_torchscript.py \
  --checkpoint "$SMALL_CHECKPOINT" \
  --output "$SMALL_TORCHSCRIPT" \
  --architecture-profile small_batchnorm \
  --expected-checkpoint-sha256 \
    915911107c82c01ff7d37746f4fcce6db39d40659cfb93e059e14b18134ba008 \
  | tee "$SMALL_RECORD/export-report.json"

test -s "$SMALL_TORCHSCRIPT"
sha256sum "$SMALL_TORCHSCRIPT" | tee "$SMALL_RECORD/torchscript.sha256"
```

Do not copy a historical or expected TorchScript SHA into the registry. Use only the digest produced
by the command above.

## Candidate private ready bundle

Copy only public text contracts from the exact validation commit:

```bash
cp model_artifacts/configs/unet-small-balanced-v1.yaml \
  "$SMALL_BUNDLE/configs/unet-small-balanced-v1.yaml"
cp model_artifacts/model_cards/unet-small-balanced-v1.md \
  "$SMALL_BUNDLE/model_cards/unet-small-balanced-v1.md"

export SMALL_TORCHSCRIPT_SHA
SMALL_TORCHSCRIPT_SHA="$(sha256sum "$SMALL_TORCHSCRIPT" | awk '{print $1}')"
```

Create `$SMALL_REGISTRY` with the computed digest:

```bash
cat > "$SMALL_REGISTRY" <<YAML
schema_version: "2.0"
models:
  - metadata:
      model_id: unet-small-balanced-v1
      family: unet
      variant: small_particle
      quality_tier: balanced
      version: "1"
      status: ready
      supports_box_prompt: false
      default_threshold: 0.30
      preprocess_profile: sem-gray-unit-crop-bottom-130-v1
      postprocess_profile: semantic-mask-v1
      inference_invalid_bottom_px: 130
      expected_input_width: 2048
      expected_input_height: 1536
      applicable_materials: []
      metrics: {}
      metric_context:
        engineering_default_threshold: 0.30
        threshold_comparison: gt
        scientific_calibration_status: pending_small_b
        cloud_validation_status: candidate
      notes: Controlled Small-A cloud-validation candidate; not Small-B scientific acceptance.
    adapter_path: app.inference.adapters.unet:UNetAdapter
    weight_path: weights/unet-small-balanced-v1.pt
    weight_sha256: ${SMALL_TORCHSCRIPT_SHA}
    config_path: configs/unet-small-balanced-v1.yaml
    model_card_path: model_cards/unet-small-balanced-v1.md
    required_modules:
      - torch
YAML
```

Verify the generated registry:

```bash
chmod 0640 "$SMALL_REGISTRY"
grep -F "weight_sha256: $SMALL_TORCHSCRIPT_SHA" "$SMALL_REGISTRY"
test "$(grep -c 'status: ready' "$SMALL_REGISTRY")" -eq 1
```

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

After every command passes, update the repository model card and registry metadata with the actual
TorchScript SHA and summarized engineering results, copy the final text contracts into the private
bundle, and rerun the registry/Gateway/smoke commands once more. Until that final rerun is recorded,
all runtime results remain `pending cloud validation` and the public registry remains
`unavailable`.
