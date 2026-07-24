# Agglomerated-A cloud validation

## Scope

This procedure validates only the real Agglomerated U-Net runtime integration:

`private registry -> immutable snapshot/bundle -> InferenceGateway -> Analysis -> canonical
artifacts -> report ZIP -> unload`.

It does not calibrate or evaluate the model. Agglomerated-B, GT, independent-test data, Dice,
IoU, precision, recall, F1, threshold selection, min-area selection, scientific tolerance, and
cross-model comparison are explicitly out of scope. The public registry remains `unavailable`.

## Frozen external identities

- Model ID: `unet-agglomerated-specialized-v1`
- TorchScript: `unet-agglomerated-specialized-v1.pt`
- TorchScript SHA-256:
  `d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9`
- Fixed input: `BiCu-3.tif`
- Fixed input SHA-256:
  `79376cc42e5cf036b1e5e1108e5eaed16c9434816772d3f130a057e643643b29`
- Config: `model_artifacts/configs/unet-agglomerated-specialized-v1.yaml`
- Model card: `model_artifacts/model_cards/unet-agglomerated-specialized-v1.md`
- Adapter: `app/inference/adapters/unet.py`

The smoke calculates the current config, model-card, and Adapter SHA-256 values at runtime. It
rejects a private bundle that differs from the checked-out repository. Do not copy historical
snapshot versions into the new bundle.

## External directory layout

Keep all private files outside the Git checkout:

```text
<PRIVATE_ROOT>/
  source/
    sem/
      BiCu-3.tif
  private-model-bundle/
    registry-preflight.yaml
    configs/
      unet-agglomerated-specialized-v1.yaml
    model_cards/
      unet-agglomerated-specialized-v1.md
    weights/
      unet-agglomerated-specialized-v1.pt
  run-record/
    environment.txt
    commands.txt
    software-manifest.json
    <timestamped-smoke-output>/
```

The source checkpoint is not required for this smoke. Checkpoints, SEM images, TorchScript,
SQLite databases, probability arrays, snapshots, and runtime outputs must remain outside public
Git.

## Create the cloud environment

Run from a clean clone of the exact branch/commit to be validated:

```bash
set -euo pipefail

export REPO_ROOT="$PWD"
export PRIVATE_ROOT="/controlled/nanoloop/agglomerated-a"
export BUNDLE_ROOT="$PRIVATE_ROOT/private-model-bundle"
export RECORD_ROOT="$PRIVATE_ROOT/run-record"
export RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export SMOKE_ROOT="$RECORD_ROOT/agglomerated-a-smoke-$RUN_STAMP"

python3.12 -m venv .venv-agglomerated-a
source .venv-agglomerated-a/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.13,<3" "torchvision>=0.28,<1"
python -m pip install -e ".[analysis,dev]"
```

For a CUDA target, replace the CPU wheel index with the approved PyTorch index matching the
target driver. Record the exact installed versions; do not silently fall back to CPU.

## Assemble the private bundle

Copy the current public config and model card into the external bundle. Copy the already verified
TorchScript from controlled storage:

```bash
mkdir -p \
  "$BUNDLE_ROOT/configs" \
  "$BUNDLE_ROOT/model_cards" \
  "$BUNDLE_ROOT/weights" \
  "$RECORD_ROOT"

cp \
  "$REPO_ROOT/model_artifacts/configs/unet-agglomerated-specialized-v1.yaml" \
  "$BUNDLE_ROOT/configs/"
cp \
  "$REPO_ROOT/model_artifacts/model_cards/unet-agglomerated-specialized-v1.md" \
  "$BUNDLE_ROOT/model_cards/"
cp \
  "/controlled/intake/unet-agglomerated-specialized-v1.pt" \
  "$BUNDLE_ROOT/weights/"

test "$(
  sha256sum "$BUNDLE_ROOT/weights/unet-agglomerated-specialized-v1.pt" |
  awk '{print $1}'
)" = "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9"

test "$(
  sha256sum "$PRIVATE_ROOT/source/sem/BiCu-3.tif" |
  awk '{print $1}'
)" = "79376cc42e5cf036b1e5e1108e5eaed16c9434816772d3f130a057e643643b29"
```

Create `registry-preflight.yaml` with paths relative to the registry:

```yaml
schema_version: "2.0"
models:
  - metadata:
      model_id: unet-agglomerated-specialized-v1
      family: unet
      variant: dense_particle
      quality_tier: balanced
      version: "1"
      status: unavailable
      supports_box_prompt: false
      default_threshold: 0.25
      default_min_area_px: 1024
      preprocess_profile: sem-gray-p1-p99-crop-bottom-130-v1
      postprocess_profile: semantic-agglomerate-mask-v1
      inference_invalid_bottom_px: 130
      expected_input_width: 2048
      expected_input_height: 1536
      applicable_materials: []
      metrics: {}
      metric_context:
        target_definition: whole_agglomerate
      notes: Agglomerated-A private preflight; runtime smoke pending.
      health_error: Current-head Gateway/Analysis smoke pending.
    adapter_path: app.inference.adapters.unet:UNetAdapter
    weight_path: weights/unet-agglomerated-specialized-v1.pt
    weight_sha256: d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9
    config_path: configs/unet-agglomerated-specialized-v1.yaml
    model_card_path: model_cards/unet-agglomerated-specialized-v1.md
    required_modules:
      - torch
```

The input registry deliberately remains `unavailable`. The smoke builds a temporary ready
manifest only after checking all current asset identities. After a successful run it writes a
repository-external `private-registry-ready.yaml` whose weight/config/card paths point to the
content-addressed snapshots inside that output.

## Record the environment

```bash
{
  date -u
  uname -a
  python --version
  python -c 'import torch, torchvision; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available())'
  git -C "$REPO_ROOT" rev-parse HEAD
  git -C "$REPO_ROOT" status --short
  nvidia-smi || true
} > "$RECORD_ROOT/environment.txt"

python -m pip freeze --all > "$RECORD_ROOT/software-manifest.txt"
python - <<'PY' > "$RECORD_ROOT/software-manifest.json"
import importlib.metadata
import json
import platform

packages = {
    dist.metadata["Name"]: dist.version
    for dist in importlib.metadata.distributions()
    if dist.metadata.get("Name")
}
print(json.dumps(
    {"python": platform.python_version(), "packages": dict(sorted(packages.items()))},
    indent=2,
    sort_keys=True,
))
PY
```

Save the exact commands actually executed in `commands.txt`. Do not place secrets in it.

## Run the real A-only smoke

`SMOKE_ROOT` must not already exist:

```bash
python scripts/models/smoke_unet_agglomerated_analysis.py \
  --image-dir "$PRIVATE_ROOT/source/sem" \
  --registry "$BUNDLE_ROOT/registry-preflight.yaml" \
  --output-root "$SMOKE_ROOT"
```

The script rejects:

- the public registry;
- an existing output directory;
- an independent-test directory;
- a wrong input filename, size, or SHA;
- the wrong TorchScript SHA;
- a config, model card, or Adapter that differs from the current checkout;
- missing or malformed canonical artifacts;
- nonzero predictions in the bottom 130 rows;
- incomplete schema-v3 execution provenance;
- an unload that leaves any Adapter cached.

## Expected output

```text
<SMOKE_ROOT>/
  analysis.sqlite3
  artifacts/
    <job-id>/
      exports/
        agglomerated-a-report.zip
      ...
  model-snapshots/
    ...
  private-registry-ready.yaml
  gateway-analysis-smoke.json
```

`gateway-analysis-smoke.json` records:

- model, weight, config, model-card, and Adapter identities;
- input SHA;
- requested and resolved device;
- Python, Torch, torchvision, Git, image, and container identity;
- health before prediction, health after prediction, unload, and health after unload;
- cache count after unload;
- run ID and final status;
- canonical artifact relative paths, sizes, and SHA-256 values;
- report ZIP relative path, size, and SHA-256;
- execution provenance and warnings.

## Verify the result

```bash
python -m json.tool "$SMOKE_ROOT/gateway-analysis-smoke.json" >/dev/null
python -m json.tool \
  "$SMOKE_ROOT"/artifacts/*/images/*/runs/*/execution_provenance.json \
  >/dev/null

python - <<'PY' "$SMOKE_ROOT/gateway-analysis-smoke.json" "$SMOKE_ROOT"
import hashlib
import json
import pathlib
import sys
import zipfile

report = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
payload = json.loads(report.read_text(encoding="utf-8"))
assert payload["model"]["weight_sha256"] == "d36cf627dc6d1eea83b769743b657e6bb3d9a1af39ddc6a00ae95acc3b20ffc9"
assert payload["input"]["sha256"] == "79376cc42e5cf036b1e5e1108e5eaed16c9434816772d3f130a057e643643b29"
assert payload["gateway_lifecycle"]["cache_count_after_unload"] == 0
assert payload["gateway_lifecycle"]["unload_completed"] is True
assert payload["runs"][0]["roi"]["prediction_bottom_check"]["bottom_nonzero_pixels"] == 0
archive = root / payload["report_zip"]["path"]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == payload["report_zip"]["sha256"]
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert "export_manifest.json" in bundle.namelist()
print("Agglomerated-A smoke evidence verified")
PY
```

## Tests

Run on Linux before the real smoke:

```bash
pytest -q tests/unit/scripts/test_smoke_unet_agglomerated_analysis.py

pytest -q \
  tests/unit/inference/test_unet_agglomerated_assets.py \
  tests/unit/scripts/test_export_unet_agglomerated_torchscript.py \
  tests/unit/scripts/test_smoke_unet_agglomerated_analysis.py \
  tests/unit/inference/test_unet_tiling.py \
  tests/unit/inference/test_registry.py

git diff --check
```

The Windows storage implementation currently cannot be imported where `os.O_DIRECTORY` is
unavailable. Do not weaken the production storage contract for this smoke; run it and its unit
test on Linux.

## Unverified and licensing boundary

This procedure does not verify scientific accuracy, GT performance, calibration, independent-test
performance, cross-material stability, or deployment suitability beyond the recorded runtime.

The current intake has no complete license, authorization, or custody ledger. Until those records
are supplied, the checkpoint, TorchScript, SEM, and outputs remain controlled private assets and
must not enter public Git. A successful private smoke does not change the public registry.

## Clean rerun

Outputs are immutable. Never reuse or overwrite an existing smoke directory. Preserve a completed
run under controlled storage, or remove an explicitly identified failed external run directory
after recording why it failed, then choose a new `RUN_STAMP` and `SMOKE_ROOT`. Recreate the bundle
from the current checkout before every new validation so stale config/card/Adapter files cannot be
silently reused.
