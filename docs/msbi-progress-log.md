# NanoLoop-MSBI progress log

## 2026-07-24 formal CUDA runtime selection

- The training host exposes eight RTX 4090 GPUs with NVIDIA driver
  `570.211.01` (CUDA driver API 12.8).
- PyPI's `torch==2.13.0` Linux wheel resolved to `torch 2.13.0+cu130`.
  A real BF16 CUDA allocation correctly refused to initialize because that
  wheel requires a newer driver; no training was attempted with it.
- Formal training therefore pins the official PyTorch CUDA 12.8 builds
  `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128`. The project lower
  bounds were adjusted to those versions, which provide all APIs used by the
  training and export paths.
- This is a runtime compatibility choice, not a model or data change. The
  independent-test split remains absent from the training host.

## 2026-07-24 formal input-pipeline validation

- Stage A retained the original on-demand loader for an unchanged first formal
  run. Profiling showed about 48 seconds per 1,024-patch epoch while the GPU
  waited on repeated TIFF normalization and NPZ decompression.
- Stages B-E enable a read-only preload cache. It keeps the stored compact
  target dtypes, performs the same crop-time casts, and leaves record
  selection, seeds, augmentation and supervision masks unchanged.
- A remote loader-only benchmark preloaded all train and validation records in
  9.72 seconds, used 3.12 GiB for the 75-record train cache, and produced the
  first finite `8×1×256×256` batch in 0.39 seconds. The host has 1 TiB RAM.
- Unit evidence compares every returned cached and on-demand tensor for the
  same record and seed. Independent-test data is not part of either cache.

This log records what actually ran. Private paths are abbreviated as `$PRIVATE_MSBI`.

## 2026-07-24 — repository and data audit

- Confirmed baseline `origin/main@7f22e93`; created `codex/msbi-instance-seg-v1`.
- Audited the supplied ZIP files and CRCs. The final `yukun_.zip` SHA-256 is
  `ad9118c9a300df0476d8be03a71f41b0b887cf4317aac18fe9ed605e15ee6f40`.
- Parsed the final train/test tables without decoding independent-test pixels.
- Generated 75 train, 18 validation, and 9 sealed-test records with
  `scripts/models/prepare_msbi_data.py`. Exit code: 0.
- Generated pseudo-instance targets for 90 labeled training-pool views. Exit code: 0.
- Cached Small/Large teacher probabilities for the three maskless `SrZn` views. Exit code: 0.

## 2026-07-24 — implementation

- Added ConvNeXt-Tiny single-channel encoder, FPN, small/large experts, learned softmax gate,
  foreground/center/boundary/SDF heads, uncertainty signals, masked multi-task loss, deterministic
  watershed, training/export/evaluation entrypoints, and an independent `MSBIAdapter`.
- Extended Analysis artifact persistence and the existing result page with optional center,
  boundary, instance-label, gate and uncertainty layers.
- Generated the OpenAPI document and TypeScript client schema. Exit code: 0.

## 2026-07-24 — smoke and export

- Synthetic dry-run: all seven heads returned the configured shapes; finite loss. Exit code: 0.
- Two-epoch micro training and one-epoch resume: checkpoint save/resume succeeded. Exit code: 0.
- Micro TorchScript export: eager/script maximum absolute difference 0; repeat difference 0;
  finite outputs. Exit code: 0.
- Micro runtime Adapter smoke ran twice on CPU: stable instance labels and zero invalid-bottom
  output. Exit code: 0.
- ConvNeXt-Tiny ImageNet initialization was downloaded from the official torchvision revision and
  adapted to one channel by averaging the RGB stem.

## 2026-07-24 — same-split U-Net baselines

Command:

```bash
.venv/bin/python scripts/models/evaluate_msbi_baselines.py \
  --registry model_artifacts/registry.yaml \
  --validation-manifest artifacts/msbi/manifests/validation.jsonl \
  --output-dir artifacts/msbi/baselines \
  --device mps
```

Exit code: 0. Small U-Net achieved validation pseudo-instance F1 0.7184 and count MAE 7.2222;
Large U-Net achieved 0.5996 and 17.1111.

## 2026-07-24 — original bounded real-data Pilot

Configuration: `model_artifacts/training_configs/msbi-stage-c-pilot.yaml`. The encoder was frozen;
three epochs used 64 training patches and 19 validation patches per epoch on MPS.

| Epoch | Train loss | Train patch Dice | Validation loss | Validation patch Dice |
|---:|---:|---:|---:|---:|
| 0 | 1.8530 | 0.2717 | 1.8017 | 0.5328 |
| 1 | 1.7286 | 0.3508 | 1.7971 | 0.6643 |
| 2 | 1.7886 | 0.3329 | 1.7899 | 0.5684 |

Checkpoint SHA-256:
`a49fb806fa210d2e1223ee3bec92c29dea36bd46564dfbdc938058ab49c85a6f`.
TorchScript SHA-256:
`250fd81c2e5ca8f4e815cd77d9e3b9385acd27f4f4e4378207e2d352b94e0a4d`.

Full 18-view validation returned pixel Dice 0.4064, boundary F1 0.0749,
pseudo-instance F1 0.1229, count MAE 19.0556 and mean runtime 5012 ms. Status:
`PILOT_NOT_ACCEPTED`.

## 2026-07-24 — augmentation defect and corrective rerun

While building the augmentation audit, a geometric-alignment defect was found: each target tensor
was drawing independent flip decisions after a common rotation. This could misalign SEM,
foreground, center, boundary and SDF targets. The code was fixed so every modality shares one
rotation and one horizontal/vertical flip decision, and a regression test was added.

The original failed run and its evidence were retained. A same-config, same-seed,
same-three-epoch corrective Pilot ran in a new output directory. It reached train patch Dice
0.5805 and validation patch Dice 0.6249. Its full 18-view validation results were pixel Dice
0.5563, boundary F1 0.2034, pseudo-instance F1 0.3485, count MAE 18.8889, and mean MPS runtime
5535 ms. The corrected checkpoint SHA-256 is
`71b7d5edd5d7a395a1d79fe505ce9baf78f649f3ab9a580fc66ed13d423b8467`; the corrected TorchScript
SHA-256 is `f03aea4d86b8e1052e3c5e0b7a0a7afeaada5a86627691ce92f3917e724743d8`.

## 2026-07-24 — bounded decoder calibration

The first corrected full-view result used the initial decoder threshold 0.50. A validation-only
bounded search then tested foreground thresholds 0.50, 0.60, 0.70, 0.80, 0.85, 0.90 and 0.95,
plus two 32-pixel minimum-area variants. Center, boundary, NMS and watershed settings stayed fixed.
The selection rule was instance F1, then count MAE, then pixel Dice.

The search stopped at and froze `foreground_threshold=0.85`, `min_area_px=32`. Final validation
metrics became pixel Dice 0.6213, boundary F1 0.2554, pseudo-instance F1 0.5136, count MAE 12.1111,
and mean MPS runtime 4883 ms. This remained below all four scientific thresholds. The search table
is stored in `artifacts/msbi/decoder-calibration.json`; no independent-test pixels were read.

## 2026-07-24 — frozen validation gate

The tolerance policy was frozen before any test-pixel access:

```bash
.venv/bin/python scripts/models/freeze_msbi_tolerance_policy.py \
  --baseline-results artifacts/msbi/baselines/baseline-metrics.json \
  --split-manifest artifacts/msbi/manifests/split-manifest.json \
  --output artifacts/msbi/tolerance-policy.json
```

Policy SHA-256:
`264265bbf80c23d7d20f611b6cf6f757aa7792c04747ddfde7bc22ba749930ec`.
Both Pilots, including the calibrated corrective candidate, failed four scientific checks and
passed only runtime/gate-collapse guardrails. The
fail-closed test entrypoint returned exit code 3 before opening a deliberately nonexistent test
manifest, proving that a failed gate does not read test data.

## 2026-07-24 — pre-formal implementation verification

- Ruff: pass.
- Mypy: 128 source files, pass.
- Pytest: 1435 passed, 1 controlled private-bundle test skipped, exit code 0.
- Alembic upgrade/downgrade and metadata drift checks: pass.
- OpenAPI regeneration: byte-identical SHA-256
  `60d689fefc0aabc34025a8ad4246d16b1895baa22114007093f38082b165bdf5`.
- TypeScript schema regeneration: byte-identical SHA-256
  `f8696bb867af509afad09757ab4f928e1b1620b8af4169bcf8213d5370b0f1b4`.
- Frontend production dependency audit: no known vulnerabilities.
- Frontend lint/typecheck: pass.
- Vitest: 21 files, 103 tests, pass.
- Next.js production build: pass.
- Playwright: 6 tests, pass.
- `docker compose config --quiet`: pass.

The aggregate `make check` and `pnpm check` commands reached their generated-contract
`git diff --exit-code` checks and returned nonzero because the newly generated OpenAPI/schema files
are intentionally uncommitted changes on this feature branch. Regeneration before/after hashes were
identical, and all remaining subchecks were run separately. Node 26 also produced an engine warning
because the project declares Node 24; build and tests nevertheless passed.

## 2026-07-24 — formal CUDA stages A–E

The owner authorized a formal run. Only the 75-record train and 18-record validation manifests and
their source pixels were transferred to the CUDA host. The nine independent-test records and pixels
were not transferred. Formal training used physical RTX 4090 GPU 2, seed 2026, BF16 AMP, EMA,
warm-up plus cosine scheduling, gradient clipping, early stopping, and exact checkpoint lineage.

| Stage | Main change | Best patch validation loss | Best checkpoint SHA-256 |
|---|---|---:|---|
| A | single expert, frozen encoder | 1.651945 | `3f5da314f423afdfa4f07c1164ea327b2bf3758b4b8bb9e0ee6102f88bf42814` |
| B | dual experts, fixed mean | 0.803364 | `929876b3c89256c9aa58f603465042fc2a8dde4316207a938628af971a935435` |
| C | learned pixel-wise gate | 0.790850 | `0697f1d2fc2095cc4eb0b7bfe7fd6d63643d577d2b33b52692db028bc6d8546b` |
| D | encoder unfreeze + teacher distillation | 0.731599 | `3052f1d69fefb4da01abf15c671ffd09f4b762eec5f572733a141f00e5a933a4` |
| E | signed-distance head | 0.731320 | `ea287f63c6a38dbd2a0868128124286657ca1666a6a7bc19737bd418c2d4e989` |

The sealed A–E orchestrator completed with
`independent_test_transferred=false`. Stage D and E were both exported with strict checkpoint
loading, zero eager/TorchScript output difference, zero deterministic repeat difference, and finite
seven-head outputs.

Full-view validation with the same initial decoder showed:

| Candidate | Pixel Dice | Boundary F1 | Pseudo-instance F1@0.5 | Count MAE | Runtime ms |
|---|---:|---:|---:|---:|---:|
| Stage D | 0.7352 | 0.4557 | 0.7185 | 10.3889 | 4284.6 |
| Stage E | 0.7435 | 0.4601 | 0.7218 | 10.2222 | 4808.1 |

Validation-only searches over Stage E decoder parameters did not produce a setting that passed all
four scientific rules. Independent-test status remained sealed.

## 2026-07-24 — corrective formal Stage F

Stage F was initialized from Stage E and targeted the two remaining failure modes. It used
`512×512` patches, morphology-balanced sampling, a differentiable contour-consistency term, higher
foreground/boundary weights, batch size 2 with four-step accumulation, EMA 0.998, and low encoder
and head learning rates. A real CUDA dry-run included forward, loss, backward, clipping and optimizer
step. Formal training early-stopped after epoch 6; epoch 0 remained the best checkpoint:
`2aab2acb179cf6826f444c5e22a53d21df6f8393ed79107daa325ad4a963dad1`.

The strict Stage F TorchScript export is
`c430a8c94a5798c7e111860ea957c09f512a44f9b9e14cf1e4c8c38bbd5cc57f`.
With the prior decoder, Stage F reached Dice 0.7636, boundary F1 0.4855,
pseudo-instance F1 0.7461, count MAE 9.2778 and mean runtime 4367.2 ms.

A final validation-only 168-combination search froze:

- foreground threshold 0.85;
- center threshold 0.50;
- center NMS radius 13;
- minimum area 256 px.

The complete 18-view rerun with that decoder produced Dice 0.7625, boundary F1 0.4882,
pseudo-instance F1 0.7674, count MAE 7.7222, runtime 4208.6 ms, and mean small/large routing
0.6401/0.3599. The instance, Dice, runtime, and anti-collapse checks passed. Count MAE exceeded
6.8611 and boundary F1 was below 0.5617, so the frozen gate status is `FAILED`.

No independent-test manifest or pixels were opened or transferred. The registry remains
`unavailable`, `ready_recommendation=false`, and the formal TorchScript is retained as a private
research artifact rather than a production weight.

## 2026-07-25 — formal handoff verification

- Final private TorchScript download SHA-256 matched the formal export manifest:
  `c430a8c94a5798c7e111860ea957c09f512a44f9b9e14cf1e4c8c38bbd5cc57f`.
- Formal runtime freeze, A–F training manifests/curves, D–F export manifests, D/E/F validation
  metrics, all decoder-search tables, the tolerance policy, and the failed gate record were copied
  into the ignored evidence directory `artifacts/msbi/formal-20260724/`.
- Final machine summary SHA-256:
  `67625effec5d4f15df95fbbda438b7c27d1019f9f5a205df5129add758c865bf`.
- Ruff: pass.
- Mypy: 128 source files, pass.
- Pytest: 1439 passed, 1 controlled private-bundle test skipped.
- Focused MSBI/registry tests: 30 passed.
- Registry status remained `unavailable`; no production promotion or independent-test evaluation
  was performed.

## 2026-07-25 — Stage I strict-superiority candidate

After the earlier ConvNeXt and MobileNet candidates failed either quality or latency, Stage I used
an efficient anchored MSBI: the verified Small U-Net semantic backbone/head stayed frozen and exact,
while trainable low-resolution small/large experts produced gate, center, boundary, and SDF
signals. Formal training ran 24 epochs on physical RTX 4090 GPU 2 with seed 2026.

- Best checkpoint SHA-256:
  `bc6ae0146e92275c03ee0ba132335997999bca5c149bbcad42a09f41dae28d47`.
- Compact TorchScript SHA-256:
  `6d68ca191f668409fb0547c7551c7b913b42d8c2e0c05cf311f51932b61b907f`.
- Parameter count: 3,369,897.
- Full, anchor, and compact eager/TorchScript outputs were all-close with maximum absolute
  difference `7.63e-06`; deterministic repeat difference was zero.

Validation-only multiscale calibration selected an `80%` tiled / `20%` full-frame probability
blend, `256×256` tiles with stride 128, threshold `0.275`, minimum area 128, and connected-component
decoding. Runtime optimization preserved that segmentation output while adding a compact forward
method, CUDA probability fusion, vectorized decoding, uncompressed compact instance labels, and a
deterministically subsampled uncertainty summary.

The final 18-view CUDA validation result was:

| Dice | Boundary F1 | Pseudo-instance F1@0.5 | PQ | Count MAE | Mean ms | P95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7513 | 0.5911 | 0.7608 | 0.6313 | 4.9444 | 362.1 | 419.2 |

All nine rules in strict policy
`61d473778c65e850c3dac0ec8678ee61077e0f64480eeeb5f503b4da6b72b025` passed. The
gate record bound the candidate metrics SHA-256
`970c7fddef51864ee2b20663c6a0407d1e01821816d6b6eb4a3124bd2c9d283e` before authorizing
independent-test access.

## 2026-07-25 — one-shot independent test and production decision

Nine sealed official test views were extracted and transferred only after the bound strict gate
passed. Every image and human binary mask was verified against its frozen SHA-256. The candidate
was then evaluated once; no test result was used for calibration.

| Model | Dice | Boundary F1 | Pseudo-instance F1@0.5 | PQ | Count MAE | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSBI Stage I | 0.6279 | 0.3774 | 0.5399 | 0.4036 | 18.7778 | 392.6 | 587.8 |
| Small U-Net | 0.6135 | 0.3804 | 0.4956 | 0.3715 | 26.4444 | 481.8 | 689.4 |
| Large U-Net | 0.7624 | 0.3619 | 0.6150 | 0.5072 | 54.7778 | 431.9 | 494.0 |

MSBI therefore achieved strict simultaneous superiority on the frozen validation set, but not on
every independent-test metric against both existing weights. The registry was updated with the
current evidence but remains `unavailable` and `ready_recommendation=false`. The consumed test set
will not be used to tune a successor; a new blind labeled holdout is required for a fresh
generalization claim.

## 2026-07-25 — Stage J validation-speed successor

Stage J addressed the Stage I latency/effect tradeoff without reopening the consumed independent
test. The compact `forward_runtime` path was fixed so a nonzero bounded foreground correction is
actually exported and used by the Adapter. Regression coverage now checks both the exact
zero-correction anchor path and the nonzero bounded-correction runtime path.

The first low-memory CUDA run warm-started from Stage I on an occupied 4090 host, using batch 4 and
gradient accumulation 8. It completed successfully and produced:

- checkpoint SHA-256:
  `8e15b7bd71e476a308de9afcd72d633f98fb76c6c1f9955364eab761d6e264ed`;
- TorchScript SHA-256:
  `6a40e2d38982d8d5b648c0bf634d0155522686d44c2ec43f2ccc0389d870e30b`;
- parameter count: 3,369,897;
- full, anchor, and compact eager/TorchScript outputs all-close with maximum absolute difference
  `9.54e-06`; deterministic repeat difference was zero.

Validation-only runtime calibration found that the raw full-frame residual path was very fast but
missed the frozen Dice rule. The selected successor therefore uses a cheaper multiscale blend:
`50%` full-frame probability plus `50%` tiled `256×256` anchor probability with stride `256`,
connected components, threshold `0.350`, and minimum area `128`.

The canonical 18-view CUDA validation result was:

| Dice | Boundary F1 | Pseudo-instance F1@0.5 | PQ | Count MAE | Mean ms | P95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7478 | 0.5837 | 0.7579 | 0.6277 | 4.5556 | 267.2 | 379.3 |

The frozen strict superiority policy passed all nine checks. Candidate metrics SHA-256:
`43b5162daed0811e12f23e81a421f711c99fce33814acc136a5e3bcce73564ce`.

Stage J is faster than Stage I and improves validation count MAE, but Stage I still has higher
validation Dice, boundary F1, instance F1, and PQ. No Stage J independent-test claim is made; the
previous nine-view test remains consumed and must not be reused as blind evidence.

## 2026-07-25 — Stage J local runtime installation

The best current speed/effect validation candidate was installed into the standard local runtime
weight path:

- default runtime weight:
  `model_artifacts/weights/msbi-instance-balanced-v1.pt`;
- default runtime SHA-256:
  `6a40e2d38982d8d5b648c0bf634d0155522686d44c2ec43f2ccc0389d870e30b`;
- private Stage J copy:
  `model_artifacts/weights/private/msbi-instance-balanced-v1/stage-j-validation-speed-v1.pt`;
- private Stage I quality-first copy:
  `model_artifacts/weights/private/msbi-instance-balanced-v1/stage-i-validation-quality-v1.pt`.

The registry now marks `msbi-instance-balanced-v1` as runtime `ready` so the UI can select it. The
scientific note remains unchanged: Stage J is validation-gated and locally selectable, but
`ready_recommendation=false` until a fresh blind holdout verifies generalization.
