# NanoLoop-MSBI data and baseline audit

Audit date: 2026-07-24
Baseline commit: `7f22e9338a29fca59f82f8138eefe42bad63a316`
Working branch: `codex/msbi-instance-seg-v1`

## Outcome

The final delivery is usable for a validation-gated development pilot, but not for a claim about
human instance segmentation. It contains 93 official training SEM views, 90 binary training masks,
and 9 independent-test views/masks. The three maskless `SrZn` training views are retained only for
teacher distillation. Independent-test pixels remain sealed.

The delivery archive `yukun_.zip` has SHA-256
`ad9118c9a300df0476d8be03a71f41b0b887cf4317aac18fe9ed605e15ee6f40`.
Private pixels are stored outside Git below the configured private asset root. This document does
not publish a personal-account path.

## Final delivery interpretation

郭境濠's final tables are treated as authoritative:

- `train_groups.csv`: all 93 rows belong to the official training pool.
- `test_groups.csv`: all 9 rows belong to the independent test.
- Official training morphology labels: 44 small, 28 large, 21 agglomerated.
- Official test morphology labels: 3 small, 3 large, 3 agglomerated.
- All audited SEM images and masks are `1536×2048`; masks are `uint8` binary `{0,255}`.
- Training masks are missing for `SrZn-1`, `SrZn-2`, and `SrZn-3`.
- No exact source-view filename occurs in both official train and test.
- Material tokens `BaCu`, `PrCu`, and `SrZr` occur on both sides. This is a documented
  material-level leakage risk in the delivered split, although the exact SEM views remain
  disjoint.
- Earlier packages incorrectly placed `BaCu-2`, `PrCu-3`, and `SrZr-3` in a training bundle.
  The final table overrides that package; these views are excluded from every training and
  validation manifest.

## Manifest and split

The frozen local manifest is generated under `artifacts/msbi/manifests/`:

| Split | Views | Split unit | Notes |
|---|---:|---|---|
| train | 75 | material group within official train pool | 72 labeled, 3 teacher-only |
| validation | 18 | material group within official train pool | all labeled |
| independent test | 9 | delivered source SEM view | pixels sealed |

Validation groups are `BaCu`, `BiCr`, `BiCu`, `GdCu`, `LaCo`, `LaCr`, and `SmNi`. A patch from one
source view can only occur in one split. The split-manifest SHA-256 is
`c1d1605018162e263ea46b6cea0e9f1472c2c6882b8d36ad6f330d6393bf7d36`.

Every record contains the source/mask hash, split, group, validity policy and supervision flag.
Chemical formulas and physical scale are intentionally unknown; neither is inferred from short
filenames or an unfrozen scale bar.

## Ground truth and valid region

The masks are binary semantic foreground masks, not independent particle-ID masks. MSBI target
generation labels connected components and derives center heatmaps, instance boundaries, normalized
distance fields and scale targets from those components. These are explicitly
**pseudo-instances**. They cannot establish performance on touching particles that a human labeled
as separate instances.

All 90 training masks are empty in the bottom 128 rows, and a visual audit found a consistent
instrument-information strip in that region. Therefore each current manifest record freezes
`invalid_bottom_px=128`; loss and evaluation exclude those pixels. This is evidence-based for the
current delivery, not a universal SEM assumption.

## Authorization boundary

The project owner supplied the archives and authorized their use for this training task. Permission
to redistribute raw SEM pixels, masks, derived private artifacts, or a trained checkpoint was not
documented. Consequently:

- raw/derived private data and checkpoints stay outside Git;
- only code, configuration, hashes, manifests without pixels, model cards, and aggregate evidence
  are candidates for source control;
- the MSBI registry entry remains `unavailable`;
- `ready_recommendation` remains false.

## Runtime environment

- Host: Apple Silicon `arm64`, macOS 15.5, Apple M4 MacBook Air, 24 GB unified memory.
- Python: 3.12.13 in the project `.venv`.
- PyTorch: 2.13.0; MPS available; CUDA and NVIDIA tooling unavailable.
- The MPS host is suitable for forward, smoke, export, and a tightly bounded pilot. It is not
  treated as the hardware for formal long training or multi-seed acceptance.

## Current U-Net baselines on the same validation split

Metrics use the same 18 views, validity mask, and connected-component pseudo-instance definition.
AP50/AP75 are null because the current binary baselines do not expose confidence-ranked instances;
matched recall at IoU 0.50/0.75 is reported instead.

| Model | Pixel Dice | Boundary F1 | Pseudo-instance F1@0.5 | PQ | Count MAE | Diameter W1 px | Mean runtime ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `unet-small-balanced-v1` | 0.7465 | 0.5817 | 0.7184 | 0.5944 | 7.2222 | 17.1027 | 2348.7 |
| `unet-large-optimized-v1` | 0.6664 | 0.3874 | 0.5996 | 0.4857 | 17.1111 | 9.4943 | 1562.9 |

The Small U-Net is the frozen acceptance reference because it has the strongest validation
pseudo-instance F1. Neither baseline is promoted here as independently scientifically accepted.

## Blocking and risk register

- No human instance IDs: touching-particle and true AP claims are blocked.
- No frozen physical calibration: nm-based size/density claims are blocked.
- No CUDA host: full staged A2–A7 ablation, multiple seeds, EMA/AMP and formal training are
  incomplete.
- Data/model redistribution permission is undocumented.
- Official train/test material-token overlap exists for three material tokens.
- The independent-test archive must remain sealed until a candidate passes the validation policy.
