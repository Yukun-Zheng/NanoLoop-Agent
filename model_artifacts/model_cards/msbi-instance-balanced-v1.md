# msbi-instance-balanced-v1

## Intended model

NanoLoop-MSBI v1 is a single-channel SEM segmentation candidate built around an exact frozen
Small U-Net semantic anchor. Low-resolution small-particle and large/agglomerate experts, a learned
softmax gate, center/boundary/SDF heads, and a calibrated full-frame/tiled probability blend add
multi-scale structure without replacing the verified semantic anchor. The production candidate
uses deterministic connected-component decoding because that matches the available binary
semantic supervision.

## Current status

- Registry status: `ready` for local runtime selection.
- Training status: formal CUDA stages A–J complete for seed 2026; multi-seed evidence is incomplete.
- Runtime status: compact TorchScript runtime and full-image CUDA Adapter verification completed.
- Scientific acceptance: `passed_strict_validation_superiority_gate`.
- Independent test: `not_accessed_for_stage_j`; the previous Stage I one-shot test did not strictly
  beat every baseline.
- Runtime selectable: `true`.
- Ready recommendation: `false` until a fresh blind holdout passes.

The supplied training masks are binary semantic masks. Connected components are explicitly marked
as pseudo-instances. They do not provide human instance IDs for touching particles, so touching
particle performance cannot be claimed from this dataset.

## Data and authorization boundary

Private SEM and masks stay outside Git. The project owner authorized their use for this task and
the current best private TorchScript has been installed locally at
`model_artifacts/weights/msbi-instance-balanced-v1.pt`. Public redistribution permission remains
undocumented. Machine-readable local manifests record content hashes and group-wise splits; public
documentation contains no private pixels or local account paths.

## Runtime outputs

The full TorchScript contract returns foreground, center, boundary, distance, small-expert,
large-expert, and gate logits. A preserved compact method returns only the foreground logits and
two gate means needed by the production path. The current Stage J validation candidate blends
`50%` tiled `256×256` anchor probabilities at stride `256` with `50%` full-frame probability,
decodes connected components at threshold `0.350`, and writes compact
uncompressed instance labels. Auxiliary raw maps remain available through the full research path.

## Formal training and validation evidence

An initial MPS Pilot exposed a geometric-augmentation defect: image and target modalities could
receive different random flips. That run is retained as failed evidence, the alignment was fixed,
and a regression test was added before formal training.

Formal stages A–H remain retained as negative or intermediate evidence. Stage I switched to the
efficient anchored design and trained the low-resolution expert, gate, center, boundary, and SDF
heads on the same 75-record train / 18-record validation split. The semantic foreground correction
was fixed at zero, making the full-frame semantic output equivalent to the verified Small U-Net
anchor while preserving the trained MSBI expert routing.

Final Stage I evidence:

- checkpoint SHA-256:
  `bc6ae0146e92275c03ee0ba132335997999bca5c149bbcad42a09f41dae28d47`;
- TorchScript SHA-256:
  `6d68ca191f668409fb0547c7551c7b913b42d8c2e0c05cf311f51932b61b907f`;
- 3,369,897 parameters;
- strict checkpoint load, eager/TorchScript all-close for full, anchor, and compact runtime methods,
  maximum absolute difference `7.63e-06`, and deterministic repeat difference zero;
- validation-only calibration froze foreground threshold `0.275`, minimum area `128 px`, tiled
  weight `0.80`, full-frame weight `0.20`, and tiled stride `128`;
- 18-view validation Dice `0.7513`, boundary F1 `0.5911`, pseudo-instance F1@0.5 `0.7608`,
  PQ `0.6313`, count MAE `4.9444`, CUDA mean `362.1 ms`, and P95 `419.2 ms`;
- mean gate routing `0.5272` small / `0.4728` large.

The frozen strict policy required every effect metric to beat the best existing validation
baseline, mean and P95 runtime to beat the fastest existing CUDA baseline, and both mean gate
weights to remain at least `0.1`. All nine checks passed before independent-test data was
transferred or opened.

## Stage J validation successor

Stage J warm-started from Stage I and enabled a bounded foreground-correction path in the compact
runtime. The exported checkpoint remained the same size, but the runtime was recalibrated to a
cheaper `50%` full-frame / `50%` tiled-anchor blend with tiled stride `256`, threshold `0.350`,
and minimum area `128 px`.

Final Stage J validation evidence:

- checkpoint SHA-256:
  `8e15b7bd71e476a308de9afcd72d633f98fb76c6c1f9955364eab761d6e264ed`;
- TorchScript SHA-256:
  `6a40e2d38982d8d5b648c0bf634d0155522686d44c2ec43f2ccc0389d870e30b`;
- installed default runtime weight:
  `model_artifacts/weights/msbi-instance-balanced-v1.pt`;
- optional private quality-first Stage I weight:
  `model_artifacts/weights/private/msbi-instance-balanced-v1/stage-i-validation-quality-v1.pt`;
- strict checkpoint load, eager/TorchScript all-close for full, anchor, and compact runtime methods,
  maximum absolute difference `9.54e-06`, and deterministic repeat difference zero;
- 18-view validation Dice `0.7478`, boundary F1 `0.5837`, pseudo-instance F1@0.5 `0.7579`,
  PQ `0.6277`, count MAE `4.5556`, CUDA mean `267.2 ms`, and P95 `379.3 ms`;
- mean gate routing `0.5365` small / `0.4635` large;
- strict superiority gate status `PASSED`, with candidate metrics SHA-256
  `43b5162daed0811e12f23e81a421f711c99fce33814acc136a5e3bcce73564ce`.

Compared with Stage I validation, Stage J trades a small amount of effect margin for a large speed
gain: mean runtime improves from `362.1 ms` to `267.2 ms`, P95 from `419.2 ms` to `379.3 ms`, and
count MAE from `4.9444` to `4.5556`. Stage I still has higher validation Dice, boundary F1,
instance F1, and PQ, so Stage J is a faster validation-successor candidate rather than a universal
quality replacement.

## One-shot independent test

After the bound validation gate passed, nine official independent-test views were authorized,
hash-verified, transferred, and evaluated once. No test result was used to tune the candidate.

| Model | Dice | Boundary F1 | Pseudo-instance F1@0.5 | PQ | Count MAE | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSBI Stage I | 0.6279 | 0.3774 | 0.5399 | 0.4036 | 18.7778 | 392.6 | 587.8 |
| Small U-Net | 0.6135 | 0.3804 | 0.4956 | 0.3715 | 26.4444 | 481.8 | 689.4 |
| Large U-Net | 0.7624 | 0.3619 | 0.6150 | 0.5072 | 54.7778 | 431.9 | 494.0 |

MSBI beat Small U-Net on Dice, instance F1, PQ, count MAE, mean latency, and P95 latency, but its
boundary F1 was lower by `0.0031`. It beat Large U-Net on boundary F1, count MAE, and mean latency,
but not Dice, instance F1, PQ, or cold-start-dominated P95. The installed Stage J successor is now
runtime-selectable, while `ready_recommendation=false` remains in force. Stage J has not opened or
reused this consumed test set.

## Known limitations

- The final checkpoint and TorchScript are installed locally as private ignored artifacts and are
  not intended for public Git redistribution.
- The nine-view independent test has now been consumed and cannot be reused as an unbiased tuning
  set for a successor candidate.
- Physical scale is not inferred from filenames or an unfrozen scale-bar calibration.
- Connected components of binary GT are pseudo-instances, not human touching-particle IDs.
- A new blind labeled holdout is required before claiming that a future revision generalizes better
  than every existing weight.
