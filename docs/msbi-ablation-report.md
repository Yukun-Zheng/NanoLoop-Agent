# NanoLoop-MSBI ablation report

> Historical A–F report. Stage I superseded this candidate on 2026-07-25, passed the frozen strict
> validation-superiority gate, and then failed to show universal superiority over both U-Nets on
> the one-shot independent test. Stage J is the current faster validation-successor candidate, but
> still lacks a fresh blind holdout. The current metrics and decision are in
> `model_artifacts/model_cards/msbi-instance-balanced-v1.md`.

## Status

`FORMAL_SINGLE_SEED_COMPLETE, SCIENTIFIC_GATE_FAILED`. Stages A–E ran sequentially on one RTX 4090
with seed 2026, exact checkpoint lineage, BF16 AMP, EMA, warm-up/cosine scheduling and early
stopping. Corrective Stage F then ran from the Stage E checkpoint. Only train and validation data
were present on the CUDA host; independent-test data remained sealed.

All full-view measurements below use the same 18-view validation split, bottom-128 validity mask,
production Adapter path and connected-component pseudo-instance definition.

## Staged evidence

| Stage | Increment over prior stage | Best patch validation loss | Full-view status |
|---|---|---:|---|
| A | single expert, frozen ConvNeXt encoder | 1.651945 | trained |
| B | small/large experts with fixed mean fusion | 0.803364 | trained |
| C | learned pixel-wise softmax gate | 0.790850 | trained |
| D | unfreeze encoder and add teacher distillation | 0.731599 | evaluated |
| E | add signed-distance supervision | 0.731320 | evaluated |
| F | `512×512`, morphology balance, contour consistency, boundary emphasis | 1.629448¹ | evaluated and calibrated |

¹ Stage F adds a weighted contour term and therefore its total loss is not numerically comparable
to A–E.

| Candidate | Pixel Dice | Boundary F1 | Pseudo-instance F1 | Count MAE | Runtime ms |
|---|---:|---:|---:|---:|---:|
| Large U-Net baseline | 0.6664 | 0.3874 | 0.5996 | 17.1111 | 1562.9 |
| Small U-Net baseline | 0.7465 | 0.5817 | 0.7184 | 7.2222 | 2348.7 |
| Stage D | 0.7352 | 0.4557 | 0.7185 | 10.3889 | 4284.6 |
| Stage E | 0.7435 | 0.4601 | 0.7218 | 10.2222 | 4808.1 |
| Stage F, prior decoder | 0.7636 | 0.4855 | 0.7461 | 9.2778 | 4367.2 |
| Stage F, frozen calibrated decoder | 0.7625 | 0.4882 | 0.7674 | 7.7222 | 4208.6 |

Stage F is the strongest MSBI candidate. Relative to Stage E, its calibrated form improves
pseudo-instance F1 by 0.0456, count MAE by 2.5, boundary F1 by 0.0281, and Dice by 0.0191 while
reducing mean runtime by about 600 ms. It also beats the Small U-Net on Dice and pseudo-instance F1.
It does not beat the frozen count and boundary tolerances, so those gains are not sufficient for
scientific acceptance.

## Decoder calibration

Two Stage E searches and one final Stage F search used validation data only. The Stage F search
evaluated 168 combinations over foreground threshold, center threshold, center NMS radius and
minimum area while keeping the remaining watershed contract fixed. The frozen selection was:

- foreground threshold 0.85;
- center threshold 0.50;
- center NMS radius 13;
- minimum area 256 px.

The selection rule first required all-rule passage, then maximized minimum relative frozen-policy
margin, number of passed scientific rules, instance F1, count error, boundary F1, Dice, and
proximity to the prior decoder. No combination passed all four scientific rules. The selected
combination passed two: pseudo-instance F1 and pixel Dice.

## Frozen gate

| Rule | Threshold | Stage F calibrated | Result |
|---|---:|---:|---|
| pseudo-instance F1@0.5 | ≥ 0.7384 | 0.7674 | pass |
| count MAE | ≤ 6.8611 | 7.7222 | fail |
| pixel Dice | ≥ 0.7265 | 0.7625 | pass |
| boundary F1 | ≥ 0.5617 | 0.4882 | fail |
| mean runtime | ≤ 8206.35 ms | 4208.61 ms | pass |
| mean small gate | ≥ 0.10 | 0.6401 | pass |
| mean large gate | ≥ 0.10 | 0.3599 | pass |

Gate status: `FAILED`. Independent test: `SEALED_NOT_ACCESSED`.

## Complexity and interpretation

MSBI contains 28,623,080 parameters. Hook-based Conv2d/Linear accounting gives 7.654 GMAC
(approximately 15.3 GFLOP under a multiply-plus-add convention) for one `256×256` patch;
interpolation, normalization, activations and watershed are excluded. Stage F uses `512×512`
patches, so its dense convolutional work scales to roughly four times the 256-patch figure before
full-image overlap and postprocessing.

The learned routing did not collapse: the final validation-average weights were 0.640 small and
0.360 large. The remaining errors are concentrated in difficult large/agglomerated views, where
particle adhesion and weak semantic boundaries produce both merges and over-segmentation. Binary
semantic masks supply only connected-component pseudo-instances, so this dataset cannot prove
touching-particle separation against human instance IDs.

Machine-readable training, export, calibration, validation and gate evidence is retained under
`artifacts/msbi/formal-20260724/`. No independent-test pixels were used.
