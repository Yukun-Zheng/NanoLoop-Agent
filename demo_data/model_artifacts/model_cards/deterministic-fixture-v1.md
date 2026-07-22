# Deterministic fixture v1

This is an engineering fixture, not a trained segmentation model. It renders
three versioned normalized ellipses and exists only to exercise the NanoLoop
API, database, immutable model bundle, task scheduler, storage, analysis, and
export path before an externally supplied checkpoint is available.

- Scientific validity: none; output must never be reported as model accuracy.
- Intended use: local development, contract tests, CI, and integration demos.
- Device: CPU only.
- Determinism: output is fully determined by the pinned adapter, YAML, marker
  file, request dimensions, ROI boxes, threshold, and minimum-area setting.
- Prohibited use: scientific evaluation, competition metrics, publications, or
  replacement of a real U-Net/YOLO-Seg/SAM2 handoff.
