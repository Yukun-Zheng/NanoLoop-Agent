# Demo data and engineering fixtures

`model_artifacts/` is an explicit, non-scientific registry used by
`scripts/mvp_fixture_smoke.py` and integration tests. It contains a tiny marker
file, a pinned YAML output description, a model card, and the deterministic
fixture adapter reference needed to exercise the same immutable bundle path as
a real model.

This registry is intentionally separate from `model_artifacts/registry.yaml`:

- production/development startup keeps missing real checkpoints unavailable;
- the fixture must be selected explicitly by the smoke script or a test;
- every fixture run records schema-v3 bundle and execution provenance;
- generated masks are simulated and cannot support scientific or competition
  performance claims.

Run from the repository root:

```bash
python scripts/mvp_fixture_smoke.py
python scripts/mvp_fixture_smoke.py --state-dir ./tmp/fixture-mvp
```
