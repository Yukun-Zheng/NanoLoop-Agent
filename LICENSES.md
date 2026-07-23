# Licenses and provenance register

The NanoLoop Agent project itself does not yet declare a public license. Do not redistribute this repository as open source until the team chooses and adds one.

## Installed direct dependencies

This development-environment snapshot was read from installed package metadata on 2026-07-18. It is an inventory aid, not a substitute for the license texts and notices that must accompany a release.

| Package | Observed version | Declared license family |
| --- | ---: | --- |
| Alembic | 1.18.5 | MIT |
| FastAPI | 0.139.2 | MIT |
| HTTPX | 0.28.1 | BSD-3-Clause |
| NumPy | 2.3.5 | BSD-3-Clause; wheel contains additional notices |
| Pillow | 12.3.0 | MIT-CMU |
| Pydantic / pydantic-settings | 2.13.4 / 2.14.2 | MIT |
| python-multipart | 0.0.32 | Apache-2.0 |
| PyYAML | 6.0.3 | MIT |
| scikit-image | 0.26.0 | BSD-3-Clause |
| SciPy | 1.18.0 | BSD-3-Clause; wheel contains additional notices |
| SQLAlchemy | 2.0.51 | MIT |
| Uvicorn | 0.51.0 | BSD-3-Clause |
| Streamlit | 1.59.2 | Apache-2.0 |
| Matplotlib | 3.11.0 | PSF-based license |
| pandas | 2.3.3 | BSD-3-Clause |

Exact versions in a release are determined by the built environment, not this table. Generate and archive a full transitive SBOM/license report from the release wheelhouse or image before distribution.

## Optional heavyweight dependencies

The default CPU image does not install `.[models]` or `.[rag]`. Before enabling or redistributing them, review at least:

- Ultralytics licensing (AGPL-3.0 or an applicable commercial license) and whether the intended deployment/distribution is compatible.
- PyMuPDF licensing (AGPL/commercial terms) for PDF extraction.
- PyTorch, torchvision, FAISS, sentence-transformers, the selected embedding model, and every transitive package/version.

Dependency availability does not grant rights to any model weights, datasets, papers, or images.

## External asset ledger requirement

Every model weight, model card, demo image, paper, knowledge document, embedding model, and generated index entering a release must record:

- origin URL or internal custodian;
- exact version and SHA-256;
- author/owner and license or written permission;
- allowed use, redistribution, and demo constraints;
- acquisition date and reviewer;
- target registry/document ID.

Placeholder directories and example JSON files are not licensed scientific assets. Never replace a missing external asset with fabricated bytes merely to make a health check green.

## Project model assets

| Asset | Identity | Current permission evidence |
| --- | --- | --- |
| `model_artifacts/weights/unet-large-optimized-v1.pt` | SHA-256 `007d9a16bf31e5f960160c52eefa938b83feeac2e6c0d7dec9c8670a38626e05`; delivered by project developer 郭境濠 on 2026-07-23 | No separate license, author/custody ledger, or written redistribution grant was included in `ModelAssets-large.zip`. It is tracked at the project owner's request for NanoLoop Agent integration. Do not infer permission for third-party redistribution, commercial use, or dataset/model sublicensing until the missing record is supplied and reviewed. |
