from __future__ import annotations

import pytest

from scripts.ingest_demo_knowledge import normalize_api_base, validate_rag_health


def test_hybrid_ingestion_requires_exposed_faiss_generation() -> None:
    component, generation = validate_rag_health(
        {
            "rag_index": {
                "status": "healthy",
                "detail": (
                    "retrieval=keyword and vector retrieval available; "
                    "vector=40 vectors, dimension=512, "
                    "generation=0123456789abcdef0123456789abcdef"
                ),
            }
        },
        allow_keyword_only=False,
    )

    assert component["status"] == "healthy"
    assert generation == "0123456789abcdef0123456789abcdef"


def test_hybrid_ingestion_rejects_degraded_or_generationless_health() -> None:
    with pytest.raises(RuntimeError, match="RAG health must"):
        validate_rag_health(
            {"rag_index": {"status": "degraded", "detail": "keyword only"}},
            allow_keyword_only=False,
        )
    with pytest.raises(RuntimeError, match="does not expose"):
        validate_rag_health(
            {"rag_index": {"status": "healthy", "detail": "generic healthy"}},
            allow_keyword_only=False,
        )


def test_keyword_only_health_requires_explicit_opt_in() -> None:
    component, generation = validate_rag_health(
        {"rag_index": {"status": "degraded", "detail": "keyword only"}},
        allow_keyword_only=True,
    )

    assert component["status"] == "degraded"
    assert generation is None


def test_demo_ingestion_rejects_plain_http_to_remote_hosts() -> None:
    assert normalize_api_base("http://localhost:8000") == "http://localhost:8000"
    with pytest.raises(ValueError, match="requires HTTPS"):
        normalize_api_base("http://example.test:8000")
