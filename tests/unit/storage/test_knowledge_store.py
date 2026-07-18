from io import BytesIO
from pathlib import Path

import pytest

from app.core.errors import StorageError, UnsupportedMediaTypeError
from app.storage import KnowledgeSourceStore, UploadSizeExceededError


def test_knowledge_source_store_is_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    store = KnowledgeSourceStore(tmp_path / "sources", max_upload_bytes=1024)

    first = store.save(BytesIO(b"grounded evidence"), "paper.md")
    second = store.save(BytesIO(b"grounded evidence"), "renamed.md")

    assert first.created is True
    assert second.created is False
    assert first.path == second.path
    assert first.path.read_bytes() == b"grounded evidence"
    assert first.path.name == f"{first.sha256}.md"


def test_knowledge_source_store_rejects_type_size_and_tampering(tmp_path: Path) -> None:
    store = KnowledgeSourceStore(tmp_path / "sources", max_upload_bytes=4)

    with pytest.raises(UnsupportedMediaTypeError):
        store.save(BytesIO(b"data"), "archive.zip")
    with pytest.raises(UploadSizeExceededError):
        store.save(BytesIO(b"12345"), "large.txt")

    stored = store.save(BytesIO(b"data"), "note.txt")
    stored.path.write_bytes(b"evil")
    with pytest.raises(StorageError, match="校验失败"):
        store.save(BytesIO(b"data"), "note.txt")
