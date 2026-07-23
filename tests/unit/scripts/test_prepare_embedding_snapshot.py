from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scripts import prepare_embedding_snapshot as snapshot


class _Vector:
    shape = (1, 512)


class _Model:
    max_seq_length = 512

    def __init__(self, model_path: str, calls: dict[str, Any], **kwargs: object) -> None:
        calls["model_path"] = model_path
        calls["model_kwargs"] = kwargs

    def encode(self, texts: list[str], **kwargs: object) -> _Vector:
        return _Vector()


def _install_fake_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    download_hook: Callable[[Path], None] | None = None,
    model_error: Exception | None = None,
) -> dict[str, Any]:
    calls: dict[str, Any] = {}
    hub = ModuleType("huggingface_hub")
    sentence_transformers = ModuleType("sentence_transformers")

    def snapshot_download(**kwargs: object) -> str:
        calls["download_kwargs"] = kwargs
        target = Path(str(kwargs["local_dir"]))
        calls["staged_snapshot"] = target
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.bin").write_bytes(b"new verified model")
        if download_hook is not None:
            download_hook(target)
        return str(target)

    def model_factory(model_path: str, **kwargs: object) -> _Model:
        if model_error is not None:
            raise model_error
        return _Model(model_path, calls, **kwargs)

    hub.__dict__["snapshot_download"] = snapshot_download
    sentence_transformers.__dict__["SentenceTransformer"] = model_factory
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    return calls


def _arguments(
    output: Path,
    manifest: Path,
    *,
    force: bool = False,
) -> list[str]:
    values = [
        "--output-dir",
        str(output),
        "--manifest",
        str(manifest),
        "--device",
        "cpu",
        "--verified-by",
        "Reviewer",
        "--verified-at",
        "2026-07-23",
    ]
    if force:
        values.append("--force")
    return values


def _transaction_artifacts(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.iterdir()
            if ".staging-" in path.name or ".backup-" in path.name
        ),
        key=lambda path: path.name,
    )


def test_prepares_in_sibling_staging_and_promotes_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "embedding-snapshot"
    manifest = tmp_path / "model-manifest.json"
    calls = _install_fake_dependencies(monkeypatch)

    result = snapshot.main(_arguments(output, manifest))

    assert result == 0
    assert (output / "model.bin").read_bytes() == b"new verified model"
    staged = calls["staged_snapshot"]
    assert isinstance(staged, Path)
    assert staged.parent == output.parent
    assert staged.name.startswith(f".{output.name}.staging-")
    assert calls["model_path"] == str(staged)
    assert calls["model_kwargs"] == {"device": "cpu", "local_files_only": True}
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert recorded["local_dir"] == str(output.resolve())
    assert recorded["tree_sha256"] == snapshot.directory_tree_sha256(output)
    assert recorded["size_bytes"] == (output / "model.bin").stat().st_size
    assert _transaction_artifacts(tmp_path) == []


def test_fingerprint_ignores_mutable_hugging_face_cache_metadata(
    tmp_path: Path,
) -> None:
    embedding = tmp_path / "embedding"
    embedding.mkdir()
    (embedding / "model.bin").write_bytes(b"stable model weights")
    metadata = embedding / ".cache" / "huggingface" / "download" / "model.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("etag\ncommit\n1784810048.001\n", encoding="utf-8")

    before = snapshot.directory_tree_sha256(embedding)
    size_before = snapshot.total_size(embedding)
    metadata.write_text("etag\ncommit\n1984810048.999\n", encoding="utf-8")

    assert snapshot.directory_tree_sha256(embedding) == before
    assert snapshot.total_size(embedding) == size_before


def test_force_preserves_previous_snapshot_when_staged_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "embedding-snapshot"
    output.mkdir()
    (output / "old.bin").write_bytes(b"previous good model")
    manifest = tmp_path / "model-manifest.json"
    old_manifest = b'{"status":"previous-good"}\n'
    manifest.write_bytes(old_manifest)

    def assert_old_snapshot_is_still_live(staged: Path) -> None:
        assert staged != output
        assert (output / "old.bin").read_bytes() == b"previous good model"

    _install_fake_dependencies(
        monkeypatch,
        download_hook=assert_old_snapshot_is_still_live,
        model_error=RuntimeError("staged smoke test failed"),
    )

    result = snapshot.main(_arguments(output, manifest, force=True))

    assert result == 1
    assert sorted(path.name for path in output.iterdir()) == ["old.bin"]
    assert (output / "old.bin").read_bytes() == b"previous good model"
    assert manifest.read_bytes() == old_manifest
    assert _transaction_artifacts(tmp_path) == []


def test_force_commits_new_snapshot_only_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "embedding-snapshot"
    output.mkdir()
    (output / "old.bin").write_bytes(b"previous good model")
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text('{"status":"previous-good"}\n', encoding="utf-8")
    _install_fake_dependencies(monkeypatch)

    result = snapshot.main(_arguments(output, manifest, force=True))

    assert result == 0
    assert sorted(path.name for path in output.iterdir()) == ["model.bin"]
    assert (output / "model.bin").read_bytes() == b"new verified model"
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert recorded["status"] == "verified"
    assert recorded["tree_sha256"] == snapshot.directory_tree_sha256(output)
    assert _transaction_artifacts(tmp_path) == []


def test_manifest_promotion_failure_rolls_back_snapshot_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "embedding-snapshot"
    output.mkdir()
    (output / "old.bin").write_bytes(b"previous good model")
    manifest = tmp_path / "model-manifest.json"
    old_manifest = b'{"status":"previous-good"}\n'
    manifest.write_bytes(old_manifest)
    _install_fake_dependencies(monkeypatch)

    real_replace = os.replace
    failure_injected = False

    def fail_new_manifest_promotion(source: str | Path, destination: str | Path) -> None:
        nonlocal failure_injected
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not failure_injected
            and destination_path == manifest
            and source_path.name.startswith(f".{manifest.name}.staging-")
        ):
            failure_injected = True
            raise OSError("simulated manifest promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_new_manifest_promotion)

    result = snapshot.main(_arguments(output, manifest, force=True))

    assert result == 1
    assert failure_injected is True
    assert sorted(path.name for path in output.iterdir()) == ["old.bin"]
    assert (output / "old.bin").read_bytes() == b"previous good model"
    assert manifest.read_bytes() == old_manifest
    assert _transaction_artifacts(tmp_path) == []


def test_incomplete_rollback_retains_named_recovery_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "embedding-snapshot"
    output.mkdir()
    (output / "old.bin").write_bytes(b"previous good model")
    manifest = tmp_path / "model-manifest.json"
    old_manifest = b'{"status":"previous-good"}\n'
    manifest.write_bytes(old_manifest)
    _install_fake_dependencies(monkeypatch)

    real_replace = os.replace
    manifest_failure_injected = False
    restore_failure_injected = False

    def fail_promotion_and_snapshot_restore(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        nonlocal manifest_failure_injected, restore_failure_injected
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not manifest_failure_injected
            and destination_path == manifest
            and source_path.name.startswith(f".{manifest.name}.staging-")
        ):
            manifest_failure_injected = True
            raise OSError("simulated manifest promotion failure")
        if (
            manifest_failure_injected
            and not restore_failure_injected
            and destination_path == output
            and source_path.name.startswith(f".{output.name}.backup-")
        ):
            restore_failure_injected = True
            raise OSError("simulated snapshot restore failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_promotion_and_snapshot_restore)

    result = snapshot.main(_arguments(output, manifest, force=True))

    assert result == 1
    assert manifest_failure_injected is True
    assert restore_failure_injected is True
    assert manifest.read_bytes() == old_manifest
    old_backups = list(tmp_path.glob(f".{output.name}.backup-*"))
    assert len(old_backups) == 1
    assert (old_backups[0] / "old.bin").read_bytes() == b"previous good model"
    retained_new = list(tmp_path.glob(f".{output.name}.staging-*"))
    assert len(retained_new) == 1
    assert (retained_new[0] / "model.bin").read_bytes() == b"new verified model"
    stderr = capsys.readouterr().err
    assert "SnapshotRecoveryRequiredError" in stderr
    assert str(old_backups[0]) in stderr
    assert str(retained_new[0]) in stderr


@pytest.mark.parametrize(
    "dangerous",
    [
        Path("/"),
        Path.home(),
        Path.cwd(),
        Path(tempfile.gettempdir()),
    ],
)
def test_rejects_broad_or_dangerous_output_targets(
    tmp_path: Path,
    dangerous: Path,
) -> None:
    with pytest.raises(ValueError, match="refusing"):
        snapshot._validate_targets(
            dangerous.resolve(),
            tmp_path / "model-manifest.json",
        )


def test_rejects_manifest_inside_snapshot_tree(tmp_path: Path) -> None:
    output = tmp_path / "embedding-snapshot"

    with pytest.raises(ValueError, match="manifest must be outside"):
        snapshot._validate_targets(output, output / "model-manifest.json")


def test_existing_assets_require_force_before_dependencies_are_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "embedding-snapshot"
    output.mkdir()
    (output / "old.bin").write_bytes(b"previous good model")
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text('{"status":"previous-good"}\n', encoding="utf-8")
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    result = snapshot.main(_arguments(output, manifest))

    assert result == 1
    assert (output / "old.bin").read_bytes() == b"previous good model"
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "previous-good"
    assert _transaction_artifacts(tmp_path) == []
