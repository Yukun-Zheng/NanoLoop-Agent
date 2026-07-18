"""Resource-lifecycle tests for descriptor-pinned file downloads."""

from pathlib import Path

import pytest
from starlette.background import BackgroundTask
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from app.api.routes.files import _PinnedStreamingResponse
from app.storage import open_pinned_managed_file


@pytest.mark.asyncio
async def test_client_disconnect_immediately_closes_pinned_descriptor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    (root / "artifact.bin").write_bytes(b"descriptor-pinned-download")
    pinned = open_pinned_managed_file(root, "artifact.bin", chunk_size=4)
    chunks = pinned.iter_chunks()
    response = _PinnedStreamingResponse(
        chunks,
        background=BackgroundTask(chunks.close),
    )

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
    }

    with pytest.raises(ClientDisconnect):
        await response(scope, receive, send)

    assert pinned.closed is True
