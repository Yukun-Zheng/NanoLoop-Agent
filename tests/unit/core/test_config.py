import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_total_request_limit_must_cover_a_single_file_limit() -> None:
    with pytest.raises(ValidationError, match="MAX_REQUEST_MB"):
        Settings(max_upload_mb=10, max_request_mb=9)

    settings = Settings(max_upload_mb=10, max_request_mb=10)
    assert settings.max_request_mb == 10
