import json

import pytest

from rag_graph_demo.api.app import application_error_handler
from rag_graph_demo.domain.exceptions import (
    ConfigurationError,
    DocumentNotFoundError,
    UnsupportedDocumentError,
    UploadTooLargeError,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "status_code"),
    [
        (DocumentNotFoundError("missing"), 404),
        (UnsupportedDocumentError("unsupported"), 415),
        (UploadTooLargeError("large"), 413),
        (ConfigurationError("not configured"), 503),
    ],
)
async def test_application_error_status_codes(exception, status_code) -> None:
    response = await application_error_handler(None, exception)

    assert response.status_code == status_code
    assert json.loads(response.body)["detail"] == str(exception)
