"""Test configuration and shared fixtures."""
import sys
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

pytest_plugins = ['pytest_asyncio']

# ---------------------------------------------------------------------------
# Pre-patch missing optional modules so that lazy imports in endpoint functions
# succeed. This is required because `tritonclient` (optional GPU dependency)
# and `unstructured` are not installed in CI/dev environments.
# ---------------------------------------------------------------------------

_MOCK_MODULES = [
    "tritonclient",
    "tritonclient.grpc",
    "unstructured",
    "unstructured.partition",
    "unstructured.partition.auto",
    "open_clip",
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


from src.api.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
