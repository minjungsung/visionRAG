"""Health endpoint tests."""
import pytest


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_method_not_allowed(client):
    resp = await client.post("/health")
    assert resp.status_code == 405


@pytest.mark.asyncio
async def test_health_no_auth_required(client):
    """Health endpoint should be publicly accessible without any auth headers."""
    resp = await client.get("/health")
    assert resp.status_code == 200
