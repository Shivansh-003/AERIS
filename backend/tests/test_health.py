import pytest
from backend.app.main import app
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health_check_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["project"] == "AERIS"
    assert data["version"] == "1.0.0"
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0
    assert "device" in data
    assert isinstance(data["models_loaded"], list)
    assert data["dataset_version"] == "city_day_v1.0"


@pytest.mark.asyncio
async def test_invalid_endpoint_rfc7807_error():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/non-existent-endpoint")

    assert response.status_code == 404
    data = response.json()
    assert "type" in data
    assert "title" in data
    assert data["status"] == 404
    assert data["instance"] == "/api/v1/non-existent-endpoint"
