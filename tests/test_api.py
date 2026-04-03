"""
Integration tests for the API endpoints.
Uses FastAPI's TestClient (httpx under the hood).
"""

import base64
import os

import pytest
from fastapi.testclient import TestClient

# Set test env vars BEFORE importing app
os.environ.setdefault("API_KEY", "test-api-key-123")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")

from app.main import app  # noqa: E402


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def api_headers():
    """Headers with valid API key."""
    return {"x-api-key": "test-api-key-123"}


# ── Health endpoints ─────────────────────────────────────────────────────────


class TestHealthEndpoints:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "supported_file_types" in data


# ── Authentication ───────────────────────────────────────────────────────────


class TestAuthentication:
    def test_missing_api_key(self, client):
        """Should return 401 when x-api-key header is missing."""
        payload = {
            "fileName": "test.pdf",
            "fileType": "pdf",
            "fileBase64": base64.b64encode(b"dummy").decode(),
        }
        resp = client.post("/document/analyze", json=payload)
        assert resp.status_code == 401

    def test_invalid_api_key(self, client):
        """Should return 401 with wrong API key."""
        payload = {
            "fileName": "test.pdf",
            "fileType": "pdf",
            "fileBase64": base64.b64encode(b"dummy").decode(),
        }
        resp = client.post(
            "/document/analyze",
            json=payload,
            headers={"x-api-key": "wrong-key"},
        )
        assert resp.status_code == 401


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidation:
    def test_unsupported_file_type(self, client, api_headers):
        """Should return 400 for unsupported file types."""
        payload = {
            "fileName": "test.xyz",
            "fileType": "xyz",
            "fileBase64": base64.b64encode(b"dummy content").decode(),
        }
        resp = client.post("/document/analyze", json=payload, headers=api_headers)
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_missing_fields(self, client, api_headers):
        """Should return 422 for missing required fields."""
        resp = client.post(
            "/document/analyze",
            json={"fileName": "test.pdf"},
            headers=api_headers,
        )
        assert resp.status_code == 422


# ── Response structure ───────────────────────────────────────────────────────


class TestResponseStructure:
    """
    These tests verify that the response JSON matches the contract.
    They require a valid GROQ_API_KEY to actually call the Groq API,
    so they are skipped in CI if the key is not set.
    """

    @pytest.mark.skipif(
        os.environ.get("GROQ_API_KEY", "").startswith("test-"),
        reason="Requires a real GROQ_API_KEY",
    )
    def test_response_has_all_fields(self, client, api_headers):
        """Verify the response structure has status, fileName, summary, entities, sentiment."""
        # Create a simple text-based "PDF" (not a real PDF — would fail extraction)
        # This test only verifies structure, not content accuracy.
        pass  # Placeholder — real integration tests need sample files
