import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import api.main as main

CLASS_NAMES = [f"class_{i}" for i in range(10)]


@pytest.fixture
def client(monkeypatch):
    """API client backed by an untrained model, so tests run without the trained weights."""
    torch.manual_seed(0)
    model = main.build_model(len(CLASS_NAMES)).eval()
    monkeypatch.setattr(main, "get_bundle", lambda: (model, CLASS_NAMES))
    return TestClient(main.app)


def image_bytes(fmt="PNG", size=(64, 64)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (30, 120, 60)).save(buffer, format=fmt)
    return buffer.getvalue()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model"] == main.MODEL_NAME


def test_predict_returns_top_three_classes(client):
    files = {"file": ("tile.png", image_bytes(), "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CLASS_NAMES
    assert len(body["top_3"]) == 3
    assert body["confidence"] == body["top_3"][0]["probability"]
    assert sum(item["probability"] for item in body["top_3"]) <= 1.0001


def test_predict_accepts_jpeg(client):
    files = {"file": ("tile.jpg", image_bytes("JPEG"), "image/jpeg")}
    assert client.post("/predict", files=files).status_code == 200


def test_predict_rejects_non_image_content_type(client):
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    assert client.post("/predict", files=files).status_code == 415


def test_predict_rejects_corrupt_image(client):
    files = {"file": ("tile.png", b"this is not an image", "image/png")}
    assert client.post("/predict", files=files).status_code == 400


def test_predict_rejects_empty_file(client):
    files = {"file": ("tile.png", b"", "image/png")}
    assert client.post("/predict", files=files).status_code == 400


def test_predict_requires_a_file(client):
    assert client.post("/predict").status_code == 422


def test_predict_returns_503_when_model_files_are_missing(monkeypatch):
    def missing():
        raise FileNotFoundError("Model files not found.")

    monkeypatch.setattr(main, "get_bundle", missing)
    files = {"file": ("tile.png", image_bytes(), "image/png")}
    response = TestClient(main.app).post("/predict", files=files)
    assert response.status_code == 503
