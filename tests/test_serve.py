import io

import pytest
import torch
from PIL import Image

from model import SimpleCNN
from serve import create_app


@pytest.fixture
def checkpoint_path(tmp_path):
    """Write a small, real checkpoint file to disk for the serving tests to load."""
    model = SimpleCNN(num_classes=10)
    path = tmp_path / "classifier_v1.pt"
    torch.save({
        "epoch": 1,
        "architecture": "simple_cnn",
        "num_classes": 10,
        "model_state_dict": model.state_dict(),
        "val_loss": 1.0,
        "val_accuracy": 0.5,
    }, path)
    return str(path)


@pytest.fixture
def client(checkpoint_path):
    """A Flask test client wired to an app that already has a model loaded."""
    app = create_app(model_path=checkpoint_path)
    app.testing = True
    return app.test_client()


def make_test_image() -> bytes:
    """Generate a small in-memory PNG to use as a /predict payload."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 60, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok_when_model_loaded(client):
    """/health should return 200 once a checkpoint has been loaded."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_health_503_when_checkpoint_missing(tmp_path):
    """/health should return 503 while no checkpoint file exists yet."""
    app = create_app(model_path=str(tmp_path / "does_not_exist.pt"))
    app.testing = True
    resp = app.test_client().get("/health")
    assert resp.status_code == 503


def test_predict_returns_probabilities(client):
    """/predict should return a full 10-class probability distribution."""
    resp = client.post(
        "/predict",
        data={"image": (io.BytesIO(make_test_image()), "test.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert "predicted_class" in body
    assert len(body["probabilities"]) == 10
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_rejects_missing_file(client):
    """/predict should return 400 when no 'image' field is sent at all."""
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_rejects_non_image(client):
    """/predict should return 400 when the uploaded file isn't a decodable image."""
    resp = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"this is not an image"), "junk.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
