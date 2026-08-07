import io

import pytest
import torch
from PIL import Image

from model import SimpleCNN
from serve import create_app


@pytest.fixture
def checkpoint_path(tmp_path):
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
    app = create_app(model_path=checkpoint_path)
    app.testing = True
    return app.test_client()


def make_test_image() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 60, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok_when_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_health_503_when_checkpoint_missing(tmp_path):
    app = create_app(model_path=str(tmp_path / "does_not_exist.pt"))
    app.testing = True
    resp = app.test_client().get("/health")
    assert resp.status_code == 503


def test_predict_returns_probabilities(client):
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
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_predict_rejects_non_image(client):
    resp = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"this is not an image"), "junk.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
