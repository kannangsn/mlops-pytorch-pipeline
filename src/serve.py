"""Flask app that serves predictions from a trained checkpoint.

Endpoints:
    GET  /health   -> 200 when the model is loaded, 503 otherwise
    POST /predict  -> multipart form with an "image" file, returns class
                      probabilities for the 10 CIFAR-10 classes
"""

import io
import logging
import os

import torch
from flask import Flask, jsonify, request
from PIL import Image

from dataset import CIFAR10_CLASSES, get_transforms
from model import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve")

DEFAULT_MODEL_PATH = "/app/checkpoints/classifier_v1.pt"


class ModelStore:
    """Holds the loaded model so /health can re-try loading lazily.

    The checkpoint may not exist yet when the container starts (e.g. the
    training job is still running), so failing to load at startup is not
    fatal - the readiness probe just keeps the pod out of rotation.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.transform = get_transforms(train=False)

    def try_load(self) -> bool:
        if self.model is not None:
            return True
        if not os.path.exists(self.model_path):
            return False
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            model = get_model(
                architecture=checkpoint.get("architecture", "resnet18"),
                num_classes=checkpoint.get("num_classes", 10),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.model = model
            logger.info("Loaded model from %s (epoch %s, val_acc %.4f)",
                        self.model_path,
                        checkpoint.get("epoch", "?"),
                        checkpoint.get("val_accuracy", float("nan")))
            return True
        except Exception:
            logger.exception("Failed to load checkpoint from %s", self.model_path)
            return False

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB").resize((32, 32))
        batch = self.transform(image).unsqueeze(0)
        probs = torch.softmax(self.model(batch), dim=1).squeeze(0)
        top = int(probs.argmax())
        return {
            "predicted_class": CIFAR10_CLASSES[top],
            "confidence": round(float(probs[top]), 4),
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(CIFAR10_CLASSES, probs)
            },
        }


def create_app(model_path: str | None = None) -> Flask:
    app = Flask(__name__)
    store = ModelStore(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    store.try_load()
    app.config["MODEL_STORE"] = store

    @app.get("/health")
    def health():
        if store.try_load():
            return jsonify({"status": "ok", "model_path": store.model_path}), 200
        return jsonify({"status": "model_not_loaded", "model_path": store.model_path}), 503

    @app.post("/predict")
    def predict():
        if not store.try_load():
            return jsonify({"error": "model not loaded"}), 503
        if "image" not in request.files:
            return jsonify({"error": "send an image file in the 'image' form field"}), 400
        try:
            image = Image.open(io.BytesIO(request.files["image"].read()))
        except Exception:
            return jsonify({"error": "could not decode the uploaded file as an image"}), 400
        return jsonify(store.predict(image)), 200

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
