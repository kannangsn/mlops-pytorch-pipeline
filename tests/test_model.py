import torch
import yaml

from model import SimpleCNN, get_model


def test_simple_cnn_output_shape():
    model = get_model("simple_cnn", num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_resnet18_output_shape():
    model = get_model("resnet18", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_unknown_architecture_raises():
    try:
        get_model("vgg99")
    except ValueError as e:
        assert "vgg99" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown architecture")


def test_softmax_gives_valid_probabilities():
    model = get_model("simple_cnn")
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.randn(1, 3, 32, 32)), dim=1)
    assert torch.all(probs >= 0)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_checkpoint_roundtrip(tmp_path):
    """Saving and reloading a checkpoint must reproduce identical outputs."""
    model = SimpleCNN(num_classes=10)
    model.eval()
    ckpt_path = tmp_path / "model.pt"
    torch.save({
        "architecture": "simple_cnn",
        "num_classes": 10,
        "model_state_dict": model.state_dict(),
    }, ckpt_path)

    restored = get_model("simple_cnn", num_classes=10)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x))


def test_training_config_is_complete():
    with open("configs/training_config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"]["num_classes"] == 10
    assert cfg["training"]["epochs"] > 0
    assert cfg["training"]["early_stopping_patience"] > 0
    assert cfg["output"]["model_name"].endswith(".pt")
