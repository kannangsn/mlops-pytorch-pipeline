"""Training entry point.

Reads hyperparameters from a YAML config (path taken from --config,
the CONFIG_PATH env var, or the default locations), trains the model,
logs metrics as JSON lines to stdout and saves the best checkpoint.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model

DEFAULT_CONFIG_PATHS = [
    "/app/configs/training_config.yaml",
    "configs/training_config.yaml",
]


def log(entry: dict) -> None:
    """Structured (JSON lines) logging to stdout."""
    print(json.dumps(entry), flush=True)


def resolve_config_path(cli_arg: str | None) -> Path:
    """Pick the config path: --config, then CONFIG_PATH env var, then defaults."""
    if cli_arg:
        return Path(cli_arg)
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        return Path(env_path)
    for candidate in DEFAULT_CONFIG_PATHS:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No training config found. Pass --config, set CONFIG_PATH, "
        f"or place a file at one of {DEFAULT_CONFIG_PATHS}"
    )


def load_config(path: Path) -> dict:
    """Parse the YAML config file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python's and PyTorch's RNGs for reproducible training runs."""
    random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """One pass over the loader. Trains if an optimizer is given."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    # Disable gradient tracking entirely during evaluation (training=False)
    # to save memory and compute.
    with torch.set_grad_enabled(training):
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Only step the optimizer when actually training.
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accumulate loss/accuracy stats for this epoch.
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

    return total_loss / total, correct / total


def main():
    """Load config, build model/data/optimizer, then run the training loop."""
    parser = argparse.ArgumentParser(description="Train the image classifier")
    parser.add_argument("--config", default=None, help="Path to training_config.yaml")
    args = parser.parse_args()

    # --- Config ---
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    log({"event": "config_loaded", "path": str(config_path)})

    # --- Reproducibility and device selection ---
    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log({"event": "device_selected", "device": str(device)})

    # --- Model ---
    model = get_model(
        architecture=cfg["model"]["architecture"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 2),
        dataset=cfg["data"].get("dataset", "cifar10"),
    )

    # --- Optimizer and loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Optional cap on batches per epoch, used for smoke tests / CI.
    max_batches = cfg["training"].get("max_batches")

    # --- Early stopping and checkpoint bookkeeping ---
    patience = cfg["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / cfg["output"]["model_name"]

    # --- Training loop ---
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, max_batches=max_batches,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, max_batches=max_batches,
        )

        # Structured (JSON-line) metric log for this epoch.
        log({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })

        # Save the checkpoint only when validation loss improves; otherwise
        # count toward early stopping and bail out once patience runs out.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "architecture": cfg["model"]["architecture"],
                "num_classes": cfg["model"]["num_classes"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
            log({"event": "checkpoint_saved", "path": str(save_path)})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log({"event": "early_stopping", "epoch": epoch})
                break

    log({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)})


if __name__ == "__main__":
    main()
