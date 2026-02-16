from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data import DatasetBundle, create_dataloaders
from src.models import BaselineCNN, build_transfer_model
from src.utils import get_device, load_config, resolve_output_dir, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CV models for assessment")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, choices=["baseline", "strong"], default="baseline")
    return parser.parse_args()


def evaluate_model(model: nn.Module, loader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    all_preds = []
    all_targets = []
    loss_list = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            loss_list.append(loss.item())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    loss_mean = float(np.mean(loss_list)) if loss_list else 0.0
    return loss_mean, acc, macro_f1


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = []
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    return float(np.mean(running_loss)) if running_loss else 0.0


def build_model(model_type: str, config: dict, num_classes: int) -> nn.Module:
    if model_type == "baseline":
        return BaselineCNN(num_classes=num_classes)
    strong_cfg = config["models"]["strong"]
    return build_transfer_model(
        num_classes=num_classes,
        freeze_backbone=bool(strong_cfg["freeze_backbone"]),
    )


def run_training(model_type: str, config: Dict) -> Path:
    seed = int(config["project"]["seed"])
    set_seed(seed)
    device = get_device()
    LOGGER.info("Using device: %s", device)

    bundle: DatasetBundle = create_dataloaders(config=config, seed=seed)
    num_classes = len(bundle.class_names)
    model = build_model(model_type=model_type, config=config, num_classes=num_classes).to(device)

    train_cfg = config["train"]
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    use_class_weights = bool(train_cfg["use_class_weights"])

    class_weights = bundle.class_weights.to(device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    output_dir = resolve_output_dir(config)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=bundle.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_loss, val_acc, val_f1 = evaluate_model(model, bundle.val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        LOGGER.info("Epoch %s | train_loss %.4f | val_loss %.4f | val_acc %.4f | val_f1 %.4f", epoch, train_loss, val_loss, val_acc, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            best_path = model_dir / f"{model_type}_best.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": bundle.class_names,
                    "config": config,
                    "model_type": model_type,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("Early stopping triggered at epoch %s", epoch)
                break

    history_path = metrics_dir / f"{model_type}_history.json"
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    LOGGER.info("Saved training history to %s", history_path)
    return model_dir / f"{model_type}_best.pt"


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    best_model_path = run_training(model_type=args.model, config=cfg)
    LOGGER.info("Best model checkpoint: %s", best_model_path)
