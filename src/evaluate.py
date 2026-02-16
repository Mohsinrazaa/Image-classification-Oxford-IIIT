from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import create_dataloaders
from src.models import BaselineCNN, build_transfer_model
from src.utils import get_device, load_config, resolve_output_dir, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, choices=["baseline", "strong"], default="strong")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    return parser.parse_args()


def build_model(model_type: str, num_classes: int, config: Dict):
    if model_type == "baseline":
        return BaselineCNN(num_classes=num_classes)
    return build_transfer_model(
        num_classes=num_classes,
        freeze_backbone=bool(config["models"]["strong"]["freeze_backbone"]),
    )


def load_checkpoint(model, ckpt_path: Path, device: torch.device) -> Dict:
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload


def collect_predictions(model, loader, device: torch.device) -> Tuple[List[int], List[int]]:
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy().tolist())
    return all_targets, all_preds


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    device = get_device()

    bundle = create_dataloaders(config=config, seed=int(config["project"]["seed"]))
    model = build_model(args.model, num_classes=len(bundle.class_names), config=config).to(device)

    output_dir = resolve_output_dir(config)
    ckpt = Path(args.checkpoint) if args.checkpoint else output_dir / "models" / f"{args.model}_best.pt"
    payload = load_checkpoint(model=model, ckpt_path=ckpt, device=device)
    class_names = payload.get("class_names", bundle.class_names)

    y_true, y_pred = collect_predictions(model, bundle.test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {"accuracy": acc, "macro_f1": macro_f1, "classification_report": report}
    metrics_path = metrics_dir / f"{args.model}_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)
    cm_path = plots_dir / f"{args.model}_confusion_matrix.png"
    save_confusion_matrix(cm=cm, class_names=class_names, out_path=cm_path)

    LOGGER.info("Test accuracy: %.4f | macro_f1: %.4f", acc, macro_f1)
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved confusion matrix to %s", cm_path)


if __name__ == "__main__":
    main()
