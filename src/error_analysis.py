from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from src.data import create_dataloaders
from src.models import BaselineCNN, build_transfer_model
from src.utils import get_device, load_config, resolve_output_dir, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("error_analysis")


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize misclassifications")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, choices=["baseline", "strong"], default="strong")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=24)
    return parser.parse_args()


def build_model(model_type: str, num_classes: int, config: Dict):
    if model_type == "baseline":
        return BaselineCNN(num_classes=num_classes)
    return build_transfer_model(
        num_classes=num_classes,
        freeze_backbone=bool(config["models"]["strong"]["freeze_backbone"]),
    )


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    image_tensor = image_tensor.cpu()
    image_tensor = image_tensor * STD + MEAN
    return torch.clamp(image_tensor, 0.0, 1.0)


def collect_misclassifications(model, loader, device: torch.device, max_samples: int) -> List[Tuple[torch.Tensor, int, int]]:
    mistakes = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            wrong_mask = preds != labels
            wrong_indices = torch.where(wrong_mask)[0]
            for idx in wrong_indices:
                mistakes.append((images[idx].detach(), int(labels[idx].item()), int(preds[idx].item())))
                if len(mistakes) >= max_samples:
                    return mistakes
    return mistakes


def save_grid(mistakes: List[Tuple[torch.Tensor, int, int]], class_names: List[str], out_path: Path) -> None:
    cols = 4
    rows = max(1, (len(mistakes) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c] if rows > 1 else axes[c]
            if idx >= len(mistakes):
                ax.axis("off")
                continue
            image, true_id, pred_id = mistakes[idx]
            vis = denormalize(image).permute(1, 2, 0).numpy()
            ax.imshow(vis)
            ax.set_title(f"T: {class_names[true_id]} | P: {class_names[pred_id]}", fontsize=9)
            ax.axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    device = get_device()

    bundle = create_dataloaders(cfg, seed=seed)
    model = build_model(args.model, num_classes=len(bundle.class_names), config=cfg).to(device)

    output_dir = resolve_output_dir(cfg)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else output_dir / "models" / f"{args.model}_best.pt"
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])

    mistakes = collect_misclassifications(model, bundle.test_loader, device, max_samples=args.num_samples)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / f"{args.model}_misclassified_samples.png"
    save_grid(mistakes, payload.get("class_names", bundle.class_names), save_path)
    LOGGER.info("Saved %s mistakes visualization to %s", len(mistakes), save_path)


if __name__ == "__main__":
    main()
