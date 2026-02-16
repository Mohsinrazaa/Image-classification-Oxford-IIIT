from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms

from src.models import BaselineCNN, build_transfer_model
from src.utils import get_device


def build_model(model_type: str, num_classes: int, freeze_backbone: bool):
    if model_type == "baseline":
        return BaselineCNN(num_classes=num_classes)
    return build_transfer_model(num_classes=num_classes, freeze_backbone=freeze_backbone)


def get_inference_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class InferenceEngine:
    def __init__(self, checkpoint_path: str, image_size: int = 224, top_k: int = 3):
        self.device = get_device()
        self.top_k = top_k
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_type = ckpt["model_type"]
        class_names = ckpt["class_names"]
        cfg = ckpt.get("config", {})
        freeze_backbone = bool(cfg.get("models", {}).get("strong", {}).get("freeze_backbone", False))
        self.model = build_model(model_type, num_classes=len(class_names), freeze_backbone=freeze_backbone).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.class_names = class_names
        self.transform = get_inference_transform(image_size=image_size)

    def predict(self, image_path: str) -> Dict:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")

        image = Image.open(path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            values, indices = torch.topk(probs, k=min(self.top_k, len(self.class_names)), dim=1)

        top_predictions: List[Dict] = []
        for score, idx in zip(values[0], indices[0]):
            class_idx = int(idx.item())
            top_predictions.append(
                {"class_id": class_idx, "class_name": self.class_names[class_idx], "confidence": round(float(score.item()), 6)}
            )

        return {"image_path": str(path), "predictions": top_predictions}
