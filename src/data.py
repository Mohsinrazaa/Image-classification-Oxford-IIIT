from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: List[str]
    class_counts: Dict[int, int]
    class_weights: torch.Tensor


def _build_transforms(image_size: int, augment: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    train_ops = [
        transforms.Resize((image_size, image_size)),
    ]
    if augment:
        train_ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_ops = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(train_ops), eval_ops


def _split_indices(total_size: int, train_split: float, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(total_size)
    rng.shuffle(indices)

    train_end = int(total_size * train_split)
    val_end = train_end + int(total_size * val_split)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def _build_subset(dataset: datasets.OxfordIIITPet, indices: np.ndarray) -> Subset:
    return Subset(dataset, indices.tolist())


def _extract_labels(dataset: datasets.OxfordIIITPet, indices: np.ndarray) -> List[int]:
    labels = [dataset._labels[idx] for idx in indices]  # pylint: disable=protected-access
    return labels


def create_dataloaders(config: dict, seed: int) -> DatasetBundle:
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    train_split = data_cfg["train_split"]
    val_split = data_cfg["val_split"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    augment = data_cfg["augment"]

    data_root = Path(data_cfg["data_dir"])
    train_tfms, eval_tfms = _build_transforms(image_size=image_size, augment=augment)

    raw_ds = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        download=True,
        transform=train_tfms,
    )

    class_names = list(raw_ds.classes)
    train_idx, val_idx, test_idx = _split_indices(
        total_size=len(raw_ds),
        train_split=train_split,
        val_split=val_split,
        seed=seed,
    )

    train_ds = _build_subset(raw_ds, train_idx)
    # Re-load for val/test to keep transforms independent.
    raw_val = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        download=True,
        transform=eval_tfms,
    )
    raw_test = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        download=True,
        transform=eval_tfms,
    )
    val_ds = _build_subset(raw_val, val_idx)
    test_ds = _build_subset(raw_test, test_idx)

    train_labels = _extract_labels(raw_ds, train_idx)
    class_counts = dict(Counter(train_labels))
    max_count = max(class_counts.values())
    label_offset = min(class_counts.keys())
    weights = []
    for class_id in range(len(class_names)):
        count = class_counts.get(class_id + label_offset, 1)
        weights.append(max_count / count)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        class_counts=class_counts,
        class_weights=class_weights,
    )
