from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_output_dir(config: Dict[str, Any]) -> Path:
    output_dir = config["project"]["output_dir"]
    return ensure_dir(output_dir)


def env_or_default(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)
