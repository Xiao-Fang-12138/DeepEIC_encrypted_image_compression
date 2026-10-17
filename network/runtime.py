"""Shared configuration, local paths and strict checkpoint loading."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from .deepelic import DeepELIC

ROOT = Path(__file__).resolve().parents[1]


def local_path(value):
    """Resolve a release-relative path; reject absolute paths and escaping links."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Use a path relative to the DeepELIC directory")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Paths must stay inside the DeepELIC directory")
    return resolved


def read_config(path):
    return json.loads(local_path(path).read_text())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    return torch.device(name)


def load_model(config, checkpoint=None, device="cpu"):
    model = DeepELIC(**config["model"])
    if checkpoint is not None:
        state = torch.load(local_path(checkpoint), map_location="cpu", weights_only=True)
        if state["model_config"] != config["model"]:
            raise ValueError("Checkpoint architecture does not match the selected configuration")
        model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device)
