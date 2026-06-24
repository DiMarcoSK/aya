"""YAML-backed configuration loading for training hyperparameters.

A YAML file under `configs/` is the single source of truth for "what we
normally run" (mirrors how most current LLM fine-tuning repos — axolotl,
llama-factory, etc. — separate config from code). CLI flags in
`training/training.py` still work and take precedence over the YAML when
explicitly passed, so quick one-off experiments don't require editing or
duplicating a YAML file.
"""
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "lora_default.yaml"


def load_yaml_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
