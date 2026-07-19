"""YAML configuration loader for MegaSlide-DiT."""

import yaml
from pathlib import Path
from typing import Union

from .config import MegaSlideConfig


def load_megaslide_config(path: Union[str, Path]) -> MegaSlideConfig:
    """Load MegaSlideConfig from YAML file.

    The YAML file should have nested sections for organization:
        model:
          frames: 256
          hidden_size: 8192
          ...
        dataset:
          path: "/path/to/data"
          ...
        training:
          batch_size: 1
          num_steps: 5000
          ...
        optimizer:
          beta1: 0.9
          ...
        memory:
          checkpoint_interval: 4
          ...
        logging:
          log_interval: 10
          ...

    These sections are flattened and merged into a single MegaSlideConfig.

    Args:
        path: Path to YAML config file.

    Returns:
        MegaSlideConfig instance with merged settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
        TypeError: If config contains invalid types.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Flatten nested sections
    flat = {}
    valid_sections = ["model", "dataset", "training", "optimizer", "memory", "logging"]

    for section in valid_sections:
        if section in data and isinstance(data[section], dict):
            flat.update(data[section])

    # Handle special type conversions
    if "dsa_kernel_size" in flat:
        # Convert list to tuple
        if isinstance(flat["dsa_kernel_size"], list):
            flat["dsa_kernel_size"] = tuple(flat["dsa_kernel_size"])

    # Convert device to int if it's numeric string
    if "device" in flat and isinstance(flat["device"], str):
        if flat["device"].isdigit():
            flat["device"] = int(flat["device"])
        # else keep as string (e.g., "cpu")

    try:
        config = MegaSlideConfig(**flat)
    except TypeError as e:
        raise TypeError(f"Invalid config in {path}: {e}") from e

    return config
