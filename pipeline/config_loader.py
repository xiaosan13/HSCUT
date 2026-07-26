"""Configuration loader for HSCUT pipeline.

Searches for paths.yaml (user config) or paths.yaml.example (template)
relative to the project root directory.
"""

import os
import yaml


def load_config(config_path=None):
    """Load YAML configuration, searching for paths.yaml or paths.yaml.example.

    Args:
        config_path: Optional explicit path to a YAML config file.

    Returns:
        dict: Parsed configuration.

    Raises:
        FileNotFoundError: If neither paths.yaml nor paths.yaml.example exists.
    """
    if config_path is not None:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # Search for project root: go up from this file until we find configs/
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        config_dir = os.path.join(current, 'configs')
        for name in ['paths.yaml', 'paths.yaml.example']:
            candidate = os.path.join(config_dir, name)
            if os.path.exists(candidate):
                with open(candidate, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        "No paths.yaml or paths.yaml.example found.\n"
        "Copy configs/paths.yaml.example to configs/paths.yaml and edit it "
        "with your local data paths."
    )
