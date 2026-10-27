"""Load config/pipeline.yaml, overlaid with config/pipeline.local.yaml if it exists.

pipeline.yaml is committed and holds no account-specific values. pipeline.local.yaml
(gitignored; copy pipeline.local.example.yaml) holds your MLflow app ARN, the EULA
acceptance and notification email, and any other per-deployment overrides. Nested keys are
merged, so the local file only needs the keys it changes.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.local.yaml"


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path=DEFAULT_CONFIG_PATH, local_path=LOCAL_CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if local_path and Path(local_path).exists():
        with open(local_path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg
