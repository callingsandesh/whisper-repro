"""Per-dataset configuration from configs/<name>.yaml.

Keeps machine-specific paths (where you unpacked SLR54, where your
recordings live) out of the code and out of git. YAML is optional -- if
PyYAML is absent or the file does not exist, adapters fall back to their
defaults and CLI flags.
"""

from __future__ import annotations

from typing import Any

from .logging_utils import get_logger
from .paths import CONFIG_DIR

log = get_logger(__name__)


def load_dataset_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        log.warning("%s exists but PyYAML is not installed -- ignoring it", path.name)
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    log.debug("loaded config %s: %s", path.name, sorted(data))
    return data
