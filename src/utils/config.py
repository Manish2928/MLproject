"""
Configuration management for DementiaVoiceAI.

Loads YAML configs with dot-access and merge support.
All paths are resolved relative to the project root.

Usage:
    from src.utils.config import load_config
    cfg = load_config()                          # loads configs/default.yaml
    cfg = load_config("configs/custom.yaml")     # loads a custom config
    print(cfg.audio.target_sample_rate)           # dot-access
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Walks up from this file's location until it finds a directory containing
    'configs/' and 'src/' — the two canonical markers of the project root.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    # Fallback: two levels up from src/utils/config.py
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()


# ---------------------------------------------------------------------------
# DotDict — dict with attribute-style access
# ---------------------------------------------------------------------------

class DotDict(dict):
    """A dictionary subclass that supports dot-notation access.

    >>> d = DotDict({"a": {"b": 1}})
    >>> d.a.b
    1
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{key}'"
            ) from None
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{key}'"
            ) from None

    @classmethod
    def from_dict(cls, data: dict) -> "DotDict":
        """Recursively convert nested dicts to DotDicts."""
        result = cls()
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = cls.from_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    cls.from_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def to_dict(self) -> dict:
        """Recursively convert back to plain dict."""
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, DotDict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a deep copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> DotDict:
    """Load a YAML configuration file and return a DotDict.

    Parameters
    ----------
    config_path : str | Path | None
        Path to the YAML file. If relative, resolved against PROJECT_ROOT.
        Defaults to ``configs/default.yaml``.
    overrides : dict | None
        Optional dict of overrides that are deep-merged on top of the
        loaded config (useful for CLI argument overrides).

    Returns
    -------
    DotDict
        The configuration with dot-access support.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raw = {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    return DotDict.from_dict(raw)


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to PROJECT_ROOT.

    If the path is already absolute, return it unchanged.
    """
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
