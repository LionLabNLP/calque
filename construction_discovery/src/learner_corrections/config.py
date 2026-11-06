
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ALLOWED_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("French", "English"),
        ("German", "English"),
        ("Japanese", "English"),
        ("Mandarin", "English"),
        ("Portuguese(Brazil)", "English"),
        ("Swedish", "English"),
    }
)

LANGUAGE_ABBR: dict[str, str] = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Mandarin": "zh",
    "Portuguese(Brazil)": "pt",
    "Swedish": "sv",
}

LANGUAGE_CODE_ALIASES: dict[str, str] = {
    "se": "sv",
}

JAPANESE_LANGUAGES: frozenset[str] = frozenset({"Japanese"})


def normalize_language_code(code: str) -> str:
    return LANGUAGE_CODE_ALIASES.get(code, code)


def is_allowed_pair(learning_language: str, native_language: str) -> bool:
    return (learning_language, native_language) in ALLOWED_PAIRS


def language_abbr(language: str) -> str:
    try:
        return LANGUAGE_ABBR[language]
    except KeyError as exc:
        raise KeyError(f"No abbreviation configured for language {language!r}") from exc



DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
TOY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "toy.yaml"


def _deep_merge(base: dict, override: Mapping) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level")
    return data


def load_config(
    extra_config_paths: Iterable[Path] = (),
    *,
    toy: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    config = load_yaml(DEFAULT_CONFIG_PATH)
    if toy and TOY_CONFIG_PATH.exists():
        config = _deep_merge(config, load_yaml(TOY_CONFIG_PATH))
    for path in extra_config_paths:
        config = _deep_merge(config, load_yaml(Path(path)))
    if overrides:
        for dotted_key, value in overrides.items():
            if value is None:
                continue
            _set_dotted(config, dotted_key, value)
    return config


def _set_dotted(config: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def get_config_value(config: Mapping, dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node
