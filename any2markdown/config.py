from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


_PACKAGE_DIR = Path(__file__).resolve().parent.parent

_SEARCH_PATHS = [
    Path("any2md.yaml"),
    _PACKAGE_DIR / "any2md.yaml",
    Path.home() / ".any2md.yaml",
]

PARSE_MODES = ("auto", "ocr", "direct")


@dataclass
class ModelConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    name: str = "xiaomi-ocr-0"
    temperature: float | None = None
    timeout: int = 60
    prompt: str | None = None


@dataclass
class RenderConfig:
    dpi: int = 200
    max_image_size: int = 4096


@dataclass
class PipelineConfig:
    concurrency: int = 4
    max_retries: int = 3
    # "html" (convert model OTSL tables → HTML) | "otsl" (keep as-is)
    table_format: str = "html"


@dataclass
class ParseConfig:
    # "auto" (direct parse when possible, else OCR) | "ocr" | "direct"
    mode: str = "auto"


@dataclass
class ServerConfig:
    """HTTP server only; the CLI ignores this section."""

    host: str = "0.0.0.0"
    port: int = 8088
    log_level: str = "INFO"
    url_timeout: float = 10.0
    max_upload_mb: int = 100
    # Processes for direct PDF parsing; None = min(CPUs, 4), 1 = in-process
    pdf_workers: int | None = None
    # Model profiles edited in the web UI (holds API keys)
    profiles_file: str = "data/model_profiles.json"
    # Image download for /convert/url with download_images=true
    media_dir: str = "/media"
    public_base_url: str | None = None
    image_max_mb: int = 5
    image_concurrency: int = 8
    image_timeout: float = 10.0
    # Seconds of silence before a streaming response sends an SSE keep-alive
    stream_heartbeat: float = 15.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# Environment variables that override the config file: ANY2MD_<NAME> → (section, key).
# Every field of every section is overridable as ANY2MD_<KEY>, except where
# the bare key would be ambiguous or misleading; those are renamed here.
_ENV_RENAMES = {
    ("model", "name"): "ANY2MD_MODEL",
    ("parse", "mode"): "ANY2MD_PARSE_MODE",
}


def load_config(config_path: str | Path | None = None) -> Config:
    """Load config from YAML, then apply ``ANY2MD_*`` environment overrides.

    Search order: explicit path > ``$ANY2MD_CONFIG`` > ./any2md.yaml >
    any2md.yaml next to the package (the repo root, /app in Docker) >
    ~/.any2md.yaml. Built-in defaults are used when no file is found.
    """
    config_path = config_path or os.environ.get("ANY2MD_CONFIG") or None
    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        cfg = _parse_config(p)
    else:
        cfg = next((_parse_config(p) for p in _SEARCH_PATHS if p.exists()), Config())

    _apply_env(cfg)
    cfg.parse.mode = str(cfg.parse.mode).lower()
    if cfg.parse.mode not in PARSE_MODES:
        raise ValueError(
            f"Invalid parse.mode: {cfg.parse.mode!r}; expected auto, ocr or direct"
        )
    return cfg


def env_names() -> dict[str, tuple[str, str]]:
    """``{"ANY2MD_BASE_URL": ("model", "base_url"), ...}``"""
    names = {}
    for section in fields(Config):
        for f in fields(section.default_factory):
            key = (section.name, f.name)
            names[_ENV_RENAMES.get(key, f"ANY2MD_{f.name.upper()}")] = key
    return names


def _parse_config(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = Config()
    for section in fields(Config):
        values = data.get(section.name) or {}
        target = getattr(cfg, section.name)
        for f in fields(target):
            if f.name in values:
                setattr(target, f.name, values[f.name])
    return cfg


def _apply_env(cfg: Config) -> None:
    for name, (section, key) in env_names().items():
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        target = getattr(cfg, section)
        # Annotations are strings here (``from __future__ import annotations``),
        # so the type comes from the default value.
        default = getattr(type(target)(), key)
        kind = _NULLABLE_TYPES.get((section, key)) or (
            str if default is None else type(default)
        )
        try:
            setattr(target, key, kind(raw))
        except ValueError:
            raise ValueError(f"{name}={raw!r} is not a valid {kind.__name__}") from None


# Fields whose default is None but whose value is not a string.
_NULLABLE_TYPES = {
    ("model", "temperature"): float,
    ("server", "pdf_workers"): int,
}
