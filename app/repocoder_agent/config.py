from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPENAI_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_OPENAI_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_COMMANDS = ["python -m pytest -q"]
DEFAULT_COMMAND_TIMEOUT_SEC = 60
DEFAULT_TRACE_ENABLED = True
DEFAULT_TRACE_DIRNAME = ".repocoder/runs"
DEFAULT_DRY_RUN_ENABLED = True

_ENV_LOADED = False
_LOADED_ENV_KEYS: set[str] = set()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_base_url: str
    openai_model: str
    default_commands: list[str]
    default_command_timeout_sec: int
    trace_enabled: bool
    trace_dirname: str
    dry_run_enabled: bool


def load_environment(start_dir: str | Path | None = None) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    candidates = _candidate_env_files(start_dir)
    for path in candidates:
        if path.exists() and path.is_file():
            _load_env_file(path)
            break

    _ENV_LOADED = True


def get_settings(start_dir: str | Path | None = None) -> Settings:
    load_environment(start_dir=start_dir)
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        default_commands=list(DEFAULT_COMMANDS),
        default_command_timeout_sec=DEFAULT_COMMAND_TIMEOUT_SEC,
        trace_enabled=_parse_bool(os.getenv("REPOCODER_TRACE_ENABLED"), DEFAULT_TRACE_ENABLED),
        trace_dirname=os.getenv("REPOCODER_TRACE_DIR", DEFAULT_TRACE_DIRNAME),
        dry_run_enabled=_parse_bool(os.getenv("REPOCODER_DRY_RUN_ENABLED"), DEFAULT_DRY_RUN_ENABLED),
    )


def reset_environment_cache(clear_loaded_values: bool = False) -> None:
    global _ENV_LOADED
    _ENV_LOADED = False
    if clear_loaded_values:
        for key in list(_LOADED_ENV_KEYS):
            os.environ.pop(key, None)
        _LOADED_ENV_KEYS.clear()


def _candidate_env_files(start_dir: str | Path | None) -> list[Path]:
    if start_dir is not None:
        base = Path(start_dir).resolve()
        if base.is_file():
            base = base.parent
        return [base / ".env"]

    bases = _walk_up(Path.cwd().resolve())
    seen: set[Path] = set()
    ordered: list[Path] = []
    for base in bases:
        env_path = base / ".env"
        if env_path not in seen:
            seen.add(env_path)
            ordered.append(env_path)
    return ordered


def _walk_up(start: Path) -> list[Path]:
    if start.is_file():
        start = start.parent
    return [start, *start.parents]


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value
        _LOADED_ENV_KEYS.add(key)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
