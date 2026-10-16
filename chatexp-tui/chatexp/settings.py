from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - optional dependency before env update
    dotenv_values = None


class Language(str, Enum):
    """Supported languages for the ChatExp interface."""
    ZH = "zh"  # Chinese
    EN = "en"  # English


@dataclass
class OpenAISettings:
    api_key: str | None
    base_url: str | None
    model: str | None
    timeout_seconds: int = 180
    max_tokens: int = 65536
    language: Language = Language.ZH

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)


@dataclass
class BOApiSettings:
    """Connection settings for the chatexp-api BO backend (chatexp-api service)."""

    endpoint: str | None = None
    api_key: str | None = None
    timeout_seconds: int = 600
    candidate_pool_size: int = 2048
    ensemble_size: int = 5
    training_steps: int = 300

    @property
    def configured(self) -> bool:
        return bool(self.endpoint)


def load_bo_api_settings(project_root: str | Path | None = None) -> BOApiSettings:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    config = _read_env_file(root / ".env")
    return BOApiSettings(
        endpoint=_first_config(config, "CHATEXP_BO_API_ENDPOINT", "CHATEXP_BO_API_BASE_URL"),
        api_key=_first_config(config, "CHATEXP_BO_API_KEY"),
        timeout_seconds=_config_int(config, "CHATEXP_BO_API_TIMEOUT_SECONDS", default=600),
        candidate_pool_size=_config_int(config, "CHATEXP_BO_RUNTIME_CANDIDATE_POOL_SIZE", default=2048),
        ensemble_size=_config_int(config, "CHATEXP_BO_RUNTIME_ENSEMBLE_SIZE", default=5),
        training_steps=_config_int(config, "CHATEXP_BO_RUNTIME_TRAINING_STEPS", default=300),
    )


def load_strategist_mode(project_root: str | Path | None = None) -> str:
    """Campaign strategist mode: ``off`` (default) or ``llm``. The strategist
    is LLM-only — there is no rule-based fallback mode."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    config = _read_env_file(root / ".env")
    mode = (_first_config(config, "CHATEXP_STRATEGIST") or "off").strip().lower()
    return mode if mode in {"off", "llm"} else "off"


def load_openai_settings(
    project_root: str | Path | None = None,
    language: str | None = None,
) -> OpenAISettings:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    config = _read_env_file(env_path)

    # Determine language: CLI arg > .env > default. Shell env is intentionally ignored.
    lang_str = language or _first_config(config, "CHATEXP_LANGUAGE", "LANGUAGE") or "zh"
    try:
        lang = Language(lang_str.lower())
    except ValueError:
        lang = Language.ZH

    return OpenAISettings(
        api_key=_first_config(config, "CHATEXP_OPENAI_API_KEY", "API_TOKEN", "OPENAI_API_KEY"),
        base_url=_first_config(config, "CHATEXP_OPENAI_BASE_URL", "BASE_URL", "OPENAI_BASE_URL"),
        model=_first_config(config, "CHATEXP_OPENAI_MODEL", "MODEL_NAME", "OPENAI_MODEL"),
        timeout_seconds=_config_int(config, "CHATEXP_OPENAI_TIMEOUT_SECONDS", "CHATEXP_OPENAI_TIMEOUT", default=180),
        max_tokens=_config_int(config, "CHATEXP_OPENAI_MAX_TOKENS", default=65536),
        language=lang,
    )


def _read_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}
    if dotenv_values is not None:
        return {str(key): str(value).strip() for key, value in dotenv_values(env_path).items() if value is not None}
    return _parse_env_file(env_path)


def _parse_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _first_config(config: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = config.get(name)
        if value is not None:
            value = value.strip()
            if value:
                return value
    return None


def _config_int(config: dict[str, str], *names: str, default: int) -> int:
    for name in names:
        value = config.get(name)
        if value is None:
            continue
        try:
            parsed = int(value.strip())
        except Exception:
            continue
        if parsed > 0:
            return parsed
    return default
