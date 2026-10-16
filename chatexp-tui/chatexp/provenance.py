"""LLM configuration provenance: record which LLM, temperature, and prompt
templates shaped each run.

Every task directory can carry an ``llm_provenance.json`` describing the exact
model settings and the sha256 of every prompt template that shaped planner
behavior. Eval scripts and the benchmark harness embed the same structure so
paper numbers are reproducible from configuration alone.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .settings import OpenAISettings

PROVENANCE_FILENAME = "llm_provenance.json"

# Single source of truth for per-stage sampling temperatures.
STAGE_TEMPERATURES: dict[str, float] = {
    "planner": 0.0,
    "respond": 0.2,
    "repair": 0.0,
    "import_mapper": 0.0,
    "import_interpreter": 0.0,
    "analysis_narrator": 0.2,
    "weight_optimizer": 0.3,
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_asset_hashes(prompt_dir: str | Path | None = None) -> dict[str, str]:
    directory = Path(prompt_dir) if prompt_dir else Path(__file__).resolve().with_name("prompt_assets")
    hashes: dict[str, str] = {}
    if not directory.exists():
        return hashes
    for path in sorted(directory.glob("*.txt")):
        try:
            hashes[path.name] = _sha256(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return hashes


def collect_llm_provenance(
    settings: OpenAISettings,
    prompt_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from urllib.parse import urlsplit

    host = None
    if settings.base_url:
        parts = urlsplit(settings.base_url)
        host = parts.netloc or None
    provenance: dict[str, Any] = {
        "v": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "llm": {
            "model": settings.model,
            # The base URL host is configuration evidence; the full URL may
            # carry staging names and is intentionally kept, but never the key.
            "base_url_host": host,
            "timeout_seconds": settings.timeout_seconds,
            "max_tokens": settings.max_tokens,
            "language": str(getattr(settings.language, "value", settings.language)),
        },
        "api_key_present": bool(settings.api_key),
        "stage_temperatures": dict(STAGE_TEMPERATURES),
        "prompt_template_sha256": prompt_asset_hashes(prompt_dir),
    }
    if extra:
        provenance.update(extra)
    return provenance


def write_llm_provenance(
    task_dir: str | Path,
    settings: OpenAISettings,
    prompt_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Persist provenance for a task. Returns the written path, or None when
    the settings are unconfigured or the write fails (best-effort)."""
    if not settings.configured:
        return None
    try:
        directory = Path(task_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / PROVENANCE_FILENAME
        if path.exists() and not overwrite:
            return path
        provenance = collect_llm_provenance(settings, prompt_dir, extra)
        path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


__all__ = [
    "PROVENANCE_FILENAME",
    "STAGE_TEMPERATURES",
    "collect_llm_provenance",
    "prompt_asset_hashes",
    "write_llm_provenance",
]
