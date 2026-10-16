"""Server-side document and image analysis agent (planner+skills+harness).

Thin entry point that delegates to the planner+skills+harness system.
The heavy lifting is done by:

- **harness.py**: Coordinator, audit loop, direction maintenance
- **planner.py**: LLM-based mission + step planning
- **skills.py**: Skills catalog with pure English prompts
- **skill_runners.py**: Skill execution functions

The LLM plans; the harness executes skills and audits each step against the
skill contracts.
"""
from __future__ import annotations

import logging
import time

from . import artifacts, harness
from .skill_runners import AnalyzeError

logger = logging.getLogger(__name__)

# Re-export AnalyzeError so callers can catch it.
__all__ = ["AnalyzeError", "analyze_document"]


def analyze_document(
    doc_id: str,
    task: str,
    images: list[dict[str, str]] | None = None,
    celery_task=None,
    llm_settings: dict[str, str] | None = None,
) -> str:
    """Analyze a parsed document and/or images using the planner+skills+harness system.

    This is a thin entry point that:
    1. Reads the document from the artifact store (if doc_id provided)
    2. Delegates to harness.analyze() for the planner+skills+harness loop
    3. Reports progress via celery update_state

    Args:
        doc_id: The parsed document ID (required — a docId is the only
            analyzable input; inline image requests are refused).
        task: The user's question or analysis task.
        images: Optional list of inline image data URLs.
        celery_task: Optional celery task for progress reporting.
        llm_settings: Optional LLM settings override.

    Returns:
        The analysis answer string.

    Raises:
        AnalyzeError: On any unrecoverable analysis failure.
    """
    from .skill_runners import _llm_settings

    start = time.monotonic()
    settings = llm_settings if llm_settings is not None else _llm_settings()
    imgs = [img for img in (images or []) if isinstance(img, dict) and img.get("dataUrl")]
    has_doc = bool(doc_id)

    # Inline images are not analyzed directly — no skill ingests them; images
    # are OCR'd by docling at parse time. Accepting them would silently analyze
    # the document while ignoring the images. The explicit contract: parse
    # first, then analyze the docId.
    if imgs:
        raise AnalyzeError(
            "Inline image analysis is not supported. Images must be parsed "
            "through the /v1/files/parse endpoint (docling OCR) first, which "
            "produces a docId. Then call /v1/files/analyze with that docId."
        )
    if not has_doc:
        raise AnalyzeError("A docId must be provided for analysis.")

    document = artifacts.read(doc_id)
    if not document:
        raise AnalyzeError(
            f"Document {doc_id} not found or evicted. Re-parse the file first."
        )

    def report(phase: str, detail: str, est_total: float = 30.0) -> None:
        if celery_task is not None:
            celery_task.update_state(
                state="PROGRESS",
                meta={
                    "phase": phase,
                    "detail": detail,
                    "elapsedSeconds": round(time.monotonic() - start, 1),
                    "estimatedTotalSeconds": est_total,
                    # The task is executing by definition, so it has been
                    # dequeued: no queue position to report (None).
                    "queuePosition": None,
                },
            )

    answer = harness.analyze(
        task=task,
        doc_id=doc_id,
        document=document,
        images=imgs,
        llm_settings=settings,
        report_progress=report,
    )

    return answer