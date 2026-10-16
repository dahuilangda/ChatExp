from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .settings import Language


SKILL_SPEC_DIR = Path(__file__).resolve().with_name("skill_specs")
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatExpSkill:
    name: str
    layer: str
    description: str
    when_to_use: str
    title_zh: str
    title_en: str
    purpose_zh: str
    purpose_en: str
    actions: tuple[str, ...]
    triggers_zh: tuple[str, ...]
    triggers_en: tuple[str, ...]
    input_schema: str
    output_schema: str
    quality_checks: tuple[str, ...]
    failure_policy: str
    decision_rule_zh: str
    decision_rule_en: str
    examples_zh: tuple[str, ...]
    examples_en: tuple[str, ...]

    @classmethod
    def from_directory(cls, skill_dir: Path) -> "ChatExpSkill":
        """Load skill from a SKILL.md + examples.md directory."""
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

        text = skill_md.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        name = str(frontmatter.get("name") or skill_dir.name)
        sections = _parse_sections(body)

        # Validate required frontmatter
        if not frontmatter.get("actions"):
            _LOG.warning("Skill %s has no actions in frontmatter", name)

        purpose_zh, purpose_en = _split_bilingual(sections.get("purpose", ""))
        decision_zh, decision_en = _split_bilingual(sections.get("decision rule", ""))
        triggers_zh, triggers_en = _parse_triggers(sections.get("triggers", ""))
        quality_checks = _parse_bullet_list(sections.get("quality checks", ""))
        examples_zh, examples_en = _load_examples(skill_dir / "examples.md") if (skill_dir / "examples.md").exists() else ((), ())

        return cls(
            name=name,
            layer=str(frontmatter.get("layer", "")),
            description=str(frontmatter.get("description", "")),
            when_to_use=str(frontmatter.get("when_to_use", "")),
            title_zh=str(frontmatter.get("title_zh", name)),
            title_en=str(frontmatter.get("title_en", name)),
            purpose_zh=purpose_zh,
            purpose_en=purpose_en,
            actions=tuple(str(a) for a in (frontmatter.get("actions") or [])),
            triggers_zh=triggers_zh,
            triggers_en=triggers_en,
            input_schema=sections.get("input contract", ""),
            output_schema=sections.get("output contract", ""),
            quality_checks=quality_checks,
            failure_policy=sections.get("failure policy", ""),
            decision_rule_zh=decision_zh,
            decision_rule_en=decision_en,
            examples_zh=examples_zh,
            examples_en=examples_en,
        )

    def render(self, language: Language | str = Language.ZH) -> str:
        is_en = _is_english(language)
        title = self.title_en if is_en else self.title_zh
        purpose = self.purpose_en if is_en else self.purpose_zh
        triggers = self.triggers_en if is_en else self.triggers_zh
        decision_rule = self.decision_rule_en if is_en else self.decision_rule_zh
        quality_text = " | ".join(self.quality_checks)
        return (
            f"- {self.name} [{self.layer}]: {title}\n"
            f"  description: {self.description}\n"
            f"  when_to_use: {self.when_to_use}\n"
            f"  actions: {', '.join(self.actions)}\n"
            f"  input: {self.input_schema}\n"
            f"  output: {self.output_schema}\n"
            f"  quality: {quality_text}\n"
            f"  failure: {self.failure_policy}\n"
            f"  decision: {decision_rule}"
        )

    def render_summary(self, language: Language | str = Language.ZH) -> str:
        is_en = _is_english(language)
        title = self.title_en if is_en else self.title_zh
        decision_rule = self.decision_rule_en if is_en else self.decision_rule_zh
        return (
            f"- {self.name}: {title}\n"
            f"  when_to_use: {self.when_to_use}\n"
            f"  actions: {', '.join(self.actions)}\n"
            f"  decision: {decision_rule}"
        )


# ---------------------------------------------------------------------------
# SKILL.md parsing helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_HEADER_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        _LOG.warning("Failed to parse YAML frontmatter")
        return {}, text
    if not isinstance(data, dict):
        _LOG.warning("YAML frontmatter is not a dict: %s", type(data))
        return {}, text
    body = text[m.end():]
    return data, body


def _parse_sections(body: str) -> dict[str, str]:
    """Parse markdown body into section-name -> content mapping."""
    parts = _SECTION_HEADER_RE.split(body)
    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        header = parts[i].strip().lower()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[header] = content
    return sections


def _split_bilingual(text: str) -> tuple[str, str]:
    """Extract zh/en values from bilingual text."""
    zh, en = "", ""
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("**zh:**"):
            zh = line.split("**zh:**", 1)[1].strip()
        elif line.lower().startswith("**en:**"):
            en = line.split("**en:**", 1)[1].strip()
    if not zh and not en:
        zh = text.strip()
    return zh, en


def _parse_triggers(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract zh/en trigger lists from triggers section."""
    zh, en = "", ""
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("**zh:**"):
            zh = line.split("**zh:**", 1)[1].strip()
        elif line.lower().startswith("**en:**"):
            en = line.split("**en:**", 1)[1].strip()
    zh_triggers = tuple(t.strip() for t in zh.split(",") if t.strip())
    en_triggers = tuple(t.strip() for t in en.split(",") if t.strip())
    return zh_triggers, en_triggers


def _parse_bullet_list(text: str) -> tuple[str, ...]:
    """Parse '- item' bullets into a tuple of strings."""
    return tuple(
        line.lstrip("- ").strip()
        for line in text.split("\n")
        if line.strip().startswith("-")
    )


def _load_examples(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load examples from examples.md with ## Chinese / ## English sections."""
    if not path.exists():
        return (), ()
    text = path.read_text(encoding="utf-8")
    sections = _parse_sections(text)
    zh_raw = sections.get("chinese", "")
    en_raw = sections.get("english", "")
    zh = tuple(_parse_numbered_list(zh_raw))
    en = tuple(_parse_numbered_list(en_raw))
    return zh, en


def _parse_numbered_list(text: str) -> list[str]:
    """Parse '1. item' numbered list."""
    items = []
    for line in text.split("\n"):
        m = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if m:
            items.append(m.group(1).strip())
    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_skills() -> tuple[ChatExpSkill, ...]:
    skills = []
    for d in sorted(SKILL_SPEC_DIR.iterdir()):
        if d.is_dir() and (d / "SKILL.md").exists():
            try:
                skills.append(ChatExpSkill.from_directory(d))
            except Exception as exc:
                _LOG.warning("Failed to load skill from %s: %s", d.name, exc)
    return tuple(skills)


def reload_skills() -> tuple[ChatExpSkill, ...]:
    """Clear cache and reload all skills from disk."""
    load_skills.cache_clear()
    return load_skills()


def skill_catalog(language: Language | str = Language.EN, names: list[str] | None = None) -> str:
    skills = load_skills()
    if names:
        wanted = set(names)
        skills = tuple(skill for skill in skills if skill.name in wanted)
    return "\n".join(skill.render(language) for skill in skills)


def _infer_campaign_stage(
    project: Any | None,
    pending: Any | None,
    ui_state: dict[str, Any] | None,
) -> str:
    """Derive the campaign stage from observable state — deterministic,
    data-driven, no keyword matching.

    Stages (progressive disclosure levels):
      bootstrap     — no project yet; the operator is starting from scratch.
      planning      — project exists but lacks factors or objectives.
      active        — project has factors + objectives; experiments may exist.
      reviewing     — a pending confirmation is staged (highest priority).
    """
    if pending is not None and getattr(pending, "action", None):
        return "reviewing"
    if project is None:
        return "bootstrap"
    if getattr(project, "factors", None) and getattr(project, "objectives", None):
        return "active"
    return "planning"


# Stage -> which skill layers to expose. Layers not listed are hidden.
# This is the progressive-disclosure policy: the LLM sees only the skills
# relevant to the current campaign stage, reducing prompt noise and
# preventing premature actions.
STAGE_LAYERS: dict[str, frozenset[str]] = {
    "bootstrap": frozenset({"conversational interface", "campaign framing", "data recognition", "design generation"}),
    "planning": frozenset({"campaign framing", "data recognition", "design generation", "conversational interface"}),
    "active": frozenset({"state mutation", "analysis and diagnosis", "display control",
                         "design generation", "conversational interface", "data recognition"}),
    "reviewing": frozenset({"conversational interface", "state mutation", "display control",
                            "analysis and diagnosis"}),
}


def build_skill_context(
    message: str,
    project: Any | None,
    pending: Any | None,
    ui_state: dict[str, Any] | None,
    language: Language | str = Language.ZH,
) -> dict[str, Any]:
    """Build the LLM-facing skill context with progressive disclosure.

    Returns a dict with:
      stage: the inferred campaign stage
      active_skills: skill names visible at this stage
      active_catalog: rendered text for visible skills (English, for the LLM)
      catalog: full catalog (for reference; the LLM should prefer active)
      routing_rule: instruction text
    """
    # The skill catalog is LLM-facing; always render in English for
    # international reproducibility.
    catalog_language = Language.EN
    skills = load_skills()
    stage = _infer_campaign_stage(project, pending, ui_state)
    allowed = STAGE_LAYERS.get(stage, frozenset(s.name for s in skills))

    visible = [s for s in skills if s.layer in allowed]
    hidden = [s for s in skills if s.layer not in allowed]
    visible_names = [s.name for s in visible]

    full_catalog = "\n".join(skill.render(catalog_language) for skill in skills)
    active_catalog = "\n".join(s.render(catalog_language) for s in visible)

    # Compact summary strip: one line per hidden skill so the LLM knows the
    # capability exists even though the full contract is not exposed.
    hidden_summary = ""
    if hidden:
        lines = []
        for skill in hidden:
            summary = skill.render_summary(catalog_language)
            last_line = summary.splitlines()[-1] if summary else skill.when_to_use
            lines.append(f"  {skill.name}: {last_line}")
        hidden_summary = (
            "\n\nSkills not active at this stage (their actions remain available "
            "via the harness if the user explicitly requests them):\n" + "\n".join(lines)
        )

    routing_rule = (
        f"Campaign stage: {stage}. "
        "Choose from the active skills below using semantic planning. "
        "Do not use keyword fallback, deterministic parsing, or hard-coded examples. "
        "Return exactly one structured action; the terminal harness will audit, "
        "risk-grade, confirm, and execute mutations."
    )
    if hidden:
        routing_rule += f" Skills not in the active set ({', '.join(s.name for s in hidden)}) are less relevant at this stage but their actions remain available if the user explicitly requests them."

    return {
        "stage": stage,
        "selected_skills": visible_names,
        "catalog": full_catalog,
        "active_layer": stage,
        "active_skills": visible_names,
        "active_catalog": active_catalog + hidden_summary,
        "hidden_skills": [s.name for s in hidden],
        "routing_rule": routing_rule,
    }


def _is_english(language: Language | str) -> bool:
    return str(language) == str(Language.EN) or str(language).lower().endswith("en")
