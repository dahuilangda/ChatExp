"""Server-side document parsing and analysis for the ChatExp SeaTable plugin.

Documents are parsed server-side by docling-serve into a structured manifest,
cached in redis, and surfaced to the plugin as section-addressable text.

Document analysis uses a planner+skills+harness architecture:
- **planner.py**: LLM-based mission + step planning
- **harness.py**: Coordinator, audit loop, direction maintenance
- **skills.py**: Skills catalog with pure English prompts
- **skill_runners.py**: Skill execution (LLM calls + data operations)
- **agent.py**: Thin entry point for the celery task
"""
