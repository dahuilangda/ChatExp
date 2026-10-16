"""Word document editing with tracked changes (修订模式).

The LLM plans edits; this module executes them deterministically on the
OOXML tree so every change lands as a native Word revision (``w:ins`` /
``w:del``) — visible in Word's Review pane, accept/reject-able, authored
"ChatExp". The original file is never mutated: the caller works on a temp
copy, and the output is re-opened and validated before it is returned.

Edit ops (planner contract):
- replace:          find → del(wrap) + ins(new text), applies to EVERY
                    occurrence within the named paragraph (find-and-replace-all)
- delete_paragraph: whole paragraph marked deleted
- insert_paragraph_after: new tracked paragraph after the anchor paragraph

Correctness model (hardened after adversarial review):
- ONE text view everywhere ("All Marks" view): direct runs + hyperlink runs +
  runs inside existing w:ins, EXCLUDING deleted text (w:delText). The planner
  sees exactly what these functions anchor on — hyperlink text and previously
  inserted tracked text are editable, enabling iterative editing rounds.
- All paragraph ids resolve to element references BEFORE any op mutates the
  tree (an insert_paragraph_after shifts indices; late ops must not drift).
- Run splitting goes through python-docx Run.text semantics; runs holding
  drawings/fields are refused explicitly instead of corrupted.
- Every failed op rolls back to its pre-op paragraph snapshot — a failed edit
  leaves no trace in the returned bytes.
"""
from __future__ import annotations

import copy
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from docx import Document
from docx.document import Document as DocumentObject
from docx.text.run import Run

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


class EditError(Exception):
    pass


# ---------------------------------------------------------------------------
# Edit plan model
# ---------------------------------------------------------------------------


@dataclass
class EditOp:
    op: str                 # replace | delete_paragraph | insert_paragraph_after
    pid: str                # paragraph id from the indexed view (e.g. "p12")
    find: str               # exact anchor substring within that paragraph ("" for insert)
    text: str = ""          # replacement / inserted text
    reason: str = ""

    def describe(self) -> str:
        return f"{self.op} pid={self.pid} find={self.find[:60]!r}"


@dataclass
class EditOutcome:
    op: EditOp
    ok: bool
    detail: str = ""


@dataclass
class EditResult:
    ok: list[EditOutcome] = field(default_factory=list)
    failed: list[EditOutcome] = field(default_factory=list)

    @property
    def all_failed(self) -> bool:
        return not self.ok

    def summary(self) -> dict[str, Any]:
        return {
            "applied": len(self.ok),
            "failed": len(self.failed),
            "operations": [
                {"op": o.op.op, "pid": o.op.pid, "reason": o.op.reason,
                 "ok": o.ok, "detail": o.detail}
                for o in self.ok + self.failed
            ],
        }


# ---------------------------------------------------------------------------
# Unified paragraph text model ("All Marks" view)
# ---------------------------------------------------------------------------


def _q(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


@dataclass
class _Unit:
    """One addressable run within a paragraph: the w:r element plus the
    parent it currently lives under (w:p, w:hyperlink, or w:ins)."""
    element: Any            # w:r element
    parent: Any             # owning w:p / w:hyperlink / w:ins element
    text: str               # Run.text semantics: w:t + w:tab→\t + w:br→\n
    start: int              # char offset in the paragraph's unified text
    end: int


def _paragraph_units(paragraph) -> list[_Unit]:
    """Ordered content units of a paragraph in the "All Marks" view.

    Includes: direct w:r children, runs inside w:hyperlink, runs inside
    existing w:ins (previously inserted tracked text — round-trip editing).
    Excludes: runs inside w:del and any w:delText (deleted text is gone from
    the view the user sees before accepting/rejecting).

    This is the SINGLE source of truth for text offsets — index_paragraphs
    and all three apply ops derive their text from these units, so the
    planner's view and the applier's anchors can never disagree.
    """
    units: list[_Unit] = []
    pos = 0

    def add_run(r_element: Any, parent: Any) -> None:
        nonlocal pos
        text = Run(r_element, paragraph).text or ""
        if text:
            units.append(_Unit(element=r_element, parent=parent, text=text, start=pos, end=pos + len(text)))
            pos += len(text)

    p = paragraph._p
    for child in p:
        tag = child.tag
        if tag == _q("r"):
            add_run(child, p)
        elif tag in (_q("hyperlink"), _q("ins")):
            # Descendant runs: containers may nest (ins>del, ins>ins, etc.).
            # Deleted runs carry w:delText only — Run.text returns "" for them
            # — so they drop out of the view naturally.
            for r in child.iter(_q("r")):
                add_run(r, child)
        # w:del and structural elements (pPr, bookmarks, proofErr) carry no
        # view text and are skipped.
    return units


def _unified_text(paragraph) -> str:
    return "".join(u.text for u in _paragraph_units(paragraph))


# ---------------------------------------------------------------------------
# Document indexing (paragraph-addressed view for the LLM)
# ---------------------------------------------------------------------------


@dataclass
class ParagraphView:
    pid: str
    index: int
    text: str
    style: str


def index_paragraphs(doc: DocumentObject, max_chars: int = 500_000) -> list[ParagraphView]:
    """Build the paragraph-indexed view the edit planner sees. Uses the
    unified "All Marks" text so previously-inserted tracked text is visible
    and editable in later rounds. Skips view-empty paragraphs but keeps
    indices stable (pid = p<index>)."""
    views: list[ParagraphView] = []
    total = 0
    for i, p in enumerate(doc.paragraphs):
        text = _unified_text(p)
        style = p.style.name if p.style is not None else ""
        if text.strip():
            views.append(ParagraphView(pid=f"p{i}", index=i, text=text, style=style))
            total += len(text)
            if total > max_chars:
                raise EditError(
                    f"Document exceeds the {max_chars}-character edit budget "
                    f"({total} chars). Split the edit into smaller files."
                )
    return views


def render_paragraph_view(views: list[ParagraphView]) -> str:
    lines = []
    for v in views:
        label = f" [{v.style}]" if v.style and v.style.lower() not in ("normal", "default") else ""
        lines.append(f"{v.pid}{label}: {v.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Track-changes primitives (OOXML surgery)
# ---------------------------------------------------------------------------


class _RevisionIds:
    def __init__(self, start: int = 1000):
        self.next = start

    def take(self) -> int:
        value = self.next
        self.next += 1
        return value


def _revision_marks(element: Any):
    """All revision-mark elements (w:ins/w:del/moveFrom/moveTo) under element."""
    for el in element.iter():
        tag = el.tag
        if tag.endswith("}ins") or tag.endswith("}del") or tag.endswith("}moveFrom") or tag.endswith("}moveTo"):
            yield el


def _max_revision_id(doc: DocumentObject) -> int:
    """Existing revision ids anywhere in the package (body + headers/footers/
    footnotes/comments — revision ids must be unique per part in practice)."""
    ids = [0]
    for part in [doc.part, *getattr(doc.part, "package", None).iter_parts()] if getattr(doc.part, "package", None) else [doc.part]:
        try:
            element = getattr(part, "element", None)
        except Exception:
            element = None
        if element is None:
            continue
        for el in _revision_marks(element):
            raw = el.get(_q("id"))
            if raw is not None and raw.isdigit():
                ids.append(int(raw))
    return max(ids)


# w:trackChanges must sit BEFORE these successors in CT_Settings' strict
# sequence (writeProtection/view/zoom/... come before it).
_TRACK_CHANGES_SUCCESSORS = {
    "documentProtection", "autoFormatOverride", "styleLockTheme", "styleLockQFSet",
    "defaultTabStop", "autoHyphenation", "consecutiveHyphenLimit", "hyphenationZone",
    "doNotHyphenateCaps", "showEnvelope", "summaryLength", "clickAndTypeStyle",
    "defaultTableStyle", "evenAndOddHeaders", "bookFoldRevPrinting", "bookFoldPrinting",
    "bookFoldPrintingSheets", "drawingGridHorizontalSpacing", "displayHorizontalDrawingGridEvery",
    "doNotUseMarginsForDrawingGridOrigin", "noPunctuationKerning", "characterSpacingControl",
    "printTwoOnOne", "strictFirstAndLastChars", "noLineBreaksAfter", "noLineBreaksBefore",
    "savePreviewPicture", "validateAgainstSchema", "saveInvalidXml", "ignoreMixedContent",
    "alwaysShowPlaceholderText", "doNotDemarcateInvalidXml", "saveXmlDataOnly",
    "useXSLTWhenSaving", "saveThroughXslt", "showXMLTags", "alwaysMergeEmptyNamespace",
    "updateFields", "hdrShapeDefaults", "footnotePr", "endnotePr", "compat",
    "docVars", "rsids", "mathPr", "attachedSchema", "themeFontLang", "clrSchemeMapping",
    "doNotIncludeSubdocsInStats", "doNotAutoCompressPictures", "forceUpgrade",
    "captions", "readModeInkLockDown", "smartTagType", "shapeDefaults", "doNotEmbedSmartTags",
    "decimalSymbol", "listSeparator",
}


def enable_track_changes(doc: DocumentObject) -> None:
    """Insert w:trackChanges in its schema-correct position (before the first
    successor element), not blindly at index 0 — strict validators and Word's
    repair prompt both care about CT_Settings ordering."""
    settings = doc.settings.element
    if settings.find(_q("trackChanges")) is not None:
        return
    track = settings.makeelement(_q("trackChanges"), {})
    for child in settings:
        local = child.tag.split("}")[-1]
        if local in _TRACK_CHANGES_SUCCESSORS:
            child.addprevious(track)
            return
    settings.append(track)


# Elements that make a run unsplittable: assigning Run.text would destroy
# them (CT_R.text setter clears all content first), so refuse explicitly.
_UNSAFE_RUN_CONTENT = {"drawing", "object", "pict", "fldChar", "fldSimple", "sym", "mc:AlternateContent"}


def _run_is_splittable(r_element: Any) -> bool:
    for child in r_element:
        local = child.tag.split("}")[-1]
        if local in _UNSAFE_RUN_CONTENT:
            return False
    return True


def _split_run_at(paragraph, unit: _Unit, offset: int) -> None:
    """Split the unit's run so that character ``offset`` (relative to the run)
    starts a new run. Uses Run.text semantics (tab/br round-trip correctly);
    runs holding drawings/fields are refused rather than corrupted."""
    run = Run(unit.element, paragraph)
    left_text = run.text[:offset]
    right_text = run.text[offset:]
    if not _run_is_splittable(unit.element):
        raise EditError(
            "cannot split a run containing drawings, fields, or symbols — "
            "the anchor would corrupt embedded content. Choose an anchor "
            "that does not cut through such a run."
        )
    new_element = copy.deepcopy(unit.element)
    unit.element.addnext(new_element)
    Run(unit.element, paragraph).text = left_text
    Run(new_element, paragraph).text = right_text


def _wrap_runs_in_del(units: list[_Unit], rev: _RevisionIds, author: str, date: str):
    """Wrap run elements in a single w:del placed at the paragraph level,
    converting w:t → w:delText. Runs are moved out of their current parents
    (direct, hyperlink, or existing w:ins) — each run carries its own rPr, so
    formatting survives; deleting previously-inserted (w:ins) text simply
    tracks the reversal. Returns the w:del element."""
    if not units:
        return None
    first_el = units[0].element
    del_el = first_el.makeelement(_q("del"), {
        _q("id"): str(rev.take()),
        _q("author"): author,
        _q("date"): date,
    })
    # The del sits directly BEFORE the first matched run, inside whatever
    # container holds it (w:p, w:hyperlink, w:ins) — document order is
    # preserved and the paired w:ins (added by the caller right after this
    # del) inherits the same position.
    first_el.addprevious(del_el)
    for unit in units:
        el = unit.element
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)
        del_el.append(el)
        for t in el.findall(_q("t")):
            t.tag = _q("delText")
            t.set(XML_SPACE, "preserve")
    return del_el


def _make_ins_run(p_element, text: str, rev: _RevisionIds, author: str, date: str):
    """Build a <w:ins><w:r><w:t>text</w:t></w:r></w:ins> element."""
    ins_el = p_element.makeelement(_q("ins"), {
        _q("id"): str(rev.take()),
        _q("author"): author,
        _q("date"): date,
    })
    r_el = p_element.makeelement(_q("r"), {})
    t_el = p_element.makeelement(_q("t"), {})
    t_el.set(XML_SPACE, "preserve")
    t_el.text = text
    r_el.append(t_el)
    ins_el.append(r_el)
    return ins_el


def _paragraph_mark_rpr(p_element):
    """Get or create pPr/rPr (paragraph-mark run properties)."""
    ppr = p_element.find(_q("pPr"))
    if ppr is None:
        ppr = p_element.makeelement(_q("pPr"), {})
        p_element.insert(0, ppr)
    rpr = ppr.find(_q("rPr"))
    if rpr is None:
        rpr = ppr.makeelement(_q("rPr"), {})
        ppr.append(rpr)
    return rpr


def _strip_revision_marks_from_rpr(rpr) -> None:
    """Remove stale w:ins/w:del marks (and their ids) from copied paragraph
    properties so duplicated revision ids never enter the document."""
    for mark in list(rpr):
        if mark.tag in (_q("ins"), _q("del")):
            rpr.remove(mark)


# ---------------------------------------------------------------------------
# Edit operations — each takes an already-resolved paragraph
# ---------------------------------------------------------------------------


def apply_replace(paragraph, op: EditOp, rev: _RevisionIds, author: str, date: str) -> str:
    """Find-and-replace with tracked changes over the unified text view.
    Every occurrence within the paragraph is replaced, right to left."""
    if not op.find:
        raise EditError("replace requires a non-empty 'find' anchor.")
    if not op.text:
        raise EditError("replace requires non-empty 'text' (use delete_paragraph instead).")

    full = _unified_text(paragraph)
    if op.find not in full:
        raise EditError(
            f"Anchor {op.find[:80]!r} not found in {op.pid}. The anchor must "
            f"be an exact substring of that paragraph."
        )

    occurrences = full.count(op.find)
    applied = 0
    for match in reversed(list(re.finditer(re.escape(op.find), full))):
        start, end = match.span()
        units = _paragraph_units(paragraph)
        # Split boundary at `start` and `end` so the anchor covers whole units.
        for boundary in (start, end):
            units = _paragraph_units(paragraph)
            split_done = False
            for u in units:
                if u.start < boundary < u.end:
                    _split_run_at(paragraph, u, boundary - u.start)
                    split_done = True
                    break
            if split_done:
                continue
        units = _paragraph_units(paragraph)
        matched = [u for u in units if u.start >= start and u.end <= end and u.end > u.start]
        if not matched:
            raise EditError(
                f"Internal: could not isolate anchor runs at [{start},{end}) in {op.pid}."
            )
        del_el = _wrap_runs_in_del(matched, rev, author, date)
        ins_el = _make_ins_run(paragraph._p, op.text, rev, author, date)
        # w:ins must be a SIBLING of w:del (nesting is invalid OOXML).
        del_el.addnext(ins_el)
        applied += 1
    return f"replaced {applied} occurrence(s) in {op.pid}" if applied else ""


def apply_delete_paragraph(paragraph, op: EditOp, rev: _RevisionIds, author: str, date: str) -> str:
    if not op.find:
        raise EditError("delete_paragraph requires a 'find' anchor to identify the paragraph.")
    if op.find not in _unified_text(paragraph):
        raise EditError(
            f"Anchor {op.find[:80]!r} not found in {op.pid}."
        )
    units = [u for u in _paragraph_units(paragraph) if u.text]
    if units:
        # Deleting the whole paragraph covers hyperlink and w:ins content too.
        _wrap_runs_in_del(units, rev, author, date)
    rpr = _paragraph_mark_rpr(paragraph._p)
    _strip_revision_marks_from_rpr(rpr)
    del_mark = rpr.makeelement(_q("del"), {
        _q("id"): str(rev.take()), _q("author"): author, _q("date"): date,
    })
    rpr.insert(0, del_mark)
    return f"paragraph {op.pid} marked deleted"


def apply_insert_paragraph_after(paragraph, op: EditOp, rev: _RevisionIds, author: str, date: str) -> str:
    if not op.find:
        raise EditError("insert_paragraph_after requires a 'find' anchor identifying the paragraph to insert after.")
    if op.find not in _unified_text(paragraph):
        raise EditError(f"Anchor {op.find[:80]!r} not found in {op.pid}.")
    if not op.text.strip():
        raise EditError("insert_paragraph_after requires non-empty 'text'.")

    new_p = copy.deepcopy(paragraph._p)
    for child in list(new_p):
        if not child.tag.endswith("}pPr"):
            new_p.remove(child)
    rpr = _paragraph_mark_rpr(new_p)
    _strip_revision_marks_from_rpr(rpr)
    ins_mark = new_p.makeelement(_q("ins"), {
        _q("id"): str(rev.take()), _q("author"): author, _q("date"): date,
    })
    rpr.insert(0, ins_mark)
    ins_el = _make_ins_run(new_p, op.text, rev, author, date)
    new_p.append(ins_el)
    paragraph._p.addnext(new_p)
    return f"tracked paragraph inserted after {op.pid}"


# ---------------------------------------------------------------------------
# Top-level edit application
# ---------------------------------------------------------------------------


def _resolve_pid(doc: DocumentObject, pid: str):
    match = re.fullmatch(r"p(\d+)", pid.strip())
    if not match:
        raise EditError(f"Invalid paragraph id {pid!r} — expected 'p<index>'.")
    idx = int(match.group(1))
    if idx >= len(doc.paragraphs):
        raise EditError(
            f"Paragraph {pid} does not exist (document has "
            f"{len(doc.paragraphs)} paragraphs)."
        )
    return doc.paragraphs[idx]


def apply_edits(
    data: bytes,
    edits: list[EditOp],
    author: str = "ChatExp",
) -> tuple[bytes, EditResult]:
    """Apply the planned edits as tracked changes and validate the output.

    Returns (edited_docx_bytes, audit). Raises EditError only for
    non-recoverable problems (unreadable input); per-op failures live in the
    audit so the caller can surface them explicitly.
    """
    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        raise EditError(f"Cannot open the document as .docx: {exc}") from exc

    result = EditResult()

    # Resolve ALL pids to paragraph references BEFORE any mutation. An
    # insert_paragraph_after shifts paragraph indices; resolving late would
    # silently retarget subsequent ops onto the wrong paragraphs.
    resolved: list[tuple[EditOp, Any]] = []
    for op in edits:
        try:
            resolved.append((op, _resolve_pid(doc, op.pid)))
        except EditError as exc:
            result.failed.append(EditOutcome(op=op, ok=False, detail=str(exc)))

    rev = _RevisionIds(_max_revision_id(doc) + 1)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for op, paragraph in resolved:
        snapshot = copy.deepcopy(paragraph._p)  # per-op rollback
        try:
            if op.op == "replace":
                detail = apply_replace(paragraph, op, rev, author, date)
            elif op.op == "delete_paragraph":
                detail = apply_delete_paragraph(paragraph, op, rev, author, date)
            elif op.op == "insert_paragraph_after":
                detail = apply_insert_paragraph_after(paragraph, op, rev, author, date)
            else:
                raise EditError(
                    f"Unknown op {op.op!r} — expected replace, delete_paragraph, "
                    f"or insert_paragraph_after."
                )
            result.ok.append(EditOutcome(op=op, ok=True, detail=detail))
        except (EditError, ValueError) as exc:
            # Roll the paragraph back to its pre-op state: a failed edit must
            # leave NO trace in the returned bytes (no half-splits, no
            # duplicated tabs/line breaks, no orphaned marks).
            parent = paragraph._p.getparent()
            if parent is not None:
                parent.replace(paragraph._p, snapshot)
            result.failed.append(EditOutcome(op=op, ok=False, detail=str(exc)))

    enable_track_changes(doc)

    # Validate: the edited document must reopen cleanly and contain the
    # expected revision marks. A corrupted output must never be returned.
    buf = io.BytesIO()
    doc.save(buf)
    out = buf.getvalue()
    try:
        check = Document(io.BytesIO(out))
        ins_count = sum(1 for el in check.element.body.iter() if el.tag == _q("ins"))
        del_count = sum(1 for el in check.element.body.iter() if el.tag == _q("del"))
        if result.ok and (ins_count + del_count) == 0:
            raise EditError(
                "Validation failed: edits reported success but no revision "
                "marks exist in the output."
            )
    except EditError:
        raise
    except Exception as exc:
        raise EditError(f"Edited document failed validation (will not be returned): {exc}") from exc

    return out, result


def parse_edit_plan(raw: Any) -> list[EditOp]:
    """Parse the LLM's edit-plan JSON into EditOp list. Strict: a malformed
    plan fails loudly instead of being partially applied."""
    if isinstance(raw, str):
        import json
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise EditError(f"Edit plan is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or "edits" not in raw:
        raise EditError("Edit plan must be an object with an 'edits' array.")
    plan = raw["edits"]
    if not isinstance(plan, list) or not plan:
        raise EditError("Edit plan 'edits' must be a non-empty array.")
    ops: list[EditOp] = []
    for i, item in enumerate(plan):
        if not isinstance(item, dict):
            raise EditError(f"Edit #{i} is not an object.")
        op_type = str(item.get("op") or "").strip()
        pid = str(item.get("pid") or "").strip()
        find = str(item.get("find") or "")
        text = str(item.get("text") or "")
        reason = str(item.get("reason") or "")
        if op_type not in ("replace", "delete_paragraph", "insert_paragraph_after"):
            raise EditError(
                f"Edit #{i}: unknown op {op_type!r}. Allowed: replace, "
                f"delete_paragraph, insert_paragraph_after."
            )
        if not re.fullmatch(r"p\d+", pid):
            raise EditError(f"Edit #{i}: 'pid' must look like 'p12', got {pid!r}.")
        ops.append(EditOp(op=op_type, pid=pid, find=find, text=text, reason=reason))
    return ops


def ensure_docx(data: bytes, name: str) -> tuple[bytes, str]:
    """Guarantee docx input. Only .docx bytes are editable: legacy Word
    formats (.doc/.rtf/.dot) are rejected explicitly (no conversion layer
    exists since the docling-only refactor), and everything else fails the
    Word-only contract."""
    from .parser import _suffix
    suffix = _suffix(name)
    if suffix == ".docx":
        return data, name
    if suffix in (".doc", ".rtf", ".dot"):
        raise EditError(
            "legacy .doc/.rtf/.dot conversion is not supported; convert the file to .docx first"
        )
    raise EditError(
        f"{name!r} is not an editable Word document (expected .docx). "
        f"Track-changes editing applies to Word documents only."
    )
