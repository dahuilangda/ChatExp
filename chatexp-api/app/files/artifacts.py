"""File-system storage for parsed document artifacts (blocks/markdown/tables)
and analysis checkpoint event logs.

Multi-MB payloads live on a shared volume (mounted in both api and worker)
rather than in redis, which stays bounded by holding only lightweight metadata.
Writes are atomic (tmp file + os.replace) so a crash never leaves a half-written
artifact that a later fetch could mistake for a valid one.

Layout:  {root}/{ab}/{docId}.json            parsed documents
         {root}/{ab}/{analysisId}.events.jsonl   analysis checkpoints
where {ab} = first two chars of the id. The two-char sharding keeps any single
directory from accumulating tens of thousands of files as the cache grows.
The distinct suffix keeps checkpoints out of the document stats/listing.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from . import config


class ArtifactError(Exception):
    pass


def _shard(doc_id: str) -> str:
    return (doc_id or "xx")[:2].lower()


def _path_for(doc_id: str) -> str:
    return os.path.join(config.parsed_files_dir(), _shard(doc_id), f"{doc_id}.json")


def exists(doc_id: str) -> bool:
    return os.path.exists(_path_for(doc_id))


def size(doc_id: str) -> int:
    try:
        return os.path.getsize(_path_for(doc_id))
    except OSError:
        return 0


def read(doc_id: str) -> dict[str, Any] | None:
    """Read a parsed artifact. Returns None if it does not exist or is corrupt
    (a half-write from an old crash would have been replaced atomically, so a
    JSON decode failure here means the file was externally corrupted)."""
    path = _path_for(doc_id)
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def write(doc_id: str, document: dict[str, Any]) -> str:
    """Atomically write an artifact and return its shard-relative path
    (``ab/docId.json``) for storage in the registry metadata.

    Atomicity: write to ``<path>.tmp.<pid>`` then os.replace onto the final
    path. os.replace is atomic on POSIX and Windows, so a reader never sees a
    partially written file. The tmp suffix includes pid to avoid collisions
    between concurrent writes to different docs in the same shard.
    """
    root = config.parsed_files_dir()
    shard_dir = os.path.join(root, _shard(doc_id))
    os.makedirs(shard_dir, exist_ok=True)
    final_path = os.path.join(shard_dir, f"{doc_id}.json")
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    # Write then fsync the tmp file before the atomic rename, so the renamed
    # file is durable even if the process is killed immediately after. Clean up
    # the tmp file on any failure (ENOSPC, etc.) so repeated failures don't
    # scatter permanent .tmp.<pid> files across shard dirs.
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return os.path.join(_shard(doc_id), f"{doc_id}.json")


def delete(doc_id: str) -> bool:
    """Remove an artifact. Returns True if a file was removed."""
    path = _path_for(doc_id)
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


def total_bytes() -> int:
    """Sum of all artifact sizes on disk, for stats and cleanup decisions."""
    root = config.parsed_files_dir()
    total = 0
    if not os.path.isdir(root):
        return 0
    for entry in os.listdir(root):
        shard_dir = os.path.join(root, entry)
        if not os.path.isdir(shard_dir):
            continue
        for name in os.listdir(shard_dir):
            if not name.endswith(".json"):
                continue
            try:
                total += os.path.getsize(os.path.join(shard_dir, name))
            except OSError:
                continue
    return total


def list_doc_ids() -> list[str]:
    """All docIds currently on disk (sorted). Used by startup reconciliation
    and stats; not hot-path."""
    root = config.parsed_files_dir()
    ids: list[str] = []
    if not os.path.isdir(root):
        return ids
    for entry in os.listdir(root):
        shard_dir = os.path.join(root, entry)
        if not os.path.isdir(shard_dir):
            continue
        for name in os.listdir(shard_dir):
            if name.endswith(".json"):
                ids.append(name[:-len(".json")])
    return sorted(ids)


# ---------------------------------------------------------------------------
# Analysis checkpoints (pi session persistence, artifact-side JSONL design)
# ---------------------------------------------------------------------------


def analysis_id(doc_id: str, task: str, llm_model: str) -> str:
    """Stable identity of ONE analysis run: same document + same question +
    same model => same checkpoint, so a crashed run resumes and a different
    question (or model) starts fresh. sha1 of the NUL-separated tuple keeps
    the components unambiguous."""
    return hashlib.sha1(
        f"{doc_id}\x00{task}\x00{llm_model}".encode("utf-8")
    ).hexdigest()


def checkpoint_path(analysis_id: str) -> str:
    """Absolute path of an analysis checkpoint, sharded exactly like document
    artifacts ({root}/{ab}/{id}.events.jsonl) so checkpoints live on the same
    shared volume under the same sharding scheme."""
    return os.path.join(
        config.parsed_files_dir(), _shard(analysis_id), f"{analysis_id}.events.jsonl"
    )


def append_checkpoint_line(path: str, obj: Any) -> None:
    """Append ONE event as a single json.dumps line (JSONL). The shard dir is
    created on demand; the line is flushed + fsynced before returning so a
    crash between events never loses the last recorded step."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_checkpoint(path: str) -> list[Any]:
    """All events from a checkpoint file, oldest first (file order). Malformed
    lines — a torn final line after a crash, external corruption — are skipped:
    resume replays only what parses. A missing file is an empty event list."""
    events: list[Any] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                text = raw.strip()
                if not text:
                    continue
                try:
                    events.append(json.loads(text))
                except ValueError:
                    continue
    except OSError:
        return []
    return events
