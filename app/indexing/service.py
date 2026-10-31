from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, select, text, update

from app.config import get_settings
from app.connectors.base import FileChange, GitError, TreeEntry
from app.connectors.local_git import LocalGitConnector
from app.connectors.registry import connector_for_repository
from app.db import session_scope
from app.indexing.chunker import chunk_source, language_for_path
from app.indexing.ignore import build_matcher
from app.indexing.limits import EMBEDDING_DIM, MAX_BLOB_BYTES, MAX_CHUNKS_PER_FILE, MAX_FILE_LINES
from app.llm.base import Embedder, ModelUnavailable
from app.llm.ollama import OllamaEmbedder
from app.models import CodeChunk, IndexJob, IndexedFile, Repository
from app.redact import redact

log = logging.getLogger(__name__)

_locks_guard = threading.Lock()
_locks: dict[uuid.UUID, threading.Lock] = {}
_threads: list[threading.Thread] = []


@dataclass
class ExistingFile:
    id: uuid.UUID
    blob_sha: str | None
    skipped_reason: str | None
    embeddings_ready: bool


def repo_lock(repository_id: uuid.UUID) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(repository_id)
        if lock is None:
            lock = threading.Lock()
            _locks[repository_id] = lock
        return lock


def join_index_threads(timeout: float = 60) -> None:
    for thread in list(_threads):
        thread.join(timeout)


def launch_index(repository_id: uuid.UUID, mode: str, force: bool) -> IndexJob:
    lock = repo_lock(repository_id)
    if not lock.acquire(blocking=False):
        with session_scope() as session:
            job = _latest_running(session, repository_id)
            if job is not None:
                session.expunge(job)
                return job
        raise RuntimeError("index already running")
    try:
        job = _create_job(repository_id, mode, force)
        thread = threading.Thread(target=_thread_main, args=(repository_id, job.id), daemon=True)
        _threads.append(thread)
        thread.start()
        return job
    except Exception:
        lock.release()
        raise


def run_index_sync(
    repository_id: uuid.UUID,
    mode: str,
    force: bool,
    embedder: Embedder | None = None,
) -> IndexJob:
    lock = repo_lock(repository_id)
    lock.acquire()
    try:
        job = _create_job(repository_id, mode, force)
        execute_index_job(repository_id, job.id, embedder=embedder)
        with session_scope() as session:
            stored = session.get(IndexJob, job.id)
            if stored is None:
                raise GitError("index job disappeared")
            session.expunge(stored)
            return stored
    finally:
        lock.release()


def execute_index_job(
    repository_id: uuid.UUID,
    job_id: uuid.UUID,
    embedder: Embedder | None = None,
) -> None:
    if embedder is None:
        embedder = OllamaEmbedder(get_settings())
    try:
        _execute(repository_id, job_id, embedder)
    except Exception as exc:
        _mark_failed(repository_id, job_id, redact(str(exc))[:500])
        log.warning("index job failed: %s", redact(str(exc))[:300])


def _thread_main(repository_id: uuid.UUID, job_id: uuid.UUID) -> None:
    try:
        execute_index_job(repository_id, job_id)
    finally:
        repo_lock(repository_id).release()


def _create_job(repository_id: uuid.UUID, mode: str, force: bool) -> IndexJob:
    with session_scope() as session:
        repo = session.get(Repository, repository_id)
        if repo is None:
            raise GitError("repository not found")
        job = IndexJob(
            workspace_id=repo.workspace_id,
            repository_id=repo.id,
            requested_mode=mode,
            force=force,
            status="queued",
            files_seen=0,
            files_updated=0,
            files_skipped=0,
            files_deleted=0,
            chunks_written=0,
        )
        repo.sync_status = "running"
        repo.last_error = None
        session.add(job)
        session.flush()
        session.expunge(job)
        return job


def _execute(repository_id: uuid.UUID, job_id: uuid.UUID, embedder: Embedder) -> None:
    with session_scope() as session:
        repo = session.get(Repository, repository_id)
        job = session.get(IndexJob, job_id)
        if repo is None or job is None:
            raise GitError("index job could not be loaded")
        local_path = repo.local_path
        provider = repo.provider
        previous = repo.latest_indexed_commit
        ignore_globs = list(repo.ignore_globs or [])
        use_defaults = repo.use_default_ignores
        workspace_id = repo.workspace_id
        force = job.force
        requested = job.requested_mode
        job.status = "running"
        job.started_at = _utcnow()
        job.heartbeat_at = job.started_at

    connector = connector_for_repository(provider, local_path)
    head = connector.resolve_commit("HEAD")
    branch = connector.branch_name()
    resolved = _decide_mode(requested, force, previous, head, connector)
    _update_job(job_id, resolved_mode=resolved, from_commit=previous, to_commit=head)

    if resolved == "refresh":
        embedding_error = _embed_pending(repository_id, embedder)
        _finish_success(repository_id, job_id, head, branch, embedding_error, files_deleted=0)
        return

    deletes: list[str] = []
    if resolved == "incremental":
        changes = connector.list_changes(previous or head, head)
        entries = _entries_for_changes(connector, head, changes)
        deletes = _delete_paths_from_changes(changes)
    else:
        entries = [
            entry
            for entry in connector.list_tree(head)
            if entry.object_type == "blob" or entry.mode in {"120000", "160000"}
        ]

    deleted = _delete_paths(repository_id, deletes) if deletes else 0
    matcher = build_matcher(local_path, ignore_globs, use_defaults)
    existing = _load_existing(repository_id)
    keep_paths = {entry.path for entry in entries} if resolved == "full" else set()
    stats = _Stats()
    needs_read: list[TreeEntry] = []

    for entry in entries:
        special = _special_skip(entry)
        path_reason = special or matcher.reason(entry.path)
        if path_reason:
            _write_skipped(
                workspace_id,
                repository_id,
                entry.path,
                head,
                path_reason,
                entry.blob_sha if entry.object_type == "blob" else None,
                job_id,
            )
            stats.seen += 1
            stats.skipped += 1
            continue
        if entry.size is not None and entry.size > MAX_BLOB_BYTES:
            _write_skipped(workspace_id, repository_id, entry.path, head, "too_large", entry.blob_sha, job_id)
            stats.seen += 1
            stats.skipped += 1
            continue
        current = existing.get(entry.path)
        if (
            not force
            and current is not None
            and current.blob_sha == entry.blob_sha
            and current.skipped_reason is None
        ):
            stats.seen += 1
            _touch_job(job_id, stats)
            continue
        needs_read.append(entry)

    for group in _groups(needs_read, 100):
        blobs = connector.read_blobs([entry.blob_sha for entry in group])
        for entry, blob in zip(group, blobs, strict=True):
            stats.seen += 1
            written = _index_blob(
                workspace_id,
                repository_id,
                entry,
                blob,
                head,
                job_id,
            )
            if written is None:
                stats.skipped += 1
            else:
                stats.updated += 1
                stats.chunks += written
            _touch_job(job_id, stats)

    if resolved == "full":
        deleted += _prune(repository_id, keep_paths)
    _update_job(
        job_id,
        files_seen=stats.seen,
        files_updated=stats.updated,
        files_skipped=stats.skipped,
        files_deleted=deleted,
        chunks_written=stats.chunks,
    )
    embedding_error = _embed_pending(repository_id, embedder)
    _finish_success(repository_id, job_id, head, branch, embedding_error, files_deleted=deleted)


def _entries_for_changes(
    connector: LocalGitConnector,
    head: str,
    changes: list[FileChange],
) -> list[TreeEntry]:
    wanted = {change.path for change in changes if change.status != "deleted"}
    if not wanted:
        return []
    return [entry for entry in connector.list_tree(head) if entry.path in wanted]


def _delete_paths_from_changes(changes: list[FileChange]) -> list[str]:
    paths: list[str] = []
    for change in changes:
        if change.status == "deleted":
            paths.append(change.path)
        elif change.status == "renamed" and change.old_path:
            paths.append(change.old_path)
    return paths


def _decide_mode(requested: str, force: bool, previous: str | None, head: str, connector: LocalGitConnector) -> str:
    if force or requested == "full" or not previous:
        return "full"
    if previous == head:
        return "refresh"
    if requested == "incremental" or requested == "auto":
        if connector.commit_exists(previous):
            return "incremental"
    return "full"


def _special_skip(entry: TreeEntry) -> str | None:
    if entry.mode in {"120000", "160000"} or entry.object_type != "blob":
        return "symlink_or_submodule"
    return None


def _load_existing(repository_id: uuid.UUID) -> dict[str, ExistingFile]:
    with session_scope() as session:
        rows = session.execute(
            select(
                IndexedFile.id,
                IndexedFile.path,
                IndexedFile.blob_sha,
                IndexedFile.skipped_reason,
                IndexedFile.embeddings_ready,
            ).where(IndexedFile.repository_id == repository_id)
        ).all()
    return {
        row.path: ExistingFile(row.id, row.blob_sha, row.skipped_reason, row.embeddings_ready)
        for row in rows
    }


def _index_blob(
    workspace_id: uuid.UUID,
    repository_id: uuid.UUID,
    entry: TreeEntry,
    blob: bytes | None,
    head: str,
    job_id: uuid.UUID,
) -> int | None:
    if blob is None:
        _write_skipped(workspace_id, repository_id, entry.path, head, "unreadable", entry.blob_sha, job_id)
        return None
    text, reason = _classify_blob(blob)
    if reason or text is None:
        _write_skipped(workspace_id, repository_id, entry.path, head, reason or "unreadable", entry.blob_sha, job_id)
        return None
    chunks = chunk_source(entry.path, text)
    if len(chunks) > MAX_CHUNKS_PER_FILE:
        _write_skipped(workspace_id, repository_id, entry.path, head, "too_many_chunks", entry.blob_sha, job_id)
        return None
    content_hash = hashlib.sha256(blob).hexdigest()
    language = language_for_path(entry.path)
    with session_scope() as session:
        session.execute(delete(CodeChunk).where(CodeChunk.repository_id == repository_id, CodeChunk.file_path == entry.path))
        session.execute(delete(IndexedFile).where(IndexedFile.repository_id == repository_id, IndexedFile.path == entry.path))
        indexed = IndexedFile(
            workspace_id=workspace_id,
            repository_id=repository_id,
            path=entry.path,
            blob_sha=entry.blob_sha,
            commit_sha=head,
            content_hash=content_hash,
            language=language,
            skipped_reason=None,
            embeddings_ready=False,
        )
        session.add(indexed)
        session.flush()
        for chunk in chunks:
            session.add(
                CodeChunk(
                    workspace_id=workspace_id,
                    repository_id=repository_id,
                    indexed_file_id=indexed.id,
                    commit_sha=head,
                    file_path=entry.path,
                    language=chunk.language or language,
                    symbol_name=chunk.symbol_name,
                    chunk_kind=chunk.chunk_kind,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    content_hash=hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
                    embedding=None,
                )
            )
        session.get(IndexJob, job_id).heartbeat_at = _utcnow()
    return len(chunks)


def _write_skipped(
    workspace_id: uuid.UUID,
    repository_id: uuid.UUID,
    path: str,
    head: str,
    reason: str,
    blob_sha: str | None,
    job_id: uuid.UUID,
) -> None:
    with session_scope() as session:
        session.execute(delete(CodeChunk).where(CodeChunk.repository_id == repository_id, CodeChunk.file_path == path))
        session.execute(delete(IndexedFile).where(IndexedFile.repository_id == repository_id, IndexedFile.path == path))
        session.add(
            IndexedFile(
                workspace_id=workspace_id,
                repository_id=repository_id,
                path=path,
                blob_sha=blob_sha,
                commit_sha=head,
                content_hash=None,
                language=language_for_path(path),
                skipped_reason=reason,
                embeddings_ready=True,
            )
        )
        job = session.get(IndexJob, job_id)
        if job is not None:
            job.heartbeat_at = _utcnow()


def _classify_blob(blob: bytes) -> tuple[str | None, str | None]:
    if b"\x00" in blob[:8192]:
        return None, "binary"
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return None, "binary_encoding"
    if len(text.splitlines()) > MAX_FILE_LINES:
        return None, "too_many_lines"
    return text, None


def _delete_paths(repository_id: uuid.UUID, paths: list[str]) -> int:
    unique = [path for path in dict.fromkeys(paths) if path]
    if not unique:
        return 0
    with session_scope() as session:
        session.execute(delete(CodeChunk).where(CodeChunk.repository_id == repository_id, CodeChunk.file_path.in_(unique)))
        result = session.execute(
            delete(IndexedFile).where(IndexedFile.repository_id == repository_id, IndexedFile.path.in_(unique))
        )
        return result.rowcount or 0


def _prune(repository_id: uuid.UUID, keep_paths: set[str]) -> int:
    with session_scope() as session:
        rows = session.execute(
            select(IndexedFile.id, IndexedFile.path).where(IndexedFile.repository_id == repository_id)
        ).all()
        stale_ids = [row.id for row in rows if row.path not in keep_paths]
    deleted = 0
    for group in _groups(stale_ids, 200):
        with session_scope() as session:
            session.execute(delete(CodeChunk).where(CodeChunk.indexed_file_id.in_(group)))
            result = session.execute(delete(IndexedFile).where(IndexedFile.id.in_(group)))
            deleted += result.rowcount or 0
    return deleted


def _embed_pending(repository_id: uuid.UUID, embedder: Embedder) -> str | None:
    error: str | None = None
    try:
        while True:
            with session_scope() as session:
                rows = session.scalars(
                    select(CodeChunk)
                    .where(CodeChunk.repository_id == repository_id, CodeChunk.embedding.is_(None))
                    .limit(16)
                ).all()
                payload = [(row.id, _embed_input(row)) for row in rows]
            if not payload:
                break
            vectors = embedder.embed([item[1] for item in payload])
            if len(vectors) != len(payload):
                error = "embedding provider returned an unexpected count"
                break
            if any(len(vector) != EMBEDDING_DIM for vector in vectors):
                found = len(vectors[0]) if vectors else 0
                error = f"embedding dimension {found} does not match the schema width {EMBEDDING_DIM}"
                break
            with session_scope() as session:
                for (chunk_id, _), vector in zip(payload, vectors, strict=True):
                    session.execute(update(CodeChunk).where(CodeChunk.id == chunk_id).values(embedding=vector))
    except ModelUnavailable as exc:
        error = redact(str(exc))[:500]
    _refresh_embedding_flags(repository_id)
    return error


def _embed_input(chunk: CodeChunk) -> str:
    header = chunk.file_path
    if chunk.symbol_name:
        header = f"{header} {chunk.symbol_name}"
    return f"{header}\n{chunk.content}"[:8000]


def _refresh_embedding_flags(repository_id: uuid.UUID) -> None:
    with session_scope() as session:
        session.execute(
            text(
                """
                UPDATE indexed_files AS f
                SET embeddings_ready = NOT EXISTS (
                    SELECT 1 FROM code_chunks AS c
                    WHERE c.indexed_file_id = f.id AND c.embedding IS NULL
                )
                WHERE f.repository_id = :repository_id
                """
            ),
            {"repository_id": repository_id},
        )


def _finish_success(
    repository_id: uuid.UUID,
    job_id: uuid.UUID,
    head: str,
    branch: str,
    embedding_error: str | None,
    files_deleted: int,
) -> None:
    with session_scope() as session:
        repo = session.get(Repository, repository_id)
        job = session.get(IndexJob, job_id)
        if repo is None or job is None:
            return
        pending = session.scalar(
            select(CodeChunk.id)
            .where(CodeChunk.repository_id == repository_id, CodeChunk.embedding.is_(None))
            .limit(1)
        )
        repo.latest_indexed_commit = head
        repo.indexed_branch = branch
        repo.default_branch = branch
        repo.last_indexed_at = _utcnow()
        repo.sync_status = "partial" if pending is not None or embedding_error else "indexed"
        repo.last_error = embedding_error
        job.status = "succeeded"
        job.finished_at = repo.last_indexed_at
        job.heartbeat_at = repo.last_indexed_at
        job.to_commit = head
        job.embedding_error = embedding_error
        job.files_deleted = files_deleted
        job.error = None


def _mark_failed(repository_id: uuid.UUID, job_id: uuid.UUID, message: str) -> None:
    with session_scope() as session:
        repo = session.get(Repository, repository_id)
        job = session.get(IndexJob, job_id)
        now = _utcnow()
        if job is not None and job.status != "succeeded":
            job.status = "failed"
            job.error = message
            job.finished_at = now
            job.heartbeat_at = now
        if repo is not None and repo.sync_status == "running":
            repo.sync_status = "error"
            repo.last_error = message


def _latest_running(session, repository_id: uuid.UUID) -> IndexJob | None:
    return session.scalar(
        select(IndexJob)
        .where(IndexJob.repository_id == repository_id, IndexJob.status.in_(("queued", "running")))
        .order_by(IndexJob.created_at.desc())
    )


def _touch_job(job_id: uuid.UUID, stats: "_Stats") -> None:
    _update_job(
        job_id,
        files_seen=stats.seen,
        files_updated=stats.updated,
        files_skipped=stats.skipped,
        chunks_written=stats.chunks,
        heartbeat_at=_utcnow(),
    )


def _update_job(job_id: uuid.UUID, **values) -> None:
    values["heartbeat_at"] = values.get("heartbeat_at") or _utcnow()
    with session_scope() as session:
        session.execute(update(IndexJob).where(IndexJob.id == job_id).values(**values))


def recover_stale_jobs() -> None:
    with session_scope() as session:
        jobs = session.scalars(select(IndexJob).where(IndexJob.status.in_(("queued", "running")))).all()
        now = _utcnow()
        message = "index interrupted; retry the index"
        for job in jobs:
            job.status = "failed"
            job.error = message
            job.finished_at = now
            repo = session.get(Repository, job.repository_id)
            if repo is not None and repo.sync_status == "running":
                repo.sync_status = "error"
                repo.last_error = message


class _Stats:
    def __init__(self) -> None:
        self.seen = 0
        self.updated = 0
        self.skipped = 0
        self.chunks = 0


def _groups(items: list, size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
