from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.config import Settings
from app.connectors.base import GitError, ProviderCapabilityError
from app.connectors.registry import connector_for_repository, open_local_git
from app.models import (
    CodeChunk,
    Environment,
    IndexJob,
    IndexedFile,
    Question,
    Repository,
    ReviewJob,
    SourceConnection,
    Tenant,
    Workspace,
)

ENVIRONMENTS = (
    ("development", "Development"),
    ("test", "Test"),
    ("stage", "Stage"),
    ("production", "Production"),
)
DEFAULT_TENANT = ("local", "Local")
DEFAULT_WORKSPACES = (("fastcrib", "FastCrib"), ("pos", "POS"))


def seed(session: Session) -> Tenant:
    tenant = session.scalar(select(Tenant).where(Tenant.slug == DEFAULT_TENANT[0]))
    if tenant is None:
        tenant = Tenant(slug=DEFAULT_TENANT[0], name=DEFAULT_TENANT[1])
        session.add(tenant)
        session.flush()
    for slug, name in DEFAULT_WORKSPACES:
        ensure_workspace(session, tenant, name, slug)
    return tenant


def ensure_workspace(session: Session, tenant: Tenant, name: str, slug: str) -> Workspace:
    workspace = session.scalar(
        select(Workspace).where(Workspace.tenant_id == tenant.id, Workspace.slug == slug)
    )
    if workspace is None:
        workspace = Workspace(tenant_id=tenant.id, name=name, slug=slug)
        session.add(workspace)
        session.flush()
        for order, (env_slug, env_name) in enumerate(ENVIRONMENTS, start=1):
            session.add(
                Environment(
                    workspace_id=workspace.id,
                    slug=env_slug,
                    name=env_name,
                    sort_order=order,
                    repository_id=None,
                    revision=None,
                    revision_source=None,
                )
            )
        session.flush()
    return workspace


def get_tenant(session: Session) -> Tenant:
    tenant = session.scalar(select(Tenant).where(Tenant.slug == DEFAULT_TENANT[0]))
    if tenant is None:
        tenant = seed(session)
        session.commit()
    return tenant


def get_workspace(session: Session, slug: str) -> Workspace | None:
    tenant = get_tenant(session)
    return session.scalar(select(Workspace).where(Workspace.tenant_id == tenant.id, Workspace.slug == slug))


def register_repository(
    session: Session,
    workspace: Workspace,
    path: str,
    settings: Settings,
    name: str | None = None,
    ignore_globs: list[str] | None = None,
    use_default_ignores: bool = True,
) -> Repository:
    connector = open_local_git(path, settings)
    identity = connector.identify()
    existing = session.scalar(
        select(Repository).where(
            Repository.workspace_id == workspace.id,
            Repository.native_id == identity.native_id,
        )
    )
    if existing is not None:
        raise FileExistsError(identity.native_id)
    globs = list(ignore_globs or [])
    connection = SourceConnection(
        workspace_id=workspace.id,
        provider=identity.provider,
        display_name=name or identity.display_name,
        native_id=identity.native_id,
        native_url=identity.native_url,
        config_json={
            "path": identity.native_id,
            "ignore_globs": globs,
            "use_default_ignores": use_default_ignores,
        },
    )
    session.add(connection)
    session.flush()
    repository = Repository(
        workspace_id=workspace.id,
        connection_id=connection.id,
        name=name or identity.display_name,
        provider=identity.provider,
        native_id=identity.native_id,
        native_url=identity.native_url,
        remote_url=identity.remote_url,
        local_path=identity.native_id,
        default_branch=identity.default_branch,
        indexed_branch=None,
        latest_indexed_commit=None,
        sync_status="never",
        last_indexed_at=None,
        last_error=None,
        ignore_globs=globs,
        use_default_ignores=use_default_ignores,
    )
    session.add(repository)
    session.commit()
    return repository


def delete_repository(session: Session, repository: Repository) -> None:
    session.execute(delete(CodeChunk).where(CodeChunk.repository_id == repository.id))
    session.execute(delete(IndexedFile).where(IndexedFile.repository_id == repository.id))
    session.execute(delete(IndexJob).where(IndexJob.repository_id == repository.id))
    session.execute(
        update(Environment)
        .where(Environment.repository_id == repository.id)
        .values(repository_id=None, revision=None, revision_source=None)
    )
    session.execute(update(Question).where(Question.repository_id == repository.id).values(repository_id=None))
    session.execute(update(ReviewJob).where(ReviewJob.repository_id == repository.id).values(repository_id=None))
    connection_id = repository.connection_id
    session.delete(repository)
    session.flush()
    session.execute(delete(SourceConnection).where(SourceConnection.id == connection_id))
    session.commit()


def reset_index(session: Session) -> None:
    session.execute(delete(CodeChunk))
    session.execute(delete(IndexedFile))
    session.execute(delete(IndexJob))
    session.execute(
        update(Repository).values(
            latest_indexed_commit=None,
            indexed_branch=None,
            sync_status="never",
            last_indexed_at=None,
            last_error=None,
        )
    )
    session.commit()


def repository_in_workspace(session: Session, workspace: Workspace, repository_id: uuid.UUID) -> Repository | None:
    return session.scalar(
        select(Repository).where(Repository.id == repository_id, Repository.workspace_id == workspace.id)
    )


def set_environment_revision(
    session: Session,
    workspace: Workspace,
    environment: Environment,
    repository_id: uuid.UUID | None,
    revision: str | None,
    clear_revision: bool,
) -> Environment:
    if clear_revision or revision is None:
        environment.repository_id = None
        environment.revision = None
        environment.revision_source = None
        session.commit()
        return environment
    if repository_id is None:
        raise GitError("repository_id is required when setting an environment revision")
    repository = repository_in_workspace(session, workspace, repository_id)
    if repository is None:
        raise GitError("repository is not in this workspace")
    connector = connector_for_repository(repository.provider, repository.local_path)
    environment.repository_id = repository.id
    environment.revision = connector.resolve_commit(revision)
    environment.revision_source = repository.provider
    session.commit()
    return environment


def repository_status(session: Session, repository: Repository) -> dict:
    job = session.scalar(
        select(IndexJob).where(IndexJob.repository_id == repository.id).order_by(IndexJob.created_at.desc())
    )
    file_count = session.scalar(
        select(func.count()).select_from(IndexedFile).where(
            IndexedFile.repository_id == repository.id,
            IndexedFile.skipped_reason.is_(None),
        )
    )
    chunk_count = session.scalar(
        select(func.count()).select_from(CodeChunk).where(CodeChunk.repository_id == repository.id)
    )
    embedded = session.scalar(
        select(func.count()).select_from(CodeChunk).where(
            CodeChunk.repository_id == repository.id,
            CodeChunk.embedding.is_not(None),
        )
    )
    head = None
    branch = repository.default_branch
    git_error = None
    try:
        connector = connector_for_repository(repository.provider, repository.local_path)
        head = connector.resolve_commit("HEAD")
        branch = connector.branch_name()
    except (GitError, ProviderCapabilityError) as exc:
        git_error = str(exc)[:300]
    running = job is not None and job.status in {"queued", "running"}
    if running:
        computed = "running"
    elif git_error:
        computed = "error"
    elif repository.latest_indexed_commit is None:
        computed = "error" if repository.sync_status == "error" else "never"
    elif head and repository.latest_indexed_commit != head:
        computed = "stale"
    elif repository.sync_status == "partial" or (chunk_count and embedded != chunk_count):
        computed = "partial"
    else:
        computed = "indexed"
    coverage = 0.0 if not chunk_count else (embedded or 0) / chunk_count
    return {
        "repository_id": repository.id,
        "name": repository.name,
        "workspace_id": repository.workspace_id,
        "provider": repository.provider,
        "native_id": repository.native_id,
        "native_url": repository.native_url,
        "remote_url": repository.remote_url,
        "branch": branch,
        "head_commit": head,
        "latest_indexed_commit": repository.latest_indexed_commit,
        "sync_status": computed,
        "stored_status": repository.sync_status,
        "git_reachable": git_error is None,
        "git_error": git_error,
        "indexed_file_count": file_count or 0,
        "chunk_count": chunk_count or 0,
        "embedding_coverage": coverage,
        "last_indexed_at": repository.last_indexed_at,
        "last_error": repository.last_error,
        "last_job": None if job is None else job_payload(job),
    }


def job_payload(job: IndexJob) -> dict:
    return {
        "id": job.id,
        "repository_id": job.repository_id,
        "requested_mode": job.requested_mode,
        "resolved_mode": job.resolved_mode,
        "force": job.force,
        "status": job.status,
        "from_commit": job.from_commit,
        "to_commit": job.to_commit,
        "files_seen": job.files_seen,
        "files_updated": job.files_updated,
        "files_skipped": job.files_skipped,
        "files_deleted": job.files_deleted,
        "chunks_written": job.chunks_written,
        "error": job.error,
        "embedding_error": job.embedding_error,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "heartbeat_at": job.heartbeat_at,
    }


def apply_config_file(session: Session, settings: Settings) -> int:
    import logging
    from pathlib import Path

    import yaml

    from app.redact import redact

    log = logging.getLogger(__name__)
    path = Path(settings.rover_config) if settings.rover_config else Path("rover.yaml")
    if not path.is_file():
        return 0
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tenant = get_tenant(session)
    added = 0
    for item in data.get("workspaces") or []:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        name = str(item.get("name") or slug).strip()
        if not slug:
            continue
        workspace = ensure_workspace(session, tenant, name or slug, slug)
        for repo in item.get("repositories") or []:
            if not isinstance(repo, dict) or not repo.get("path"):
                continue
            try:
                register_repository(
                    session,
                    workspace,
                    str(repo["path"]),
                    settings,
                    name=repo.get("name"),
                    ignore_globs=list(repo.get("ignore_globs") or []),
                    use_default_ignores=bool(repo.get("use_default_ignores", True)),
                )
                added += 1
            except FileExistsError:
                continue
            except (GitError, OSError) as exc:
                log.warning("skipped configured repository: %s", redact(str(exc))[:300])
    session.commit()
    return added


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
