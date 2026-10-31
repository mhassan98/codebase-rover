from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.catalog import (
    delete_repository,
    ensure_workspace,
    get_tenant,
    get_workspace,
    job_payload,
    register_repository,
    repository_in_workspace,
    repository_status,
    reset_index,
    seed,
    set_environment_revision,
)
from app.config import get_settings
from app.connectors.base import GitError, ProviderCapabilityError
from app.connectors.registry import connector_for_repository
from app.db import get_db
from app.indexing.service import launch_index
from app.llm.base import HostedModelDisabled
from app.llm.factory import models_payload
from app.models import Environment, IndexJob, Question, Repository, ReviewJob, Workspace
from app.qa.service import answer_question, enforce_chat_provider
from app.review.diffparse import DiffParseError
from app.review.service import publish_review_comments, run_review

router = APIRouter()


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")


class RepositoryCreate(BaseModel):
    path: str = Field(min_length=1, max_length=2000)
    name: str | None = Field(default=None, max_length=200)
    ignore_globs: list[str] = Field(default_factory=list)
    use_default_ignores: bool = True


class IndexRequest(BaseModel):
    mode: str = Field(default="auto", pattern="^(auto|full|incremental)$")
    force: bool = False


class EnvironmentUpdate(BaseModel):
    repository_id: uuid.UUID | None = None
    revision: str | None = Field(default=None, max_length=200)
    clear_revision: bool = False


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    repository_id: uuid.UUID | None = None
    model_preference: str = Field(default="ollama", pattern="^(ollama|hosted)$")
    confirm_send_context: bool = False


class ReviewIn(BaseModel):
    diff: str | None = None
    head_sha: str | None = Field(default=None, max_length=200)
    base_sha: str | None = Field(default=None, max_length=200)
    repository_id: uuid.UUID | None = None
    title: str | None = Field(default=None, max_length=200)
    model_preference: str = Field(default="ollama", pattern="^(ollama|hosted)$")
    confirm_send_context: bool = False
    native_pr_id: str | None = Field(default=None, max_length=200)


class ResetIn(BaseModel):
    confirm: str


@router.get("/health")
def health(session: Session = Depends(get_db)) -> JSONResponse:
    database_ok = True
    try:
        session.execute(select(1))
    except Exception:
        database_ok = False
    return JSONResponse(
        status_code=200 if database_ok else 503,
        content={
            "status": "ok" if database_ok else "degraded",
            "database": database_ok,
            "ollama": _ollama_reachable() if database_ok else False,
            "version": "0.1.0",
        },
    )


@router.get("/models")
def models() -> dict:
    return models_payload(get_settings())


@router.get("/tenant")
def tenant(session: Session = Depends(get_db)) -> dict:
    record = get_tenant(session)
    return {"id": record.id, "name": record.name, "slug": record.slug}


@router.get("/workspaces")
def list_workspaces(session: Session = Depends(get_db)) -> list[dict]:
    record = get_tenant(session)
    rows = session.scalars(select(Workspace).where(Workspace.tenant_id == record.id).order_by(Workspace.slug)).all()
    return [_workspace_payload(session, row) for row in rows]


@router.post("/workspaces", status_code=201)
def create_workspace(body: WorkspaceCreate, session: Session = Depends(get_db)) -> dict:
    record = get_tenant(session)
    existing = get_workspace(session, body.slug)
    if existing is not None and existing.tenant_id == record.id:
        raise HTTPException(status_code=409, detail="workspace already exists")
    workspace = ensure_workspace(session, record, body.name, body.slug)
    session.commit()
    return _workspace_payload(session, workspace)


@router.get("/workspaces/{slug}")
def workspace_detail(slug: str, session: Session = Depends(get_db)) -> dict:
    return _workspace_payload(session, _workspace_or_404(session, slug))


@router.get("/workspaces/{slug}/environments")
def list_environments(slug: str, session: Session = Depends(get_db)) -> list[dict]:
    workspace = _workspace_or_404(session, slug)
    rows = session.scalars(
        select(Environment).where(Environment.workspace_id == workspace.id).order_by(Environment.sort_order)
    ).all()
    return [_environment_payload(row) for row in rows]


@router.patch("/workspaces/{slug}/environments/{env_slug}")
def patch_environment(
    slug: str,
    env_slug: str,
    body: EnvironmentUpdate,
    session: Session = Depends(get_db),
) -> dict:
    workspace = _workspace_or_404(session, slug)
    environment = session.scalar(
        select(Environment).where(Environment.workspace_id == workspace.id, Environment.slug == env_slug)
    )
    if environment is None:
        raise HTTPException(status_code=404, detail="environment not found")
    try:
        updated = set_environment_revision(
            session,
            workspace,
            environment,
            body.repository_id,
            body.revision,
            body.clear_revision,
        )
    except GitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _environment_payload(updated)


@router.get("/workspaces/{slug}/repositories")
def list_repositories(slug: str, session: Session = Depends(get_db)) -> list[dict]:
    workspace = _workspace_or_404(session, slug)
    rows = session.scalars(select(Repository).where(Repository.workspace_id == workspace.id).order_by(Repository.name)).all()
    return [_repository_summary(row) for row in rows]


@router.post("/workspaces/{slug}/repositories", status_code=201)
def create_repository(slug: str, body: RepositoryCreate, session: Session = Depends(get_db)) -> dict:
    workspace = _workspace_or_404(session, slug)
    try:
        repository = register_repository(
            session,
            workspace,
            body.path,
            get_settings(),
            name=body.name,
            ignore_globs=body.ignore_globs,
            use_default_ignores=body.use_default_ignores,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="repository is already connected to this workspace") from exc
    except GitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _repository_summary(repository)


@router.get("/workspaces/{slug}/repositories/{repository_id}")
def get_repository(slug: str, repository_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    repository = _repository_or_404(session, slug, repository_id)
    return repository_status(session, repository)


@router.delete("/workspaces/{slug}/repositories/{repository_id}", status_code=204)
def remove_repository(slug: str, repository_id: uuid.UUID, session: Session = Depends(get_db)) -> None:
    repository = _repository_or_404(session, slug, repository_id)
    from app.indexing.service import repo_lock

    lock = repo_lock(repository.id)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="repository is indexing")
    try:
        delete_repository(session, repository)
    finally:
        lock.release()


@router.post("/workspaces/{slug}/repositories/{repository_id}/index", status_code=202)
def index_repository(
    slug: str,
    repository_id: uuid.UUID,
    body: IndexRequest,
    session: Session = Depends(get_db),
) -> dict:
    repository = _repository_or_404(session, slug, repository_id)
    try:
        job = launch_index(repository.id, body.mode, body.force)
    except GitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job_payload(job)


@router.get("/workspaces/{slug}/repositories/{repository_id}/status")
def repository_sync(slug: str, repository_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    repository = _repository_or_404(session, slug, repository_id)
    return repository_status(session, repository)


@router.get("/workspaces/{slug}/index-jobs/{job_id}")
def get_job(slug: str, job_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    workspace = _workspace_or_404(session, slug)
    job = session.get(IndexJob, job_id)
    if job is None or job.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail="index job not found")
    return job_payload(job)


@router.post("/workspaces/{slug}/questions", status_code=201)
def ask(slug: str, body: QuestionIn, session: Session = Depends(get_db)) -> dict:
    settings = get_settings()
    try:
        provider = enforce_chat_provider(settings, body.model_preference, body.confirm_send_context)
    except HostedModelDisabled as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    workspace = _workspace_or_404(session, slug)
    if body.repository_id is not None and repository_in_workspace(session, workspace, body.repository_id) is None:
        raise HTTPException(status_code=404, detail="repository is not in this workspace")
    row = answer_question(
        session,
        workspace,
        body.question,
        body.repository_id,
        provider,
        settings,
    )
    return _question_out(row)


@router.get("/workspaces/{slug}/questions/{question_id}")
def get_question(slug: str, question_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    workspace = _workspace_or_404(session, slug)
    row = session.get(Question, question_id)
    if row is None or row.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail="question not found")
    return _question_out(row)


@router.post("/workspaces/{slug}/reviews", status_code=201)
def create_review(slug: str, body: ReviewIn, session: Session = Depends(get_db)) -> dict:
    settings = get_settings()
    try:
        provider = enforce_chat_provider(settings, body.model_preference, body.confirm_send_context)
    except HostedModelDisabled as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    workspace = _workspace_or_404(session, slug)
    diff_text = body.diff
    head_sha = body.head_sha
    base_sha = body.base_sha
    title = body.title
    if body.native_pr_id:
        if body.repository_id is None:
            raise HTTPException(status_code=400, detail="repository_id is required to fetch a pull request")
        repository = repository_in_workspace(session, workspace, body.repository_id)
        if repository is None:
            raise HTTPException(status_code=404, detail="repository is not in this workspace")
        try:
            snapshot = connector_for_repository(repository.provider, repository.local_path).fetch_pull_request(
                body.native_pr_id
            )
        except ProviderCapabilityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        diff_text = snapshot.diff
        head_sha = snapshot.head_sha
        base_sha = snapshot.base_sha
        title = title or snapshot.title
    try:
        row = run_review(
            session,
            workspace,
            diff_text=diff_text,
            head_sha=head_sha,
            base_sha=base_sha,
            repository_id=body.repository_id,
            title=title,
            provider=provider,
            settings=settings,
        )
    except (GitError, DiffParseError, ProviderCapabilityError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _review_out(row)


@router.get("/workspaces/{slug}/reviews/{review_id}")
def get_review(slug: str, review_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    workspace = _workspace_or_404(session, slug)
    row = session.get(ReviewJob, review_id)
    if row is None or row.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail="review not found")
    return _review_out(row)


@router.post("/workspaces/{slug}/reviews/{review_id}/publish")
def publish_review(slug: str, review_id: uuid.UUID, session: Session = Depends(get_db)) -> dict:
    workspace = _workspace_or_404(session, slug)
    row = session.get(ReviewJob, review_id)
    if row is None or row.workspace_id != workspace.id:
        raise HTTPException(status_code=404, detail="review not found")
    try:
        publish_review_comments(session, workspace, row)
    except ProviderCapabilityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"published": True, "review_id": row.id}


@router.post("/maintenance/reset-index")
def reset_local_index(body: ResetIn, session: Session = Depends(get_db)) -> dict:
    if body.confirm != "reset-index":
        raise HTTPException(status_code=400, detail='confirm must be "reset-index"')
    reset_index(session)
    seed(session)
    session.commit()
    return {"reset": True}


def _workspace_or_404(session: Session, slug: str) -> Workspace:
    workspace = get_workspace(session, slug)
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return workspace


def _repository_or_404(session: Session, slug: str, repository_id: uuid.UUID) -> Repository:
    workspace = _workspace_or_404(session, slug)
    repository = repository_in_workspace(session, workspace, repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="repository not found")
    return repository


def _workspace_payload(session: Session, workspace: Workspace) -> dict:
    repositories = session.scalars(
        select(Repository).where(Repository.workspace_id == workspace.id).order_by(Repository.name)
    ).all()
    environments = session.scalars(
        select(Environment).where(Environment.workspace_id == workspace.id).order_by(Environment.sort_order)
    ).all()
    return {
        "id": workspace.id,
        "name": workspace.name,
        "slug": workspace.slug,
        "tenant_id": workspace.tenant_id,
        "repositories": [_repository_summary(row) for row in repositories],
        "environments": [_environment_payload(row) for row in environments],
    }


def _repository_summary(repository: Repository) -> dict:
    return {
        "id": repository.id,
        "name": repository.name,
        "provider": repository.provider,
        "native_id": repository.native_id,
        "native_url": repository.native_url,
        "remote_url": repository.remote_url,
        "local_path": repository.local_path,
        "default_branch": repository.default_branch,
        "indexed_branch": repository.indexed_branch,
        "latest_indexed_commit": repository.latest_indexed_commit,
        "sync_status": repository.sync_status,
        "last_indexed_at": repository.last_indexed_at,
        "last_error": repository.last_error,
        "ignore_globs": repository.ignore_globs,
        "use_default_ignores": repository.use_default_ignores,
    }


def _environment_payload(environment: Environment) -> dict:
    return {
        "id": environment.id,
        "name": environment.name,
        "slug": environment.slug,
        "sort_order": environment.sort_order,
        "repository_id": environment.repository_id,
        "revision": environment.revision,
        "revision_source": environment.revision_source,
    }


def _question_out(row: Question) -> dict:
    payload = row.citations_json or {}
    return {
        "id": row.id,
        "workspace_id": row.workspace_id,
        "repository_id": row.repository_id,
        "question": row.question_text,
        "answer": row.answer_text,
        "sufficient": row.sufficient,
        "synthesis": row.synthesis,
        "citations": payload.get("citations", []),
        "related_excerpts": payload.get("related_excerpts", []),
        "model_provider": row.model_provider,
        "model_name": row.model_name,
        "created_at": row.created_at,
    }


def _review_out(row: ReviewJob) -> dict:
    payload = row.result_json or {}
    return {
        "id": row.id,
        "workspace_id": row.workspace_id,
        "repository_id": row.repository_id,
        "title": row.title,
        "head_sha": row.head_sha,
        "submitted_head": row.submitted_head,
        "base_sha": row.base_sha,
        "status": row.status,
        "synthesis": row.synthesis,
        "verified_against_git": row.verified_against_git,
        "summary": row.summary or payload.get("summary", ""),
        "findings": payload.get("findings", []),
        "assumptions": payload.get("assumptions", []),
        "limitations": payload.get("limitations", []),
        "rejected": payload.get("rejected", []),
        "diff_files": payload.get("diff_files", []),
        "context_excerpts": payload.get("context_excerpts", []),
        "model_provider": row.model_provider,
        "model_name": row.model_name,
        "error": row.error,
        "created_at": row.created_at,
        "finished_at": row.finished_at,
    }


def _ollama_reachable() -> bool:
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code < 500
