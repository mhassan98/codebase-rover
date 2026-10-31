from __future__ import annotations

import argparse
import json
import sys

from app.catalog import get_workspace, register_repository, reset_index, repository_in_workspace, seed
from app.config import get_settings
from app.db import init_db, session_scope
from app.indexing.service import run_index_sync
from app.logging_config import configure_logging
from app.models import Repository
from app.qa.service import answer_question, enforce_chat_provider
from app.review.service import run_review
from sqlalchemy import select


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="codebase-rover")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the local API and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--reload", action="store_true")

    sub.add_parser("seed", help="create the default tenant and workspaces")

    reset = sub.add_parser("reset-index", help="delete indexed chunks and jobs")
    reset.add_argument("--yes", action="store_true")

    index = sub.add_parser("index", help="connect a local git repository and index it")
    index.add_argument("--workspace", required=True)
    index.add_argument("--path", required=True)
    index.add_argument("--name")
    index.add_argument("--full", action="store_true")
    index.add_argument("--force", action="store_true")

    ask = sub.add_parser("ask", help="ask a question in a workspace")
    ask.add_argument("--workspace", required=True)
    ask.add_argument("question")

    review = sub.add_parser("review", help="review a unified diff")
    review.add_argument("--workspace", required=True)
    review.add_argument("--diff", required=True, help="path to a diff file, or - for stdin")
    review.add_argument("--head", required=True)
    review.add_argument("--base")
    review.add_argument("--repository")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
        return
    init_db()
    if args.command == "seed":
        with session_scope() as session:
            seed(session)
        print("seeded tenant local with workspaces fastcrib and pos")
        return
    if args.command == "reset-index":
        if not args.yes:
            print('refusing to reset without --yes', file=sys.stderr)
            raise SystemExit(2)
        with session_scope() as session:
            reset_index(session)
        print("index reset")
        return
    if args.command == "index":
        _cmd_index(args)
        return
    if args.command == "ask":
        _cmd_ask(args)
        return
    if args.command == "review":
        _cmd_review(args)


def _cmd_index(args) -> None:
    settings = get_settings()
    with session_scope() as session:
        seed(session)
        workspace = get_workspace(session, args.workspace)
        if workspace is None:
            raise SystemExit(f"unknown workspace: {args.workspace}")
        from app.connectors.local_git import LocalGitConnector

        identity = LocalGitConnector(args.path).identify()
        repository = session.scalar(
            select(Repository).where(
                Repository.workspace_id == workspace.id,
                Repository.native_id == identity.native_id,
            )
        )
        if repository is None:
            repository = register_repository(session, workspace, args.path, settings, name=args.name)
        repository_id = repository.id
    mode = "full" if args.full or args.force else "auto"
    job = run_index_sync(repository_id, mode, args.force)
    print(
        json.dumps(
            {
                "job_id": str(job.id),
                "status": job.status,
                "resolved_mode": job.resolved_mode,
                "to_commit": job.to_commit,
                "files_updated": job.files_updated,
                "chunks_written": job.chunks_written,
                "embedding_error": job.embedding_error,
                "error": job.error,
            },
            indent=2,
        )
    )
    if job.status != "succeeded":
        raise SystemExit(1)


def _cmd_ask(args) -> None:
    settings = get_settings()
    with session_scope() as session:
        seed(session)
        workspace = get_workspace(session, args.workspace)
        if workspace is None:
            raise SystemExit(f"unknown workspace: {args.workspace}")
        provider = enforce_chat_provider(settings, "ollama", False)
        row = answer_question(session, workspace, args.question, None, provider, settings)
        print(row.answer_text)
        for citation in row.citations_json.get("citations") or row.citations_json.get("related_excerpts") or []:
            print(
                f"- {citation['repository_name']} {citation['file_path']}:"
                f"{citation['start_line']}-{citation['end_line']} @{citation['commit_sha'][:12]}"
            )


def _cmd_review(args) -> None:
    settings = get_settings()
    if args.diff == "-":
        diff_text = sys.stdin.read()
    else:
        with open(args.diff, encoding="utf-8") as handle:
            diff_text = handle.read()
    with session_scope() as session:
        seed(session)
        workspace = get_workspace(session, args.workspace)
        if workspace is None:
            raise SystemExit(f"unknown workspace: {args.workspace}")
        repository_id = None
        if args.repository:
            repository = repository_in_workspace(session, workspace, _parse_uuid(args.repository))
            if repository is None:
                raise SystemExit("repository is not in this workspace")
            repository_id = repository.id
        provider = enforce_chat_provider(settings, "ollama", False)
        row = run_review(
            session,
            workspace,
            diff_text=diff_text,
            head_sha=args.head,
            base_sha=args.base,
            repository_id=repository_id,
            title=None,
            provider=provider,
            settings=settings,
        )
    print(row.summary)
    for finding in row.result_json.get("findings") or []:
        print(
            f"[{finding['severity']}/{finding['confidence']}] {finding['file_path']}:"
            f"{finding['start_line']}-{finding['end_line']} {finding['title']}"
        )
    for assumption in row.result_json.get("assumptions") or []:
        print(f"assumption: {assumption}")


def _parse_uuid(value: str):
    import uuid

    return uuid.UUID(value)


if __name__ == "__main__":
    main()
