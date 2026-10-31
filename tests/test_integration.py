from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from sqlalchemy import select

from app.db import session_scope
from app.indexing.limits import EMBEDDING_DIM
from app.indexing.service import join_index_threads, run_index_sync
from app.models import CodeChunk, Workspace
from app.qa.service import INSUFFICIENT_EVIDENCE, MODEL_UNAVAILABLE
from app.retrieval.search import search_chunks
from tests.gitutil import git, init_repo, write_commit


class HashEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            block = hashlib.sha256(text.encode()).digest()
            values: list[float] = []
            while len(values) < EMBEDDING_DIM:
                for byte in block:
                    values.append((byte - 128) / 128)
                    if len(values) == EMBEDDING_DIM:
                        break
                block = hashlib.sha256(block).digest()
            vectors.append(values)
        return vectors


def _connect(api, slug: str, path: Path, name: str, ignore_globs: list[str] | None = None) -> dict:
    response = api.post(
        f"/api/workspaces/{slug}/repositories",
        json={"path": str(path), "name": name, "ignore_globs": ignore_globs or []},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _index(api, slug: str, repository_id: str) -> None:
    response = api.post(
        f"/api/workspaces/{slug}/repositories/{repository_id}/index",
        json={"mode": "full", "force": True},
    )
    assert response.status_code == 202, response.text
    join_index_threads(60)
    status = api.get(f"/api/workspaces/{slug}/repositories/{repository_id}/status")
    assert status.status_code == 200, status.text
    body = status.json()
    assert body["sync_status"] in {"indexed", "partial"}, body
    assert body["latest_indexed_commit"]
    assert body["chunk_count"] > 0


def test_seeded_workspaces_are_separate(api) -> None:
    response = api.get("/api/workspaces")
    assert response.status_code == 200
    payload = {item["slug"]: item for item in response.json()}
    assert set(payload) >= {"fastcrib", "pos"}
    assert payload["fastcrib"]["name"] == "FastCrib"
    assert payload["pos"]["name"] == "POS"
    assert [item["slug"] for item in payload["fastcrib"]["environments"]] == [
        "development",
        "test",
        "stage",
        "production",
    ]
    assert all(item["revision"] is None for item in payload["fastcrib"]["environments"])
    health = api.get("/api/health")
    assert health.status_code == 200
    assert health.json()["database"] is True
    models = api.get("/api/models")
    assert models.json()["hosted"]["enabled"] is False


def test_hosted_question_does_not_call_a_model(api, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("model client was created")

    monkeypatch.setattr("app.llm.ollama.httpx.Client", boom)
    monkeypatch.setattr("app.llm.hosted.httpx.Client", boom)
    response = api.post(
        "/api/workspaces/fastcrib/questions",
        json={
            "question": "How does checkout calculate tax?",
            "model_preference": "hosted",
            "confirm_send_context": True,
        },
    )
    assert response.status_code == 400
    assert "disabled" in response.json()["detail"].lower()


def test_index_is_workspace_scoped_and_citations_use_the_indexed_revision(api, tmp_path: Path) -> None:
    fastcrib = tmp_path / "fastcrib"
    pos = tmp_path / "pos"
    init_repo(fastcrib)
    init_repo(pos)
    first = write_commit(
        fastcrib,
        {
            "src/tax.py": "def tax(amount):\n    return f'FASTCRIB_TAX_MARKER {amount}'\n",
            "vendor/lib.py": "VENDOR_ONLY_MARKER = True\n",
            "secrets/keys.py": "SECRET_MARKER = 'nope'\n",
        },
        "init fastcrib",
    )
    write_commit(pos, {"src/tender.py": "def tender():\n    return 'POS_TENDER_MARKER'\n"}, "init pos")
    fast_repo = _connect(api, "fastcrib", fastcrib, "fastcrib", ["secrets/**"])
    pos_repo = _connect(api, "pos", pos, "pos")
    _index(api, "fastcrib", fast_repo["id"])
    _index(api, "pos", pos_repo["id"])

    fast_answer = api.post(
        "/api/workspaces/fastcrib/questions",
        json={"question": "Where is FASTCRIB_TAX_MARKER returned?"},
    )
    assert fast_answer.status_code == 201, fast_answer.text
    fast_body = fast_answer.json()
    assert fast_body["answer"] in {MODEL_UNAVAILABLE, INSUFFICIENT_EVIDENCE} or fast_body["sufficient"] is False
    excerpts = fast_body["related_excerpts"] or fast_body["citations"]
    assert any("FASTCRIB_TAX_MARKER" in item["text"] and item["commit_sha"] == first for item in excerpts)
    assert all("POS_TENDER_MARKER" not in item["text"] for item in excerpts)
    assert all("VENDOR_ONLY_MARKER" not in item["text"] for item in excerpts)
    assert all("SECRET_MARKER" not in item["text"] for item in excerpts)

    pos_answer = api.post(
        "/api/workspaces/pos/questions",
        json={"question": "Where is POS_TENDER_MARKER returned?"},
    )
    pos_excerpts = pos_answer.json()["related_excerpts"] or pos_answer.json()["citations"]
    assert any("POS_TENDER_MARKER" in item["text"] for item in pos_excerpts)
    assert all("FASTCRIB_TAX_MARKER" not in item["text"] for item in pos_excerpts)

    write_commit(fastcrib, {"src/tax.py": "def tax(amount):\n    return 'REWRITTEN_MARKER'\n"}, "rewrite")
    stale = api.post(
        "/api/workspaces/fastcrib/questions",
        json={"question": "FASTCRIB_TAX_MARKER"},
    )
    stale_excerpts = stale.json()["related_excerpts"] or stale.json()["citations"]
    assert any(item["text"].find("FASTCRIB_TAX_MARKER") >= 0 and "REWRITTEN_MARKER" not in item["text"] for item in stale_excerpts)

    missing = api.post("/api/workspaces/fastcrib/repositories", json={"path": str(tmp_path / "missing")})
    assert missing.status_code == 400


def test_incremental_index_drops_deleted_files(api, tmp_path: Path) -> None:
    repo_path = tmp_path / "fastcrib"
    init_repo(repo_path)
    write_commit(
        repo_path,
        {
            "src/keep.py": "def keep():\n    return 'KEEP_MARKER'\n",
            "src/drop.py": "def drop():\n    return 'DELETE_ME_MARKER'\n",
        },
        "init",
    )
    repo = _connect(api, "fastcrib", repo_path, "fastcrib")
    _index(api, "fastcrib", repo["id"])
    (repo_path / "src" / "drop.py").unlink()
    git(repo_path, "add", "-A")
    git(repo_path, "commit", "-m", "delete drop")
    response = api.post(
        f"/api/workspaces/fastcrib/repositories/{repo['id']}/index",
        json={"mode": "incremental", "force": False},
    )
    assert response.status_code == 202, response.text
    join_index_threads(60)
    status = api.get(f"/api/workspaces/fastcrib/repositories/{repo['id']}/status")
    assert status.json()["last_job"]["resolved_mode"] == "incremental"
    gone = api.post("/api/workspaces/fastcrib/questions", json={"question": "DELETE_ME_MARKER"})
    excerpts = gone.json()["related_excerpts"] + gone.json()["citations"]
    assert all("DELETE_ME_MARKER" not in item["text"] for item in excerpts)
    kept = api.post("/api/workspaces/fastcrib/questions", json={"question": "KEEP_MARKER"})
    kept_excerpts = kept.json()["related_excerpts"] or kept.json()["citations"]
    assert any("KEEP_MARKER" in item["text"] for item in kept_excerpts)


def test_review_reads_head_and_local_git_cannot_publish(api, tmp_path: Path) -> None:
    repo_path = tmp_path / "fastcrib"
    init_repo(repo_path)
    base = write_commit(repo_path, {"src/tax.py": "def tax(amount):\n    return amount\n"}, "base")
    (repo_path / "src" / "tax.py").write_text(
        "def tax(amount):\n    if amount < 0:\n        raise ValueError('negative amount')\n    return amount\n"
    )
    git(repo_path, "add", "-A")
    git(repo_path, "commit", "-m", "guard")
    head = git(repo_path, "rev-parse", "HEAD").stdout.strip()
    diff = git(repo_path, "diff", "--find-renames", base, head).stdout
    repo = _connect(api, "fastcrib", repo_path, "fastcrib")
    _index(api, "fastcrib", repo["id"])
    response = api.post(
        "/api/workspaces/fastcrib/reviews",
        json={
            "title": "Reject negative tax",
            "repository_id": repo["id"],
            "head_sha": head,
            "base_sha": base,
            "diff": diff,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["verified_against_git"] is True
    assert body["head_sha"] == head
    assert body["findings"] == []
    assert body["synthesis"] == "unavailable"
    assert any(item["path"] == "src/tax.py" for item in body["diff_files"])
    assert any("negative amount" in item["text"] for item in body["context_excerpts"])
    missing_sha = api.post(
        "/api/workspaces/fastcrib/reviews",
        json={"repository_id": repo["id"], "head_sha": "deadbeef", "diff": diff},
    )
    assert missing_sha.status_code == 400
    native = api.post(
        "/api/workspaces/fastcrib/reviews",
        json={"repository_id": repo["id"], "native_pr_id": "15", "head_sha": head},
    )
    assert native.status_code == 400
    assert "pull request" in native.json()["detail"].lower()
    published = api.post(f"/api/workspaces/fastcrib/reviews/{body['id']}/publish")
    assert published.status_code == 400
    assert "publish" in published.json()["detail"].lower()

    environment = api.patch(
        "/api/workspaces/fastcrib/environments/production",
        json={"repository_id": repo["id"], "revision": head},
    )
    assert environment.status_code == 200, environment.text
    assert environment.json()["revision"] == head
    assert environment.json()["revision_source"] == "local_git"


def test_vector_search_stays_inside_the_workspace(api, tmp_path: Path) -> None:
    fastcrib = tmp_path / "fastcrib"
    pos = tmp_path / "pos"
    init_repo(fastcrib)
    init_repo(pos)
    write_commit(fastcrib, {"src/a.py": "def alpha():\n    return 'FAST_VECTOR_MARKER'\n"}, "a")
    write_commit(pos, {"src/b.py": "def beta():\n    return 'POS_VECTOR_MARKER'\n"}, "b")
    fast_repo = _connect(api, "fastcrib", fastcrib, "fastcrib")
    pos_repo = _connect(api, "pos", pos, "pos")
    fast_job = run_index_sync(uuid.UUID(fast_repo["id"]), "full", True, embedder=HashEmbedder())
    pos_job = run_index_sync(uuid.UUID(pos_repo["id"]), "full", True, embedder=HashEmbedder())
    assert fast_job.status == "succeeded"
    assert pos_job.status == "succeeded"
    assert fast_job.embedding_error is None
    with session_scope() as session:
        pos_workspace = session.scalar(select(Workspace).where(Workspace.slug == "pos"))
        fast_workspace = session.scalar(select(Workspace).where(Workspace.slug == "fastcrib"))
        pos_chunk = session.scalar(select(CodeChunk).where(CodeChunk.workspace_id == pos_workspace.id))
        assert pos_chunk is not None and pos_chunk.embedding is not None
        leaked = search_chunks(
            session,
            workspace_id=fast_workspace.id,
            query="POS_VECTOR_MARKER",
            query_embedding=list(pos_chunk.embedding),
        )
        found = search_chunks(
            session,
            workspace_id=pos_workspace.id,
            query="POS_VECTOR_MARKER",
            query_embedding=list(pos_chunk.embedding),
        )
    assert all(item.chunk.workspace_id == fast_workspace.id for item in leaked)
    assert all("POS_VECTOR_MARKER" not in item.chunk.content for item in leaked)
    assert any("POS_VECTOR_MARKER" in item.chunk.content for item in found)


def test_reset_index_keeps_workspaces(api, tmp_path: Path) -> None:
    repo_path = tmp_path / "fastcrib"
    init_repo(repo_path)
    write_commit(repo_path, {"src/a.py": "def alpha():\n    return 1\n"}, "init")
    repo = _connect(api, "fastcrib", repo_path, "fastcrib")
    _index(api, "fastcrib", repo["id"])
    rejected = api.post("/api/maintenance/reset-index", json={"confirm": "no"})
    assert rejected.status_code == 400
    reset = api.post("/api/maintenance/reset-index", json={"confirm": "reset-index"})
    assert reset.status_code == 200, reset.text
    status = api.get(f"/api/workspaces/fastcrib/repositories/{repo['id']}/status")
    assert status.json()["latest_indexed_commit"] is None
    assert status.json()["chunk_count"] == 0
    assert {item["slug"] for item in api.get("/api/workspaces").json()} >= {"fastcrib", "pos"}
