from __future__ import annotations

import os

import pytest

os.environ["DATABASE_URL"] = "postgresql+psycopg://rover:rover@localhost:5432/rover_test"
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:9")

TEST_URL = os.environ["DATABASE_URL"]


def _postgres_up() -> bool:
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(TEST_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def client():
    if not _postgres_up():
        pytest.skip("PostgreSQL with the test database is not available")
    from app.config import get_settings
    from app.db import Base, get_engine, init_db, reset_engine

    get_settings.cache_clear()
    reset_engine()
    import app.models  # noqa: F401

    Base.metadata.drop_all(get_engine())
    init_db()
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    reset_engine()


@pytest.fixture
def api(client):
    from sqlalchemy import delete

    from app.catalog import seed
    from app.db import session_scope
    from app.indexing.service import join_index_threads
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

    join_index_threads()
    with session_scope() as session:
        for model in (
            CodeChunk,
            IndexedFile,
            IndexJob,
            Question,
            ReviewJob,
            Environment,
            Repository,
            SourceConnection,
            Workspace,
            Tenant,
        ):
            session.execute(delete(model))
        seed(session)
    return client
