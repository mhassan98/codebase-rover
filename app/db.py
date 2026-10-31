from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

log = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


class Base(DeclarativeBase):
    pass


def reset_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, autoflush=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    from app import models  # noqa: F401

    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    except Exception as exc:
        raise RuntimeError(
            "PostgreSQL could not load vector and pg_trgm. "
            "Use the pgvector image from docker-compose, or create the extensions as a superuser."
        ) from exc
    Base.metadata.create_all(engine)
    _optional_index(
        """
        CREATE INDEX IF NOT EXISTS ix_code_chunks_content_tsv
        ON code_chunks USING gin (to_tsvector('english', content))
        """
    )
    _optional_index(
        """
        CREATE INDEX IF NOT EXISTS ix_code_chunks_content_trgm
        ON code_chunks USING gin (content gin_trgm_ops)
        """
    )
    _optional_index(
        """
        CREATE INDEX IF NOT EXISTS ix_code_chunks_embedding_hnsw
        ON code_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )


def _optional_index(sql: str) -> None:
    try:
        with get_engine().begin() as conn:
            conn.execute(text(sql))
    except Exception:
        log.warning("optional search index was not created")
