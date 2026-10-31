from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.connectors.base import GitError
from app.connectors.local_git import LocalGitConnector

PROVIDERS = ("local_git",)


def open_local_git(path: str, settings: Settings) -> LocalGitConnector:
    resolved = Path(path).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in settings.allowed_roots()]
    if roots and not any(_is_within(resolved, root) for root in roots):
        raise GitError("repository path is outside ROVER_ALLOWED_ROOTS")
    return LocalGitConnector(resolved)


def connector_for_repository(provider: str, local_path: str) -> LocalGitConnector:
    if provider != "local_git":
        raise GitError(
            f"source provider '{provider}' is not installed. The local git connector is the only implementation."
        )
    return LocalGitConnector(local_path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
