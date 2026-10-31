from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class ProviderCapabilityError(RuntimeError):
    """The connector does not implement this source-control capability."""


class GitError(RuntimeError):
    """A local git read failed or the path is not a repository."""


@dataclass(frozen=True)
class SourceIdentity:
    provider: str
    native_id: str
    native_url: str
    remote_url: str | None
    default_branch: str | None
    head_sha: str
    display_name: str


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    object_type: str
    blob_sha: str
    size: int | None
    path: str


@dataclass(frozen=True)
class FileChange:
    status: str
    path: str
    old_path: str | None = None


@dataclass(frozen=True)
class PullRequestSnapshot:
    native_id: str
    native_url: str | None
    title: str
    head_sha: str
    base_sha: str | None
    diff: str


@dataclass(frozen=True)
class ReviewCommentDraft:
    file_path: str
    line: int
    body: str


@dataclass(frozen=True)
class ReviewDraft:
    summary: str
    comments: list[ReviewCommentDraft] = field(default_factory=list)


class SourceConnector(Protocol):
    """Identifies a source, reads a revision, and reports file changes.

    Implementations keep the provider's native id and URL. They do not assume
    a particular host such as GitHub, GitLab, or Azure DevOps.
    """

    provider_name: str

    def identify(self) -> SourceIdentity: ...

    def read_text(self, revision: str, path: str) -> str: ...

    def list_changes(self, from_revision: str, to_revision: str) -> list[FileChange]: ...


class SourceControlConnector(SourceConnector, Protocol):
    """Optional pull-request operations for a native source-control provider."""

    def fetch_pull_request(self, native_pr_id: str) -> PullRequestSnapshot: ...

    def publish_review(self, native_pr_id: str, review: ReviewDraft) -> None: ...
