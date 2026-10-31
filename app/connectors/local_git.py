from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from app.connectors.base import (
    FileChange,
    GitError,
    ProviderCapabilityError,
    PullRequestSnapshot,
    ReviewDraft,
    SourceIdentity,
    TreeEntry,
)
from app.redact import redact

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_REVISION = re.compile(r"^[A-Za-z0-9._/-]+$")


def _validate_revision(revision: str) -> str:
    if (
        not revision
        or len(revision) > 200
        or revision.startswith("-")
        or ".." in revision
        or not _REVISION.fullmatch(revision)
    ):
        raise GitError("invalid revision")
    return revision


def _validate_blob(sha: str) -> str:
    if not _HEX40.fullmatch(sha):
        raise GitError("invalid blob id")
    return sha


def _validate_path(path: str) -> str:
    if not path or "\x00" in path or "\\" in path or path.startswith("/"):
        raise GitError("invalid path")
    parts = path.split("/")
    if any(part == ".." for part in parts):
        raise GitError("invalid path")
    return path


def git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def parse_ls_tree(data: bytes) -> list[TreeEntry]:
    entries: list[TreeEntry] = []
    for record in data.split(b"\0"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        pieces = meta.decode("utf-8", errors="replace").split(" ")
        if len(pieces) < 4:
            continue
        mode, object_type, sha, size_token = pieces[0], pieces[1], pieces[2], pieces[3]
        size = int(size_token) if size_token.isdigit() else None
        entries.append(
            TreeEntry(
                mode=mode,
                object_type=object_type,
                blob_sha=sha,
                size=size,
                path=raw_path.decode("utf-8", errors="replace"),
            )
        )
    return entries


def parse_cat_file_batch(data: bytes, count: int) -> list[bytes | None]:
    results: list[bytes | None] = []
    index = 0
    for _ in range(count):
        newline = data.find(b"\n", index)
        if newline < 0:
            raise GitError("truncated git cat-file response")
        header = data[index:newline].decode("utf-8", errors="replace")
        index = newline + 1
        if header.endswith(" missing"):
            results.append(None)
            continue
        parts = header.split(" ")
        if len(parts) < 3 or not parts[-1].isdigit():
            raise GitError("unexpected git cat-file response")
        size = int(parts[-1])
        blob = data[index : index + size]
        if len(blob) != size:
            raise GitError("truncated git blob")
        index += size + 1
        results.append(blob)
    return results


def parse_name_status_z(data: bytes) -> list[FileChange]:
    parts = data.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    changes: list[FileChange] = []
    index = 0
    while index < len(parts):
        status = parts[index].decode("utf-8", errors="replace")
        index += 1
        code = status[:1]
        if code in {"R", "C"}:
            if index + 1 >= len(parts):
                break
            old_path = parts[index].decode("utf-8", errors="replace")
            new_path = parts[index + 1].decode("utf-8", errors="replace")
            index += 2
            changes.append(FileChange("renamed" if code == "R" else "copied", new_path, old_path))
            continue
        if index >= len(parts):
            break
        path = parts[index].decode("utf-8", errors="replace")
        index += 1
        kind = {"A": "added", "M": "modified", "D": "deleted", "T": "modified"}.get(code, "modified")
        changes.append(FileChange(kind, path, None))
    return changes


class LocalGitConnector:
    """Reads a local git working tree. It does not publish review comments."""

    provider_name = "local_git"

    def __init__(self, path: str | Path, *, timeout: int = 120):
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise GitError(f"not a directory: {root}")
        probe = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
            env=git_env(),
            check=False,
        )
        if probe.returncode != 0:
            raise GitError(f"not a git repository: {root}")
        toplevel = Path(probe.stdout.strip()).resolve()
        if toplevel != root:
            raise GitError(f"path is not the top level of the git repository: {root}")
        self.root = root
        self.timeout = timeout

    def _run(
        self,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
        check: bool = True,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        for arg in args:
            if "\n" in arg or "\x00" in arg:
                raise GitError("invalid git argument")
        proc = subprocess.run(
            ["git", "-C", str(self.root), *args],
            input=input_bytes,
            capture_output=True,
            timeout=timeout or self.timeout,
            env=git_env(),
            check=False,
        )
        if check and proc.returncode != 0:
            message = proc.stderr.decode("utf-8", errors="replace").strip()
            raise GitError(redact(message)[:500] or "git command failed")
        return proc

    def identify(self) -> SourceIdentity:
        head = self.resolve_commit("HEAD")
        return SourceIdentity(
            provider=self.provider_name,
            native_id=str(self.root),
            native_url=self.root.as_uri(),
            remote_url=self.remote_url(),
            default_branch=self.branch_name(),
            head_sha=head,
            display_name=self.root.name,
        )

    def resolve_commit(self, revision: str) -> str:
        _validate_revision(revision)
        proc = self._run(["rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"])
        sha = proc.stdout.decode("utf-8", errors="replace").strip()
        if not _HEX40.fullmatch(sha):
            raise GitError("revision did not resolve to a commit")
        return sha

    def commit_exists(self, revision: str) -> bool:
        try:
            self.resolve_commit(revision)
        except GitError:
            return False
        return True

    def branch_name(self) -> str:
        proc = self._run(["symbolic-ref", "--short", "HEAD"], check=False)
        if proc.returncode != 0:
            return "HEAD"
        name = proc.stdout.decode("utf-8", errors="replace").strip()
        return name or "HEAD"

    def remote_url(self) -> str | None:
        proc = self._run(["remote", "get-url", "origin"], check=False)
        if proc.returncode != 0:
            return None
        url = proc.stdout.decode("utf-8", errors="replace").strip()
        return url or None

    def list_tree(self, revision: str) -> list[TreeEntry]:
        sha = revision if _HEX40.fullmatch(revision) else self.resolve_commit(revision)
        proc = self._run(["ls-tree", "-r", "-l", "-z", sha], timeout=max(self.timeout, 300))
        return parse_ls_tree(proc.stdout)

    def read_blobs(self, shas: list[str]) -> list[bytes | None]:
        if not shas:
            return []
        for sha in shas:
            _validate_blob(sha)
        payload = b"".join(sha.encode("ascii") + b"\n" for sha in shas)
        proc = self._run(["cat-file", "--batch"], input_bytes=payload, timeout=max(self.timeout, 300))
        return parse_cat_file_batch(proc.stdout, len(shas))

    def read_bytes(self, revision: str, path: str) -> bytes:
        _validate_path(path)
        sha = revision if _HEX40.fullmatch(revision) else self.resolve_commit(revision)
        spec = f"{sha}:{path}"
        proc = self._run(["cat-file", "--batch"], input_bytes=spec.encode("utf-8") + b"\n")
        blobs = parse_cat_file_batch(proc.stdout, 1)
        if not blobs or blobs[0] is None:
            raise GitError("path not found at revision")
        return blobs[0]

    def read_text(self, revision: str, path: str) -> str:
        data = self.read_bytes(revision, path)
        if b"\x00" in data[:8192]:
            raise GitError("binary file")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitError("file is not utf-8 text") from exc

    def list_changes(self, from_revision: str, to_revision: str) -> list[FileChange]:
        old = self.resolve_commit(from_revision)
        new = self.resolve_commit(to_revision)
        if old == new:
            return []
        proc = self._run(
            ["diff", "--name-status", "--find-renames", "-z", old, new],
            timeout=max(self.timeout, 300),
        )
        return parse_name_status_z(proc.stdout)

    def diff(self, base: str, head: str) -> str:
        base_sha = self.resolve_commit(base)
        head_sha = self.resolve_commit(head)
        proc = self._run(
            ["diff", "--find-renames", "--no-color", "--no-ext-diff", base_sha, head_sha],
            timeout=max(self.timeout, 300),
        )
        return proc.stdout.decode("utf-8", errors="replace")

    def fetch_pull_request(self, native_pr_id: str) -> PullRequestSnapshot:
        raise ProviderCapabilityError(
            "local_git has no pull request object. Submit a diff and a repository revision to the review API."
        )

    def publish_review(self, native_pr_id: str, review: ReviewDraft) -> None:
        raise ProviderCapabilityError(
            "local_git cannot publish review comments. A native source-control connector can implement this later."
        )
