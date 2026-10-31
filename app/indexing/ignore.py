from __future__ import annotations

from pathlib import Path

from pathspec import GitIgnoreSpec

DENY_DIR_NAMES = {
    "node_modules",
    "vendor",
    "third_party",
    "third-party",
    "dist",
    "build",
    "target",
    "out",
    "__pycache__",
    ".venv",
    "venv",
    "site-packages",
    "coverage",
    ".next",
    ".turbo",
    "bower_components",
    "deps",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    "bin",
    "obj",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".mp3",
    ".mp4",
    ".mov",
    ".wasm",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".class",
    ".jar",
    ".pyc",
    ".pyo",
    ".a",
    ".o",
    ".obj",
}

DEFAULT_GLOBS = [
    "**/*.min.js",
    "**/*.min.css",
    "**/*.map",
    "**/*.lock",
    "**/package-lock.json",
    "**/yarn.lock",
    "**/pnpm-lock.yaml",
    "**/Cargo.lock",
    "**/poetry.lock",
    "**/go.sum",
    "**/*_generated.*",
    "**/*.pb.go",
    "**/*.pyc",
    "**/*.pyo",
]


class IgnoreMatcher:
    def __init__(self, repo_root: Path, extra_globs: list[str], use_defaults: bool):
        self.repo_root = repo_root
        self.use_defaults = use_defaults
        patterns = list(extra_globs)
        if use_defaults:
            patterns.extend(DEFAULT_GLOBS)
        self.extra = GitIgnoreSpec.from_lines(patterns)
        self.gitignore_specs = _load_gitignores(repo_root)

    def reason(self, relative_path: str) -> str | None:
        path = relative_path.replace("\\", "/").lstrip("/")
        if not path or path.startswith(".git/") or "/.git/" in path or path == ".git":
            return "git_metadata"
        suffix = Path(path).suffix.lower()
        if suffix in BINARY_EXTENSIONS:
            return "binary_extension"
        if self.use_defaults:
            parts = [part for part in path.split("/") if part]
            if any(part in DENY_DIR_NAMES for part in parts):
                return "vendored_or_build"
            if self.extra.match_file(path):
                return "generated_or_ignored"
        elif self.extra.match_file(path):
            return "ignored_glob"
        if _matches_gitignore(self.gitignore_specs, path):
            return "gitignore"
        return None


def build_matcher(repo_root: str | Path, extra_globs: list[str] | None, use_defaults: bool = True) -> IgnoreMatcher:
    return IgnoreMatcher(Path(repo_root), list(extra_globs or []), use_defaults)


def _load_gitignores(repo_root: Path) -> list[tuple[str, GitIgnoreSpec]]:
    specs: list[tuple[str, pathspec.PathSpec]] = []
    if not repo_root.is_dir():
        return specs
    for dirpath, dirnames, filenames in os_walk_safe(repo_root):
        dirnames[:] = [name for name in dirnames if name != ".git" and name not in DENY_DIR_NAMES]
        if ".gitignore" not in filenames:
            continue
        gitignore_path = Path(dirpath) / ".gitignore"
        relative = gitignore_path.parent.relative_to(repo_root).as_posix()
        prefix = "" if relative == "." else relative
        try:
            lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        specs.append((prefix, GitIgnoreSpec.from_lines(lines)))
    return specs


def os_walk_safe(repo_root: Path):
    import os

    return os.walk(repo_root, followlinks=False)


def _matches_gitignore(specs: list[tuple[str, GitIgnoreSpec]], path: str) -> bool:
    for prefix, spec in specs:
        if prefix:
            lead = prefix + "/"
            if path != prefix and not path.startswith(lead):
                continue
            remainder = path[len(lead) :] if path.startswith(lead) else ""
        else:
            remainder = path
        if not remainder:
            continue
        if spec.match_file(remainder):
            return True
        parts = remainder.split("/")
        accumulated: list[str] = []
        for part in parts[:-1]:
            accumulated.append(part)
            if spec.match_file("/".join(accumulated) + "/"):
                return True
    return False
