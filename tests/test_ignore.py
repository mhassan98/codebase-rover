from pathlib import Path

from app.indexing.ignore import build_matcher


def test_default_ignores_skip_vendored_build_and_generated(tmp_path: Path) -> None:
    matcher = build_matcher(tmp_path, ["secrets/**"], use_defaults=True)
    assert matcher.reason("src/app.py") is None
    assert matcher.reason("src/builder.py") is None
    assert matcher.reason("vendor/lib.py") == "vendored_or_build"
    assert matcher.reason("node_modules/pkg/index.js") == "vendored_or_build"
    assert matcher.reason("dist/app.js") == "vendored_or_build"
    assert matcher.reason("build/output.py") == "vendored_or_build"
    assert matcher.reason("pkg/app.min.js") == "generated_or_ignored"
    assert matcher.reason("go.sum") == "generated_or_ignored"
    assert matcher.reason("secrets/keys.py") == "generated_or_ignored"
    assert matcher.reason("assets/logo.png") == "binary_extension"


def test_gitignore_and_optional_defaults(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("local_only.py\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "local_only.py").write_text("x = 1\n")
    matcher = build_matcher(tmp_path, [], use_defaults=True)
    assert matcher.reason("pkg/local_only.py") == "gitignore"

    relaxed = build_matcher(tmp_path, ["keep-out/**"], use_defaults=False)
    assert relaxed.reason("vendor/lib.py") is None
    assert relaxed.reason("keep-out/notes.py") == "ignored_glob"
    assert relaxed.reason("logo.png") == "binary_extension"
