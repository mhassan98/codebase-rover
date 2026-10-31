from pathlib import Path

import pytest

from app.connectors.base import ProviderCapabilityError
from app.connectors.local_git import LocalGitConnector
from tests.gitutil import git, init_repo, write_commit


def test_reads_exact_revision_and_lists_changes(tmp_path: Path) -> None:
    init_repo(tmp_path)
    first = write_commit(tmp_path, {"src/app.py": "def tax():\n    return 1\n", "src/note.py": "x = 1\n"}, "init")
    (tmp_path / "src" / "link.py").symlink_to("app.py")
    git(tmp_path, "add", "src/link.py")
    git(tmp_path, "commit", "-m", "link")
    (tmp_path / "src" / "app.py").write_text("def tax():\n    return 2\n")
    (tmp_path / "src" / "extra.py").write_text("def extra():\n    return 3\n")
    (tmp_path / "src" / "note.py").unlink()
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-m", "edit")
    head = git(tmp_path, "rev-parse", "HEAD").stdout.strip()

    connector = LocalGitConnector(tmp_path)
    identity = connector.identify()
    assert identity.provider == "local_git"
    assert identity.native_id == str(tmp_path.resolve())
    assert identity.native_url.startswith("file:")
    assert connector.read_text(first, "src/app.py").startswith("def tax():\n    return 1")
    assert "return 2" not in connector.read_text(first, "src/app.py")
    assert "return 2" in connector.read_text(head, "src/app.py")

    changes = connector.list_changes(first, head)
    by_path = {change.path: change.status for change in changes}
    assert by_path["src/app.py"] == "modified"
    assert by_path["src/extra.py"] == "added"
    assert by_path["src/note.py"] == "deleted"
    tree = connector.list_tree(head)
    link = next(entry for entry in tree if entry.path == "src/link.py")
    assert link.mode == "120000"
    with pytest.raises(ProviderCapabilityError):
        connector.fetch_pull_request("1")
    with pytest.raises(ProviderCapabilityError):
        connector.publish_review("1", None)
