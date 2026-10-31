from app.review.diffparse import parse_unified_diff

DIFF = """diff --git a/billing/tax.py b/billing/tax.py
index 1111111..2222222 100644
--- a/billing/tax.py
+++ b/billing/tax.py
@@ -3,6 +3,8 @@ def ready():
     return True
 
 def tax(amount):
-    return round(amount * 0.05, 2)
+    if amount < 0:
+        raise ValueError("negative amount")
+    return round(amount * 0.08, 2)
 
 def currency():
"""


def test_parse_modified_python_hunk() -> None:
    files = parse_unified_diff(DIFF)
    assert len(files) == 1
    parsed = files[0]
    assert parsed.path == "billing/tax.py"
    assert parsed.status == "modified"
    added = [line for hunk in parsed.hunks for line in hunk.lines if line.kind == "add"]
    assert [line.text for line in added] == [
        "    if amount < 0:",
        '        raise ValueError("negative amount")',
        "    return round(amount * 0.08, 2)",
    ]
    assert all(line.new_line is not None for line in added)
    assert added[0].new_line == added[1].new_line - 1


def test_new_and_deleted_files() -> None:
    diff = """diff --git a/new.py b/new.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+def created():
+    return 1
diff --git a/old.py b/old.py
deleted file mode 100644
index 2222222..0000000
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def removed():
-    return 0
"""
    files = parse_unified_diff(diff)
    assert [item.status for item in files] == ["added", "deleted"]
    assert files[0].new_path == "new.py"
    assert files[1].old_path == "old.py"
    assert files[1].new_path is None


def test_rejects_non_diff() -> None:
    try:
        parse_unified_diff("this is not a diff")
    except ValueError as exc:
        assert "unified diff" in str(exc)
    else:
        raise AssertionError("expected parse failure")
