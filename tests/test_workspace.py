from pathlib import Path, PurePosixPath

from guardedpy.workspace import Workspace

from conftest import safe_config


GOOD_MULTI_FILE_DIFF = """--- a/src/first.py
+++ b/src/first.py
@@ -1 +1 @@
-before
+after
--- a/tests/second.py
+++ b/tests/second.py
@@ -1 +1 @@
-old
+new
"""

BAD_MULTI_FILE_DIFF = """--- a/src/first.py
+++ b/src/first.py
@@ -1 +1 @@
-before
+after
--- a/tests/second.py
+++ b/tests/second.py
@@ -1 +1 @@
-missing
+new
"""


def test_read_rejects_project_escape(tmp_path: Path) -> None:
    """Catches a read that resolves a parent path outside the selected root."""
    result = Workspace(tmp_path, safe_config(tmp_path)).read_file(
        PurePosixPath("../secret"), 0, 100
    )

    assert result.ok is False
    assert result.data["reason"] == "path_outside_project"


def test_list_files_returns_only_root_relative_files(tmp_path: Path) -> None:
    """Catches a listing that leaks parent paths or omits nested project files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("pass\n")
    (tmp_path / "README.md").write_text("project\n")

    result = Workspace(tmp_path, safe_config(tmp_path)).list_files(PurePosixPath("."))

    assert result.ok is True
    assert result.data["files"] == ("README.md", "src/module.py")


def test_read_file_returns_a_bounded_line_page(tmp_path: Path) -> None:
    """Catches a reader that ignores the requested page offset or limit."""
    (tmp_path / "notes.txt").write_text("zero\none\ntwo\nthree\n")

    result = Workspace(tmp_path, safe_config(tmp_path)).read_file(
        PurePosixPath("notes.txt"), 1, 2
    )

    assert result.ok is True
    assert result.data == {"path": "notes.txt", "content": "one\ntwo\n", "next_offset": 3}


def test_patch_mismatch_writes_nothing(tmp_path: Path) -> None:
    """Catches a patcher that commits an earlier file before a later hunk fails."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "first.py").write_text("before\n")
    (tmp_path / "tests" / "second.py").write_text("old\n")

    result = Workspace(tmp_path, safe_config(tmp_path)).apply_patch(BAD_MULTI_FILE_DIFF)

    assert result.ok is False
    assert result.data["reason"] == "hunk_mismatch"
    assert (tmp_path / "src" / "first.py").read_text() == "before\n"
    assert (tmp_path / "tests" / "second.py").read_text() == "old\n"


def test_matching_multi_file_patch_updates_all_files(tmp_path: Path) -> None:
    """Catches a patcher that validates but does not apply every matched file hunk."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "first.py").write_text("before\n")
    (tmp_path / "tests" / "second.py").write_text("old\n")

    result = Workspace(tmp_path, safe_config(tmp_path)).apply_patch(GOOD_MULTI_FILE_DIFF)

    assert result.ok is True
    assert (tmp_path / "src" / "first.py").read_text() == "after\n"
    assert (tmp_path / "tests" / "second.py").read_text() == "new\n"


def test_patch_can_create_a_file_from_an_empty_hunk(tmp_path: Path) -> None:
    """Catches a parser that treats a zero-line creation hunk as one old line."""
    (tmp_path / "src").mkdir()

    result = Workspace(tmp_path, safe_config(tmp_path)).apply_patch(
        """--- /dev/null
+++ b/src/new.py
@@ -0,0 +1 @@
+created
"""
    )

    assert result.ok is True
    assert (tmp_path / "src" / "new.py").read_text() == "created\n"


def test_patch_rejects_a_path_outside_the_project(tmp_path: Path) -> None:
    """Catches a diff header that lets a patch write outside the selected root."""
    result = Workspace(tmp_path, safe_config(tmp_path)).apply_patch(
        """--- a/../secret
+++ b/../secret
@@ -0,0 +1 @@
+leak
"""
    )

    assert result.ok is False
    assert result.data["reason"] == "path_outside_project"


def test_delete_path_removes_a_project_file(tmp_path: Path) -> None:
    """Catches a delete tool that reports success without removing its target."""
    target = tmp_path / "obsolete.txt"
    target.write_text("obsolete\n")

    result = Workspace(tmp_path, safe_config(tmp_path)).delete_path(PurePosixPath("obsolete.txt"))

    assert result.ok is True
    assert target.exists() is False


def test_delete_rejects_project_escape(tmp_path: Path) -> None:
    """Catches a delete that resolves parent paths outside the selected root."""
    result = Workspace(tmp_path, safe_config(tmp_path)).delete_path(PurePosixPath("../secret"))

    assert result.ok is False
    assert result.data["reason"] == "path_outside_project"
