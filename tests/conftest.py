from pathlib import Path, PurePosixPath
import sys

import pytest

from guardedpy.config import HarnessConfig
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import TaskMode, TaskState, TddPhase


def safe_config(tmp_path: Path) -> HarnessConfig:
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    return HarnessConfig(
        profile=ProjectProfile(
            root=tmp_path.resolve(),
            discovery_source="tests_dir",
            source_dirs=(PurePosixPath("src"),),
            test_dirs=(PurePosixPath("tests"),),
            pytest_command=(sys.executable, "-m", "pytest"),
        )
    )


@pytest.fixture
def feature_task(tmp_path: Path) -> TaskState:
    return TaskState(
        description="Add a feature",
        mode=TaskMode.FEATURE,
        config=safe_config(tmp_path),
    )


@pytest.fixture
def ready_bugfix_task(tmp_path: Path) -> TaskState:
    return TaskState(
        description="Fix a failing test",
        mode=TaskMode.BUGFIX,
        bugfix_target="tests/test_example.py::test_before",
        config=safe_config(tmp_path),
        tdd_phase=TddPhase.RED_OBSERVED,
    )


SOURCE_DIFF = """--- a/src/example.py
+++ b/src/example.py
@@ -1 +1 @@
-before
+after
"""

TEST_DIFF = """--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -1 +1 @@
-def test_before(): pass
+def test_after(): pass
"""
