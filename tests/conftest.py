from pathlib import Path

import pytest

from guardedpy.config import HarnessConfig
from guardedpy.domain import TaskMode, TaskState, TddPhase


def safe_config(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig(
        source_dirs=("src",),
        test_dirs=("tests",),
        pytest_command=("pytest",),
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
