from pathlib import Path

import pytest

from guardedpy.actions import (
    ApplyPatchAction,
    DeletePathAction,
    FinishAction,
    ReadFileAction,
    RunCommandAction,
)
from guardedpy.domain import PolicyVerdict, TaskMode, TaskState, TddPhase
from guardedpy.policy import PolicyEngine


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


@pytest.fixture
def policy(tmp_path: Path) -> PolicyEngine:
    return PolicyEngine(tmp_path)


def test_source_patch_before_red_is_denied(policy: PolicyEngine, feature_task: TaskState) -> None:
    """Catches a policy change that lets production code bypass the red-test gate."""
    result = policy.decide(
        feature_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "tdd.red_required")


def test_feature_task_records_red_only_after_a_test_patch(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches a transition that accepts an unrelated failing test as feature TDD evidence."""
    first = policy.record_pytest(feature_task, passed=False)

    policy.record_read(
        feature_task, ReadFileAction(kind="read_file", summary="read test", path="tests/test_example.py")
    )
    policy.decide(
        feature_task, ApplyPatchAction(kind="apply_patch", summary="add test", diff=TEST_DIFF)
    )
    second = policy.record_pytest(feature_task, passed=False)

    assert (first.verdict, first.rule_id) == (PolicyVerdict.DENY, "tdd.test_change_required")
    assert second.verdict is PolicyVerdict.ALLOW
    assert feature_task.tdd_phase is TddPhase.RED_OBSERVED


def test_source_patch_requires_a_current_read_after_red(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches a read-before-patch bypass for a production file."""
    result = policy.decide(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "patch.read_required")


def test_read_source_then_patch_transitions_to_implementation(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches a policy that allows implementation without recording its state transition."""
    policy.record_read(
        ready_bugfix_task, ReadFileAction(kind="read_file", summary="read source", path="src/example.py")
    )
    result = policy.decide(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    )

    assert result.verdict is PolicyVerdict.ALLOW
    assert ready_bugfix_task.tdd_phase is TddPhase.IMPLEMENTATION


def test_passing_pytest_allows_completed_finish_only_after_implementation(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches an early completed finish that skips the green-test observation."""
    policy.record_read(
        ready_bugfix_task, ReadFileAction(kind="read_file", summary="read source", path="src/example.py")
    )
    policy.decide(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    )
    policy.record_pytest(ready_bugfix_task, passed=True)

    result = policy.decide(
        ready_bugfix_task, FinishAction(kind="finish", summary="all tests pass", status="completed")
    )

    assert result.verdict is PolicyVerdict.ALLOW
    assert ready_bugfix_task.tdd_phase is TddPhase.FINISHED


def test_patch_outside_configured_directories_is_denied(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches a path fence that permits automatic edits to project documentation."""
    diff = """--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-before
+after
"""

    result = policy.decide(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="edit docs", diff=diff)
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.APPROVAL_REQUIRED, "patch.non_code")


def test_outside_path_is_directly_denied(policy: PolicyEngine, feature_task: TaskState) -> None:
    """Catches a root-boundary bypass through a traversal path."""
    result = policy.record_read(
        feature_task, ReadFileAction(kind="read_file", summary="read outside", path="../credentials.txt")
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "path.outside_root")


def test_delete_approval_is_exact_and_single_use(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches an approval replay that authorizes a dangerous delete more than once."""
    action = DeletePathAction(kind="delete_path", summary="remove test", path="tests/test_example.py")
    pending = policy.request_approval(ready_bugfix_task, action)

    accepted = policy.apply_approval(pending, action, approved=True)
    replayed = policy.apply_approval(pending, action, approved=True)

    assert accepted.verdict is PolicyVerdict.ALLOW
    assert (replayed.verdict, replayed.rule_id) == (PolicyVerdict.DENY, "approval.already_used")


def test_approval_does_not_match_a_different_action(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches an approval binding that ignores the exact proposed action hash."""
    pending = policy.request_approval(
        ready_bugfix_task,
        DeletePathAction(kind="delete_path", summary="test", path="tests/test_example.py"),
    )

    result = policy.apply_approval(
        pending,
        DeletePathAction(kind="delete_path", summary="source", path="src/example.py"),
        approved=True,
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "approval.action_mismatch")


def test_approval_cannot_override_a_direct_denial(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches an approval route that turns a TDD violation into permission."""
    action = ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    pending = policy.request_approval(feature_task, action)

    result = policy.apply_approval(pending, action, approved=True)

    assert (pending.verdict, pending.rule_id) == (PolicyVerdict.DENY, "tdd.red_required")
    assert result.verdict is PolicyVerdict.DENY


def test_dependency_command_requires_exact_approval(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches an automatic dependency installation through the generic command tool."""
    action = RunCommandAction(kind="run_command", summary="install", args=("pip", "install", "ruff"))

    result = policy.decide(feature_task, action)

    assert (result.verdict, result.rule_id) == (
        PolicyVerdict.APPROVAL_REQUIRED,
        "command.dependency_install",
    )


@pytest.mark.parametrize(
    ("action", "rule_id"),
    [
        (
            RunCommandAction(
                kind="run_command", summary="edit source", args=("sed", "-i", "s/a/b/", "src/example.py")
            ),
            "command.source_or_test_write",
        ),
        (
            RunCommandAction(kind="run_command", summary="elevate", args=("sudo", "id")),
            "command.privilege",
        ),
        (
            RunCommandAction(
                kind="run_command", summary="read key", args=("keyring", "get", "service", "user")
            ),
            "command.credentials",
        ),
    ],
)
def test_dangerous_command_is_directly_denied(
    policy: PolicyEngine, feature_task: TaskState, action: RunCommandAction, rule_id: str
) -> None:
    """Catches generic command paths that bypass source, privilege, or credential safeguards."""
    result = policy.decide(feature_task, action)

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, rule_id)
