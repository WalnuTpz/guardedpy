from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import safe_config
from guardedpy.actions import (
    Action,
    ApplyPatchAction,
    DeletePathAction,
    FinishAction,
    ProposeMemoryAction,
    ReadFileAction,
    RequestApprovalAction,
    RunCommandAction,
    RunPytestAction,
)
from guardedpy.command_rules import CommandRuleStore
from guardedpy.domain import (
    ApprovalDecision,
    FeedbackKind,
    PolicyDecision,
    PolicyVerdict,
    TaskIntent,
    TaskPath,
    TaskState,
    TddPhase,
)
from guardedpy.feedback import FeedbackCollector, PytestFeedback, PytestRun
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

ROOT_INTERNAL_SOURCE_DIFF = """--- a/tests/../src/example.py
+++ b/tests/../src/example.py
@@ -1 +1 @@
-before
+after
"""

NEW_TEST_DIFF = """--- /dev/null
+++ b/tests/test_created.py
@@ -0,0 +1,2 @@
+def test_created() -> None:
+    assert False
"""

NEW_SOURCE_DIFF = """--- /dev/null
+++ b/src/created.py
@@ -0,0 +1 @@
+VALUE = "created"
"""

SECOND_NEW_TEST_DIFF = """--- /dev/null
+++ b/tests/test_second_created.py
@@ -0,0 +1,2 @@
+def test_second_created() -> None:
+    assert False
"""


def _record_passing_baseline(policy: PolicyEngine, task: TaskState) -> None:
    decision = policy.record_pytest(
        task,
        RunPytestAction(kind="run_pytest", summary="establish baseline", targets=()),
        PytestFeedback(FeedbackKind.PASSED, (), ""),
    )
    assert decision.verdict is PolicyVerdict.ALLOW


def _record_bugfix_red_baseline(policy: PolicyEngine, task: TaskState) -> None:
    assert task.repair_targets
    target = task.repair_targets[0]
    task.path = TaskPath.BASELINE_PENDING
    task.repair_targets = ()
    task.tdd_phase = TddPhase.TEST_DESIGN
    decision = policy.record_pytest(
        task,
        RunPytestAction(kind="run_pytest", summary="establish broken baseline", targets=()),
        PytestFeedback(
            FeedbackKind.ASSERTION_FAILURE,
            (target,),
            "assert actual == expected",
        ),
    )
    assert decision.verdict is PolicyVerdict.ALLOW


def _request_approval(
    policy: PolicyEngine,
    task: TaskState,
    action: Action,
) -> PolicyDecision:
    decision = policy.decide(task, action)
    return policy.request_approval(task, action, decision)


@pytest.fixture
def policy(tmp_path: Path) -> PolicyEngine:
    return PolicyEngine(tmp_path)


def test_source_patch_before_red_is_denied(policy: PolicyEngine, feature_task: TaskState) -> None:
    """Catches a policy change that lets production code bypass the red-test gate."""
    result = policy.decide(
        feature_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "tdd.red_required")


def test_feature_can_create_test_before_red_without_prior_read(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches new tests inheriting the read requirement for existing files."""
    _record_passing_baseline(policy, feature_task)
    result = policy.decide(
        feature_task, ApplyPatchAction(kind="apply_patch", summary="add test", diff=NEW_TEST_DIFF)
    )

    assert result.verdict is PolicyVerdict.ALLOW


def test_feature_records_target_free_passing_suite_as_task_start_baseline(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches a healthy starting suite being rejected or mistaken for final green."""
    baseline = policy.record_pytest(
        feature_task,
        RunPytestAction(kind="run_pytest", summary="establish baseline", targets=()),
        PytestFeedback(FeedbackKind.PASSED, (), ""),
    )

    assert (baseline.verdict, baseline.rule_id) == (
        PolicyVerdict.ALLOW,
        "tdd.baseline_recorded",
    )
    assert feature_task.tdd_phase is TddPhase.TEST_DESIGN


def test_feature_test_patch_requires_target_free_passing_baseline(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches feature tests changing before the configured suite is proven healthy."""
    patch = ApplyPatchAction(kind="apply_patch", summary="add test", diff=NEW_TEST_DIFF)

    before_baseline = policy.decide(feature_task, patch)
    policy.record_pytest(
        feature_task,
        RunPytestAction(
            kind="run_pytest",
            summary="targeted pass is not a baseline",
            targets=("tests/test_existing.py",),
        ),
        PytestFeedback(FeedbackKind.PASSED, (), ""),
    )
    after_targeted_pass = policy.decide(feature_task, patch)
    policy.record_pytest(
        feature_task,
        RunPytestAction(kind="run_pytest", summary="establish baseline", targets=()),
        PytestFeedback(FeedbackKind.PASSED, (), ""),
    )
    after_full_suite_pass = policy.decide(feature_task, patch)

    assert (before_baseline.verdict, before_baseline.rule_id) == (
        PolicyVerdict.DENY,
        "tdd.baseline_required",
    )
    assert (after_targeted_pass.verdict, after_targeted_pass.rule_id) == (
        PolicyVerdict.DENY,
        "tdd.baseline_required",
    )
    assert after_full_suite_pass.verdict is PolicyVerdict.ALLOW


def test_new_source_file_requires_a_created_test_then_selected_red(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches source creation skipping the successful new-test and red gates."""
    test_patch = ApplyPatchAction(kind="apply_patch", summary="add test", diff=NEW_TEST_DIFF)
    source_patch = ApplyPatchAction(kind="apply_patch", summary="add source", diff=NEW_SOURCE_DIFF)

    assert policy.decide(feature_task, source_patch).rule_id == "tdd.red_required"
    _record_passing_baseline(policy, feature_task)
    assert policy.decide(feature_task, test_patch).verdict is PolicyVerdict.ALLOW
    assert policy.record_patch(feature_task, test_patch).verdict is PolicyVerdict.ALLOW
    policy.record_new_test_path(feature_task, "tests/test_created.py")
    assert (
        policy.record_pytest(
            feature_task,
            RunPytestAction(kind="run_pytest", summary="observe red", targets=("tests/test_created.py",)),
            PytestFeedback(
                FeedbackKind.ASSERTION_FAILURE,
                ("tests/test_created.py",),
                "assert False",
            ),
        ).verdict
        is PolicyVerdict.ALLOW
    )

    assert policy.decide(feature_task, source_patch).verdict is PolicyVerdict.ALLOW


def test_memory_proposal_requires_nonblank_text(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches whitespace-only model memory becoming a user-review candidate."""
    result = policy.decide(
        feature_task,
        ProposeMemoryAction(kind="propose_memory", summary="remember", text="   "),
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "memory.text_required")


def test_feature_task_records_red_only_after_a_test_patch(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches a transition that accepts an unrelated failing test as feature TDD evidence."""
    _record_passing_baseline(policy, feature_task)
    target_run = RunPytestAction(
        kind="run_pytest", summary="run new test", targets=("tests/test_created.py",)
    )
    first = policy.record_pytest(
        feature_task,
        target_run,
        PytestFeedback(FeedbackKind.ASSERTION_FAILURE, ("tests/test_created.py",), "assertion failed"),
    )

    proposed = policy.decide(
        feature_task, ApplyPatchAction(kind="apply_patch", summary="add test", diff=NEW_TEST_DIFF)
    )
    second = policy.record_pytest(
        feature_task,
        target_run,
        PytestFeedback(FeedbackKind.ASSERTION_FAILURE, ("tests/test_created.py",), "assertion failed"),
    )
    recorded = policy.record_patch(
        feature_task, ApplyPatchAction(kind="apply_patch", summary="add test", diff=NEW_TEST_DIFF)
    )
    policy.record_new_test_path(feature_task, "tests/test_created.py")
    third = policy.record_pytest(
        feature_task,
        target_run,
        PytestFeedback(FeedbackKind.ASSERTION_FAILURE, ("tests/test_created.py",), "assertion failed"),
    )

    assert (first.verdict, first.rule_id) == (PolicyVerdict.DENY, "tdd.test_change_required")
    assert proposed.verdict is PolicyVerdict.ALLOW
    assert (second.verdict, second.rule_id) == (PolicyVerdict.DENY, "tdd.test_change_required")
    assert recorded.verdict is PolicyVerdict.ALLOW
    assert third.verdict is PolicyVerdict.ALLOW
    assert feature_task.tdd_phase is TddPhase.RED_OBSERVED


def test_feature_red_requires_registered_changed_test_target_and_feedback(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches an unrelated assertion failure unlocking a new source file."""
    _record_passing_baseline(policy, feature_task)
    created_test = ApplyPatchAction(kind="apply_patch", summary="add test", diff=NEW_TEST_DIFF)
    source = ApplyPatchAction(kind="apply_patch", summary="add source", diff=NEW_SOURCE_DIFF)
    policy.record_patch(feature_task, created_test)
    policy.record_new_test_path(feature_task, "tests/test_created.py")

    unrelated = policy.record_pytest(
        feature_task,
        RunPytestAction(kind="run_pytest", summary="run unrelated", targets=("tests/test_existing.py",)),
        PytestFeedback(
            FeedbackKind.ASSERTION_FAILURE,
            ("tests/test_existing.py::test_existing",),
            "assert False",
        ),
    )

    assert (unrelated.verdict, unrelated.rule_id) == (
        PolicyVerdict.DENY,
        "tdd.changed_test_required",
    )
    assert policy.decide(feature_task, source).rule_id == "tdd.red_required"

    matching = policy.record_pytest(
        feature_task,
        RunPytestAction(kind="run_pytest", summary="run created", targets=("tests/test_created.py",)),
        PytestFeedback(
            FeedbackKind.ASSERTION_FAILURE,
            ("tests/test_created.py::test_created",),
            "assert False",
        ),
    )

    assert matching.verdict is PolicyVerdict.ALLOW
    assert policy.decide(feature_task, source).verdict is PolicyVerdict.ALLOW


def test_feature_red_feedback_must_match_its_own_changed_test_target(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches feedback from a second created test unlocking the first test's run."""
    _record_passing_baseline(policy, feature_task)
    first = ApplyPatchAction(kind="apply_patch", summary="add first test", diff=NEW_TEST_DIFF)
    second = ApplyPatchAction(kind="apply_patch", summary="add second test", diff=SECOND_NEW_TEST_DIFF)
    policy.record_patch(feature_task, first)
    policy.record_new_test_path(feature_task, "tests/test_created.py")
    policy.record_patch(feature_task, second)
    policy.record_new_test_path(feature_task, "tests/test_second_created.py")
    target_first = RunPytestAction(
        kind="run_pytest", summary="run first", targets=("tests/test_created.py",)
    )
    source = ApplyPatchAction(kind="apply_patch", summary="add source", diff=NEW_SOURCE_DIFF)

    crossed = policy.record_pytest(
        feature_task,
        target_first,
        PytestFeedback(
            FeedbackKind.ASSERTION_FAILURE,
            ("tests/test_second_created.py::test_second_created",),
            "assert False",
        ),
    )

    assert (crossed.verdict, crossed.rule_id) == (
        PolicyVerdict.DENY,
        "tdd.changed_test_required",
    )
    assert policy.decide(feature_task, source).rule_id == "tdd.red_required"
    matching = policy.record_pytest(
        feature_task,
        target_first,
        PytestFeedback(
            FeedbackKind.ASSERTION_FAILURE,
            ("tests/test_created.py::test_created",),
            "assert False",
        ),
    )
    assert matching.verdict is PolicyVerdict.ALLOW
    assert policy.decide(feature_task, source).verdict is PolicyVerdict.ALLOW


def test_feature_red_from_modified_existing_test_unlocks_source_patch(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches feature RED evidence being restricted to newly created tests."""
    _record_passing_baseline(policy, feature_task)
    changed_test = ApplyPatchAction(kind="apply_patch", summary="change test", diff=TEST_DIFF)
    source = ApplyPatchAction(kind="apply_patch", summary="change source", diff=SOURCE_DIFF)
    policy.record_read(
        feature_task,
        ReadFileAction(kind="read_file", summary="read test", path="tests/test_example.py"),
    )
    policy.record_read(
        feature_task,
        ReadFileAction(kind="read_file", summary="read source", path="src/example.py"),
    )

    assert policy.record_patch(feature_task, changed_test).verdict is PolicyVerdict.ALLOW
    matching = policy.record_pytest(
        feature_task,
        RunPytestAction(
            kind="run_pytest", summary="run changed test", targets=("tests/test_example.py",)
        ),
        PytestFeedback(
            FeedbackKind.ASSERTION_FAILURE,
            ("tests/test_example.py::test_after",),
            "assert actual == expected",
        ),
    )

    assert matching.verdict is PolicyVerdict.ALLOW
    assert feature_task.tdd_phase is TddPhase.RED_OBSERVED
    assert policy.decide(feature_task, source).verdict is PolicyVerdict.ALLOW

    unrelated_task = TaskState(
        description="Reject unrelated feature red",
        intent=TaskIntent.CODING,
        config=feature_task.config,
    )
    _record_passing_baseline(policy, unrelated_task)
    policy.record_read(
        unrelated_task,
        ReadFileAction(kind="read_file", summary="read test", path="tests/test_example.py"),
    )
    assert policy.record_patch(unrelated_task, changed_test).verdict is PolicyVerdict.ALLOW
    unrelated = policy.record_pytest(
        unrelated_task,
        RunPytestAction(
            kind="run_pytest", summary="run changed test", targets=("tests/test_example.py",)
        ),
        PytestFeedback(
            FeedbackKind.ASSERTION_FAILURE,
            ("tests/test_other.py::test_other",),
            "assert actual == expected",
        ),
    )

    assert unrelated.verdict is PolicyVerdict.DENY
    assert unrelated_task.tdd_phase is TddPhase.TEST_DESIGN


def test_complete_assertion_baseline_records_all_repair_targets_and_red(
    policy: PolicyEngine, tmp_path: Path
) -> None:
    """Catches automatic repair classification dropping a baseline assertion node."""
    targets = (
        "tests/test_parser.py::test_bad_input",
        "tests/test_parser.py::test_empty_input",
    )
    unbaselined = TaskState(
        description="Persisted red without baseline evidence",
        intent=TaskIntent.CODING,
        config=safe_config(tmp_path),
    )
    unbaselined.path = TaskPath.REPAIR
    unbaselined.repair_targets = targets
    unbaselined.tdd_phase = TddPhase.RED_OBSERVED
    policy.record_read(
        unbaselined,
        ReadFileAction(kind="read_file", summary="read parser", path="src/example.py"),
    )
    source_patch = ApplyPatchAction(
        kind="apply_patch", summary="repair parser", diff=SOURCE_DIFF
    )

    without_baseline = policy.decide(unbaselined, source_patch)

    assert (without_baseline.verdict, without_baseline.rule_id) == (
        PolicyVerdict.DENY,
        "tdd.baseline_required",
    )

    task = TaskState(
        description="Repair parser",
        intent=TaskIntent.CODING,
        config=safe_config(tmp_path),
    )

    recorded = policy.record_pytest(
        task,
        RunPytestAction(kind="run_pytest", summary="run configured suite", targets=()),
        PytestFeedback(FeedbackKind.ASSERTION_FAILURE, targets, "assert actual == expected"),
    )
    policy.record_read(
        task,
        ReadFileAction(kind="read_file", summary="read parser", path="src/example.py"),
    )
    source = policy.decide(task, source_patch)

    assert (recorded.verdict, recorded.rule_id) == (
        PolicyVerdict.ALLOW,
        "tdd.red_recorded",
    )
    assert task.tdd_phase is TddPhase.RED_OBSERVED
    assert task.path is TaskPath.REPAIR
    assert task.repair_targets == targets
    assert source.verdict is PolicyVerdict.ALLOW


def test_targeted_assertion_failure_does_not_establish_automatic_baseline(
    policy: PolicyEngine, tmp_path: Path
) -> None:
    """Catches a targeted run claiming that the unrelated starting suite is healthy."""
    selected = "tests/test_parser.py::test_bad_input"
    task = TaskState(
        description="Repair parser",
        intent=TaskIntent.CODING,
        config=safe_config(tmp_path),
    )

    targeted = policy.record_pytest(
        task,
        RunPytestAction(kind="run_pytest", summary="run selected test", targets=(selected,)),
        PytestFeedback(FeedbackKind.ASSERTION_FAILURE, (selected,), "assert actual == expected"),
    )

    assert (targeted.verdict, targeted.rule_id) == (
        PolicyVerdict.DENY,
        "tdd.baseline_required",
    )
    assert task.tdd_phase is TddPhase.TEST_DESIGN


def test_repair_iteration_rejects_assertion_outside_automatic_repair_set(
    policy: PolicyEngine, tmp_path: Path
) -> None:
    """Catches later pytest feedback expanding the immutable automatic repair set."""
    selected = "tests/test_parser.py::test_bad_input"
    task = TaskState(
        description="Repair parser",
        intent=TaskIntent.CODING,
        config=safe_config(tmp_path),
    )
    baseline = policy.record_pytest(
        task,
        RunPytestAction(kind="run_pytest", summary="run configured suite", targets=()),
        PytestFeedback(FeedbackKind.ASSERTION_FAILURE, (selected,), "assert actual == expected"),
    )
    unrelated = policy.record_pytest(
        task,
        RunPytestAction(kind="run_pytest", summary="retry suite", targets=()),
        PytestFeedback(
            FeedbackKind.ASSERTION_FAILURE,
            ("tests/test_parser.py::test_other_input",),
            "assert actual == expected",
        ),
    )

    assert baseline.verdict is PolicyVerdict.ALLOW
    assert (unrelated.verdict, unrelated.rule_id) == (
        PolicyVerdict.DENY,
        "tdd.repair_target_required",
    )
    assert task.repair_targets == (selected,)


def test_automatic_policy_refuses_runtime_feedback_with_a_source_assert_line(
    policy: PolicyEngine, tmp_path: Path
) -> None:
    """Catches a source assertion in a TypeError traceback advancing the bugfix red gate."""
    target = "tests/test_parser.py::test_bad_input"
    task = TaskState(
        description="Repair parser",
        intent=TaskIntent.CODING,
        config=safe_config(tmp_path),
    )
    feedback = FeedbackCollector().collect(
        PytestRun(
            1,
            "FAILED tests/test_parser.py::test_bad_input - TypeError: unsupported input\n"
            "    assert payload is not None\n"
            "E   TypeError: unsupported input\n",
            "",
            False,
        )
    )
    decision = policy.record_pytest(
        task,
        RunPytestAction(kind="run_pytest", summary="run configured suite", targets=()),
        feedback,
    )

    assert feedback.kind is FeedbackKind.EXECUTION_ERROR
    assert decision.rule_id == "tdd.baseline_invalid"
    assert task.tdd_phase is TddPhase.TEST_DESIGN


def test_source_patch_requires_a_current_read_after_red(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches a read-before-patch bypass for a production file."""
    _record_bugfix_red_baseline(policy, ready_bugfix_task)
    result = policy.decide(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "patch.read_required")


def test_deciding_source_patch_does_not_transition_to_implementation(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches decide() treating an unexecuted patch as a successful write."""
    _record_bugfix_red_baseline(policy, ready_bugfix_task)
    policy.record_read(
        ready_bugfix_task, ReadFileAction(kind="read_file", summary="read source", path="src/example.py")
    )
    result = policy.decide(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    )

    assert result.verdict is PolicyVerdict.ALLOW
    assert ready_bugfix_task.tdd_phase is TddPhase.RED_OBSERVED
    recorded = policy.record_patch(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    )
    assert recorded.verdict is PolicyVerdict.ALLOW
    assert ready_bugfix_task.tdd_phase is TddPhase.IMPLEMENTATION


def test_passing_target_pytest_does_not_allow_completed_finish(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches finish.completed accepting target-only green test evidence."""
    _record_bugfix_red_baseline(policy, ready_bugfix_task)
    policy.record_read(
        ready_bugfix_task, ReadFileAction(kind="read_file", summary="read source", path="src/example.py")
    )
    source_patch = ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    policy.decide(
        ready_bugfix_task, source_patch
    )
    policy.record_patch(ready_bugfix_task, source_patch)
    target_run = RunPytestAction(
        kind="run_pytest", summary="run selected test", targets=("tests/test_example.py",)
    )
    policy.record_pytest(ready_bugfix_task, target_run, PytestFeedback(FeedbackKind.PASSED, (), ""))

    result = policy.decide(
        ready_bugfix_task, FinishAction(kind="finish", summary="all tests pass", status="completed")
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "pytest.full_suite_required")
    full_run = RunPytestAction(kind="run_pytest", summary="run configured suite")
    policy.record_pytest(ready_bugfix_task, full_run, PytestFeedback(FeedbackKind.PASSED, (), ""))
    completed = policy.decide(
        ready_bugfix_task, FinishAction(kind="finish", summary="all tests pass", status="completed")
    )
    assert completed.verdict is PolicyVerdict.ALLOW


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
    action = DeletePathAction(kind="delete_path", summary="remove file", path="obsolete.txt")
    pending = _request_approval(policy, ready_bugfix_task, action)

    accepted = policy.apply_approval(pending, action, decision="once")
    replayed = policy.apply_approval(pending, action, decision="once")

    assert accepted.verdict is PolicyVerdict.ALLOW
    assert (replayed.verdict, replayed.rule_id) == (PolicyVerdict.DENY, "approval.already_used")


@pytest.mark.parametrize("phase", [TddPhase.IMPLEMENTATION, TddPhase.GREEN_OBSERVED])
def test_test_deletion_is_directly_denied_after_test_design(
    policy: PolicyEngine, ready_bugfix_task: TaskState, phase: TddPhase
) -> None:
    """Catches approval bypassing the TDD phase boundary for deleting tests."""
    ready_bugfix_task.tdd_phase = phase

    result = policy.decide(
        ready_bugfix_task,
        DeletePathAction(
            kind="delete_path",
            summary="remove regression",
            path="tests/test_example.py",
        ),
    )

    assert (result.verdict, result.rule_id) == (
        PolicyVerdict.DENY,
        "tdd.test_delete_phase",
    )


def test_explicit_approval_request_is_exact_hitl_and_not_permanent_eligible(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches a non-executable model request being recorded as ordinary allowed work."""
    action = RequestApprovalAction(
        kind="request_approval",
        summary="ask user",
        reason="model-controlled reason must not become policy evidence",
    )

    pending = _request_approval(policy, feature_task, action)

    assert pending.verdict is PolicyVerdict.APPROVAL_REQUIRED
    assert pending.action_hash == action.stable_hash()
    assert pending.permanent_eligible is False


@pytest.mark.parametrize("invalid_decision", ["invalid", True])
def test_invalid_approval_decision_is_denied_without_consuming_pending_action(
    policy: PolicyEngine,
    ready_bugfix_task: TaskState,
    invalid_decision: object,
) -> None:
    """Catches unknown or wrong-type decisions falling through as one-time approval."""
    action = DeletePathAction(kind="delete_path", summary="remove file", path="obsolete.txt")
    pending = _request_approval(policy, ready_bugfix_task, action)

    invalid = policy.apply_approval(
        pending,
        action,
        decision=invalid_decision,  # type: ignore[arg-type]
    )
    retry = policy.apply_approval(pending, action, decision="once")

    assert (invalid.verdict, invalid.rule_id) == (
        PolicyVerdict.DENY,
        "approval.invalid_decision",
    )
    assert retry.verdict is PolicyVerdict.ALLOW


def test_approval_does_not_match_a_different_action(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches an approval binding that ignores the exact proposed action hash."""
    first = DeletePathAction(kind="delete_path", summary="first", path="obsolete.txt")
    pending = _request_approval(policy, ready_bugfix_task, first)

    result = policy.apply_approval(
        pending,
        DeletePathAction(kind="delete_path", summary="other", path="other.txt"),
        decision="once",
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "approval.action_mismatch")


def test_approval_cannot_override_a_direct_denial(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches an approval route that turns a TDD violation into permission."""
    action = ApplyPatchAction(kind="apply_patch", summary="change", diff=SOURCE_DIFF)
    pending = _request_approval(policy, feature_task, action)

    result = policy.apply_approval(pending, action, decision="once")

    assert (pending.verdict, pending.rule_id) == (PolicyVerdict.DENY, "tdd.red_required")
    assert result.verdict is PolicyVerdict.DENY


def test_dependency_command_is_directly_denied(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches a dependency command being sent to approval instead of rejected."""
    action = RunCommandAction(kind="run_command", summary="install", args=("pip", "install", "ruff"))

    result = policy.decide(feature_task, action)

    assert result.verdict is PolicyVerdict.DENY


def test_only_exact_read_only_git_diff_can_wait_for_approval(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches an allowed command family that can modify the repository or run helpers."""
    action = RunCommandAction(
        kind="run_command",
        summary="check diff whitespace",
        args=("git", "diff", "--no-ext-diff", "--check"),
    )

    result = policy.decide(feature_task, action)

    assert (result.verdict, result.rule_id) == (
        PolicyVerdict.APPROVAL_REQUIRED,
        "command.read_only_approval_required",
    )
    assert result.permanent_eligible is True


@pytest.mark.parametrize(
    ("args", "current_branch"),
    [
        (("git", "diff", "--no-ext-diff", "--check"), None),
        (("git", "push", "origin", "main"), "main"),
        (("python", "-m", "pip", "install", "ruff", "httpx>=0.27"), None),
    ],
)
def test_each_exact_command_family_requires_approval_without_a_rule(
    tmp_path: Path,
    feature_task: TaskState,
    args: tuple[str, ...],
    current_branch: str | None,
) -> None:
    """Catches an eligible command family executing before any human decision."""
    result = PolicyEngine(
        tmp_path,
        current_branch_provider=lambda: current_branch,
    ).decide(
        feature_task,
        RunCommandAction(kind="run_command", summary="eligible command", args=args),
    )

    assert result.verdict is PolicyVerdict.APPROVAL_REQUIRED


@pytest.mark.parametrize(
    "action",
    [
        RunCommandAction(
            kind="run_command",
            summary="long but valid package install",
            args=(
                "python",
                "-m",
                "pip",
                "install",
                *(f"package{index}==1.2.3" for index in range(40)),
            ),
        ),
        DeletePathAction(
            kind="delete_path",
            summary="long but root-contained delete",
            path="/".join(("directory",) * 60) + "/obsolete.txt",
        ),
        ApplyPatchAction(
            kind="apply_patch",
            summary="long but valid non-code patch",
            diff=(
                "--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n".format(
                    path="/".join(("directory",) * 60) + "/README.md"
                )
            ),
        ),
    ],
    ids=("pip", "delete", "patch"),
)
def test_action_that_cannot_be_fully_projected_is_denied_before_approval(
    policy: PolicyEngine,
    feature_task: TaskState,
    action: RunCommandAction | DeletePathAction | ApplyPatchAction,
) -> None:
    """Catches approval of an action whose decision inputs would be truncated in the UI."""
    result = policy.decide(feature_task, action)

    assert (result.verdict, result.rule_id) == (
        PolicyVerdict.DENY,
        "approval.projection_too_long",
    )


def test_matching_always_allowed_git_push_rule_bypasses_new_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feature_task: TaskState
) -> None:
    """Catches a valid permanent push rule being ignored on the same branch and root."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    store = CommandRuleStore(tmp_path)
    approved = RunCommandAction(
        kind="run_command", summary="first wording", args=("git", "push", "origin", "main")
    )
    store.add_from(approved, "main")
    same_push = RunCommandAction(
        kind="run_command", summary="new wording", args=("git", "push", "origin", "main")
    )

    result = PolicyEngine(
        tmp_path,
        current_branch_provider=lambda: "main",
        command_rules=store,
    ).decide(
        feature_task, same_push
    )

    assert (result.verdict, result.rule_id) == (
        PolicyVerdict.ALLOW,
        "command.persistent_rule",
    )


def test_command_decision_uses_one_fresh_branch_for_family_and_rule_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feature_task: TaskState
) -> None:
    """Catches either caching a branch or reading two branch values in one decision."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    action = RunCommandAction(
        kind="run_command",
        summary="push current branch",
        args=("git", "push", "origin", "main"),
    )
    store = CommandRuleStore(tmp_path)
    store.add_from(action, "main")
    branch = "main"
    reads: list[str] = []

    def current_branch() -> str:
        reads.append(branch)
        return branch

    policy = PolicyEngine(
        tmp_path,
        current_branch_provider=current_branch,
        command_rules=store,
    )

    allowed = policy.decide(feature_task, action)
    branch = "feature"
    changed = policy.decide(feature_task, action)

    assert (allowed.verdict, allowed.rule_id) == (
        PolicyVerdict.ALLOW,
        "command.persistent_rule",
    )
    assert changed.verdict is PolicyVerdict.DENY
    assert reads == ["main", "feature"]


@pytest.mark.parametrize("approval_decision", ("once", "always"))
def test_changed_branch_invalidates_approval_before_it_is_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_task: TaskState,
    approval_decision: ApprovalDecision,
) -> None:
    """Catches exact approval authorizing a push after the repository branch changes."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    branch = "main"
    policy = PolicyEngine(tmp_path, current_branch_provider=lambda: branch)
    action = RunCommandAction(
        kind="run_command",
        summary="push current branch",
        args=("git", "push", "origin", "main"),
    )
    pending = _request_approval(policy, feature_task, action)

    branch = "feature"
    invalidated = policy.apply_approval(pending, action, decision=approval_decision)
    branch = "main"
    retry = policy.apply_approval(pending, action, decision=approval_decision)

    assert invalidated.verdict is PolicyVerdict.DENY
    assert retry.verdict is PolicyVerdict.ALLOW


def test_policy_rejects_a_rule_store_injected_from_another_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches dependency injection moving durable permission across project roots."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_store = CommandRuleStore(first)
    first_store.add_from(
        RunCommandAction(
            kind="run_command",
            summary="check first project",
            args=("git", "diff", "--no-ext-diff", "--check"),
        ),
        None,
    )

    with pytest.raises(ValueError, match="project root"):
        PolicyEngine(second, command_rules=first_store)


@pytest.mark.parametrize(
    ("args", "rule_id"),
    [
        (("git", "push", "origin", "feature"), "command.not_allowed"),
        (("git", "push", "--force", "origin", "main"), "command.not_allowed"),
        (("python", "-m", "pip", "install", "https://example.invalid/pkg.whl"), "command.not_allowed"),
        (("python", "-m", "pip", "install", ".env"), "command.credentials"),
        (("python", "-m", "pip", "install", "../local-package"), "path.outside_root"),
        (("python", "-m", "pip", "install", "ruff;id"), "command.metacharacter"),
        (("git", "push", "origin", "main>release"), "command.metacharacter"),
    ],
)
def test_branch_url_force_and_direct_denies_run_before_persistent_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature_task: TaskState,
    args: tuple[str, ...],
    rule_id: str,
) -> None:
    """Catches stored approval weakening branch, grammar, credential, path, or shell checks."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    store = CommandRuleStore(tmp_path)
    store.add_from(
        RunCommandAction(
            kind="run_command", summary="approved push", args=("git", "push", "origin", "main")
        ),
        "main",
    )

    result = PolicyEngine(
        tmp_path,
        current_branch_provider=lambda: "main",
        command_rules=store,
    ).decide(
        feature_task,
        RunCommandAction(kind="run_command", summary="unsafe variant", args=args),
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, rule_id)


@pytest.mark.parametrize(
    "args",
    [
        ("git", "show", "src/example.py"),
        ("git", "diff", "--no-ext-diff", "--check", "harness.yaml"),
        ("curl", "https://example.invalid"),
    ],
)
def test_generic_command_cannot_bypass_protected_paths_or_network(
    policy: PolicyEngine, feature_task: TaskState, args: tuple[str, ...]
) -> None:
    """Catches approval of commands outside the narrowly read-only command contract."""
    result = policy.decide(
        feature_task,
        RunCommandAction(kind="run_command", summary="generic command", args=args),
    )

    assert result.verdict is PolicyVerdict.DENY


@pytest.mark.parametrize(
    ("targets", "rule_id"),
    [
        (("src/example.py",), "pytest.target_not_test"),
        (("../outside_test.py",), "pytest.target_outside_root"),
        (("-k", "name"), "pytest.option_not_allowed"),
    ],
)
def test_pytest_rejects_non_test_targets_and_options(
    policy: PolicyEngine,
    feature_task: TaskState,
    targets: tuple[str, ...],
    rule_id: str,
) -> None:
    """Catches pytest arguments escaping the configured test-directory contract."""
    result = policy.decide(
        feature_task,
        RunPytestAction(kind="run_pytest", summary="run selected tests", targets=targets),
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, rule_id)


def test_pytest_rejects_source_hidden_behind_test_parent(
    policy: PolicyEngine, feature_task: TaskState
) -> None:
    """Catches path classification that treats a normalized source target as a test."""
    result = policy.decide(
        feature_task,
        RunPytestAction(
            kind="run_pytest",
            summary="run disguised source",
            targets=("tests/../src/example.py",),
        ),
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "pytest.target_not_test")


def test_read_and_patch_equivalent_source_paths_share_a_normalized_identity(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches read records and patches treating one root-internal source as different paths."""
    _record_bugfix_red_baseline(policy, ready_bugfix_task)
    policy.record_read(
        ready_bugfix_task,
        ReadFileAction(kind="read_file", summary="read source", path="src/example.py"),
    )

    result = policy.decide(
        ready_bugfix_task,
        ApplyPatchAction(kind="apply_patch", summary="change", diff=ROOT_INTERNAL_SOURCE_DIFF),
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.ALLOW, "patch.source_allowed")


def test_patch_rejects_deletion_hidden_beside_a_valid_test_change(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches a parser that validates only +++ paths and misses a deleted sensitive file."""
    diff = """--- a/.env
+++ /dev/null
@@ -1 +0,0 @@
-token=secret
--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -1 +1 @@
-before
+after
"""

    result = policy.decide(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=diff)
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "patch.delete_unsupported")


def test_patch_rejects_rename_hidden_beside_a_valid_test_change(
    policy: PolicyEngine, ready_bugfix_task: TaskState
) -> None:
    """Catches a parser that ignores an old path while admitting a renamed file."""
    diff = """--- a/src/old.py
+++ b/src/new.py
@@ -1 +1 @@
-before
+after
--- a/tests/test_example.py
+++ b/tests/test_example.py
@@ -1 +1 @@
-before
+after
"""

    result = policy.decide(
        ready_bugfix_task, ApplyPatchAction(kind="apply_patch", summary="change", diff=diff)
    )

    assert (result.verdict, result.rule_id) == (PolicyVerdict.DENY, "patch.rename_unsupported")


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
            "command.privileged",
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
