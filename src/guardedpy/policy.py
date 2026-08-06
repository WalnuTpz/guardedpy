"""Deterministic safety and TDD decisions for model-proposed actions."""

from __future__ import annotations

from pathlib import Path, PurePath
import re
from typing import Callable
from uuid import UUID

from guardedpy.actions import (
    Action,
    ApplyPatchAction,
    DeletePathAction,
    FinishAction,
    ListFilesAction,
    MEMORY_PROPOSAL_TEXT_MAX_LENGTH,
    ProposeMemoryAction,
    ReadFileAction,
    RequestApprovalAction,
    RunCommandAction,
    RunPytestAction,
    stable_hash,
)
from guardedpy.command_rules import CommandRuleStore, command_rule_kind, has_shell_metacharacter
from guardedpy.domain import (
    ApprovalDecision,
    CommandRuleKind,
    FeedbackKind,
    PolicyDecision,
    PolicyVerdict,
    TaskMode,
    TaskState,
    TddPhase,
    is_approval_decision,
)
from guardedpy.feedback import PytestFeedback


APPROVAL_ACTION_PROJECTION_MAX_LENGTH = 500
_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)


def patch_operations(diff: str) -> tuple[tuple[tuple[str, bool], ...], str | None]:
    """Return file operations from complete unified-diff header and hunk boundaries."""
    lines = diff.splitlines()
    if any(line.startswith(("rename from ", "rename to ", "similarity index ")) for line in lines):
        return (), "patch.rename_unsupported"

    operations: list[tuple[str, bool]] = []
    index = 0
    while index < len(lines):
        old_header = lines[index]
        if not old_header.startswith("--- ") or index + 1 >= len(lines):
            return (), "patch.invalid"
        new_header = lines[index + 1]
        if not new_header.startswith("+++ "):
            return (), "patch.invalid"
        old_path = _diff_path(old_header, "--- a/")
        new_path = _diff_path(new_header, "+++ b/")
        if old_header == "--- /dev/null":
            if new_path is None:
                return (), "patch.invalid"
            operations.append((new_path, True))
        elif new_header == "+++ /dev/null":
            return (), "patch.delete_unsupported"
        elif old_path is None or new_path is None:
            return (), "patch.invalid"
        elif old_path != new_path:
            return (), "patch.rename_unsupported"
        else:
            operations.append((old_path, False))
        index += 2

        parsed_hunk = False
        while index < len(lines) and lines[index].startswith("@@ "):
            match = _HUNK_HEADER.fullmatch(lines[index])
            if match is None:
                return (), "patch.invalid"
            _old_start, old_count, _new_start, new_count = match.groups()
            expected_old = int(old_count) if old_count is not None else 1
            expected_new = int(new_count) if new_count is not None else 1
            observed_old = 0
            observed_new = 0
            index += 1
            while observed_old < expected_old or observed_new < expected_new:
                if index >= len(lines) or not lines[index].startswith((" ", "+", "-")):
                    return (), "patch.invalid"
                marker = lines[index][0]
                if marker != "+":
                    observed_old += 1
                if marker != "-":
                    observed_new += 1
                if observed_old > expected_old or observed_new > expected_new:
                    return (), "patch.invalid"
                index += 1
            parsed_hunk = True
        if not parsed_hunk or (index < len(lines) and not lines[index].startswith("--- ")):
            return (), "patch.invalid"

    return (tuple(operations), None) if operations else ((), "patch.invalid")


def approval_action_projection(action: Action) -> str | None:
    """Return the complete deterministic command or path projection for approval."""
    if isinstance(action, RunCommandAction):
        return f"Command: {' '.join(action.args)}"
    if isinstance(action, DeletePathAction):
        return f"Path: {action.path}"
    if isinstance(action, ApplyPatchAction):
        operations, error_rule = patch_operations(action.diff)
        if error_rule is not None:
            return None
        return f"Paths: {', '.join(path for path, _created in operations)}"
    if isinstance(action, RequestApprovalAction):
        return "Approval request"
    return None


def _diff_path(header: str, prefix: str) -> str | None:
    path = header.removeprefix(prefix).split("\t", maxsplit=1)[0]
    return path if header.startswith(prefix) and path else None


class PolicyEngine:
    """Evaluate actions without executing them or persisting raw pending actions."""

    def __init__(
        self,
        project_root: Path,
        *,
        current_branch_provider: Callable[[], str | None] | None = None,
        command_rules: CommandRuleStore | None = None,
    ) -> None:
        self._project_root = project_root.resolve()
        if command_rules is not None and command_rules.project_root != self._project_root:
            raise ValueError("command rule store belongs to another project root")
        self._current_branch_provider = current_branch_provider or (lambda: None)
        self._command_rules = command_rules or CommandRuleStore(self._project_root)
        self._read_paths: dict[UUID, set[str]] = {}
        self._changed_test_paths: dict[UUID, set[str]] = {}
        self._new_test_paths: dict[UUID, set[str]] = {}
        self._baseline_recorded: set[UUID] = set()
        self._full_suite_green: set[UUID] = set()
        self._pending_approvals: set[tuple[UUID, str]] = set()

    def decide(self, task: TaskState, action: Action) -> PolicyDecision:
        """Return the deterministic result for one action without executing it."""
        if isinstance(action, (ListFilesAction, ReadFileAction)):
            return self._read_decision(task, action.path, action)
        if isinstance(action, ApplyPatchAction):
            return self._patch_decision(task, action)
        if isinstance(action, DeletePathAction):
            return self._delete_decision(task, action)
        if isinstance(action, RunPytestAction):
            return self._pytest_decision(task, action)
        if isinstance(action, RunCommandAction):
            return self._command_decision(task, action)
        if isinstance(action, ProposeMemoryAction):
            return self._memory_proposal_decision(task, action)
        if isinstance(action, RequestApprovalAction):
            return self._approval_required(
                task,
                action,
                "approval.requested",
                "explicit user approval is required",
            )
        if isinstance(action, FinishAction):
            return self._finish_decision(task, action)
        raise TypeError(f"unsupported action: {type(action).__name__}")

    def record_read(self, task: TaskState, action: ReadFileAction) -> PolicyDecision:
        """Record a permitted current-version read for later patch authorization."""
        decision = self.decide(task, action)
        if decision.verdict is PolicyVerdict.ALLOW:
            normalized_path = self._normalized_project_path(action.path)
            assert normalized_path is not None
            self._read_paths.setdefault(task.id, set()).add(normalized_path)
        return decision

    def record_patch(self, task: TaskState, action: ApplyPatchAction) -> PolicyDecision:
        """Record a successful patch after the tool executor has applied it."""
        decision = self.decide(task, action)
        if decision.verdict is not PolicyVerdict.ALLOW:
            return decision

        paths, _ = self._patch_paths(action.diff)
        normalized_paths = self._normalized_paths(paths)
        assert normalized_paths is not None
        category = self._path_category(task, normalized_paths[0])
        self._full_suite_green.discard(task.id)
        if category == "test":
            self._changed_test_paths.setdefault(task.id, set()).update(normalized_paths)
        else:
            task.tdd_phase = TddPhase.IMPLEMENTATION
        self._invalidate_reads(task, normalized_paths)
        return decision

    def record_new_test_path(self, task: TaskState, path: str) -> None:
        """Record a test created by a successful atomic workspace patch."""
        normalized_path = self._normalized_project_path(path)
        if normalized_path is None or self._path_category(task, normalized_path) != "test":
            raise ValueError("created path must be a root-contained test path")
        self._new_test_paths.setdefault(task.id, set()).add(normalized_path)

    def audit_feedback_node(self, task: TaskState, node_id: str) -> str | None:
        """Project an untrusted pytest node to its root-relative configured test path."""
        path = node_id.split("::", maxsplit=1)[0]
        normalized_path = self._normalized_project_path(path)
        if (
            normalized_path is None
            or self._path_category(task, normalized_path) != "test"
            or not (self._project_root / normalized_path).is_file()
        ):
            return None
        return normalized_path

    def record_delete(
        self, task: TaskState, action: DeletePathAction, decision: PolicyDecision
    ) -> PolicyDecision:
        """Invalidate green-suite evidence only after an allowed deletion has succeeded."""
        if decision.verdict is not PolicyVerdict.ALLOW or decision.action_hash != stable_hash(action):
            return decision
        self._full_suite_green.discard(task.id)
        return decision

    def record_pytest(
        self, task: TaskState, action: RunPytestAction, feedback: PytestFeedback
    ) -> PolicyDecision:
        """Advance the TDD state only from an observed pytest outcome."""
        decision = self.decide(task, action)
        if decision.verdict is not PolicyVerdict.ALLOW:
            return decision
        if task.tdd_phase is TddPhase.TEST_DESIGN and task.id not in self._baseline_recorded:
            return self._record_starting_baseline(task, action, feedback)
        if feedback.kind is FeedbackKind.ASSERTION_FAILURE:
            if task.tdd_phase is not TddPhase.TEST_DESIGN:
                return self._deny(task, None, "tdd.red_out_of_sequence", "red result is out of sequence")
            if task.mode is TaskMode.FEATURE and not self._changed_test_paths.get(task.id):
                return self._deny(
                    task,
                    None,
                    "tdd.test_change_required",
                    "a feature task must change a test before observing red",
                )
            if task.mode is TaskMode.FEATURE and not self._feature_red_matches_changed_test(
                task, action, feedback
            ):
                return self._deny(
                    task,
                    None,
                    "tdd.changed_test_required",
                    "a feature red result must target its successfully changed test",
                )
            if task.mode is TaskMode.BUGFIX and (
                action.targets
                or task.bugfix_target is None
                or feedback.node_ids != (task.bugfix_target,)
            ):
                return self._deny(
                    task,
                    None,
                    "tdd.bugfix_target_assertion_required",
                    "a bugfix red result must be an assertion failure for its selected target",
                )
            task.tdd_phase = TddPhase.RED_OBSERVED
            return self._allow(task, None, "tdd.red_recorded", "red pytest result recorded")
        if feedback.kind is not FeedbackKind.PASSED:
            return self._deny(
                task,
                None,
                "tdd.bugfix_target_assertion_required"
                if task.mode is TaskMode.BUGFIX
                else "tdd.assertion_failure_required",
                "a red result must be an assertion failure"
                if task.mode is TaskMode.FEATURE
                else "a bugfix red result must be an assertion failure for its selected target",
            )
        if task.tdd_phase is TddPhase.IMPLEMENTATION:
            task.tdd_phase = TddPhase.GREEN_OBSERVED
        elif task.tdd_phase is not TddPhase.GREEN_OBSERVED:
            return self._deny(task, None, "tdd.green_out_of_sequence", "green result is out of sequence")
        if not action.targets:
            self._full_suite_green.add(task.id)
        return self._allow(task, None, "tdd.green_recorded", "green pytest result recorded")

    def _record_starting_baseline(
        self, task: TaskState, action: RunPytestAction, feedback: PytestFeedback
    ) -> PolicyDecision:
        """Record only target-free evidence that proves the task's starting suite state."""
        if action.targets:
            return self._deny(
                task,
                None,
                "tdd.baseline_required",
                "a target-free configured suite run is required before changes",
            )
        if feedback.kind is FeedbackKind.PASSED:
            self._baseline_recorded.add(task.id)
            return self._allow(
                task,
                None,
                "tdd.baseline_recorded",
                "passing task-start baseline recorded",
            )
        if (
            task.mode is TaskMode.BUGFIX
            and feedback.kind is FeedbackKind.ASSERTION_FAILURE
            and task.bugfix_target is not None
            and feedback.node_ids == (task.bugfix_target,)
        ):
            self._baseline_recorded.add(task.id)
            task.tdd_phase = TddPhase.RED_OBSERVED
            return self._allow(task, None, "tdd.red_recorded", "red pytest result recorded")
        if task.mode is TaskMode.BUGFIX:
            return self._deny(
                task,
                None,
                "tdd.bugfix_target_assertion_required",
                "a bugfix baseline must pass or fail only its selected assertion target",
            )
        return self._deny(
            task,
            None,
            "tdd.baseline_pass_required",
            "a feature baseline must pass before tests change",
        )

    def request_approval(
        self,
        task: TaskState,
        action: Action,
        decision: PolicyDecision,
    ) -> PolicyDecision:
        """Register the exact policy decision already bound to this task and action."""
        if decision.task_id != task.id or decision.action_hash != stable_hash(action):
            return self._deny(
                task,
                action,
                "approval.decision_mismatch",
                "approval registration must use the exact policy decision",
            )
        if decision.verdict is PolicyVerdict.APPROVAL_REQUIRED:
            self._pending_approvals.add((task.id, stable_hash(action)))
        return decision

    def apply_approval(
        self, pending: PolicyDecision, action: Action, decision: ApprovalDecision
    ) -> PolicyDecision:
        """Consume an exact pending approval as rejection, one-time, or constrained durable access."""
        if not is_approval_decision(decision):
            return self._deny(
                None,
                action,
                "approval.invalid_decision",
                "approval decision must be reject, once, or always",
            )
        if pending.verdict is not PolicyVerdict.APPROVAL_REQUIRED or pending.task_id is None:
            return self._deny(None, action, "approval.not_pending", "action has no pending approval")
        if pending.action_hash != stable_hash(action):
            return self._deny(None, action, "approval.action_mismatch", "approval is bound to another action")
        approval_key = (pending.task_id, pending.action_hash)
        if approval_key not in self._pending_approvals:
            return self._deny(None, action, "approval.already_used", "approval was already consumed")
        if decision == "reject":
            self._pending_approvals.remove(approval_key)
            return self._deny(None, action, "approval.declined", "user declined the action")
        current_branch: str | None = None
        if isinstance(action, RunCommandAction):
            current_branch = self._current_branch_provider()
            if command_rule_kind(action, current_branch) is None:
                return self._deny(
                    None,
                    action,
                    "approval.command_invalidated",
                    "the command no longer matches its constrained approval family",
                )
        if decision == "always":
            if not isinstance(action, RunCommandAction):
                return self._deny(
                    None,
                    action,
                    "approval.permanent_command_only",
                    "only constrained command families support permanent approval",
                )
        self._pending_approvals.remove(approval_key)
        return PolicyDecision(
            verdict=PolicyVerdict.ALLOW,
            rule_id="approval.granted_always" if decision == "always" else "approval.granted",
            reason=(
                "user approved a constrained persistent command rule"
                if decision == "always"
                else "user approved this exact action once"
            ),
            task_id=pending.task_id,
            action_hash=pending.action_hash,
        )

    def finalize_command_approval(
        self,
        task: TaskState,
        action: RunCommandAction,
        *,
        permanent: bool,
    ) -> PolicyDecision:
        """Revalidate immediately before dispatch and then persist an eligible rule."""
        current_branch = self._current_branch_provider()
        decision = self._command_decision_with_branch(task, action, current_branch)
        if decision.verdict is PolicyVerdict.DENY or not permanent:
            return decision
        try:
            self._command_rules.add_from(action, current_branch)
        except ValueError:
            return self._deny(
                task,
                action,
                "approval.permanent_rule_invalid",
                "action cannot derive a constrained permanent rule",
            )
        return decision

    def _read_decision(self, task: TaskState, path: str, action: Action) -> PolicyDecision:
        normalized_path = self._normalized_project_path(path)
        if normalized_path is None:
            return self._deny(task, action, "path.outside_root", "path must stay inside the project root")
        if self._is_sensitive_path(normalized_path):
            return self._deny(task, action, "path.sensitive", "credentials and harness configuration are unavailable")
        return self._allow(task, action, "read.allowed", "project read is allowed")

    def _patch_decision(self, task: TaskState, action: ApplyPatchAction) -> PolicyDecision:
        operations, error_rule = self._patch_operations(action.diff)
        if error_rule is not None:
            return self._deny(task, action, error_rule, "patch file operation is not supported")
        paths = tuple(path for path, _created in operations)
        normalized_paths = self._normalized_paths(paths)
        if normalized_paths is None:
            return self._deny(task, action, "path.outside_root", "patch path must stay inside the project root")
        if any(self._is_sensitive_path(path) for path in normalized_paths):
            return self._deny(task, action, "path.sensitive", "credentials and harness configuration are unavailable")

        categories = {self._path_category(task, path) for path in normalized_paths}
        if "other" in categories:
            return self._approval_required(task, action, "patch.non_code", "non-code files require approval")
        if len(categories) != 1:
            return self._deny(task, action, "patch.mixed_code_and_test", "source and tests must change in separate TDD actions")

        category = categories.pop()
        created_paths = tuple(
            normalized_path
            for (_path, created), normalized_path in zip(operations, normalized_paths, strict=True)
            if created
        )
        modified_paths = tuple(
            normalized_path
            for (_path, created), normalized_path in zip(operations, normalized_paths, strict=True)
            if not created
        )
        if category == "source" and task.tdd_phase is not TddPhase.RED_OBSERVED:
            return self._deny(task, action, "tdd.red_required", "source patch requires an observed red test")
        if category == "test" and task.tdd_phase is not TddPhase.TEST_DESIGN:
            return self._deny(task, action, "tdd.test_design_required", "test changes must start the TDD sequence")
        if category == "test" and task.mode is TaskMode.FEATURE and task.id not in self._baseline_recorded:
            return self._deny(
                task,
                action,
                "tdd.baseline_required",
                "feature tests require a passing task-start baseline",
            )
        if category == "source" and task.id not in self._baseline_recorded:
            return self._deny(
                task,
                action,
                "tdd.baseline_required",
                "source changes require recorded task-start baseline evidence",
            )
        if category == "source" and created_paths and not self._new_test_paths.get(task.id):
            return self._deny(
                task,
                action,
                "tdd.test_creation_required",
                "new source files require a successfully created test",
            )
        if not self._all_paths_were_read(task, modified_paths):
            return self._deny(task, action, "patch.read_required", "each patched file must first be read")
        if category == "test":
            return self._allow(task, action, "patch.test_allowed", "test patch is allowed before red")
        return self._allow(task, action, "patch.source_allowed", "source patch follows observed red")

    def created_test_paths(self, task: TaskState, action: ApplyPatchAction) -> tuple[str, ...]:
        """Return policy-normalized test paths created by an already allowed patch."""
        operations, error_rule = self._patch_operations(action.diff)
        assert error_rule is None
        normalized_paths = self._normalized_paths(tuple(path for path, _created in operations))
        assert normalized_paths is not None
        return tuple(
            normalized_path
            for (_path, created), normalized_path in zip(operations, normalized_paths, strict=True)
            if created and self._path_category(task, normalized_path) == "test"
        )

    def _delete_decision(self, task: TaskState, action: DeletePathAction) -> PolicyDecision:
        normalized_path = self._normalized_project_path(action.path)
        if normalized_path is None:
            return self._deny(task, action, "path.outside_root", "path must stay inside the project root")
        if self._is_sensitive_path(normalized_path):
            return self._deny(task, action, "path.sensitive", "credentials and harness configuration are unavailable")
        category = self._path_category(task, normalized_path)
        if category == "test" and task.tdd_phase is not TddPhase.TEST_DESIGN:
            return self._deny(
                task,
                action,
                "tdd.test_delete_phase",
                "tests may be deleted only during test design",
            )
        if category == "test" and task.mode is TaskMode.FEATURE and task.id not in self._baseline_recorded:
            return self._deny(
                task,
                action,
                "tdd.baseline_required",
                "feature tests require a passing task-start baseline",
            )
        if category == "source" and task.tdd_phase is not TddPhase.RED_OBSERVED:
            return self._deny(task, action, "tdd.red_required", "source deletion requires an observed red test")
        if category == "source" and task.id not in self._baseline_recorded:
            return self._deny(
                task,
                action,
                "tdd.baseline_required",
                "source changes require recorded task-start baseline evidence",
            )
        return self._approval_required(task, action, "delete.approval_required", "deletion requires approval")

    def _command_decision(self, task: TaskState, action: RunCommandAction) -> PolicyDecision:
        return self._command_decision_with_branch(
            task,
            action,
            self._current_branch_provider(),
        )

    def _command_decision_with_branch(
        self,
        task: TaskState,
        action: RunCommandAction,
        current_branch: str | None,
    ) -> PolicyDecision:
        command = action.args[0] if action.args else ""
        if command in {"sudo", "doas", "su"}:
            return self._deny(task, action, "command.privileged", "privilege escalation is forbidden")
        if command == "keyring" or any(".env" in argument for argument in action.args):
            return self._deny(task, action, "command.credentials", "credential access is forbidden")
        if any(self._argument_escapes_root(argument) for argument in action.args):
            return self._deny(task, action, "path.outside_root", "command path must stay inside root")
        if has_shell_metacharacter(action.args):
            return self._deny(
                task,
                action,
                "command.metacharacter",
                "shell metacharacters are forbidden in command arguments",
            )
        for argument in action.args:
            normalized_path = self._normalized_project_path(argument)
            if normalized_path is not None and self._is_sensitive_path(normalized_path):
                return self._deny(task, action, "command.sensitive", "harness configuration is unavailable")
        if command in {"rm", "rmdir", "unlink"}:
            return self._deny(task, action, "command.delete", "delete_path is the only deletion action")
        if any(self._path_category(task, argument) in {"source", "test"} for argument in action.args):
            return self._deny(
                task,
                action,
                "command.source_or_test_write",
                "generic commands cannot target source or test directories",
            )
        kind = command_rule_kind(action, current_branch)
        if kind is None:
            return self._deny(
                task,
                action,
                "command.not_allowed",
                "command does not match an exact approvable family",
            )
        projection_denial = self._approval_projection_denial(task, action)
        if projection_denial is not None:
            return projection_denial
        if self._command_rules.matches(action, current_branch):
            return self._allow(
                task,
                action,
                "command.persistent_rule",
                "a matching constrained project rule allows this command",
            )
        if kind is CommandRuleKind.GIT_DIFF_CHECK:
            return self._approval_required(
                task,
                action,
                "command.read_only_approval_required",
                "the read-only Git whitespace check requires approval",
            )
        return self._approval_required(
            task,
            action,
            "command.approval_required",
            "the constrained command requires approval",
        )

    @staticmethod
    def _argument_escapes_root(argument: str) -> bool:
        path = PurePath(argument)
        return path.is_absolute() or ".." in path.parts or argument.startswith("~")

    def _memory_proposal_decision(
        self, task: TaskState, action: ProposeMemoryAction
    ) -> PolicyDecision:
        if not action.text.strip():
            return self._deny(task, action, "memory.text_required", "memory proposal text must be nonblank")
        if len(action.text) > MEMORY_PROPOSAL_TEXT_MAX_LENGTH:
            return self._deny(task, action, "memory.text_too_long", "memory proposal text is too long")
        return self._allow(task, action, "memory.proposal_allowed", "memory proposal is queued for user review")

    def _finish_decision(self, task: TaskState, action: FinishAction) -> PolicyDecision:
        if action.status == "completed" and task.tdd_phase is not TddPhase.GREEN_OBSERVED:
            return self._deny(task, action, "tdd.green_required", "completed finish requires observed green")
        if action.status == "completed" and task.id not in self._full_suite_green:
            return self._deny(
                task,
                action,
                "pytest.full_suite_required",
                "completed finish requires a passing configured test suite",
            )
        task.tdd_phase = TddPhase.FINISHED
        return self._allow(task, action, "finish.allowed", "task may finish")

    def _pytest_decision(self, task: TaskState, action: RunPytestAction) -> PolicyDecision:
        for target in action.targets:
            if target.startswith("-"):
                return self._deny(task, action, "pytest.option_not_allowed", "pytest options are fixed by config")
            path = target.split("::", maxsplit=1)[0]
            normalized_path = self._normalized_project_path(path)
            if normalized_path is None:
                return self._deny(task, action, "pytest.target_outside_root", "pytest target must stay inside root")
            if self._path_category(task, normalized_path) != "test":
                return self._deny(task, action, "pytest.target_not_test", "pytest target must be in a test directory")
        return self._allow(task, action, "pytest.allowed", "restricted pytest is allowed")

    def _feature_red_matches_changed_test(
        self, task: TaskState, action: RunPytestAction, feedback: PytestFeedback
    ) -> bool:
        changed_paths = self._changed_test_paths.get(task.id, set())
        target_paths = tuple(
            self._normalized_project_path(target.split("::", maxsplit=1)[0])
            for target in action.targets
        )
        node_paths = tuple(
            self._normalized_project_path(node_id.split("::", maxsplit=1)[0])
            for node_id in feedback.node_ids
        )
        return bool(target_paths and node_paths) and all(
            path in changed_paths for path in target_paths
        ) and all(path in target_paths for path in node_paths)

    def _path_category(self, task: TaskState, path: str) -> str:
        normalized_path = self._normalized_project_path(path)
        if normalized_path is None:
            return "other"
        normalized = PurePath(normalized_path)
        if any(self._is_within(normalized, directory) for directory in task.config.source_dirs):
            return "source"
        if any(self._is_within(normalized, directory) for directory in task.config.test_dirs):
            return "test"
        return "other"

    @staticmethod
    def _is_within(path: PurePath, directory: Path) -> bool:
        directory_path = PurePath(directory)
        return path == directory_path or directory_path in path.parents

    @staticmethod
    def _patch_paths(diff: str) -> tuple[tuple[str, ...], str | None]:
        """Return modified or added paths, rejecting unsupported file operations."""
        operations, error_rule = PolicyEngine._patch_operations(diff)
        return tuple(path for path, _created in operations), error_rule

    @staticmethod
    def _patch_operations(diff: str) -> tuple[tuple[tuple[str, bool], ...], str | None]:
        """Return unified-diff paths and whether each operation creates a new path."""
        return patch_operations(diff)

    @staticmethod
    def _diff_path(header: str, prefix: str) -> str | None:
        return _diff_path(header, prefix)

    def _normalized_project_path(self, path: str) -> str | None:
        candidate = (self._project_root / path).resolve(strict=False)
        if not candidate.is_relative_to(self._project_root):
            return None
        return candidate.relative_to(self._project_root).as_posix()

    def _normalized_paths(self, paths: tuple[str, ...]) -> tuple[str, ...] | None:
        normalized_paths = tuple(self._normalized_project_path(path) for path in paths)
        if any(path is None for path in normalized_paths):
            return None
        return tuple(path for path in normalized_paths if path is not None)

    @staticmethod
    def _is_sensitive_path(path: str) -> bool:
        parts = PurePath(path).parts
        return ".env" in parts or "harness.yaml" in parts

    def _all_paths_were_read(self, task: TaskState, paths: tuple[str, ...]) -> bool:
        return set(paths).issubset(self._read_paths.get(task.id, set()))

    def _invalidate_reads(self, task: TaskState, paths: tuple[str, ...]) -> None:
        self._read_paths.get(task.id, set()).difference_update(paths)

    @staticmethod
    def _allow(
        task: TaskState, action: Action | None, rule_id: str, reason: str
    ) -> PolicyDecision:
        return PolicyDecision(
            verdict=PolicyVerdict.ALLOW,
            rule_id=rule_id,
            reason=reason,
            task_id=task.id,
            action_hash=stable_hash(action) if action is not None else None,
        )

    @classmethod
    def _approval_required(
        cls, task: TaskState, action: Action, rule_id: str, reason: str
    ) -> PolicyDecision:
        projection_denial = cls._approval_projection_denial(task, action)
        if projection_denial is not None:
            return projection_denial
        return PolicyDecision(
            verdict=PolicyVerdict.APPROVAL_REQUIRED,
            rule_id=rule_id,
            reason=reason,
            task_id=task.id,
            action_hash=stable_hash(action),
            permanent_eligible=isinstance(action, RunCommandAction),
        )

    @staticmethod
    def _approval_projection_denial(
        task: TaskState, action: Action
    ) -> PolicyDecision | None:
        projection = approval_action_projection(action)
        if projection is None:
            return PolicyEngine._deny(
                task,
                action,
                "approval.projection_unavailable",
                "action cannot be represented safely for approval",
            )
        if len(projection) > APPROVAL_ACTION_PROJECTION_MAX_LENGTH:
            return PolicyEngine._deny(
                task,
                action,
                "approval.projection_too_long",
                "action is too long to represent completely for approval",
            )
        return None

    @staticmethod
    def _deny(
        task: TaskState | None, action: Action | None, rule_id: str, reason: str
    ) -> PolicyDecision:
        return PolicyDecision(
            verdict=PolicyVerdict.DENY,
            rule_id=rule_id,
            reason=reason,
            task_id=task.id if task is not None else None,
            action_hash=stable_hash(action) if action is not None else None,
        )
