"""Deterministic safety and TDD decisions for model-proposed actions."""

from __future__ import annotations

from pathlib import Path, PurePath
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
from guardedpy.domain import FeedbackKind, PolicyDecision, PolicyVerdict, TaskMode, TaskState, TddPhase
from guardedpy.feedback import PytestFeedback


class PolicyEngine:
    """Evaluate actions without executing them or retaining approval details on disk."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        self._read_paths: dict[UUID, set[str]] = {}
        self._feature_test_changes: set[UUID] = set()
        self._new_test_paths: dict[UUID, set[str]] = {}
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
            return self._allow(task, action, "approval.requested", "approval request is recorded")
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
            self._feature_test_changes.add(task.id)
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
        if feedback.kind is FeedbackKind.ASSERTION_FAILURE:
            if task.tdd_phase is not TddPhase.TEST_DESIGN:
                return self._deny(task, None, "tdd.red_out_of_sequence", "red result is out of sequence")
            if task.mode is TaskMode.FEATURE and task.id not in self._feature_test_changes:
                return self._deny(
                    task,
                    None,
                    "tdd.test_change_required",
                    "a feature task must change a test before observing red",
                )
            if task.mode is TaskMode.BUGFIX and (
                task.bugfix_target is None or feedback.node_ids != (task.bugfix_target,)
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

    def request_approval(self, task: TaskState, action: Action) -> PolicyDecision:
        """Register one pending approval when the proposed action needs one."""
        decision = self.decide(task, action)
        if decision.verdict is PolicyVerdict.APPROVAL_REQUIRED:
            self._pending_approvals.add((task.id, stable_hash(action)))
        return decision

    def apply_approval(
        self, pending: PolicyDecision, action: Action, approved: bool
    ) -> PolicyDecision:
        """Apply one user decision to the exact pending action and consume it on acceptance."""
        if pending.verdict is not PolicyVerdict.APPROVAL_REQUIRED or pending.task_id is None:
            return self._deny(None, action, "approval.not_pending", "action has no pending approval")
        if pending.action_hash != stable_hash(action):
            return self._deny(None, action, "approval.action_mismatch", "approval is bound to another action")
        approval_key = (pending.task_id, pending.action_hash)
        if approval_key not in self._pending_approvals:
            return self._deny(None, action, "approval.already_used", "approval was already consumed")
        if not approved:
            self._pending_approvals.remove(approval_key)
            return self._deny(None, action, "approval.declined", "user declined the action")
        self._pending_approvals.remove(approval_key)
        return PolicyDecision(
            verdict=PolicyVerdict.ALLOW,
            rule_id="approval.granted",
            reason="user approved this exact action once",
            task_id=pending.task_id,
            action_hash=pending.action_hash,
        )

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

    def _delete_decision(self, task: TaskState, action: DeletePathAction) -> PolicyDecision:
        normalized_path = self._normalized_project_path(action.path)
        if normalized_path is None:
            return self._deny(task, action, "path.outside_root", "path must stay inside the project root")
        if self._is_sensitive_path(normalized_path):
            return self._deny(task, action, "path.sensitive", "credentials and harness configuration are unavailable")
        category = self._path_category(task, normalized_path)
        if category == "source" and task.tdd_phase is not TddPhase.RED_OBSERVED:
            return self._deny(task, action, "tdd.red_required", "source deletion requires an observed red test")
        return self._approval_required(task, action, "delete.approval_required", "deletion requires approval")

    def _command_decision(self, task: TaskState, action: RunCommandAction) -> PolicyDecision:
        command = action.args[0] if action.args else ""
        if command in {"sudo", "doas", "su"}:
            return self._deny(task, action, "command.privilege", "privilege escalation is forbidden")
        if command == "keyring" or any(".env" in argument for argument in action.args):
            return self._deny(task, action, "command.credentials", "credential access is forbidden")
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
        if action.args == ("git", "diff", "--no-ext-diff", "--check"):
            return self._approval_required(
                task,
                action,
                "command.read_only_approval_required",
                "the read-only Git whitespace check requires approval",
            )
        return self._deny(
            task,
            action,
            "command.not_allowed",
            "only the fixed read-only Git diff check may request approval",
        )

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
        lines = diff.splitlines()
        if any(line.startswith(("rename from ", "rename to ", "similarity index ")) for line in lines):
            return (), "patch.rename_unsupported"

        operations: list[tuple[str, bool]] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.startswith("--- "):
                index += 1
                continue
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                return (), "patch.invalid"
            old_path = PolicyEngine._diff_path(line, "--- a/")
            new_path = PolicyEngine._diff_path(lines[index + 1], "+++ b/")
            if line == "--- /dev/null":
                if new_path is None:
                    return (), "patch.invalid"
                operations.append((new_path, True))
            elif lines[index + 1] == "+++ /dev/null":
                return (), "patch.delete_unsupported"
            elif old_path is None or new_path is None:
                return (), "patch.invalid"
            elif old_path != new_path:
                return (), "patch.rename_unsupported"
            else:
                operations.append((old_path, False))
            index += 2

        if not operations:
            return (), "patch.invalid"
        return tuple(operations), None

    @staticmethod
    def _diff_path(header: str, prefix: str) -> str | None:
        path = header.removeprefix(prefix).split("\t", maxsplit=1)[0]
        return path if header.startswith(prefix) and path else None

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

    @staticmethod
    def _approval_required(task: TaskState, action: Action, rule_id: str, reason: str) -> PolicyDecision:
        return PolicyDecision(
            verdict=PolicyVerdict.APPROVAL_REQUIRED,
            rule_id=rule_id,
            reason=reason,
            task_id=task.id,
            action_hash=stable_hash(action),
        )

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
