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
    ReadFileAction,
    RequestApprovalAction,
    RunCommandAction,
    RunPytestAction,
    stable_hash,
)
from guardedpy.domain import PolicyDecision, PolicyVerdict, TaskMode, TaskState, TddPhase


class PolicyEngine:
    """Evaluate actions without executing them or retaining approval details on disk."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        self._read_paths: dict[UUID, set[str]] = {}
        self._feature_test_changes: set[UUID] = set()
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
            return self._allow(task, action, "pytest.allowed", "restricted pytest is allowed")
        if isinstance(action, RunCommandAction):
            return self._command_decision(task, action)
        if isinstance(action, RequestApprovalAction):
            return self._allow(task, action, "approval.requested", "approval request is recorded")
        if isinstance(action, FinishAction):
            return self._finish_decision(task, action)
        raise TypeError(f"unsupported action: {type(action).__name__}")

    def record_read(self, task: TaskState, action: ReadFileAction) -> PolicyDecision:
        """Record a permitted current-version read for later patch authorization."""
        decision = self.decide(task, action)
        if decision.verdict is PolicyVerdict.ALLOW:
            self._read_paths.setdefault(task.id, set()).add(action.path)
        return decision

    def record_pytest(self, task: TaskState, *, passed: bool) -> PolicyDecision:
        """Advance the TDD state only from an observed pytest outcome."""
        if not passed:
            if task.tdd_phase is not TddPhase.TEST_DESIGN:
                return self._deny(task, None, "tdd.red_out_of_sequence", "red result is out of sequence")
            if task.mode is TaskMode.FEATURE and task.id not in self._feature_test_changes:
                return self._deny(
                    task,
                    None,
                    "tdd.test_change_required",
                    "a feature task must change a test before observing red",
                )
            task.tdd_phase = TddPhase.RED_OBSERVED
            return self._allow(task, None, "tdd.red_recorded", "red pytest result recorded")
        if task.tdd_phase is not TddPhase.IMPLEMENTATION:
            return self._deny(task, None, "tdd.green_out_of_sequence", "green result is out of sequence")
        task.tdd_phase = TddPhase.GREEN_OBSERVED
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
        if not self._is_project_path(path):
            return self._deny(task, action, "path.outside_root", "path must stay inside the project root")
        if self._is_sensitive_path(path):
            return self._deny(task, action, "path.sensitive", "credentials and harness configuration are unavailable")
        return self._allow(task, action, "read.allowed", "project read is allowed")

    def _patch_decision(self, task: TaskState, action: ApplyPatchAction) -> PolicyDecision:
        paths = self._patch_paths(action.diff)
        if not paths:
            return self._deny(task, action, "patch.invalid", "patch must name at least one project path")
        if any(not self._is_project_path(path) for path in paths):
            return self._deny(task, action, "path.outside_root", "patch path must stay inside the project root")
        if any(self._is_sensitive_path(path) for path in paths):
            return self._deny(task, action, "path.sensitive", "credentials and harness configuration are unavailable")

        categories = {self._path_category(task, path) for path in paths}
        if "other" in categories:
            return self._approval_required(task, action, "patch.non_code", "non-code files require approval")
        if len(categories) != 1:
            return self._deny(task, action, "patch.mixed_code_and_test", "source and tests must change in separate TDD actions")

        category = categories.pop()
        if category == "source" and task.tdd_phase is not TddPhase.RED_OBSERVED:
            return self._deny(task, action, "tdd.red_required", "source patch requires an observed red test")
        if category == "test" and task.tdd_phase is not TddPhase.TEST_DESIGN:
            return self._deny(task, action, "tdd.test_design_required", "test changes must start the TDD sequence")
        if not self._all_paths_were_read(task, paths):
            return self._deny(task, action, "patch.read_required", "each patched file must first be read")
        if category == "test":
            self._feature_test_changes.add(task.id)
            self._invalidate_reads(task, paths)
            return self._allow(task, action, "patch.test_allowed", "test patch is allowed before red")
        task.tdd_phase = TddPhase.IMPLEMENTATION
        self._invalidate_reads(task, paths)
        return self._allow(task, action, "patch.source_allowed", "source patch follows observed red")

    def _delete_decision(self, task: TaskState, action: DeletePathAction) -> PolicyDecision:
        if not self._is_project_path(action.path):
            return self._deny(task, action, "path.outside_root", "path must stay inside the project root")
        if self._is_sensitive_path(action.path):
            return self._deny(task, action, "path.sensitive", "credentials and harness configuration are unavailable")
        category = self._path_category(task, action.path)
        if category == "source" and task.tdd_phase is not TddPhase.RED_OBSERVED:
            return self._deny(task, action, "tdd.red_required", "source deletion requires an observed red test")
        return self._approval_required(task, action, "delete.approval_required", "deletion requires approval")

    def _command_decision(self, task: TaskState, action: RunCommandAction) -> PolicyDecision:
        command = action.args[0] if action.args else ""
        if command in {"sudo", "doas", "su"}:
            return self._deny(task, action, "command.privilege", "privilege escalation is forbidden")
        if command == "keyring" or any(".env" in argument for argument in action.args):
            return self._deny(task, action, "command.credentials", "credential access is forbidden")
        if command in {"rm", "rmdir", "unlink"}:
            return self._deny(task, action, "command.delete", "delete_path is the only deletion action")
        if any(self._path_category(task, argument) in {"source", "test"} for argument in action.args):
            return self._deny(
                task,
                action,
                "command.source_or_test_write",
                "generic commands cannot target source or test directories",
            )
        if command in {"pip", "uv", "poetry"} and "install" in action.args:
            return self._approval_required(
                task, action, "command.dependency_install", "dependency installation requires approval"
            )
        if command in {"curl", "wget"}:
            return self._approval_required(task, action, "command.network", "network access requires approval")
        if command == "git" and any(argument in {"push", "publish"} for argument in action.args):
            return self._approval_required(task, action, "command.git_publish", "Git publishing requires approval")
        return self._approval_required(
            task, action, "command.approval_required", "generic commands require approval"
        )

    def _finish_decision(self, task: TaskState, action: FinishAction) -> PolicyDecision:
        if action.status == "completed" and task.tdd_phase is not TddPhase.GREEN_OBSERVED:
            return self._deny(task, action, "tdd.green_required", "completed finish requires observed green")
        task.tdd_phase = TddPhase.FINISHED
        return self._allow(task, action, "finish.allowed", "task may finish")

    def _path_category(self, task: TaskState, path: str) -> str:
        normalized = PurePath(path)
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
    def _patch_paths(diff: str) -> tuple[str, ...]:
        return tuple(
            line.removeprefix("+++ b/")
            for line in diff.splitlines()
            if line.startswith("+++ b/")
        )

    def _is_project_path(self, path: str) -> bool:
        candidate = (self._project_root / path).resolve(strict=False)
        return candidate.is_relative_to(self._project_root)

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
