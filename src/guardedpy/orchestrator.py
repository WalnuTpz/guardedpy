"""The explicit, deterministic coding-agent control loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
from threading import RLock
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from guardedpy.actions import (
    Action,
    ApplyPatchAction,
    DeletePathAction,
    FinishAction,
    ListFilesAction,
    ProposeMemoryAction,
    ReadFileAction,
    RequestApprovalAction,
    RunCommandAction,
    RunPytestAction,
    parse_action,
    stable_hash,
)
from guardedpy.context import ContextBuilder, LlmContext
from guardedpy.command_rules import CommandRuleStore
from guardedpy.domain import (
    ApprovalDecision,
    PolicyDecision,
    PolicyVerdict,
    TaskState,
    TaskStatus,
    is_approval_decision,
)
from guardedpy.events import EventStore, FeedbackAudit, RunEvent, StopReason
from guardedpy.feedback import FeedbackCollector, PytestFeedback
from guardedpy.llm import LLMClient, TemporaryProviderFailure
from guardedpy.memory import MemoryStore
from guardedpy.policy import PolicyEngine
from guardedpy.workspace import ToolResult, Workspace


@dataclass
class _LoopState:
    """In-process control state that must survive an approval pause."""

    seen_actions: set[str] = field(default_factory=set)
    next_round: int = 0


class TaskOrchestrator:
    """Run one-action LLM rounds through deterministic policy and workspace tools."""

    def __init__(
        self,
        project_root: Path,
        llm: LLMClient,
        *,
        max_rounds: int = 20,
        memory_store: MemoryStore | None = None,
        current_branch: str | None = None,
        command_rules: CommandRuleStore | None = None,
    ) -> None:
        self._project_root = project_root.resolve()
        self._llm = llm
        self._max_rounds = max_rounds
        self._memory_store = memory_store or MemoryStore(self._project_root)
        self._policy = PolicyEngine(
            self._project_root,
            current_branch=current_branch,
            command_rules=command_rules,
        )
        self._workspace_by_task: dict[UUID, Workspace] = {}
        self._tasks: dict[UUID, TaskState] = {}
        self._feedback: dict[UUID, dict[str, Any] | None] = {}
        self._loop_by_task: dict[UUID, _LoopState] = {}
        self._pending: dict[tuple[UUID, str], tuple[Action, PolicyDecision, int]] = {}
        self._cancelled_task_ids: set[UUID] = set()
        self._state_lock = RLock()
        self._events = EventStore(self._project_root)
        self._events.mark_unfinished_interrupted()
        self._feedback_collector = FeedbackCollector()

    def run(self, task: TaskState) -> TaskState:
        """Advance a task until it reaches a terminal, approval, or bounded stop state."""
        with self._state_lock:
            self._reject_second_active_task(task)
            self._tasks[task.id] = task
            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.BLOCKED,
                TaskStatus.CANCELLED,
                TaskStatus.INTERRUPTED,
            }:
                return task
            if task.status is TaskStatus.WAITING_APPROVAL:
                return task
            self._workspace_by_task.setdefault(task.id, Workspace(self._project_root, task.config))
            if task.status is not TaskStatus.RUNNING:
                task.status = TaskStatus.RUNNING
                self._events.append(RunEvent(task_id=task.id, task_status=task.status))
        loop = self._loop_by_task.setdefault(task.id, _LoopState())

        while loop.next_round < self._max_rounds:
            if task.status is TaskStatus.CANCELLED:
                return task
            round_number = loop.next_round
            try:
                completion = self._llm.complete(self._context(task))
            except TemporaryProviderFailure:
                if self._is_cancelled(task):
                    return task
                return self._stop(task, StopReason.PROVIDER_TEMPORARY_FAILURE, round_number)
            except (ValidationError, ValueError, json.JSONDecodeError):
                if self._is_cancelled(task):
                    return task
                return self._stop(task, StopReason.INVALID_MODEL_OUTPUT, round_number)
            except Exception:
                if self._is_cancelled(task):
                    return task
                return self._stop(task, StopReason.UNRECOVERABLE_ERROR, round_number)
            if self._is_cancelled(task):
                return task
            try:
                action = parse_action(completion)
            except (ValidationError, ValueError, json.JSONDecodeError):
                return self._stop(task, StopReason.INVALID_MODEL_OUTPUT, round_number)
            if self._is_cancelled(task):
                return task
            loop.next_round += 1

            action_hash = stable_hash(action)
            repeat_key = self._repeat_key(action)
            if repeat_key in loop.seen_actions:
                with self._state_lock:
                    if self._is_cancelled(task):
                        return task
                    self._events.append(
                        RunEvent(
                            task_id=task.id,
                            task_status=TaskStatus.BLOCKED,
                            action=action,
                            retry_count=round_number,
                            stop_reason=StopReason.REPEATED_ACTION,
                        )
                    )
                    task.status = TaskStatus.BLOCKED
                return task
            loop.seen_actions.add(repeat_key)

            decision = self._policy.decide(task, action)
            if decision.verdict is PolicyVerdict.DENY:
                self._record_decision(task, action, decision, round_number)
                self._feedback[task.id] = {
                    "type": "policy_denial",
                    "rule_id": decision.rule_id,
                    "reason": decision.reason,
                }
                continue
            if decision.verdict is PolicyVerdict.APPROVAL_REQUIRED:
                with self._state_lock:
                    if self._is_cancelled(task):
                        return task
                    self._policy.request_approval(task, action)
                    self._pending[(task.id, action_hash)] = (action, decision, round_number)
                    task.status = TaskStatus.WAITING_APPROVAL
                    self._events.append(
                        RunEvent(
                            task_id=task.id,
                            task_status=task.status,
                            action=action,
                            policy_verdict=decision.verdict,
                            retry_count=round_number,
                        )
                    )
                return task
            if isinstance(action, FinishAction):
                with self._state_lock:
                    if self._is_cancelled(task):
                        return task
                    task.status = (
                        TaskStatus.COMPLETED if action.status == "completed" else TaskStatus.BLOCKED
                    )
                return self._stop(
                    task,
                    StopReason.COMPLETED if task.status is TaskStatus.COMPLETED else StopReason.BLOCKED,
                    round_number,
                    action=action,
                    decision=decision,
                )

            try:
                with self._state_lock:
                    if self._is_cancelled(task):
                        return task
                    self._execute_allowed(task, action, decision, round_number)
            except Exception:
                return self._stop(
                    task,
                    StopReason.UNRECOVERABLE_ERROR,
                    round_number,
                    action=action,
                    decision=decision,
                )
            if task.status is not TaskStatus.RUNNING:
                return task

        return self._stop(task, StopReason.ROUND_LIMIT, loop.next_round)

    def submit(self, task: TaskState) -> TaskState:
        """Register a pending task before a caller starts its background run."""
        with self._state_lock:
            self._reject_second_active_task(task)
            self._tasks[task.id] = task
        return task

    def cancel(self, task_id: UUID) -> TaskState:
        """Cancel one known non-terminal task and discard its in-memory pending action."""
        with self._state_lock:
            task = self._tasks[task_id]
            if task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.BLOCKED,
                TaskStatus.CANCELLED,
                TaskStatus.INTERRUPTED,
            }:
                return task
            self._cancelled_task_ids.add(task_id)
            self._discard_pending(task_id)
            task.status = TaskStatus.CANCELLED
            self._events.append(
                RunEvent(
                    task_id=task.id,
                    task_status=task.status,
                    stop_reason=StopReason.CANCELLED,
                )
            )
        return task

    def resolve_approval(
        self,
        task_id: UUID,
        action_hash: str,
        *,
        decision: ApprovalDecision | None = None,
        approved: bool | None = None,
    ) -> bool:
        """Consume an exact approval decision and execute only the bound allowed action."""
        if decision is None:
            if type(approved) is not bool:
                return False
            decision = "once" if approved else "reject"
        elif approved is not None or not is_approval_decision(decision):
            return False
        with self._state_lock:
            key = (task_id, action_hash)
            pending = self._pending.pop(key, None)
            task = self._tasks.get(task_id)
            if pending is None or task is None or task.status is not TaskStatus.WAITING_APPROVAL:
                return False
            action, pending_decision, round_number = pending
            approval = self._policy.apply_approval(pending_decision, action, decision=decision)
            if approval.verdict is not PolicyVerdict.ALLOW:
                task.status = TaskStatus.BLOCKED
                self._events.append(
                    RunEvent(
                        task_id=task.id,
                        task_status=task.status,
                        action=action,
                        policy_verdict=approval.verdict,
                        approval_granted=False,
                        stop_reason=StopReason.BLOCKED,
                    )
                )
                return False
            if self._is_cancelled(task):
                return False
            task.status = TaskStatus.RUNNING
            self._events.append(
                RunEvent(
                    task_id=task.id,
                    task_status=task.status,
                    action=action,
                    policy_verdict=approval.verdict,
                    approval_granted=True,
                )
            )
            try:
                self._execute_allowed(task, action, approval, round_number)
            except Exception:
                self._stop(
                    task,
                    StopReason.UNRECOVERABLE_ERROR,
                    round_number,
                    action=action,
                    decision=approval,
                )
            return True

    def _execute_allowed(
        self, task: TaskState, action: Action, decision: PolicyDecision, round_number: int
    ) -> None:
        if self._is_cancelled(task):
            return
        workspace = self._workspace_by_task[task.id]
        if isinstance(action, ListFilesAction):
            result = workspace.list_files(PurePosixPath(action.path))
            self._record_tool_result(task, action, decision, result, round_number)
            return
        if isinstance(action, ReadFileAction):
            result = workspace.read_file(PurePosixPath(action.path), action.offset, action.limit)
            if result.ok:
                self._policy.record_read(task, action)
            self._record_tool_result(task, action, decision, result, round_number)
            return
        if isinstance(action, ApplyPatchAction):
            created_test_paths = self._policy.created_test_paths(task, action)
            result = workspace.apply_patch(action.diff)
            if result.ok:
                self._policy.record_patch(task, action)
                for path in created_test_paths:
                    self._policy.record_new_test_path(task, path)
            self._record_tool_result(task, action, decision, result, round_number)
            return
        if isinstance(action, DeletePathAction):
            result = workspace.delete_path(PurePosixPath(action.path))
            if result.ok:
                self._policy.record_delete(task, action, decision)
            self._record_tool_result(task, action, decision, result, round_number)
            return
        if isinstance(action, RunPytestAction):
            run = workspace.run_pytest(action.targets)
            feedback = self._feedback_collector.collect(run)
            self._policy.record_pytest(task, action, feedback)
            self._feedback[task.id] = self._pytest_feedback(feedback)
            self._events.append(
                RunEvent(
                    task_id=task.id,
                    task_status=task.status,
                    action=action,
                    policy_verdict=decision.verdict,
                    feedback=FeedbackAudit(
                        kind=feedback.kind,
                        node_id=feedback.node_ids[0] if feedback.node_ids else None,
                    ),
                    retry_count=round_number,
                )
            )
            return
        if isinstance(action, RunCommandAction):
            result = self._run_command(action)
            self._record_tool_result(task, action, decision, result, round_number)
            return
        if isinstance(action, RequestApprovalAction):
            self._feedback[task.id] = {"type": "approval_request", "recorded": True}
            self._record_decision(task, action, decision, round_number)
            return
        if isinstance(action, ProposeMemoryAction):
            self._memory_store.propose(task.id, action.text)
            self._feedback[task.id] = {"type": "memory_proposal", "recorded": True}
            self._record_decision(task, action, decision, round_number)
            return
        raise TypeError(f"unsupported allowed action: {type(action).__name__}")

    def _record_tool_result(
        self,
        task: TaskState,
        action: Action,
        decision: PolicyDecision,
        result: ToolResult,
        round_number: int,
    ) -> None:
        self._feedback[task.id] = {
            "type": "tool_result",
            "ok": result.ok,
            "summary": result.summary,
            "data": result.data,
        }
        self._record_decision(task, action, decision, round_number)

    def _record_decision(
        self, task: TaskState, action: Action, decision: PolicyDecision, round_number: int
    ) -> None:
        self._events.append(
            RunEvent(
                task_id=task.id,
                task_status=task.status,
                action=action,
                policy_verdict=decision.verdict,
                retry_count=round_number,
            )
        )

    def _stop(
        self,
        task: TaskState,
        reason: StopReason,
        round_number: int,
        *,
        action: Action | None = None,
        decision: PolicyDecision | None = None,
    ) -> TaskState:
        with self._state_lock:
            if self._is_cancelled(task):
                return task
            if reason in {
                StopReason.INVALID_MODEL_OUTPUT,
                StopReason.PROVIDER_TEMPORARY_FAILURE,
                StopReason.UNRECOVERABLE_ERROR,
                StopReason.ROUND_LIMIT,
                StopReason.REPEATED_ACTION,
                StopReason.BLOCKED,
            }:
                task.status = TaskStatus.BLOCKED
            self._events.append(
                RunEvent(
                    task_id=task.id,
                    task_status=task.status,
                    action=action,
                    policy_verdict=decision.verdict if decision else None,
                    retry_count=round_number,
                    stop_reason=reason,
                )
            )
        return task

    def _context(self, task: TaskState) -> LlmContext:
        return ContextBuilder(self._project_root).build(
            task,
            self._feedback.get(task.id),
            self._memory_store.search(task.description),
        )

    @staticmethod
    def _pytest_feedback(feedback: PytestFeedback) -> dict[str, object]:
        return {
            "type": "pytest_feedback",
            "kind": feedback.kind.value,
            "node_ids": feedback.node_ids,
            "excerpt": feedback.excerpt,
        }

    @staticmethod
    def _repeat_key(action: Action) -> str:
        """Identify repeated operations without letting user-facing wording alter the key."""
        canonical = json.dumps(
            action.model_dump(mode="json", exclude={"summary"}),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(canonical.encode()).hexdigest()

    def _discard_pending(self, task_id: UUID) -> None:
        for key in tuple(self._pending):
            if key[0] == task_id:
                self._pending.pop(key)

    def _is_cancelled(self, task: TaskState) -> bool:
        with self._state_lock:
            return task.id in self._cancelled_task_ids or task.status is TaskStatus.CANCELLED

    def _reject_second_active_task(self, task: TaskState) -> None:
        """Keep one task's unsafe pause and control state isolated from every other task."""
        if task.id in self._tasks:
            return
        active_statuses = {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}
        if task.status in active_statuses and any(
            current.status in active_statuses for current in self._tasks.values()
        ):
            raise ValueError("another task is already active")

    def _run_command(self, action: RunCommandAction) -> ToolResult:
        try:
            completed = subprocess.run(
                action.args,
                cwd=self._project_root,
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
        except OSError:
            return ToolResult(False, "Approved command could not run", {"reason": "command_failed"})
        if completed.returncode != 0:
            return ToolResult(False, "Approved command failed", {"reason": "command_failed"})
        return ToolResult(True, "Approved command completed", {})
