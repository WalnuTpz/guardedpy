"""Domain values shared by GuardedPy components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypeAlias, TypeGuard
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from guardedpy.config import HarnessConfig


class TaskIntent(StrEnum):
    """The immutable user intent, independent of automatic coding classification."""

    CODING = "coding"
    PLAN = "plan"
    REVIEW = "review"


class TaskPath(StrEnum):
    """The deterministic path selected by intent and complete-suite evidence."""

    BASELINE_PENDING = "baseline_pending"
    FEATURE = "feature"
    REPAIR = "repair"
    READ_ONLY = "read_only"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class TddPhase(StrEnum):
    TEST_DESIGN = "test_design"
    RED_OBSERVED = "red_observed"
    IMPLEMENTATION = "implementation"
    GREEN_OBSERVED = "green_observed"
    FINISHED = "finished"


class PolicyVerdict(StrEnum):
    ALLOW = "allow"
    APPROVAL_REQUIRED = "approval_required"
    DENY = "deny"


ApprovalDecision: TypeAlias = Literal["reject", "once", "always"]
_APPROVAL_DECISIONS = frozenset({"reject", "once", "always"})


def is_approval_decision(value: object) -> TypeGuard[ApprovalDecision]:
    """Recognize only the three exact string decisions at runtime."""
    return type(value) is str and value in _APPROVAL_DECISIONS


class CommandRuleKind(StrEnum):
    """The only command families eligible for durable approval."""

    GIT_DIFF_CHECK = "git_diff_check"
    GIT_PUSH = "git_push"
    PIP_INSTALL = "pip_install"


@dataclass(frozen=True, slots=True)
class CommandApprovalRule:
    """An immutable, structured permission without a raw pending action."""

    id: str
    kind: CommandRuleKind
    project_hash: str
    branch: str | None = None
    package_specs: tuple[str, ...] = ()


class PolicyDecision(BaseModel):
    """A deterministic policy result for one proposed action."""

    verdict: PolicyVerdict
    rule_id: str
    reason: str
    task_id: UUID | None = None
    action_hash: str | None = None
    permanent_eligible: bool = False


class FeedbackKind(StrEnum):
    PASSED = "passed"
    ASSERTION_FAILURE = "assertion_failure"
    COLLECTION_ERROR = "collection_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"


class TaskState(BaseModel):
    """The persisted state needed to govern one coding task."""

    description: str
    intent: TaskIntent = Field(default=TaskIntent.CODING, frozen=True)
    config: HarnessConfig
    id: UUID = Field(default_factory=uuid4)
    status: TaskStatus = TaskStatus.PENDING
    tdd_phase: TddPhase = TddPhase.TEST_DESIGN
    path: TaskPath = TaskPath.BASELINE_PENDING
    repair_targets: tuple[str, ...] = ()

    def model_post_init(self, __context: object) -> None:
        """Select the read-only path without admitting user-selected coding paths."""
        if self.intent in {TaskIntent.PLAN, TaskIntent.REVIEW}:
            self.path = TaskPath.READ_ONLY
