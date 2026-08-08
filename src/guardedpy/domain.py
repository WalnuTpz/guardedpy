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
    review_path: str | None = Field(default=None, frozen=True)

    def model_post_init(self, __context: object) -> None:
        """Derive internal path state and validate the optional review scope."""
        if self.intent is TaskIntent.CODING:
            self.path = TaskPath.BASELINE_PENDING
            self.repair_targets = ()
            self.tdd_phase = TddPhase.TEST_DESIGN
            if self.review_path is not None:
                raise ValueError("review path is valid only for review intent")
            return
        self.path = TaskPath.READ_ONLY
        if self.intent is TaskIntent.PLAN:
            if self.review_path is not None:
                raise ValueError("review path is valid only for review intent")
            return
        if self.review_path is None:
            return
        if not self.review_path.strip():
            raise ValueError("review path must be a nonblank existing project path")
        root = self.config.profile.root.resolve()
        candidate = (root / self.review_path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("review path must be an existing path inside the project root")
        normalized = candidate.relative_to(root).as_posix()
        restoring_history = (
            isinstance(__context, dict)
            and __context.get("source") == "event_store_history"
        )
        if restoring_history and self.review_path != normalized:
            raise ValueError("persisted review path must be a normalized project-relative path")
        if not restoring_history and not candidate.exists():
            raise ValueError("review path must be an existing path inside the project root")
        object.__setattr__(self, "review_path", normalized)
