"""Domain values shared by GuardedPy components."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from guardedpy.config import HarnessConfig


class TaskMode(StrEnum):
    FEATURE = "feature"
    BUGFIX = "bugfix"


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


class PolicyDecision(BaseModel):
    """A deterministic policy result for one proposed action."""

    verdict: PolicyVerdict
    rule_id: str
    reason: str
    task_id: UUID | None = None
    action_hash: str | None = None


class FeedbackKind(StrEnum):
    PASSED = "passed"
    ASSERTION_FAILURE = "assertion_failure"
    COLLECTION_ERROR = "collection_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"


class TaskState(BaseModel):
    """The persisted state needed to govern one coding task."""

    description: str
    mode: TaskMode
    config: HarnessConfig
    id: UUID = Field(default_factory=uuid4)
    status: TaskStatus = TaskStatus.PENDING
    tdd_phase: TddPhase = TddPhase.TEST_DESIGN
    bugfix_target: str | None = None

    @model_validator(mode="after")
    def require_bugfix_target(self) -> "TaskState":
        """Require one explicit pytest node before admitting a bugfix task."""
        if self.mode is TaskMode.BUGFIX and not (self.bugfix_target and self.bugfix_target.strip()):
            raise ValueError("bugfix target must be a nonblank pytest node")
        return self
