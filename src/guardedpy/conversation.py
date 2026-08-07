"""In-memory continuous conversation protocol owned by GuardedPy."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Literal, Protocol, TypeAlias
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict


TurnMode: TypeAlias = Literal["normal", "plan", "review", "goal"]
TurnStatus: TypeAlias = Literal[
    "queued",
    "running",
    "waiting_approval",
    "cancelling",
    "completed",
    "interrupted",
    "failed",
]
EventKind: TypeAlias = Literal[
    "user_message",
    "turn_started",
    "assistant_item_started",
    "assistant_text_delta",
    "assistant_item_completed",
    "tool_item_started",
    "tool_output",
    "tool_item_completed",
    "approval_requested",
    "approval_resolved",
    "turn_completed",
    "turn_interrupted",
    "turn_failed",
]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    json_schema: Mapping[str, object]


@dataclass(frozen=True)
class ProviderMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None


@dataclass(frozen=True)
class SessionEvent:
    session_id: UUID
    turn_id: UUID
    sequence: int
    kind: EventKind
    item_id: UUID | None = None
    text: str = ""
    data: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None
    name: str | None
    arguments_fragment: str


@dataclass(frozen=True)
class ResponseFinished:
    finish_reason: Literal["stop", "tool_calls"]


ModelChunk: TypeAlias = (
    TextDelta | ReasoningDelta | ToolCallDelta | ResponseFinished
)


class ConversationModel(Protocol):
    def stream(
        self,
        messages: tuple[ProviderMessage, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> Iterator[ModelChunk]: ...


class TemporaryProviderFailure(RuntimeError):
    """A bounded provider retry failed or a stream broke after yielding."""


class TurnNotActiveError(RuntimeError):
    """The requested state transition does not own the active Turn."""


class SafeTurnSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    terminal_status: Literal["completed", "interrupted", "failed"]
    changed_paths: tuple[str, ...]
    pytest_outcome: Literal[
        "passed",
        "assertion_failure",
        "collection_error",
        "execution_error",
        "timeout",
        "not_run",
    ]
    approval_outcome: Literal["none", "approved", "rejected"]
    final_text: str


class ConversationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    project_title: str
    created_at: datetime
    updated_at: datetime
    turns: tuple[SafeTurnSummary, ...]


@dataclass(frozen=True)
class ReadRecord:
    path: str
    sha256: str
    complete: bool


@dataclass(frozen=True)
class PendingApproval:
    approval_id: UUID
    item_id: UUID
    call: ToolCall
    later_calls: tuple[ToolCall, ...]


@dataclass
class Turn:
    id: UUID
    session_id: UUID
    initial_text: str
    mode: TurnMode
    status: TurnStatus = "running"
    sequence: int = 0
    provider_responses: int = 0
    tool_calls: int = 0
    reads: dict[str, ReadRecord] = field(default_factory=dict)
    needs_full_verification: bool = False
    waiting_approval: PendingApproval | None = None
    pending_steers: deque[ProviderMessage] = field(default_factory=deque)
    cancelled: bool = False


@dataclass
class Session:
    id: UUID
    provider_messages: list[ProviderMessage]
    safe_summary: ConversationSummary | None
    turns: dict[UUID, Turn] = field(default_factory=dict)
    active_turn_id: UUID | None = None
    queued_turn_ids: deque[UUID] = field(default_factory=deque)


@dataclass
class _ToolCallParts:
    id: str | None = None
    name: str | None = None
    arguments: str = ""


class _ProviderProtocolError(RuntimeError):
    pass


class ScriptedConversationModel:
    """Deterministic streaming model for offline conversation tests and demos."""

    def __init__(self, responses: list[list[ModelChunk] | Exception]) -> None:
        self._responses = deque(responses)
        self.received_messages: list[tuple[ProviderMessage, ...]] = []

    def stream(
        self,
        messages: tuple[ProviderMessage, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> Iterator[ModelChunk]:
        del tools
        self.received_messages.append(messages)
        if not self._responses:
            raise RuntimeError("script exhausted")
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise _ProviderProtocolError from response
        yield from response


class ConversationAgent:
    """Own Session/Turn state, provider history, events, and stop decisions."""

    def __init__(
        self,
        model: ConversationModel,
        tools: tuple[ToolDefinition, ...] = (),
        governor: ToolGovernor | None = None,
        executor: ToolExecutor | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._governor = governor
        self._executor = executor
        self._sessions: dict[UUID, Session] = {}

    def create_session(
        self, safe_summary: ConversationSummary | None = None
    ) -> UUID:
        session_id = uuid4()
        messages: list[ProviderMessage] = []
        if safe_summary is not None:
            messages.append(
                ProviderMessage(
                    role="system",
                    content=(
                        "Safe prior-session summary (not raw provider history):\n"
                        f"{safe_summary.model_dump_json()}"
                    ),
                )
            )
        self._sessions[session_id] = Session(
            id=session_id,
            provider_messages=messages,
            safe_summary=safe_summary,
        )
        return session_id

    def begin_turn(
        self, session_id: UUID, text: str, mode: TurnMode = "normal"
    ) -> tuple[UUID, SessionEvent]:
        session = self._session(session_id)
        initial_text = _nonblank(text)
        if session.active_turn_id is not None:
            raise TurnNotActiveError("session already has an active turn")
        turn = Turn(
            id=uuid4(),
            session_id=session_id,
            initial_text=initial_text,
            mode=mode,
        )
        session.turns[turn.id] = turn
        session.active_turn_id = turn.id
        session.provider_messages.append(
            ProviderMessage(role="user", content=initial_text)
        )
        return turn.id, self._event(
            turn,
            "user_message",
            item_id=uuid4(),
            text=initial_text,
        )

    def run_turn(
        self, session_id: UUID, turn_id: UUID
    ) -> Iterator[SessionEvent]:
        session = self._session(session_id)
        turn = self._active_turn(session, turn_id, allow_cancelling=True)
        while True:
            yield self._event(turn, "turn_started")
            paused = yield from self._advance(session, turn)
            if paused or session.active_turn_id is None:
                return
            turn = session.turns[session.active_turn_id]

    def _advance(self, session: Session, turn: Turn) -> Iterator[SessionEvent]:
        while True:
            if turn.cancelled:
                terminal, _ = self._terminate(session, turn, "interrupted", "turn_interrupted")
                yield terminal
                return False
            while turn.pending_steers:
                session.provider_messages.append(turn.pending_steers.popleft())
            if turn.provider_responses >= 20:
                terminal, _ = self._terminate(session, turn, "failed", "turn_failed", code="round_limit")
                yield terminal
                return False
            assistant_item_id = uuid4()
            yield self._event(turn, "assistant_item_started", item_id=assistant_item_id)
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            call_parts: dict[int, _ToolCallParts] = {}
            finish_reason: Literal["stop", "tool_calls"] | None = None
            failure_code: str | None = None
            chunks: Iterator[ModelChunk] | None = None

            try:
                chunks = iter(
                    self._model.stream(
                        tuple(session.provider_messages), self._tools
                    )
                )
                while True:
                    if turn.cancelled:
                        break
                    try:
                        chunk = next(chunks)
                    except StopIteration:
                        break
                    if finish_reason is not None:
                        raise _ProviderProtocolError
                    if isinstance(chunk, TextDelta):
                        if not isinstance(chunk.text, str):
                            raise _ProviderProtocolError
                        text_parts.append(chunk.text)
                        yield self._event(
                            turn,
                            "assistant_text_delta",
                            item_id=assistant_item_id,
                            text=chunk.text,
                        )
                    elif isinstance(chunk, ReasoningDelta):
                        if not isinstance(chunk.text, str):
                            raise _ProviderProtocolError
                        reasoning_parts.append(chunk.text)
                    elif isinstance(chunk, ToolCallDelta):
                        self._join_tool_delta(call_parts, chunk)
                    elif isinstance(chunk, ResponseFinished):
                        if chunk.finish_reason not in ("stop", "tool_calls"):
                            raise _ProviderProtocolError
                        finish_reason = chunk.finish_reason
                    else:
                        raise _ProviderProtocolError
            except TemporaryProviderFailure:
                failure_code = "provider_temporary_failure"
            except Exception:
                failure_code = "provider_protocol_error"

            if turn.cancelled or failure_code is not None:
                _close_iterator(chunks)
            yield self._event(turn, "assistant_item_completed", item_id=assistant_item_id)
            if turn.cancelled:
                terminal, _ = self._terminate(session, turn, "interrupted", "turn_interrupted")
                yield terminal
                return False
            elif failure_code is not None or finish_reason is None:
                terminal, _ = self._terminate(session, turn, "failed", "turn_failed", code=failure_code or "provider_protocol_error")
                yield terminal
                return False
            try:
                calls = self._complete_tool_calls(call_parts, finish_reason)
            except _ProviderProtocolError:
                terminal, _ = self._terminate(session, turn, "failed", "turn_failed", code="provider_protocol_error")
                yield terminal
                return False
            turn.provider_responses += 1
            session.provider_messages.append(
                ProviderMessage(
                    role="assistant", content="".join(text_parts), tool_calls=calls,
                    reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
                )
            )
            if turn.tool_calls + len(calls) > 50:
                terminal, _ = self._terminate(session, turn, "failed", "turn_failed", code="round_limit")
                yield terminal
                return False
            if not calls:
                if turn.needs_full_verification:
                    terminal, _ = self._terminate(session, turn, "failed", "turn_failed", code="verification_required")
                else:
                    terminal, _ = self._terminate(session, turn, "completed", "turn_completed")
                yield terminal
                return False
            turn.tool_calls += len(calls)
            if self._governor is None or self._executor is None:
                terminal, _ = self._terminate(session, turn, "failed", "turn_failed", code="tool_execution_unavailable")
                yield terminal
                return False
            for index, call in enumerate(calls):
                item_id = uuid4()
                yield self._event(turn, "tool_item_started", item_id=item_id, data={"tool": call.name})
                decision = self._governor.decide(turn, item_id, call)
                if decision.verdict == "approval_required":
                    assert decision.approval_id is not None
                    turn.waiting_approval = PendingApproval(decision.approval_id, item_id, call, calls[index + 1 :])
                    turn.status = "waiting_approval"
                    yield self._event(
                        turn, "approval_requested", item_id=item_id,
                        data={"approval_id": str(decision.approval_id), "tool": call.name, "rule_id": decision.rule_id},
                    )
                    return True
                if decision.verdict == "deny":
                    payload = {"ok": False, "code": decision.code, "summary": decision.code}
                    self._append_tool_result(session, call, payload)
                    yield self._event(turn, "tool_item_completed", item_id=item_id, data={"code": decision.code, "verdict": "deny"})
                    continue
                execution = self._executor.execute(turn, item_id, call)
                yield from self._execution_events(session, turn, item_id, call, execution)

    def resolve_approval(
        self,
        session_id: UUID,
        turn_id: UUID,
        approval_id: UUID,
        accepted: bool,
    ) -> Iterator[SessionEvent]:
        session = self._session(session_id)
        turn = self._active_turn(session, turn_id)
        pending = turn.waiting_approval
        if (
            turn.status != "waiting_approval"
            or pending is None
            or pending.approval_id != approval_id
        ):
            raise TurnNotActiveError("approval is not active")
        normalized = self._governor.normalized_call(pending.call)
        decision = self._governor.resolve(
            session_id, turn_id, pending.item_id, normalized, approval_id, accepted
        )
        if decision.code == "approval_stale":
            raise TurnNotActiveError("approval is stale")
        yield self._event(
            turn, "approval_resolved", item_id=pending.item_id,
            data={"approval_id": str(approval_id), "accepted": str(accepted).lower()},
        )
        if decision.verdict == "allow":
            execution = self._executor.execute(turn, pending.item_id, pending.call)
            yield from self._execution_events(session, turn, pending.item_id, pending.call, execution)
        else:
            payload = {"ok": False, "code": decision.code, "summary": decision.code}
            self._append_tool_result(session, pending.call, payload)
            yield self._event(turn, "tool_item_completed", item_id=pending.item_id, data={"code": decision.code, "verdict": "deny"})
        for call in pending.later_calls:
            item_id = uuid4()
            yield self._event(turn, "tool_item_started", item_id=item_id, data={"tool": call.name})
            payload = {"ok": False, "code": "not_executed_after_approval", "summary": "not_executed_after_approval"}
            self._append_tool_result(session, call, payload)
            yield self._event(turn, "tool_item_completed", item_id=item_id, data={"code": "not_executed_after_approval", "verdict": "deny"})
        turn.waiting_approval = None
        turn.status = "running"
        paused = yield from self._advance(session, turn)
        if paused:
            return
        while session.active_turn_id is not None:
            promoted = session.turns[session.active_turn_id]
            yield self._event(promoted, "turn_started")
            if (yield from self._advance(session, promoted)):
                return

    @staticmethod
    def _append_tool_result(session: Session, call: ToolCall, payload: Mapping[str, object]) -> None:
        session.provider_messages.append(
            ProviderMessage(role="tool", content=json.dumps(payload, sort_keys=True, separators=(",", ":")), tool_call_id=call.id)
        )

    def _execution_events(self, session: Session, turn: Turn, item_id: UUID, call: ToolCall, execution: object) -> Iterator[SessionEvent]:
        payload = execution.provider_result
        self._append_tool_result(session, call, payload)
        summary = str(execution.summary)
        if summary:
            yield self._event(turn, "tool_output", item_id=item_id, text=summary)
        data = {"code": str(execution.code), "verdict": str(execution.verdict)}
        if execution.changed_paths:
            data["changed_paths"] = json.dumps(execution.changed_paths)
        if execution.feedback is not None:
            data["pytest_outcome"] = execution.feedback.kind.value
        yield self._event(turn, "tool_item_completed", item_id=item_id, data=data)

    def steer(
        self, session_id: UUID, turn_id: UUID, text: str
    ) -> SessionEvent:
        session = self._session(session_id)
        turn = self._active_turn(session, turn_id)
        message = _nonblank(text)
        if turn.status != "running":
            raise TurnNotActiveError("turn is not running")
        turn.pending_steers.append(ProviderMessage(role="user", content=message))
        return self._event(
            turn, "user_message", item_id=uuid4(), text=message
        )

    def queue(
        self, session_id: UUID, text: str, mode: TurnMode = "normal"
    ) -> tuple[UUID, SessionEvent]:
        session = self._session(session_id)
        initial_text = _nonblank(text)
        if session.active_turn_id is None:
            raise TurnNotActiveError("session has no active turn")
        active = session.turns[session.active_turn_id]
        if active.status not in ("running", "waiting_approval"):
            raise TurnNotActiveError("active turn cannot accept a queued turn")
        turn = Turn(
            id=uuid4(),
            session_id=session_id,
            initial_text=initial_text,
            mode=mode,
            status="queued",
        )
        session.turns[turn.id] = turn
        session.queued_turn_ids.append(turn.id)
        return turn.id, self._event(
            turn,
            "user_message",
            item_id=uuid4(),
            text=initial_text,
            data={"queued": "true"},
        )

    def interrupt(
        self, session_id: UUID, turn_id: UUID
    ) -> SessionEvent | None:
        session = self._session(session_id)
        turn = self._active_turn(session, turn_id, allow_cancelling=True)
        if turn.status == "running":
            turn.cancelled = True
            turn.status = "cancelling"
            return None
        if turn.status == "waiting_approval":
            turn.cancelled = True
            turn.waiting_approval = None
            event, _ = self._terminate(
                session, turn, "interrupted", "turn_interrupted"
            )
            return event
        raise TurnNotActiveError("turn cannot be interrupted")

    def _session(self, session_id: UUID) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise TurnNotActiveError("session is not active") from None

    @staticmethod
    def _active_turn(
        session: Session,
        turn_id: UUID,
        *,
        allow_cancelling: bool = False,
    ) -> Turn:
        if session.active_turn_id != turn_id:
            raise TurnNotActiveError("turn does not own the session")
        try:
            turn = session.turns[turn_id]
        except KeyError:
            raise TurnNotActiveError("turn does not belong to the session") from None
        allowed = ("running", "waiting_approval")
        if allow_cancelling:
            allowed += ("cancelling",)
        if turn.status not in allowed:
            raise TurnNotActiveError("turn is not active")
        return turn

    @staticmethod
    def _event(
        turn: Turn,
        kind: EventKind,
        *,
        item_id: UUID | None = None,
        text: str = "",
        data: Mapping[str, str] | None = None,
    ) -> SessionEvent:
        turn.sequence += 1
        return SessionEvent(
            session_id=turn.session_id,
            turn_id=turn.id,
            sequence=turn.sequence,
            kind=kind,
            item_id=item_id,
            text=text,
            data={} if data is None else data,
        )

    @staticmethod
    def _join_tool_delta(
        call_parts: dict[int, _ToolCallParts], chunk: ToolCallDelta
    ) -> None:
        if not isinstance(chunk.index, int) or not isinstance(
            chunk.arguments_fragment, str
        ):
            raise _ProviderProtocolError
        parts = call_parts.setdefault(chunk.index, _ToolCallParts())
        if chunk.id is not None:
            if not isinstance(chunk.id, str) or (
                parts.id is not None and parts.id != chunk.id
            ):
                raise _ProviderProtocolError
            parts.id = chunk.id
        if chunk.name is not None:
            if not isinstance(chunk.name, str) or (
                parts.name is not None and parts.name != chunk.name
            ):
                raise _ProviderProtocolError
            parts.name = chunk.name
        parts.arguments += chunk.arguments_fragment

    @staticmethod
    def _complete_tool_calls(
        call_parts: dict[int, _ToolCallParts],
        finish_reason: Literal["stop", "tool_calls"],
    ) -> tuple[ToolCall, ...]:
        if (finish_reason == "stop" and call_parts) or (
            finish_reason == "tool_calls" and not call_parts
        ):
            raise _ProviderProtocolError
        calls: list[ToolCall] = []
        seen_ids: set[str] = set()
        for index in sorted(call_parts):
            parts = call_parts[index]
            if not parts.id or not parts.id.strip() or not parts.name or not parts.name.strip():
                raise _ProviderProtocolError
            if parts.id in seen_ids:
                raise _ProviderProtocolError
            seen_ids.add(parts.id)
            try:
                arguments = json.loads(parts.arguments)
            except (TypeError, json.JSONDecodeError):
                raise _ProviderProtocolError from None
            if not isinstance(arguments, dict):
                raise _ProviderProtocolError
            calls.append(
                ToolCall(
                    id=parts.id,
                    name=parts.name,
                    arguments_json=parts.arguments,
                )
            )
        return tuple(calls)

    def _terminate(
        self,
        session: Session,
        turn: Turn,
        status: Literal["completed", "interrupted", "failed"],
        kind: Literal["turn_completed", "turn_interrupted", "turn_failed"],
        *,
        code: str | None = None,
    ) -> tuple[SessionEvent, Turn | None]:
        turn.status = status
        turn.waiting_approval = None
        event = self._event(
            turn, kind, data={} if code is None else {"code": code}
        )
        if session.active_turn_id == turn.id:
            session.active_turn_id = None
        promoted: Turn | None = None
        if session.queued_turn_ids:
            promoted = session.turns[session.queued_turn_ids.popleft()]
            promoted.status = "running"
            session.active_turn_id = promoted.id
            session.provider_messages.append(
                ProviderMessage(role="user", content=promoted.initial_text)
            )
        return event, promoted


def _nonblank(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise ValueError("text must not be blank")
    return normalized


def _close_iterator(chunks: Iterator[ModelChunk] | None) -> None:
    if chunks is not None:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()
