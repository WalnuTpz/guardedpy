"""Local project settings and continuous-session ownership."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import UUID

import yaml

from guardedpy.config import (
    HarnessConfig,
    load_config,
    load_or_create_discovered_config,
    local_state_path,
    project_config_path,
    save_discovered_config,
    update_future_defaults as updated_future_defaults,
)
from guardedpy.conversation import (
    ConversationAgent,
    ConversationSummary,
    SafeTurnSummary,
    SessionEvent,
    TurnMode,
)
from guardedpy.conversations import ConversationStore
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile
from guardedpy.feedback import FeedbackCollector
from guardedpy.workspace import Workspace


_MAX_SUMMARY_TEXT = 1200


class CredentialPort(Protocol):
    """The non-secret credential operations required by the local CLI."""

    def status(self) -> CredentialStatus: ...

    def set_key(self, key: str) -> None: ...

    def clear_key(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    """Injectable non-secret services used by the local runtime."""

    credentials: CredentialPort


class RuntimeNotConfiguredError(RuntimeError):
    """Raised when an operation requires a selected project configuration."""

    def __init__(self) -> None:
        super().__init__("尚未完成设置。")


class ConversationRuntime:
    """Runtime ownership boundary for one in-memory continuous Agent."""

    def __init__(self, agent: ConversationAgent, store: ConversationStore) -> None:
        self._agent = agent
        self.store = store
        self._summaries: dict[UUID, ConversationSummary] = {}
        self._texts: dict[tuple[UUID, UUID], list[str]] = {}
        self._facts: dict[tuple[UUID, UUID], dict[str, object]] = {}

    def create_session(self, project_title: str, summary_id: UUID | None = None) -> UUID:
        prior = None if summary_id is None else self.store.load_summary(summary_id)
        session_id = self._agent.create_session(prior)
        now = datetime.now(timezone.utc)
        self._summaries[session_id] = ConversationSummary(
            id=session_id, project_title=project_title, created_at=now, updated_at=now,
            turns=() if prior is None else prior.turns,
        )
        self.store.save_summary(self._summaries[session_id])
        return session_id

    def summary(self, session_id: UUID) -> ConversationSummary:
        try:
            return self._summaries[session_id]
        except KeyError:
            summary = self.store.load_summary(session_id)
            self._summaries[session_id] = summary
            return summary

    def begin_turn(
        self, session_id: UUID, text: str, mode: TurnMode = "normal", *, goal: str | None = None
    ) -> tuple[UUID, SessionEvent]:
        return self._agent.begin_turn(session_id, text, mode, goal=goal)

    def run_turn(self, session_id: UUID, turn_id: UUID):
        return self._capture(session_id, turn_id, self._agent.run_turn(session_id, turn_id))

    def resolve_approval(self, session_id: UUID, turn_id: UUID, approval_id: UUID, accepted: bool):
        return self._capture(session_id, turn_id, self._agent.resolve_approval(session_id, turn_id, approval_id, accepted))

    def steer(self, session_id: UUID, turn_id: UUID, text: str) -> SessionEvent:
        return self._agent.steer(session_id, turn_id, text)

    def queue(self, session_id: UUID, text: str, mode: TurnMode = "normal") -> tuple[UUID, SessionEvent]:
        return self._agent.queue(session_id, text, mode)

    def interrupt(self, session_id: UUID, turn_id: UUID) -> SessionEvent | None:
        event = self._agent.interrupt(session_id, turn_id)
        if event is not None:
            self._record_terminal(session_id, turn_id, event)
        return event

    def _capture(self, session_id: UUID, turn_id: UUID, events: object):
        for event in events:  # type: ignore[union-attr]
            facts = self._facts.setdefault((session_id, turn_id), {
                "changed_paths": set(), "pytest_outcome": "not_run", "approval_outcome": "none",
            })
            if event.kind == "tool_item_completed":
                raw_paths = event.data.get("changed_paths")
                if raw_paths is not None:
                    facts["changed_paths"].update(json.loads(raw_paths))  # type: ignore[union-attr]
                if "pytest_outcome" in event.data:
                    facts["pytest_outcome"] = event.data["pytest_outcome"]
            if event.kind == "approval_resolved":
                facts["approval_outcome"] = "approved" if event.data["accepted"] == "true" else "rejected"
            if event.kind in {"turn_completed", "turn_interrupted", "turn_failed"}:
                self._record_terminal(session_id, turn_id, event)
            yield event

    def _record_terminal(self, session_id: UUID, turn_id: UUID, event: SessionEvent) -> None:
        summary = self.summary(session_id)
        status = {"turn_completed": "completed", "turn_interrupted": "interrupted", "turn_failed": "failed"}[event.kind]
        facts = self._facts.pop((session_id, turn_id), {
            "changed_paths": set(), "pytest_outcome": "not_run", "approval_outcome": "none",
        })
        turn = SafeTurnSummary(
            terminal_status=status, changed_paths=tuple(sorted(facts["changed_paths"])),
            pytest_outcome=facts["pytest_outcome"], approval_outcome=facts["approval_outcome"],
            final_text=_bounded_safe_summary(_safe_summary_text(status, facts)),
        )
        updated = summary.model_copy(update={"updated_at": datetime.now(timezone.utc), "turns": (*summary.turns, turn)})
        self._summaries[session_id] = updated
        self.store.save_summary(updated)

def _safe_summary_text(status: str, facts: dict[str, object]) -> str:
    """Persist only deterministic lifecycle and tool facts, never model text."""
    text = {
        "completed": "本轮已完成。",
        "interrupted": "本轮已中断。",
        "failed": "本轮未完成。",
    }[status]
    paths = facts["changed_paths"]
    if paths:
        text += f"已修改 {', '.join(sorted(paths))}。"
    pytest_outcome = facts["pytest_outcome"]
    if pytest_outcome != "not_run":
        text += {
            "passed": "pytest：通过。",
            "assertion_failure": "pytest：发现断言失败。",
            "collection_error": "pytest：收集失败。",
            "execution_error": "pytest：执行失败。",
            "timeout": "pytest：超时。",
        }[pytest_outcome]
    approval_outcome = facts["approval_outcome"]
    if approval_outcome == "approved":
        text += "危险操作已批准。"
    elif approval_outcome == "rejected":
        text += "危险操作已拒绝。"
    return text


def _bounded_safe_summary(text: str) -> str:
    return text if len(text) <= _MAX_SUMMARY_TEXT else text[: _MAX_SUMMARY_TEXT - 1] + "…"


class LocalRuntime:
    """Own the current project configuration and keyring boundary for the CLI."""

    def __init__(self, services: RuntimeServices) -> None:
        self._services = services
        self._project_root: Path | None = None
        self._config: HarnessConfig | None = None
        self._lock = RLock()
        self._restore_selected_project()

    @property
    def project_root(self) -> Path | None:
        return self._project_root

    @property
    def config(self) -> HarnessConfig | None:
        return self._config

    def setup(self, profile: ProjectProfile, api_key: str | None = None) -> None:
        with self._lock:
            root = profile.root.resolve()
            config = load_or_create_discovered_config(profile, project_config_path(root).parent.parent)
            _write_selected_project(root)
            if api_key is not None and api_key.strip():
                self._services.credentials.set_key(api_key)
            self._project_root = root
            self._config = config

    def update_future_defaults(self, *, model: str | None = None, reasoning_effort: str | None = None) -> HarnessConfig:
        with self._lock:
            _, config = self._configured()
            changed = updated_future_defaults(config, model=model, reasoning_effort=reasoning_effort)
            save_discovered_config(changed, project_config_path(changed.profile.root).parent.parent)
            self._config = changed
            return changed

    def credential_status(self) -> CredentialStatus:
        return self._services.credentials.status()

    def update_credential(self, api_key: str) -> None:
        self._services.credentials.set_key(api_key)

    def clear_credential(self) -> None:
        self._services.credentials.clear_key()

    def local_check(self, name: str) -> str:
        """Run one explicitly requested, bounded local diagnostic."""
        root, config = self._configured()
        workspace = Workspace(root, config)
        if name == "tests":
            feedback = FeedbackCollector().collect(workspace.run_pytest(()))
            return f"pytest：{feedback.kind.value}"
        if name == "diff":
            result = workspace.git_diff()
            if not result.ok:
                return "Git diff：不可用"
            return "Git diff：有变更" if result.data.get("output") else "Git diff：无变更"
        if name == "doctor":
            configured = self.credential_status().configured
            return f"诊断：项目已识别；凭据{'已配置' if configured else '未配置'}"
        raise ValueError("unknown local check")

    def _configured(self) -> tuple[Path, HarnessConfig]:
        if self._project_root is None or self._config is None:
            raise RuntimeNotConfiguredError()
        return self._project_root, self._config

    def _restore_selected_project(self) -> None:
        try:
            root = _read_selected_project()
            self._project_root = root
            self._config = load_config(project_config_path(root), root)
        except (OSError, ValueError, yaml.YAMLError):
            return


def _write_selected_project(project_root: Path) -> None:
    path = local_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"selected_project": str(project_root)}, sort_keys=False))


def _read_selected_project() -> Path:
    payload = yaml.safe_load(local_state_path().read_text())
    if not isinstance(payload, dict) or set(payload) != {"selected_project"}:
        raise ValueError("local state has invalid fields")
    value = payload["selected_project"]
    if not isinstance(value, str):
        raise ValueError("local state has invalid values")
    root = Path(value)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("stored project root is invalid")
    return root.resolve()
