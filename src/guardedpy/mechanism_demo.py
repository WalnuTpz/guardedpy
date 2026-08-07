"""Offline, new-core continuous-agent mechanism evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import UUID, uuid4

from guardedpy.config import HarnessConfig
from guardedpy.conversation import (
    ConversationAgent, ResponseFinished, ScriptedConversationModel, TextDelta,
    ToolCallDelta, TurnNotActiveError,
)
from guardedpy.discovery import discover_project
from guardedpy.executor import ToolExecutor
from guardedpy.governor import ToolGovernor, governed_tool_definitions


ScenarioName = Literal["delete_approval_rejected", "feedback_repair", "stale_approval_denied"]
_SCENARIOS: tuple[ScenarioName, ...] = (
    "delete_approval_rejected", "feedback_repair", "stale_approval_denied",
)


@dataclass(frozen=True)
class ScenarioResult:
    name: ScenarioName
    status: str
    event_kinds: tuple[str, ...]
    workspace_value: str
    stale_approval_denied: bool = False


def run_scenario(name: ScenarioName) -> ScenarioResult:
    if name not in _SCENARIOS:
        raise KeyError(name)
    with TemporaryDirectory(prefix="guardedpy-continuous-demo-") as directory:
        root = Path(directory)
        _fixture(root, repair=name == "feedback_repair")
        config = HarnessConfig(profile=discover_project(root))
        model = ScriptedConversationModel(_responses(name))
        agent = ConversationAgent(
            model, governed_tool_definitions(), ToolGovernor(config), ToolExecutor(root, config)
        )
        session_id = agent.create_session()
        turn_id, _ = agent.begin_turn(session_id, "Run fixed offline scenario")
        events = list(agent.run_turn(session_id, turn_id))
        stale = False
        if name != "feedback_repair":
            approval_id = UUID(next(event.data["approval_id"] for event in events if event.kind == "approval_requested"))
            if name == "stale_approval_denied":
                try:
                    list(agent.resolve_approval(session_id, turn_id, uuid4(), True))
                except TurnNotActiveError:
                    stale = True
            events.extend(agent.resolve_approval(session_id, turn_id, approval_id, False))
        return ScenarioResult(
            name=name,
            status=events[-1].kind.removeprefix("turn_"),
            event_kinds=_facts(events),
            workspace_value=(root / "src" / "value.py").read_text().strip(),
            stale_approval_denied=stale,
        )


def run_all_scenarios() -> tuple[ScenarioResult, ...]:
    return tuple(run_scenario(name) for name in _SCENARIOS)


def _fixture(root: Path, *, repair: bool) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "value.py").write_text("broken\n" if repair else "present\n")
    if repair:
        (root / "tests" / "test_value.py").write_text(
            "from pathlib import Path\n\ndef test_value():\n"
            "    assert Path('src/value.py').read_text() == 'fixed\\n'\n"
        )
    else:
        (root / "tests" / "test_value.py").write_text("def test_value(): assert True\n")


def _responses(name: ScenarioName) -> list[list[object]]:
    if name != "feedback_repair":
        return [
            [ToolCallDelta(0, "delete", "delete_path", '{"path":"src/value.py"}'), ResponseFinished("tool_calls")],
            [TextDelta("Kept the file."), ResponseFinished("stop")],
        ]
    patch = "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-broken\n+fixed\n"
    return [
        [ToolCallDelta(0, "read", "read_file", '{"path":"src/value.py"}'), ResponseFinished("tool_calls")],
        [ToolCallDelta(0, "red", "run_pytest", "{}"), ResponseFinished("tool_calls")],
        [ToolCallDelta(0, "patch", "apply_patch", json.dumps({"unified_diff": patch})), ResponseFinished("tool_calls")],
        [ToolCallDelta(0, "green", "run_pytest", "{}"), ResponseFinished("tool_calls")],
        [TextDelta("Repaired and verified."), ResponseFinished("stop")],
    ]


def _facts(events: list[object]) -> tuple[str, ...]:
    facts: list[str] = []
    for event in events:
        if event.kind in {"approval_requested", "approval_resolved", "tool_item_completed"}:
            facts.append(event.kind)
        if event.data.get("pytest_outcome") == "assertion_failure":
            facts.append("assertion_failure")
        if event.data.get("changed_paths"):
            facts.append("patch_applied")
        if event.data.get("pytest_outcome") == "passed":
            facts.append("pytest_passed")
    return tuple(facts)
