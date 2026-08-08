"""Offline, new-core continuous-agent mechanism evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Callable
from typing import Literal
from uuid import UUID, uuid4

from guardedpy.config import HarnessConfig
from guardedpy.conversation import (
    ConversationAgent, ResponseFinished, ScriptedConversationModel, SessionEvent, TextDelta,
    ToolCallDelta, TurnNotActiveError,
)
from guardedpy.discovery import discover_project
from guardedpy.executor import ToolExecutor
from guardedpy.governor import ToolGovernor, governed_tool_definitions


ScenarioName = Literal["delete_requires_approval", "feedback_repair", "stale_approval_denied"]
_SCENARIOS: tuple[ScenarioName, ...] = (
    "delete_requires_approval", "feedback_repair", "stale_approval_denied",
)
_REQUESTS: dict[ScenarioName, str] = {
    "delete_requires_approval": "删除 src/value.py。",
    "feedback_repair": "修复当前失败的测试。",
    "stale_approval_denied": "使用过期审批尝试删除 src/value.py。",
}


@dataclass(frozen=True)
class ScenarioResult:
    name: ScenarioName
    status: str
    event_kinds: tuple[str, ...]
    workspace_value: str
    stale_approval_denied: bool = False


def scenario_request(name: ScenarioName) -> str:
    """Return the sole fixed request shown for one offline demo scenario."""
    try:
        return _REQUESTS[name]
    except KeyError:
        raise KeyError(name) from None


def run_scenario(
    name: ScenarioName,
    *,
    on_event: Callable[[SessionEvent], None] | None = None,
    approval_resolver: Callable[[SessionEvent], bool] | None = None,
) -> ScenarioResult:
    """Run one isolated mock scenario, optionally exposing its real event sequence."""
    if name not in _SCENARIOS:
        raise KeyError(name)
    with TemporaryDirectory(prefix="guardedpy-continuous-demo-") as directory:
        root = Path(directory)
        _fixture(root, repair=name == "feedback_repair")
        config = HarnessConfig(profile=discover_project(root))
        model = ScriptedConversationModel(_responses(name))
        agent = ConversationAgent(
            model, governed_tool_definitions(config), ToolGovernor(config), ToolExecutor(root, config)
        )
        session_id = agent.create_session()
        turn_id, submitted = agent.begin_turn(session_id, scenario_request(name))
        events: list[SessionEvent] = []

        def record(event: SessionEvent) -> None:
            events.append(event)
            if on_event is not None:
                on_event(event)

        record(submitted)
        for event in agent.run_turn(session_id, turn_id):
            record(event)
        stale = False
        if name != "feedback_repair":
            approval_event = next(event for event in events if event.kind == "approval_requested")
            approval_id = UUID(approval_event.data["approval_id"])
            accepted = False
            if name == "delete_requires_approval" and approval_resolver is not None:
                accepted = approval_resolver(approval_event)
            if name == "stale_approval_denied":
                try:
                    list(agent.resolve_approval(session_id, turn_id, uuid4(), True))
                except TurnNotActiveError:
                    stale = True
            for event in agent.resolve_approval(session_id, turn_id, approval_id, accepted):
                record(event)
        target = root / "src" / "value.py"
        return ScenarioResult(
            name=name,
            status=events[-1].kind.removeprefix("turn_"),
            event_kinds=_facts(events),
            workspace_value=target.read_text().strip() if target.exists() else "<deleted>",
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
    if name == "delete_requires_approval":
        return [
            [ToolCallDelta(0, "delete", "delete_path", '{"path":"src/value.py"}'), ResponseFinished("tool_calls")],
            [TextDelta("已根据你的审批决定完成处理。"), ResponseFinished("stop")],
        ]
    if name == "stale_approval_denied":
        return [
            [ToolCallDelta(0, "delete", "delete_path", '{"path":"src/value.py"}'), ResponseFinished("tool_calls")],
            [TextDelta("过期审批 ID 已被确定性拒绝，src/value.py 保持不变。"), ResponseFinished("stop")],
        ]
    patch = "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-broken\n+fixed\n"
    return [
        [ToolCallDelta(0, "read", "read_file", '{"path":"src/value.py"}'), ResponseFinished("tool_calls")],
        [ToolCallDelta(0, "red", "run_pytest", "{}"), ResponseFinished("tool_calls")],
        [ToolCallDelta(0, "patch", "apply_patch", json.dumps({"unified_diff": patch})), ResponseFinished("tool_calls")],
        [ToolCallDelta(0, "green", "run_pytest", "{}"), ResponseFinished("tool_calls")],
        [TextDelta("已根据 pytest 失败反馈修复 src/value.py，并验证通过。"), ResponseFinished("stop")],
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
