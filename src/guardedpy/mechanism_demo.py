"""Provider-free, headless evidence from GuardedPy's retained Harness core."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Iterator, Literal

from guardedpy.actions import RunCommandAction
from guardedpy.config import HarnessConfig
from guardedpy.context import LlmContext
from guardedpy.domain import FeedbackKind, PolicyVerdict, TaskMode, TaskState
from guardedpy.events import EventStore, StoredRunEvent
from guardedpy.llm import ScriptedLLM
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.workspace import ToolResult


ScenarioName = Literal[
    "dangerous_action_denied",
    "failure_feedback_corrects",
    "tdd_source_patch_denied",
]
_SCENARIOS: tuple[ScenarioName, ...] = (
    "dangerous_action_denied",
    "failure_feedback_corrects",
    "tdd_source_patch_denied",
)
_PYTEST_CONTROL_VARIABLES = (
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Bounded facts derived from one fresh Harness run."""

    name: ScenarioName
    status: str
    rule_id: str | None
    feedback_kind: str | None
    dispatched_command: bool
    event_kinds: tuple[str, ...]
    workspace_value: str


class FeedbackAwareDemoLLM(ScriptedLLM):
    """Return the repair patch only after trusted assertion feedback is present."""

    def __init__(self) -> None:
        super().__init__(_corrective_responses())

    def complete(self, context: LlmContext) -> str:
        if len(self.contexts) == 2:
            feedback = context.trusted_data.get("feedback")
            assert isinstance(feedback, dict) and feedback.get("type") == "pytest_feedback" and (
                feedback.get("kind") == "assertion_failure"
            ), "repair requires trusted assertion feedback"
        return super().complete(context)


class _DemoOrchestrator(TaskOrchestrator):
    """Record generic command dispatch attempts without launching a process."""

    def __init__(self, project_root: Path, llm: ScriptedLLM) -> None:
        super().__init__(project_root, llm)
        self.dispatched_commands: list[tuple[str, ...]] = []

    def _run_command(self, action: RunCommandAction) -> ToolResult:
        self.dispatched_commands.append(action.args)
        return ToolResult(True, "Demo command dispatch was recorded", {})


def run_scenario(name: ScenarioName) -> ScenarioResult:
    """Execute one fixed scenario through the actual governed Harness loop."""
    if name not in _SCENARIOS:
        raise KeyError(name)

    with _isolated_demo_root() as root:
        _write_fixture(root)
        task = TaskState(
            description=_description_for(name),
            mode=TaskMode.BUGFIX,
            bugfix_target="tests/test_value.py::test_value_is_fixed",
            config=HarnessConfig(
                source_dirs=(Path("src"),),
                test_dirs=(Path("tests"),),
                pytest_command=(sys.executable, "-m", "pytest", "-q"),
            ),
        )
        llm = FeedbackAwareDemoLLM() if name == "failure_feedback_corrects" else ScriptedLLM(
            _responses_for(name)
        )
        orchestrator = _DemoOrchestrator(root, llm)
        completed = orchestrator.run(task)
        events = tuple(EventStore(root).events_for(task.id))
        return ScenarioResult(
            name=name,
            status=completed.status.value,
            rule_id=_denial_rule(events),
            feedback_kind=_assertion_feedback_kind(events),
            dispatched_command=bool(orchestrator.dispatched_commands),
            event_kinds=_event_kinds(events),
            workspace_value=(root / "src" / "value.py").read_text().strip(),
        )


def run_all_scenarios() -> tuple[ScenarioResult, ...]:
    """Execute the three literal scenarios in their stable evidence order."""
    return tuple(run_scenario(name) for name in _SCENARIOS)


@contextmanager
def _isolated_demo_root() -> Iterator[Path]:
    original_state_home = os.environ.get("XDG_STATE_HOME")
    original_pytest_environment = {
        name: os.environ.get(name) for name in _PYTEST_CONTROL_VARIABLES
    }
    with TemporaryDirectory(prefix="guardedpy-demo-") as directory:
        temporary = Path(directory)
        os.environ["XDG_STATE_HOME"] = str(temporary / "state")
        os.environ.pop("PYTEST_ADDOPTS", None)
        os.environ.pop("PYTEST_PLUGINS", None)
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        try:
            root = temporary / "project"
            root.mkdir()
            yield root
        finally:
            if original_state_home is None:
                os.environ.pop("XDG_STATE_HOME", None)
            else:
                os.environ["XDG_STATE_HOME"] = original_state_home
            for name, value in original_pytest_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _write_fixture(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "value.py").write_text("broken\n")
    (root / "tests" / "test_value.py").write_text(
        "from pathlib import Path\n\n"
        "def test_value_is_fixed() -> None:\n"
        "    assert Path('src/value.py').read_text() == 'fixed\\n'\n"
    )


def _description_for(name: ScenarioName) -> str:
    return {
        "dangerous_action_denied": "Attempt a prohibited privileged action.",
        "failure_feedback_corrects": "Correct the selected assertion failure.",
        "tdd_source_patch_denied": "Attempt a source patch before observing red.",
    }[name]


def _responses_for(name: ScenarioName) -> list[str]:
    if name == "dangerous_action_denied":
        return [
            _action(kind="run_command", summary="attempt privilege escalation", args=["sudo", "id"]),
            _action(kind="finish", summary="stop after policy denial", status="blocked"),
        ]
    if name == "tdd_source_patch_denied":
        return [
            _action(kind="read_file", summary="inspect value", path="src/value.py"),
            _repair_action(),
            _action(kind="finish", summary="stop after TDD denial", status="blocked"),
        ]
    raise KeyError(name)


def _corrective_responses() -> list[str]:
    return [
        _action(kind="read_file", summary="inspect value", path="src/value.py"),
        _action(kind="run_pytest", summary="observe assertion feedback", targets=[]),
        _repair_action(),
        _action(kind="run_pytest", summary="run full suite", targets=[]),
        _action(kind="finish", summary="finish after green", status="completed"),
    ]


def _repair_action() -> str:
    return _action(
        kind="apply_patch",
        summary="repair value",
        diff="--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n-broken\n+fixed\n",
    )


def _action(**payload: object) -> str:
    return json.dumps(payload)


def _denial_rule(events: tuple[StoredRunEvent, ...]) -> str | None:
    return next(
        (
            event.policy_rule_id
            for event in events
            if event.policy_verdict is PolicyVerdict.DENY
            and event.policy_rule_id is not None
        ),
        None,
    )


def _assertion_feedback_kind(events: tuple[StoredRunEvent, ...]) -> str | None:
    return next(
        (
            event.feedback_kind.value
            for event in events
            if event.feedback_kind is FeedbackKind.ASSERTION_FAILURE
        ),
        None,
    )


def _event_kinds(events: tuple[StoredRunEvent, ...]) -> tuple[str, ...]:
    kinds: list[str] = []
    for event in events:
        if event.policy_verdict is PolicyVerdict.DENY:
            kinds.append("policy_denial")
        if event.feedback_kind is FeedbackKind.ASSERTION_FAILURE:
            kinds.append("assertion_feedback")
        if event.action_summary == "apply source patch" and event.policy_verdict is PolicyVerdict.ALLOW:
            kinds.append("source_patch")
        if event.feedback_kind is FeedbackKind.PASSED and event.action_summary == "run configured tests":
            kinds.append("full_suite_pass")
    return tuple(kinds)
