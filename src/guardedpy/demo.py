"""A fixed, offline public demonstration of GuardedPy's safety mechanisms."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from guardedpy.actions import RunCommandAction
from guardedpy.config import HarnessConfig
from guardedpy.domain import TaskMode, TaskState, TaskStatus
from guardedpy.events import EventStore, StoredRunEvent
from guardedpy.llm import ScriptedLLM
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.workspace import ToolResult


SCENARIOS = (
    "dangerous_action_denied",
    "failure_feedback_corrects",
    "tdd_source_patch_denied",
)

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """The read-only evidence emitted by one fresh fixed demo run."""

    name: str
    status: TaskStatus
    events: tuple[StoredRunEvent, ...]
    command_dispatches: tuple[tuple[str, ...], ...]
    source_value: str


class _DemoOrchestrator(TaskOrchestrator):
    """Keep generic command actions observable without ever launching a command."""

    def __init__(self, project_root: Path, llm: ScriptedLLM) -> None:
        super().__init__(project_root, llm)
        self.command_dispatches: list[tuple[str, ...]] = []

    def _run_command(self, action: RunCommandAction) -> ToolResult:
        self.command_dispatches.append(action.args)
        return ToolResult(True, "Demo command dispatch was simulated", {})


def create_demo_app() -> FastAPI:
    """Create the public, fixed-scenario surface without any local-control services."""
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "demo.html", {"scenarios": SCENARIOS})

    @app.get("/demo/scenarios")
    async def scenarios() -> list[str]:
        return list(SCENARIOS)

    @app.get("/demo/scenarios/{name}")
    async def scenario(name: str) -> dict[str, object]:
        try:
            result = run_scenario(name)
        except KeyError:
            raise HTTPException(status_code=404, detail="未找到演示场景。") from None
        return _result_payload(result)

    return app


def run_scenario(name: str) -> ScenarioResult:
    """Run one hard-coded scenario in a new temporary project and state directory."""
    if name not in SCENARIOS:
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
        orchestrator = _DemoOrchestrator(root, ScriptedLLM(_responses_for(name)))
        completed = orchestrator.run(task)
        return ScenarioResult(
            name=name,
            status=completed.status,
            events=tuple(EventStore(root).events_for(task.id)),
            command_dispatches=tuple(orchestrator.command_dispatches),
            source_value=(root / "src" / "value.py").read_text(),
        )


@contextmanager
def _isolated_demo_root() -> Iterator[Path]:
    """Isolate every demo run from the caller's state and project filesystem."""
    original_state_home = os.environ.get("XDG_STATE_HOME")
    with TemporaryDirectory(prefix="guardedpy-demo-") as directory:
        temporary = Path(directory)
        os.environ["XDG_STATE_HOME"] = str(temporary / "state")
        try:
            root = temporary / "project"
            root.mkdir()
            yield root
        finally:
            if original_state_home is None:
                os.environ.pop("XDG_STATE_HOME", None)
            else:
                os.environ["XDG_STATE_HOME"] = original_state_home


def _write_fixture(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "value.py").write_text("VALUE = 'broken'\n")
    (root / "tests" / "test_value.py").write_text(
        "from pathlib import Path\n\n"
        "def test_value_is_fixed() -> None:\n"
        "    assert Path('src/value.py').read_text() == \"VALUE = 'fixed'\\n\"\n"
    )


def _description_for(name: str) -> str:
    return {
        "dangerous_action_denied": "Attempt a prohibited privileged action.",
        "failure_feedback_corrects": "Correct the selected assertion failure.",
        "tdd_source_patch_denied": "Attempt a source patch before observing red.",
    }[name]


def _responses_for(name: str) -> list[str]:
    dangerous = [
        _action(kind="run_command", summary="attempt privilege escalation", args=["sudo", "id"]),
        _action(kind="finish", summary="stop after policy denial", status="blocked"),
    ]
    corrective = [
        _action(kind="read_file", summary="inspect value", path="src/value.py"),
        _action(kind="run_pytest", summary="observe failure", targets=[]),
        _action(
            kind="apply_patch",
            summary="repair value",
            diff=(
                "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n"
                "-VALUE = 'broken'\n+VALUE = 'fixed'\n"
            ),
        ),
        _action(kind="run_pytest", summary="run full suite", targets=[]),
        _action(kind="finish", summary="finish after green", status="completed"),
    ]
    tdd_denied = [
        _action(kind="read_file", summary="inspect value", path="src/value.py"),
        _action(
            kind="apply_patch",
            summary="patch source before red",
            diff=(
                "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n"
                "-VALUE = 'broken'\n+VALUE = 'fixed'\n"
            ),
        ),
        _action(kind="finish", summary="stop after TDD denial", status="blocked"),
    ]
    return {
        "dangerous_action_denied": dangerous,
        "failure_feedback_corrects": corrective,
        "tdd_source_patch_denied": tdd_denied,
    }[name]


def _action(**payload: object) -> str:
    return json.dumps(payload)


def _result_payload(result: ScenarioResult) -> dict[str, object]:
    return {
        "name": result.name,
        "status": result.status.value,
        "events": [event.model_dump(mode="json") for event in result.events],
        "command_dispatches": [list(args) for args in result.command_dispatches],
        "source_value": result.source_value,
    }
