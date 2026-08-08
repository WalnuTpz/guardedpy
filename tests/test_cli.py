"""Public contracts for GuardedPy's sole terminal entry point."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import TaskIntent, TaskState, TaskStatus


class _Runtime:
    def __init__(self) -> None:
        self.config: HarnessConfig | None = None
        self.project_root: Path | None = None
        self.created: list[TaskState] = []
        self.setup_profiles: list[ProjectProfile] = []

    def setup(self, profile: ProjectProfile, api_key: str | None) -> None:
        assert api_key is None
        self.project_root = profile.root
        self.config = HarnessConfig(profile=profile)
        self.setup_profiles.append(profile)

    def create_task(self, description: str, intent: TaskIntent, review_path: str | None = None) -> TaskState:
        task = TaskState(description=description, intent=intent, config=self.config, review_path=review_path)
        self.created.append(task)
        return task

    def run(self, task_id: object) -> TaskState:
        task = next(task for task in self.created if task.id == task_id)
        task.status = TaskStatus.COMPLETED
        return task

    def events(self, task_id: object) -> list[object]:
        del task_id
        return []

    def credential_status(self) -> CredentialStatus:
        return CredentialStatus(configured=False)


def _project(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n")
    (tmp_path / "tests").mkdir()


def test_help_exposes_only_the_cli_only_surface(capsys: object) -> None:
    """Catches a help path reintroducing manual init, server, or shell modes."""
    from guardedpy.cli import main

    assert main(["--help"]) == 0
    rendered = capsys.readouterr().out.lower()  # type: ignore[attr-defined]
    assert "task" in rendered
    assert "demo" in rendered
    for retired in ("serve", "server", "webui", "api", "/init", "--prompt"):
        assert retired not in rendered


def test_direct_task_discovers_cwd_and_uses_safe_non_tty_lifecycle(tmp_path: Path, monkeypatch: object) -> None:
    """Catches direct CLI work using stale project setup or a separate task runner."""
    from guardedpy.cli import main

    _project(tmp_path)
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    runtime = _Runtime()
    output = StringIO()

    code = main(["inspect project"], runtime_factory=lambda: runtime, stdin=StringIO(), stdout=output)

    assert code == 0
    assert runtime.setup_profiles[0].root == tmp_path.resolve()
    assert runtime.created[0].description == "inspect project"
    assert "completed" in output.getvalue()


def test_demo_non_tty_is_offline_and_never_composes_project_runtime(monkeypatch: object) -> None:
    """Catches demo acquiring a provider, keyring, or caller-project runtime."""
    from guardedpy.cli import main

    output = StringIO()
    assert main(["demo"], runtime_factory=lambda: (_ for _ in ()).throw(AssertionError()), stdin=StringIO(), stdout=output) == 0
    assert output.getvalue().splitlines() == [
        "dangerous_action_denied status=blocked",
        "failure_feedback_corrects status=completed",
        "tdd_source_patch_denied status=blocked",
    ]
