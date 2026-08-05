"""Rendered-DOM contracts for the GuardedPy local console foundation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

from guardedpy.credentials import CredentialStatus


@dataclass
class FakeCredentials:
    """Keep rendered-page tests independent from the operating-system keyring."""

    configured: bool = False

    def status(self) -> CredentialStatus:
        return CredentialStatus(configured=self.configured)

    def set_key(self, key: str) -> None:
        del key
        self.configured = True

    def clear_key(self) -> None:
        self.configured = False


@dataclass
class FakeOrchestrator:
    """Accept task setup without starting a real harness run."""

    def submit(self, task: Any) -> Any:
        return task

    def run(self, task: Any) -> Any:
        return task

    def cancel(self, task_id: Any) -> Any:
        del task_id
        raise AssertionError("task cancellation is outside this rendered-page contract")


class RenderedDocument(HTMLParser):
    """Extract semantic page contracts from the response users actually receive."""

    def __init__(self) -> None:
        super().__init__()
        self.lang: str | None = None
        self.landmarks: set[str] = set()
        self.navigation_hrefs: list[str] = []
        self.current_href: str | None = None
        self.form_actions: list[str] = []
        self.input_names: set[str] = set()
        self.section_labels: set[str] = set()
        self.data_od_ids: set[str] = set()
        self.badge_statuses: set[str] = set()
        self.bugfix_target_copies: set[str] = set()
        self.has_task_mode_control = False
        self._inside_navigation = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html":
            self.lang = attributes.get("lang")
        if tag == "nav":
            self.landmarks.add("navigation")
            self._inside_navigation += 1
        if tag == "main":
            self.landmarks.add("main")
        if attributes.get("data-od-id"):
            self.data_od_ids.add(attributes["data-od-id"])
        if tag == "a" and self._inside_navigation and attributes.get("href"):
            href = attributes["href"]
            self.navigation_hrefs.append(href)
            if attributes.get("aria-current") == "page":
                self.current_href = href
        if tag == "form":
            self.form_actions.append(attributes.get("action", ""))
        if tag in {"input", "select", "textarea"} and attributes.get("name"):
            self.input_names.add(attributes["name"])
        if tag == "select" and "data-task-mode" in attributes:
            self.has_task_mode_control = True
        if tag == "label" and attributes.get("data-feature-copy"):
            self.bugfix_target_copies.add(attributes["data-feature-copy"])
            self.bugfix_target_copies.add(attributes["data-bugfix-copy"])
        if "badge" in attributes.get("class", "").split() and attributes.get("data-status"):
            self.badge_statuses.add(attributes["data-status"])
        if tag == "section" and attributes.get("aria-label"):
            self.section_labels.add(attributes["aria-label"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav":
            self._inside_navigation -= 1


async def _request(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _document(response: httpx.Response) -> RenderedDocument:
    document = RenderedDocument()
    document.feed(response.text)
    return document


def _app() -> Any:
    from guardedpy import web

    return web.create_app(
        "local",
        web.WebServices(
            credentials=FakeCredentials(),
            orchestrator_factory=lambda root, config, memory: FakeOrchestrator(),
        ),
    )


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    return root


def _setup_data(root: Path) -> dict[str, str]:
    return {
        "project_root": str(root),
        "source_dirs": "src",
        "test_dirs": "tests",
        "pytest_command": "pytest",
        "model": "deepseek-chat",
        "timeout_seconds": "30",
        "api_key": "test-only-key",
    }


def test_shell_exposes_chinese_landmarks_navigation_and_explicit_current_page(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches a shell that loses its keyboard landmarks or page location."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = _app()
    root = _project_root(tmp_path)

    setup_page = asyncio.run(_request(app, "GET", "/setup"))
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303
    task_page = asyncio.run(_request(app, "GET", "/tasks/new"))
    credentials_page = asyncio.run(_request(app, "GET", "/settings/credentials"))

    expected_navigation = [
        "/",
        "/tasks/new",
        "/memories",
        "/settings/command-rules",
        "/settings/credentials",
    ]
    for response, expected_current_href, expected_form_actions in [
        (setup_page, "/", ["/setup"]),
        (task_page, "/tasks/new", ["/tasks"]),
        (credentials_page, "/settings/credentials", ["/settings/credentials", "/settings/credentials/clear"]),
    ]:
        document = _document(response)
        assert document.lang == "zh-CN"
        assert document.landmarks == {"navigation", "main"}
        assert document.navigation_hrefs == expected_navigation
        assert document.current_href == expected_current_href
        assert document.form_actions == expected_form_actions
    assert {"app-rail", "primary-navigation", "context-header", "main-content"} <= _document(task_page).data_od_ids
    assert _document(credentials_page).badge_statuses == {"configured"}


def test_setup_keeps_one_submission_form_with_its_security_fields(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches a visual setup rebuild splitting or dropping persisted setup inputs."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    document = _document(asyncio.run(_request(_app(), "GET", "/setup")))

    assert document.form_actions == ["/setup"]
    assert document.input_names == {
        "project_root",
        "source_dirs",
        "test_dirs",
        "pytest_command",
        "model",
        "timeout_seconds",
        "api_key",
    }
    assert {"项目边界", "测试策略", "模型运行时"} <= document.section_labels


def test_task_workspace_keeps_submission_contract_in_three_labelled_regions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches task-workspace layout changes hiding lifecycle controls or history."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = _app()
    root = _project_root(tmp_path)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303
    document = _document(asyncio.run(_request(app, "GET", "/tasks/new")))

    assert document.form_actions == ["/tasks"]
    assert {"mode", "description", "bugfix_target"} <= document.input_names
    assert {"当前任务", "新建任务", "任务历史"} <= document.section_labels
    assert document.has_task_mode_control
    assert document.bugfix_target_copies == {"功能开发不需要缺陷测试目标。", "缺陷修复必须填写唯一的 pytest 测试目标。"}
    assert {"task-current", "task-create", "task-history"} <= document.data_od_ids


def test_shared_stylesheet_provides_neutral_modern_accessible_shell_primitives() -> None:
    """Catches a shared stylesheet that removes the responsive console guarantees."""
    stylesheet = (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.css").read_text()

    for token, value in {
        "--bg": "#FAFAFA",
        "--surface": "#FFFFFF",
        "--fg": "#111111",
        "--muted": "#6B6B6B",
        "--border": "#E5E5E5",
        "--accent": "#2F6FEB",
        "--success": "#17A34A",
        "--warn": "#EAB308",
        "--danger": "#DC2626",
        "--rail-width": "240px",
        "--container-max": "1200px",
    }.items():
        assert f"{token}: {value}" in stylesheet
    assert ".rail" in stylesheet and "position: fixed" in stylesheet
    assert "@media (max-width: 639px)" in stylesheet
    assert ":focus-visible" in stylesheet
    assert "min-height: 44px" in stylesheet
    assert "overflow-wrap: anywhere" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet
    assert "input:hover" in stylesheet
    assert "input:active" in stylesheet
    assert "input:disabled" in stylesheet
    assert "[aria-invalid=\"true\"]" in stylesheet
    assert ".empty-state" in stylesheet
    assert "border-left: 2px solid var(--accent)" in stylesheet
    assert "border-top: 4px solid var(--warn)" in stylesheet
