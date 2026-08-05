"""Rendered-DOM contracts for the GuardedPy local console foundation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from guardedpy.actions import RunCommandAction
from guardedpy.command_rules import CommandRuleStore
from guardedpy.credentials import CredentialBackendUnavailableError, CredentialStatus


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


@dataclass
class UnavailableCredentials:
    """Exercise the fixed safe-error presentation without a real keyring."""

    def status(self) -> CredentialStatus:
        raise CredentialBackendUnavailableError("unavailable")

    def set_key(self, key: str) -> None:
        del key
        raise CredentialBackendUnavailableError("unavailable")

    def clear_key(self) -> None:
        raise CredentialBackendUnavailableError("unavailable")


class RenderedDocument(HTMLParser):
    """Extract semantic page contracts from the response users actually receive."""

    def __init__(self) -> None:
        super().__init__()
        self.lang: str | None = None
        self.landmarks: set[str] = set()
        self.navigation_hrefs: list[str] = []
        self.current_href: str | None = None
        self.forms: list[tuple[str, set[str]]] = []
        self.section_labels: set[str] = set()
        self.data_od_ids: set[str] = set()
        self.badge_statuses: set[str] = set()
        self.bugfix_target_copies: set[str] = set()
        self.has_task_mode_control = False
        self.task_link_hrefs: list[str] = []
        self.anchors: list[tuple[str, str]] = []
        self.header_texts: list[str] = []
        self._anchor_captures: list[list[str]] = []
        self._header_captures: list[list[str]] = []
        self._inside_navigation = 0
        self._form_stack: list[tuple[str, set[str]]] = []
        self._script_fragments: list[str] = []
        self._mono_text_captures: list[tuple[str, list[str]]] = []
        self.mono_texts: list[str] = []
        self._inside_script = False

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
        if tag == "a" and "task-link" in attributes.get("class", "").split():
            self.task_link_hrefs.append(attributes.get("href", ""))
        if tag == "a":
            self._anchor_captures.append([attributes.get("href", "")])
        if tag == "header":
            self._header_captures.append([])
        if tag == "form":
            form = (attributes.get("action", ""), set())
            self.forms.append(form)
            self._form_stack.append(form)
        if tag in {"input", "select", "textarea"} and attributes.get("name"):
            if self._form_stack:
                self._form_stack[-1][1].add(attributes["name"])
        if tag == "select" and "data-task-mode" in attributes:
            self.has_task_mode_control = True
        if tag == "label" and attributes.get("data-feature-copy"):
            self.bugfix_target_copies.add(attributes["data-feature-copy"])
            self.bugfix_target_copies.add(attributes["data-bugfix-copy"])
        if "badge" in attributes.get("class", "").split() and attributes.get("data-status"):
            self.badge_statuses.add(attributes["data-status"])
        if tag == "section" and attributes.get("aria-label"):
            self.section_labels.add(attributes["aria-label"])
        if "mono" in attributes.get("class", "").split():
            self._mono_text_captures.append((tag, []))
        if tag == "script":
            self._inside_script = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav":
            self._inside_navigation -= 1
        if tag == "form":
            self._form_stack.pop()
        if self._mono_text_captures and self._mono_text_captures[-1][0] == tag:
            _, fragments = self._mono_text_captures.pop()
            self.mono_texts.append("".join(fragments).strip())
        if tag == "script":
            self._inside_script = False
        if tag == "a":
            href, *fragments = self._anchor_captures.pop()
            self.anchors.append((href, "".join(fragments).strip()))
        if tag == "header":
            self.header_texts.append("".join(self._header_captures.pop()).strip())

    def handle_data(self, data: str) -> None:
        for _, fragments in self._mono_text_captures:
            fragments.append(data)
        for fragments in self._anchor_captures:
            fragments.append(data)
        for fragments in self._header_captures:
            fragments.append(data)
        if self._inside_script:
            self._script_fragments.append(data)

    def form_fields(self, action: str) -> list[set[str]]:
        return [fields for form_action, fields in self.forms if form_action == action]

    @property
    def task_mode_wiring(self) -> str:
        return "".join(self._script_fragments)


async def _request(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _document(response: httpx.Response) -> RenderedDocument:
    document = RenderedDocument()
    document.feed(response.text)
    return document


def _app(credentials: Any | None = None) -> Any:
    from guardedpy import web

    return web.create_app(
        "local",
        web.WebServices(
            credentials=credentials or FakeCredentials(),
            orchestrator_factory=lambda root, config, memory: FakeOrchestrator(),
        ),
    )


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    return root


def _setup_data(root: Path, **overrides: str) -> dict[str, str]:
    data = {
        "project_root": str(root),
        "source_dirs": "src",
        "test_dirs": "tests",
        "pytest_command": "pytest",
        "model": "deepseek-chat",
        "timeout_seconds": "30",
        "api_key": "test-only-key",
    }
    data.update(overrides)
    return data


def _css_rules(stylesheet: str, media_condition: str | None = None) -> dict[str, dict[str, str]]:
    """Parse flat rules and selected media contents without CSS-format assumptions."""
    def declarations(content: str) -> dict[str, str]:
        return {
            name.strip(): value.strip()
            for name, value in (
                declaration.split(":", 1)
                for declaration in content.split(";")
                if ":" in declaration
            )
        }

    rules: dict[str, dict[str, str]] = {}
    start = 0
    while start < len(stylesheet):
        brace = stylesheet.find("{", start)
        if brace < 0:
            break
        header = stylesheet[start:brace].strip()
        depth = 1
        end = brace + 1
        while depth and end < len(stylesheet):
            if stylesheet[end] == "{":
                depth += 1
            elif stylesheet[end] == "}":
                depth -= 1
            end += 1
        content = stylesheet[brace + 1 : end - 1]
        if header.startswith("@media"):
            if media_condition and media_condition in header:
                rules.update(_css_rules(content))
        elif media_condition is None and not header.startswith("@"):
            for selector in header.split(","):
                rules[selector.strip()] = declarations(content)
        start = end
    return rules


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
    for response, expected_current_href, expected_forms in [
        (setup_page, "/", [("/setup", {"project_root", "source_dirs", "test_dirs", "pytest_command", "model", "timeout_seconds", "api_key"})]),
        (task_page, "/tasks/new", [("/tasks", {"mode", "description", "bugfix_target"})]),
        (credentials_page, "/settings/credentials", [("/settings/credentials", {"api_key"}), ("/settings/credentials/clear", set())]),
    ]:
        document = _document(response)
        assert document.lang == "zh-CN"
        assert document.landmarks == {"navigation", "main"}
        assert document.navigation_hrefs == expected_navigation
        assert document.current_href == expected_current_href
        assert document.forms == expected_forms
    assert {"app-rail", "primary-navigation", "context-header", "main-content"} <= _document(task_page).data_od_ids
    assert _document(credentials_page).badge_statuses == {"configured"}


def test_setup_keeps_one_submission_form_with_its_security_fields(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches a visual setup rebuild splitting or dropping persisted setup inputs."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    document = _document(asyncio.run(_request(_app(), "GET", "/setup")))

    assert document.forms == [("/setup", {
        "project_root",
        "source_dirs",
        "test_dirs",
        "pytest_command",
        "model",
        "timeout_seconds",
        "api_key",
    })]
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

    assert document.forms == [("/tasks", {"mode", "description", "bugfix_target"})]
    assert {"当前任务", "新建任务", "任务历史"} <= document.section_labels
    assert document.has_task_mode_control
    assert document.bugfix_target_copies == {"功能开发不需要缺陷测试目标。", "缺陷修复必须填写唯一的 pytest 测试目标。"}
    assert {"task-current", "task-create", "task-history"} <= document.data_od_ids


def test_task_mode_copy_and_requiredness_wiring_is_complete(tmp_path: Path, monkeypatch: Any) -> None:
    """Catches mode changes leaving stale copy or bugfix target requiredness behind."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = _app()
    root = _project_root(tmp_path)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303
    wiring = _document(asyncio.run(_request(app, "GET", "/tasks/new"))).task_mode_wiring

    assert all(
        fragment in wiring
        for fragment in (
            'mode.addEventListener("change", updateBugfixTarget)',
            'mode.value === "bugfix"',
            "copy.textContent = isBugfix",
            "target.required = isBugfix",
        )
    )


def test_task_workspace_detail_links_have_the_local_44px_target_contract(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches current-task or history detail links shrinking to ordinary text targets."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = _app()
    root = _project_root(tmp_path)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303
    created = asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "feature", "description": "可访问任务链接"})
    )
    document = _document(asyncio.run(_request(app, "GET", "/tasks/new")))
    stylesheet = (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.css").read_text()

    assert created.status_code == 303
    assert document.task_link_hrefs == [created.headers["location"], created.headers["location"]]
    assert document.forms == [
        (f"/tasks/{created.headers['location'].rsplit('/', 1)[1]}/cancel", set()),
        ("/tasks", {"mode", "description", "bugfix_target"}),
    ]
    assert _css_rules(stylesheet)[".task-link"]["min-height"] == "44px"


def test_task14_pages_render_the_complete_fixed_error_set_in_chinese(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches Task 14 surfaces exposing fixed English errors after the Chinese rebuild."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = _app()
    root = _project_root(tmp_path)
    invalid_setup = asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root, api_key="")))
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303
    invalid_task = asyncio.run(_request(app, "POST", "/tasks", data={"mode": "feature", "description": ""}))
    created = asyncio.run(
        _request(app, "POST", "/tasks", data={"mode": "feature", "description": "保持活跃"})
    )
    active_setup = asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root)))
    missing_task = asyncio.run(_request(app, "POST", f"/tasks/{uuid4()}/cancel"))
    invalid_credential = asyncio.run(
        _request(app, "POST", "/settings/credentials", data={"api_key": ""})
    )
    unavailable = asyncio.run(_request(_app(UnavailableCredentials()), "GET", "/settings/credentials"))

    assert created.status_code == 303
    for response, status_code, chinese, english in [
        (invalid_setup, 422, "无法保存设置。", "Setup could not be saved."),
        (invalid_task, 422, "无法启动任务。", "Task could not be started."),
        (active_setup, 409, "已有任务正在运行。", "Another task is active."),
        (missing_task, 404, "未找到任务。", "Task was not found."),
        (invalid_credential, 422, "无法更新凭据。", "Credential could not be updated."),
        (unavailable, 503, "凭据存储不可用。", "Credential store is unavailable."),
    ]:
        assert response.status_code == status_code
        assert chinese in response.text
        assert english not in response.text


def test_memory_review_uses_chinese_pending_and_approved_regions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches consent-controlled memory lists losing their review-stage meaning."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = _app()
    root = _project_root(tmp_path)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303
    memory_store = app.state.local.memory_store
    assert memory_store is not None
    proposal = memory_store.propose(uuid4(), "记住解析器命名约定")

    document = _document(asyncio.run(_request(app, "GET", "/memories")))

    assert {"待审记忆", "已批准记忆"} <= document.section_labels
    assert document.forms == [
        (f"/memories/{proposal.id}/approve", set()),
        (f"/memories/{proposal.id}/delete", set()),
    ]


def test_security_settings_explain_current_rules_without_changing_their_controls(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches command-rule settings becoming an unexplained or non-semantic control surface."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = _app()
    root = _project_root(tmp_path)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303

    command_rules = _document(asyncio.run(_request(app, "GET", "/settings/command-rules")))
    credentials = _document(asyncio.run(_request(app, "GET", "/settings/credentials")))

    assert {"命令规则状态", "已保存的命令规则"} <= command_rules.section_labels
    assert {"凭据状态", "更新凭据", "清除凭据"} <= credentials.section_labels
    assert credentials.forms == [
        ("/settings/credentials", {"api_key"}),
        ("/settings/credentials/clear", set()),
    ]


def test_command_rule_projection_keeps_raw_command_in_monospace(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Catches a persisted command projection rendering as ordinary prose."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = _app()
    root = _project_root(tmp_path)
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303
    rule = CommandRuleStore(root).add_from(
        RunCommandAction(
            kind="run_command",
            summary="not rendered",
            args=("python", "-m", "pip", "install", "example-package==1.2.3"),
        ),
        None,
    )

    document = _document(asyncio.run(_request(app, "GET", "/settings/command-rules")))

    assert document.forms == [(f"/settings/command-rules/{rule.id}/delete", set())]
    assert "Python package install: example-package==1.2.3" in document.mono_texts


def test_shared_stylesheet_provides_neutral_modern_accessible_shell_primitives() -> None:
    """Catches a shared stylesheet that removes the responsive console guarantees."""
    stylesheet = (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.css").read_text()

    rules = _css_rules(stylesheet)
    root = rules[":root"]
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
        assert root[token] == value
    assert rules[".rail"]["position"] == "fixed"
    assert rules[".rail"]["width"] == "var(--rail-width)"
    assert rules[".task-link"]["min-height"] == "44px"
    assert rules["a:focus-visible"]["box-shadow"] == "var(--focus-ring)"
    assert rules["input:focus-visible"]["border-color"] == "var(--accent)"
    assert rules[".timeline"]["border-left"] == "2px solid var(--accent)"
    assert rules[".approval-card"]["border-top"] == "4px solid var(--warn)"
    assert rules[".empty-state"]["text-align"] == "center"
    phone_rules = _css_rules(stylesheet, "max-width: 639px")
    assert phone_rules[".rail"]["position"] == "static"
    assert phone_rules[".rail-nav"]["flex-wrap"] == "wrap"
    assert "prefers-reduced-motion: reduce" not in stylesheet


def test_public_demo_is_a_chinese_read_only_surface_with_only_fixed_scenario_links() -> None:
    """Catches the public demo regressing into local-console navigation or controls."""
    from guardedpy.demo import create_demo_app

    document = _document(asyncio.run(_request(create_demo_app(), "GET", "/")))
    stylesheet = (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.css").read_text()

    assert document.lang == "zh-CN"
    assert any("只读演示" in text for text in document.header_texts)
    assert document.landmarks == {"main"}
    assert document.forms == []
    assert document.anchors == [
        ("/demo/scenarios/dangerous_action_denied", "危险动作被拒绝"),
        ("/demo/scenarios/failure_feedback_corrects", "失败反馈促成修正"),
        ("/demo/scenarios/tdd_source_patch_denied", "未观察失败前拒绝源码修改"),
    ]
    assert _css_rules(stylesheet)[".demo-scenario-link"]["min-height"] == "44px"
    phone_rules = _css_rules(stylesheet, "max-width: 639px")
    assert phone_rules[".demo-header"]["align-items"] == "flex-start"
