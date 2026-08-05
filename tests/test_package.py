"""Packaging and delivery contracts checked without invoking a shell."""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_package_metadata_declares_web_console_entrypoint_and_ui_assets() -> None:
    """Catches a wheel that omits the installed WebUI entrypoint or templates."""
    project = tomllib.loads(_text("pyproject.toml"))

    assert project["project"]["scripts"] == {"guardedpy": "guardedpy.web:serve"}
    assert project["tool"]["setuptools"]["package-data"]["guardedpy"] == [
        "templates/*.html",
        "static/*.css",
        "static/*.js",
    ]


def test_delivery_automation_runs_the_same_offline_test_demo_and_build_contract() -> None:
    """Catches drift between local Make targets and the two CI definitions."""
    makefile = _text("Makefile")
    assert "test:" in makefile
    assert "demo:" in makefile
    assert "build:" in makefile
    assert "pytest tests -q" in makefile
    for scenario in (
        "dangerous_action_denied",
        "failure_feedback_corrects",
        "tdd_source_patch_denied",
    ):
        assert scenario in makefile
    assert "assert actual == expected" in makefile
    assert "$(PYTHON) -m build" in makefile

    github = yaml.safe_load(_text(".github/workflows/ci.yml"))
    github_runs = [step.get("run") for step in github["jobs"]["unit-test"]["steps"] if "run" in step]
    assert "pip install -e \".[dev]\"" in github_runs
    assert github_runs[-3:] == ["make test", "make demo", "make build"]

    gitlab = yaml.safe_load(_text(".gitlab-ci.yml"))
    assert gitlab["unit-test"]["script"][0] == 'pip install -e ".[dev]"'
    assert gitlab["unit-test"]["script"][-3:] == ["make test", "make demo", "make build"]


def test_render_blueprint_starts_the_isolated_demo_not_local_controls() -> None:
    """Catches a public deployment missing demo runtime or composing local controls."""
    render = yaml.safe_load(_text("render.yaml"))
    service = render["services"][0]

    assert service["name"] == "guardedpy-demo"
    assert service["buildCommand"] == 'pip install ".[dev]"'
    assert "guardedpy.demo:create_demo_app" in service["startCommand"]
    assert "guardedpy.web:serve" not in service["startCommand"]
    assert "guardedpy serve" not in service["startCommand"]


def test_readme_documents_real_local_demo_security_delivery_and_course_context() -> None:
    """Catches missing operating guidance or fabricated deployment/CI evidence."""
    readme = _text("README.md")
    for heading in (
        "## Installation",
        "## Local operation",
        "## Demo operation",
        "## Keyring lifecycle",
        "## Safety boundaries",
        "## Tests and build",
        "## Directory structure",
        "## Repository and PR evidence",
        "## Render demo wake-up",
        "## CI evidence",
        "## Open Design attribution",
        "## Known limitations",
    ):
        assert heading in readme
    assert ".env" in readme
    assert "127.0.0.1" in readme
    assert "not deployed" in readme
    assert "GitHub Actions has recorded successful validation runs" in readme
    assert "GitLab workflow has not yet been executed" in readme
    assert "Open Design" in readme
    assert "Agentic" in readme
    assert "git@github.com:WalnuTpz/guardedpy.git" in readme
    assert "https://github.com/WalnuTpz/guardedpy/pull/1" in readme
    assert "https://github.com/WalnuTpz/guardedpy/pull/12" in readme
    assert "present-time reconstruction records" in readme
    assert "src/guardedpy/" in readme
    assert "tests/" in readme


@pytest.mark.parametrize("mode", ["serve", "demo"])
def test_console_help_exits_before_composing_or_starting_a_server(
    mode: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a console help request accidentally starting local or demo Uvicorn."""
    from guardedpy import web

    monkeypatch.setattr(web, "local_services", lambda: pytest.fail("help must not compose local services"))
    monkeypatch.setattr(web, "create_demo_app", lambda: pytest.fail("help must not compose demo services"))
    monkeypatch.setattr(web.uvicorn, "run", lambda *args, **kwargs: pytest.fail("help must not start Uvicorn"))
    monkeypatch.setattr(sys, "argv", ["guardedpy", mode, "--help"])

    with pytest.raises(SystemExit) as exit_code:
        web.serve()

    assert exit_code.value.code == 0
    assert "{serve,demo}" in capsys.readouterr().out
