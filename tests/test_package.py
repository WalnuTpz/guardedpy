"""CLI-only delivery contracts that do not require a published release."""

from __future__ import annotations

from pathlib import Path
import tomllib

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_package_metadata_exposes_only_the_guardedpy_cli() -> None:
    """Catches a distribution exposing a retired server or web entry point."""
    project = tomllib.loads(_text("pyproject.toml"))

    assert project["project"]["scripts"] == {"guardedpy": "guardedpy.cli:main"}
    assert "fastapi" not in {dependency.lower() for dependency in project["project"]["dependencies"]}
    assert "uvicorn" not in {dependency.lower() for dependency in project["project"]["dependencies"]}


def test_make_and_ci_run_the_three_cli_delivery_commands() -> None:
    """Catches CI drifting from the local test, demo, and distribution build contract."""
    makefile = _text("Makefile")
    assert ".PHONY: test demo build" in makefile
    assert "cli-check" not in makefile
    assert "demo-assets" not in makefile
    assert "pytest tests -q" in makefile
    assert "scripts/run_mechanism_demo.py" in makefile
    assert "-m build --no-isolation" in makefile

    github = yaml.safe_load(_text(".github/workflows/ci.yml"))
    github_runs = [step["run"] for step in github["jobs"]["unit-test"]["steps"] if "run" in step]
    assert github_runs[-3:] == ["make test", "make demo", "make build"]

    gitlab = yaml.safe_load(_text(".gitlab-ci.yml"))
    assert list(gitlab) == ["image", "unit-test"]
    assert gitlab["unit-test"]["script"][-3:] == ["make test", "make demo", "make build"]


def test_readme_is_a_cli_only_installation_and_operation_guide() -> None:
    """Catches user guidance retaining obsolete setup, server, deployment, or web claims."""
    readme = _text("README.md").lower()

    for heading in (
        "## installation",
        "## run in a project",
        "## credentials, model, and effort",
        "## mechanism demo",
        "## test, build, and local release artifact",
        "## safety boundaries and limitations",
    ):
        assert heading in readme
    for command in ("guardedpy", "/credentials", "/model", "/effort", "guardedpy demo", "make test", "make demo", "make build"):
        assert command in readme
    for retired in ("guardedpy-server", "guardedpy serve", "webui", "render.yaml", "fastapi", "/init", "guardedpy-cli"):
        assert retired not in readme
    assert "teacher clarification" in readme
    assert "not uploaded" in readme
