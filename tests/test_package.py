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


def test_documentation_covers_cli_installation_and_operation() -> None:
    """Keep the project guides reachable from the package README."""
    readme = _text("README.md").lower()
    tutorial = _text("docs/getting-started.md").lower()
    usage = _text("docs/usage.md").lower()
    design = _text("docs/design.md").lower()

    for path in ("docs/getting-started.md", "docs/usage.md", "docs/design.md"):
        assert f"]({path.lower()})" in readme
        assert (ROOT / path).is_file()
    for command in ("git clone", "python3 -m venv", "python -m pip install", "guardedpy", "/credentials"):
        assert command in readme
        assert command in tutorial
    for command in ("/model", "/effort", "/conversations", "/tests", "/diff"):
        assert command in usage
    documentation = "\n".join((readme, tutorial, usage, design))
    for retired in ("guardedpy-server", "guardedpy serve", "render.yaml", "fastapi", "guardedpy-cli"):
        assert retired not in documentation
