"""Packaging and delivery contracts checked without invoking a shell."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import venv
import zipfile

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_package_metadata_declares_cli_and_loopback_server_entrypoints() -> None:
    """Catches a wheel that routes either public command away from its local adapter."""
    project = tomllib.loads(_text("pyproject.toml"))

    assert project["project"]["scripts"] == {
        "guardedpy": "guardedpy.cli:main",
        "guardedpy-cli": "guardedpy.cli:main",
        "guardedpy-server": "guardedpy.cli:server_main",
    }
    assert project["tool"]["setuptools"]["package-data"]["guardedpy"] == [
        "templates/*.html",
        "static/*.css",
        "static/*.js",
    ]


def test_distribution_wheel_includes_pytest_required_by_the_public_demo(tmp_path: Path) -> None:
    """Catches a standard wheel install whose fixed demo cannot run its pytest fixture."""
    project_copy = tmp_path / "project"
    shutil.copytree(
        ROOT,
        project_copy,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            ".superpowers",
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    wheelhouse = tmp_path / "wheelhouse"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheelhouse),
        ],
        cwd=project_copy,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    wheel = next(wheelhouse.glob("guardedpy-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()

    assert "Requires-Dist: pytest\n" in metadata


def test_cli_check_runs_each_installed_console_entrypoint_without_composition(tmp_path: Path) -> None:
    """Catches cli-check bypassing the three installed local console scripts."""
    project_copy = tmp_path / "project"
    shutil.copytree(
        ROOT,
        project_copy,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            ".superpowers",
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    wheelhouse = tmp_path / "wheelhouse"
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheelhouse),
        ],
        cwd=project_copy,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(next(wheelhouse.glob("*.whl")))],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert install.returncode == 0, install.stderr

    sitecustomize = tmp_path / "instrumentation" / "sitecustomize.py"
    sitecustomize.parent.mkdir()
    sitecustomize.write_text(
        "from guardedpy import web\n"
        "def fail(*args, **kwargs):\n"
        "    raise AssertionError('help must not compose local services')\n"
        "web.local_services = fail\n"
        "web.uvicorn.run = fail\n",
        encoding="utf-8",
    )
    entrypoint_bin = tmp_path / "entrypoints"
    entrypoint_bin.mkdir()
    entrypoint_log = tmp_path / "entrypoints.log"
    entrypoint_log.touch()
    for name in ("guardedpy", "guardedpy-cli", "guardedpy-server"):
        wrapper = entrypoint_bin / name
        wrapper.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' {name} >> \"$GUARDEDPY_ENTRYPOINT_LOG\"\n"
            f"exec \"$GUARDEDPY_ENTRYPOINT_BIN/{name}\" \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)

    dependency_paths = [path for path in sys.path if "site-packages" in path]
    environment_variables = {
        **os.environ,
        "GUARDEDPY_ENTRYPOINT_BIN": str(environment / "bin"),
        "GUARDEDPY_ENTRYPOINT_LOG": str(entrypoint_log),
        "PATH": f"{entrypoint_bin}{os.pathsep}{os.environ['PATH']}",
        "PYTHON": str(python),
        "PYTHONPATH": os.pathsep.join([str(sitecustomize.parent), *dependency_paths]),
    }
    result = subprocess.run(
        ["make", "cli-check"],
        cwd=project_copy,
        capture_output=True,
        text=True,
        env=environment_variables,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert entrypoint_log.read_text(encoding="utf-8").splitlines() == [
        "guardedpy",
        "guardedpy-cli",
        "guardedpy-server",
    ]


def test_delivery_automation_runs_the_same_offline_test_demo_and_build_contract() -> None:
    """Catches drift between local Make targets and the two CI definitions."""
    makefile = _text("Makefile")
    assert "test:" in makefile
    assert "demo:" in makefile
    assert "build:" in makefile
    assert "cli-check:" in makefile
    assert "\tguardedpy --help" in makefile
    assert "\tguardedpy-cli --help" in makefile
    assert "\tguardedpy-server --help" in makefile
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
    assert "https://github.com/WalnuTpz/guardedpy/pull/13" in readme
    assert "present-time reconstruction records" in readme
    assert "src/guardedpy/" in readme
    assert "tests/" in readme


def test_readme_documents_local_cli_server_and_pending_release_assets() -> None:
    """Catches an operator guide that promotes a public UI or invents a release asset."""
    readme = _text("README.md")

    assert "guardedpy-cli" in readme
    assert "guardedpy-server" in readme
    assert "公网 WebUI" in readme
    assert "Release asset URL is pending" in readme
    assert "releases/download" not in readme


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
