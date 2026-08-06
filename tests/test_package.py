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

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_package_metadata_declares_exact_cli_only_boundary() -> None:
    """Catches a wheel exposing a retired surface or depending on a Web stack."""
    project = tomllib.loads(_text("pyproject.toml"))

    assert project["project"]["scripts"] == {"guardedpy": "guardedpy.cli:main"}
    assert project["project"]["dependencies"] == [
        "pydantic>=2,<3",
        "keyring",
        "openai",
        "pytest",
        "pyyaml",
        "textual",
    ]
    assert "package-data" not in project["tool"]["setuptools"]


def test_distribution_wheel_includes_pytest_required_by_the_headless_demo(tmp_path: Path) -> None:
    """Catches a standard wheel install missing the demo fixture's test runner."""
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


def test_cli_check_runs_the_installed_guardedpy_entrypoint(tmp_path: Path) -> None:
    """Catches cli-check bypassing the sole installed console script."""
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

    entrypoint_bin = tmp_path / "entrypoints"
    entrypoint_bin.mkdir()
    entrypoint_log = tmp_path / "entrypoints.log"
    entrypoint_log.touch()
    wrapper = entrypoint_bin / "guardedpy"
    wrapper.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' guardedpy >> \"$GUARDEDPY_ENTRYPOINT_LOG\"\n"
        "exec \"$GUARDEDPY_ENTRYPOINT_BIN/guardedpy\" \"$@\"\n",
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
        "PYTHONPATH": os.pathsep.join(dependency_paths),
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
    assert entrypoint_log.read_text(encoding="utf-8").splitlines() == ["guardedpy"]


def test_delivery_automation_runs_the_same_offline_test_demo_and_build_contract() -> None:
    """Catches drift between local Make targets and the two CI definitions."""
    makefile = _text("Makefile")
    assert "test:" in makefile
    assert "demo:" in makefile
    assert "build:" in makefile
    assert "cli-check:" in makefile
    assert "\tguardedpy --help" in makefile
    assert "guardedpy-cli" not in makefile
    assert "guardedpy-server" not in makefile
    assert "demo-assets" not in makefile
    assert "pytest tests -q" in makefile
    assert "PYTHONPATH=src $(PYTHON) scripts/run_mechanism_demo.py" in makefile
    assert "$(PYTHON) -m build" in makefile

    github = yaml.safe_load(_text(".github/workflows/ci.yml"))
    github_runs = [step.get("run") for step in github["jobs"]["unit-test"]["steps"] if "run" in step]
    assert "pip install -e \".[dev]\"" in github_runs
    assert github_runs[-3:] == ["make test", "make demo", "make build"]

    gitlab = yaml.safe_load(_text(".gitlab-ci.yml"))
    assert gitlab["unit-test"]["script"][0] == 'pip install -e ".[dev]"'
    assert gitlab["unit-test"]["script"][-3:] == ["make test", "make demo", "make build"]


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
