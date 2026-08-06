"""End-to-end local release-artifact checks for the sole GuardedPy CLI."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tarfile
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_SITE = next(
    Path(path) for path in site.getsitepackages() if Path(path).is_dir()
)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, env=env)


def _copy_and_build(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_copy = tmp_path / "source-copy"
    shutil.copytree(
        ROOT,
        project_copy,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", ".superpowers", "__pycache__", "*.egg-info", "build", "dist"
        ),
    )
    artifact_dir = tmp_path / "artifacts"
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    built = _run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(artifact_dir)],
        cwd=project_copy,
        env=environment,
    )
    assert built.returncode == 0, built.stderr
    wheel = next(artifact_dir.glob("guardedpy-*.whl"))
    sdist = next(artifact_dir.glob("guardedpy-*.tar.gz"))
    return project_copy, wheel, sdist


def _assert_no_retired_package_surface(wheel: Path, sdist: Path) -> None:
    forbidden_suffixes = (
        "/src/guardedpy/api.py",
        "/src/guardedpy/web.py",
        "/src/guardedpy/demo.py",
        "/guardedpy/api.py",
        "/guardedpy/web.py",
        "/guardedpy/demo.py",
    )
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    with tarfile.open(sdist) as archive:
        sdist_names = archive.getnames()
    names = [*wheel_names, *sdist_names]
    assert not [
        name
        for name in names
        if name.endswith(forbidden_suffixes)
        or "/src/guardedpy/templates/" in name
        or "/src/guardedpy/static/" in name
        or "/guardedpy/templates/" in name
        or "/guardedpy/static/" in name
    ]


def _installed_environment(tmp_path: Path, wheel: Path) -> tuple[Path, Path, dict[str, str]]:
    environment = tmp_path / "isolated-environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    installed = _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        cwd=tmp_path,
        env={name: value for name, value in os.environ.items() if name != "PYTHONPATH"},
    )
    assert installed.returncode == 0, installed.stderr
    isolated_site = Path(
        subprocess.run(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    # The test runs offline.  The wheel itself is installed in the fresh venv;
    # this dependency-only path never exposes the source checkout to its importer.
    (isolated_site / "guardedpy_test_dependencies.pth").write_text(
        f"{DEPENDENCY_SITE}\n", encoding="utf-8"
    )
    command_env = {
        **{name: value for name, value in os.environ.items() if name != "PYTHONPATH"},
        "PATH": f"{environment / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "XDG_STATE_HOME": str(tmp_path / "runtime-state"),
    }
    origin = _run(
        [str(python), "-c", "import guardedpy; print(guardedpy.__file__)"], cwd=tmp_path, env=command_env
    )
    assert origin.returncode == 0, origin.stderr
    assert str(isolated_site) in origin.stdout
    return python, environment / "bin" / "guardedpy", command_env


def _temporary_pytest_project(tmp_path: Path) -> Path:
    project = tmp_path / "target-project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "tests").mkdir()
    (project / "tests" / "test_sample.py").write_text(
        "def test_sample_value():\n    assert 1 == 1\n", encoding="utf-8"
    )
    return project


def test_local_wheel_and_sdist_install_run_the_cli_without_the_source_checkout(tmp_path: Path) -> None:
    """Catches release artifacts that include retired UI files or cannot run the installed CLI."""
    project_copy, wheel, sdist = _copy_and_build(tmp_path)
    assert project_copy != ROOT
    _assert_no_retired_package_surface(wheel, sdist)
    _python, guardedpy, command_env = _installed_environment(tmp_path, wheel)
    target = _temporary_pytest_project(tmp_path)

    help_result = _run([str(guardedpy), "--help"], cwd=target, env=command_env)
    assert help_result.returncode == 0, help_result.stderr
    assert "usage: guardedpy" in help_result.stdout.lower()
    assert "server" not in help_result.stdout.lower()

    task_result = _run([str(guardedpy), "inspect this project"], cwd=target, env=command_env)
    assert task_result.returncode == 0, task_result.stderr
    assert "blocked" in task_result.stdout.lower()

    demo_result = _run([str(guardedpy), "demo"], cwd=target, env=command_env)
    assert demo_result.returncode == 0, demo_result.stderr
    assert demo_result.stdout.splitlines() == [
        "dangerous_action_denied status=blocked",
        "failure_feedback_corrects status=completed",
        "tdd_source_patch_denied status=blocked",
    ]
