"""End-to-end local release-artifact checks for the sole GuardedPy CLI."""

from __future__ import annotations

import base64
import hashlib
from importlib import metadata
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tarfile
import venv
import zipfile
from email.parser import BytesParser

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


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


def _wheel_requirement_names(wheel: Path) -> set[str]:
    return {Requirement(value).name.lower() for value in _wheel_requirements(wheel)}


def _build_local_wheelhouse(wheel: Path, wheelhouse: Path) -> None:
    """Repackage the installed dependency closure so pip can resolve it offline."""
    available = {
        canonicalize_name(distribution.metadata["Name"]): distribution
        for distribution in metadata.distributions(path=site.getsitepackages())
        if distribution.metadata.get("Name")
    }
    pending = [Requirement(requirement) for requirement in _wheel_requirements(wheel)]
    selected: dict[str, metadata.Distribution] = {}
    environment = {**default_environment(), "extra": ""}
    selected_extras: dict[str, set[str]] = {}
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        active_extras = set(requirement.extras) or {""}
        if active_extras <= selected_extras.get(name, set()):
            continue
        distribution = available.get(name)
        assert distribution is not None, f"missing installed dependency for {requirement.name}"
        selected[name] = distribution
        selected_extras.setdefault(name, set()).update(active_extras)
        for value in distribution.metadata.get_all("Requires-Dist", []):
            dependency = Requirement(value)
            if _matches_marker(dependency, environment, active_extras):
                pending.append(dependency)
    for distribution in selected.values():
        _repackage_distribution(distribution, wheelhouse)


def _wheel_requirements(wheel: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata_message = BytesParser().parsebytes(archive.read(metadata_name))
    environment = {**default_environment(), "extra": ""}
    return tuple(
        value
        for value in metadata_message.get_all("Requires-Dist", [])
        if _matches_marker(Requirement(value), environment, {""})
    )


def _matches_marker(requirement: Requirement, environment: dict[str, str], extras: set[str]) -> bool:
    return requirement.marker is None or any(
        requirement.marker.evaluate({**environment, "extra": extra}) for extra in extras
    )


def _repackage_distribution(distribution: metadata.Distribution, wheelhouse: Path) -> None:
    """Create a local wheel from one already-installed third-party distribution."""
    wheel_metadata = distribution.read_text("WHEEL")
    assert wheel_metadata is not None
    tags = [line.removeprefix("Tag: ") for line in wheel_metadata.splitlines() if line.startswith("Tag: ")]
    assert tags
    python_tag, abi_tag, platform_tag = tags[0].split("-", maxsplit=2)
    filename = (
        f"{distribution.metadata['Name'].replace('-', '_')}-{distribution.version}-"
        f"{python_tag}-{abi_tag}-{platform_tag}.whl"
    )
    files = distribution.files
    assert files is not None
    record_path = next(
        path for path in files if path.as_posix().endswith(".dist-info/RECORD")
    )
    records: list[tuple[str, str, int]] = []
    with zipfile.ZipFile(wheelhouse / filename, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path == record_path or ".." in path.parts:
                continue
            source = Path(distribution.locate_file(path))
            if not source.is_file():
                continue
            payload = source.read_bytes()
            archive_name = path.as_posix()
            archive.writestr(archive_name, payload)
            digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            records.append((archive_name, digest, len(payload)))
        content = "".join(
            f"{name},sha256={digest},{size}\n" for name, digest, size in records
        ) + f"{record_path.as_posix()},,\n"
        archive.writestr(record_path.as_posix(), content)


def _installed_environment(
    tmp_path: Path, wheel: Path, wheelhouse: Path
) -> tuple[Path, Path, dict[str, str]]:
    environment = tmp_path / "isolated-environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    installed = _run(
        [
            str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), str(wheel),
        ],
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


def test_wheel_metadata_declares_every_runtime_dependency(tmp_path: Path) -> None:
    """Catches a built wheel silently dropping a module required by the installed CLI."""
    _project_copy, wheel, _sdist = _copy_and_build(tmp_path)

    assert _wheel_requirement_names(wheel) == {
        "keyring",
        "openai",
        "pydantic",
        "pytest",
        "pyyaml",
        "textual",
    }


def test_wheel_install_resolves_declared_dependencies_without_host_path_injection(tmp_path: Path) -> None:
    """Catches an artifact check that runs only because controller site-packages leak into its venv."""
    _project_copy, wheel, _sdist = _copy_and_build(tmp_path)
    environment = tmp_path / "dependency-check-environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    wheelhouse = tmp_path / "offline-wheelhouse"
    wheelhouse.mkdir()
    _build_local_wheelhouse(wheel, wheelhouse)

    installed = _run(
        [
            str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheelhouse), str(wheel),
        ],
        cwd=tmp_path,
        env={name: value for name, value in os.environ.items() if name != "PYTHONPATH"},
    )

    assert installed.returncode == 0, installed.stderr


def test_local_wheel_runs_the_cli_and_sdist_is_scanned_without_the_source_checkout(tmp_path: Path) -> None:
    """Catches release artifacts that include retired UI files or cannot run the installed CLI."""
    project_copy, wheel, sdist = _copy_and_build(tmp_path)
    assert project_copy != ROOT
    _assert_no_retired_package_surface(wheel, sdist)
    wheelhouse = tmp_path / "offline-wheelhouse"
    wheelhouse.mkdir()
    _build_local_wheelhouse(wheel, wheelhouse)
    _python, guardedpy, command_env = _installed_environment(tmp_path, wheel, wheelhouse)
    target = _temporary_pytest_project(tmp_path)

    help_result = _run([str(guardedpy), "--help"], cwd=target, env=command_env)
    assert help_result.returncode == 0, help_result.stderr
    assert "usage: guardedpy" in help_result.stdout.lower()
    assert "server" not in help_result.stdout.lower()

    task_result = _run([str(guardedpy), "inspect this project"], cwd=target, env=command_env)
    assert task_result.returncode == 1, task_result.stderr
    assert task_result.stdout == "需要先在交互终端配置凭据。\n"

    demo_result = _run([str(guardedpy), "demo"], cwd=target, env=command_env)
    assert demo_result.returncode == 0, demo_result.stderr
    assert demo_result.stdout.splitlines() == [
        "dangerous_action_denied status=blocked",
        "failure_feedback_corrects status=completed",
        "tdd_source_patch_denied status=blocked",
    ]
