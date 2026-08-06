"""Deterministic, fail-closed discovery of local Python pytest projects."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import sys
import tomllib
from typing import Literal


DiscoverySource = Literal["pytest_config", "tests_dir", "root_tests"]
DiscoveryErrorCode = Literal[
    "invalid_pytest_config", "configured_testpath_missing", "unsupported_project"
]


class ProjectDiscoveryError(RuntimeError):
    """A bounded project-discovery failure safe for adapter-level diagnostics."""

    def __init__(self, code: DiscoveryErrorCode) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ProjectProfile:
    """Immutable root-relative facts discovered without consulting an LLM."""

    root: Path
    discovery_source: DiscoverySource
    source_dirs: tuple[PurePosixPath, ...]
    test_dirs: tuple[PurePosixPath, ...]
    pytest_command: tuple[str, str, str]

    def __post_init__(self) -> None:
        if self.discovery_source not in ("pytest_config", "tests_dir", "root_tests"):
            raise ValueError("project profile has an invalid discovery source")
        if self.root != self.root.resolve() or not self.root.is_dir():
            raise ValueError("project root must be an existing resolved directory")
        if self.pytest_command != (sys.executable, "-m", "pytest"):
            raise ValueError("pytest command must use the current interpreter")
        if not self.source_dirs or not self.test_dirs:
            raise ValueError("project directories must be non-empty")
        for relative in (*self.source_dirs, *self.test_dirs):
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("project directories must stay inside the root")
            candidate = (self.root / relative).resolve()
            if not candidate.is_dir() or not candidate.is_relative_to(self.root):
                raise ValueError("project directories must be existing root-relative directories")


def discover_project(root: Path) -> ProjectProfile:
    """Discover a supported pytest layout from one resolved project root."""
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ProjectDiscoveryError("unsupported_project")

    configured_testpaths = _configured_testpaths(resolved_root)
    if configured_testpaths is not None:
        test_dirs = _validated_testpaths(resolved_root, configured_testpaths)
        discovery_source: DiscoverySource = "pytest_config"
    else:
        tests_dir = resolved_root / "tests"
        if _contained_directory(resolved_root, tests_dir):
            test_dirs = (PurePosixPath("tests"),)
            discovery_source = "tests_dir"
        elif _has_root_test_evidence(resolved_root):
            test_dirs = (PurePosixPath("."),)
            discovery_source = "root_tests"
        else:
            raise ProjectDiscoveryError("unsupported_project")

    source_dirs = _source_dirs(resolved_root)
    if source_dirs is None:
        raise ProjectDiscoveryError("unsupported_project")
    return ProjectProfile(
        root=resolved_root,
        discovery_source=discovery_source,
        source_dirs=source_dirs,
        test_dirs=test_dirs,
        pytest_command=(sys.executable, "-m", "pytest"),
    )


def _configured_testpaths(root: Path) -> tuple[str, ...] | None:
    for filename in ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg"):
        path = root / filename
        if path.is_symlink() and not path.exists():
            raise ProjectDiscoveryError("invalid_pytest_config")
        if not path.exists():
            continue
        try:
            valid_file = path.is_file() and _contained_path(root, path)
        except (OSError, RuntimeError):
            raise ProjectDiscoveryError("invalid_pytest_config") from None
        if not valid_file:
            raise ProjectDiscoveryError("invalid_pytest_config")
        if filename == "pyproject.toml":
            return _toml_testpaths(path)
        return _ini_testpaths(path)
    return None


def _toml_testpaths(path: Path) -> tuple[str, ...] | None:
    try:
        payload = tomllib.loads(path.read_text())
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ProjectDiscoveryError("invalid_pytest_config") from None
    tool = payload.get("tool")
    if tool is None:
        return None
    if not isinstance(tool, dict):
        raise ProjectDiscoveryError("invalid_pytest_config")
    pytest_options = tool.get("pytest")
    if pytest_options is None:
        return None
    if not isinstance(pytest_options, dict):
        raise ProjectDiscoveryError("invalid_pytest_config")
    options = pytest_options.get("ini_options")
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ProjectDiscoveryError("invalid_pytest_config")
    if "testpaths" not in options:
        return None
    value = options["testpaths"]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ProjectDiscoveryError("invalid_pytest_config")
    return tuple(value)


def _ini_testpaths(path: Path) -> tuple[str, ...] | None:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(path.read_text())
    except (OSError, UnicodeError, configparser.Error):
        raise ProjectDiscoveryError("invalid_pytest_config") from None
    if not parser.has_option("pytest", "testpaths"):
        return None
    paths = tuple(parser.get("pytest", "testpaths").split())
    if not paths:
        raise ProjectDiscoveryError("invalid_pytest_config")
    return paths


def _validated_testpaths(root: Path, values: tuple[str, ...]) -> tuple[PurePosixPath, ...]:
    validated: list[PurePosixPath] = []
    for value in values:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ProjectDiscoveryError("invalid_pytest_config")
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            raise ProjectDiscoveryError("invalid_pytest_config") from None
        if not resolved.is_relative_to(root):
            raise ProjectDiscoveryError("invalid_pytest_config")
        if candidate.is_symlink() and not candidate.exists():
            raise ProjectDiscoveryError("invalid_pytest_config")
        if not candidate.exists():
            raise ProjectDiscoveryError("configured_testpath_missing")
        if not candidate.is_dir():
            raise ProjectDiscoveryError("invalid_pytest_config")
        validated.append(relative)
    return tuple(validated)


def _source_dirs(root: Path) -> tuple[PurePosixPath, ...] | None:
    src = root / "src"
    if src.exists():
        if _contained_directory(root, src):
            return (PurePosixPath("src"),)
        return None
    if _has_root_source(root):
        return (PurePosixPath("."),)
    return None


def _has_root_source(root: Path) -> bool:
    for candidate in root.iterdir():
        if candidate.is_file() and candidate.suffix == ".py":
            if _is_test_module(candidate.name) or not _contained_path(root, candidate):
                continue
            return True
        if not candidate.is_dir() or _is_test_package(candidate.name):
            continue
        initializer = candidate / "__init__.py"
        if initializer.is_file() and _contained_path(root, candidate) and _contained_path(
            root, initializer
        ):
            return True
    return False


def _has_root_test_evidence(root: Path) -> bool:
    for pattern in ("test_*.py", "*_test.py"):
        for candidate in root.rglob(pattern):
            if candidate.is_file() and _contained_path(root, candidate):
                return True
    return False


def _is_test_module(name: str) -> bool:
    return name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py")


def _is_test_package(name: str) -> bool:
    return name in {"test", "tests"} or name.startswith("test_") or name.endswith("_test")


def _contained_directory(root: Path, candidate: Path) -> bool:
    return candidate.is_dir() and _contained_path(root, candidate)


def _contained_path(root: Path, candidate: Path) -> bool:
    return candidate.resolve().is_relative_to(root)
