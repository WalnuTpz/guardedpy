"""Immutable task configuration derived from deterministic project discovery."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml

from guardedpy.discovery import ProjectProfile


ModelName = Literal["deepseek-v4-flash", "deepseek-v4-pro"]
ReasoningEffort = Literal["high", "max"]


class HarnessConfig(BaseModel):
    """Frozen non-secret facts captured independently for each task."""

    model_config = ConfigDict(frozen=True)

    profile: ProjectProfile
    model: ModelName = "deepseek-v4-flash"
    thinking_enabled: Literal[True] = True
    reasoning_effort: ReasoningEffort = "high"
    timeout_seconds: int = Field(default=120, ge=5, le=120)

    @property
    def source_dirs(self) -> tuple[PurePosixPath, ...]:
        return self.profile.source_dirs

    @property
    def test_dirs(self) -> tuple[PurePosixPath, ...]:
        return self.profile.test_dirs

    @property
    def pytest_command(self) -> tuple[str, str, str]:
        return self.profile.pytest_command


def load_or_create_discovered_config(
    profile: ProjectProfile, state_dir: Path
) -> HarnessConfig:
    """Combine fresh discovery with persisted non-secret future-task selections."""
    path = _snapshot_path(profile.root, state_dir)
    selections: dict[str, object] = {}
    if path.exists():
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict) or set(payload) != {
            "profile",
            "model",
            "thinking_enabled",
            "reasoning_effort",
            "timeout_seconds",
        }:
            raise ValueError("configuration snapshot has invalid fields")
        selections = {
            "model": payload["model"],
            "thinking_enabled": payload["thinking_enabled"],
            "reasoning_effort": payload["reasoning_effort"],
            "timeout_seconds": payload["timeout_seconds"],
        }
    config = HarnessConfig(profile=profile, **selections)
    save_discovered_config(config, state_dir)
    return config


def update_future_defaults(
    config: HarnessConfig,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> HarnessConfig:
    """Return validated selections for later tasks without mutating the input snapshot."""
    return HarnessConfig(
        profile=config.profile,
        model=config.model if model is None else model,
        thinking_enabled=config.thinking_enabled,
        reasoning_effort=(
            config.reasoning_effort if reasoning_effort is None else reasoning_effort
        ),
        timeout_seconds=config.timeout_seconds,
    )


def save_discovered_config(config: HarnessConfig, state_dir: Path) -> None:
    """Atomically persist the sole non-secret discovered configuration schema."""
    _write_config(_snapshot_path(config.profile.root, state_dir), config)


def load_config(path: Path, project_root: Path) -> HarnessConfig:
    """Load a persisted non-secret discovered configuration snapshot."""
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {
        "profile",
        "model",
        "thinking_enabled",
        "reasoning_effort",
        "timeout_seconds",
    }:
        raise ValueError("configuration snapshot has invalid fields")
    profile_payload = payload["profile"]
    if not isinstance(profile_payload, dict) or set(profile_payload) != {
        "root",
        "discovery_source",
        "source_dirs",
        "test_dirs",
        "pytest_command",
    }:
        raise ValueError("configuration profile has invalid fields")
    profile = ProjectProfile(
        root=Path(profile_payload["root"]),
        discovery_source=profile_payload["discovery_source"],
        source_dirs=tuple(PurePosixPath(value) for value in profile_payload["source_dirs"]),
        test_dirs=tuple(PurePosixPath(value) for value in profile_payload["test_dirs"]),
        pytest_command=tuple(profile_payload["pytest_command"]),
    )
    if profile.root != project_root.resolve():
        raise ValueError("configuration snapshot belongs to another project")
    return HarnessConfig(
        profile=profile,
        model=payload["model"],
        thinking_enabled=payload["thinking_enabled"],
        reasoning_effort=payload["reasoning_effort"],
        timeout_seconds=payload["timeout_seconds"],
    )


def app_state_dir(project_root: Path) -> Path:
    """Return the short-hash-isolated external state directory for one root."""
    return _snapshot_directory(project_root.resolve(), _application_state_root())


def project_config_path(project_root: Path) -> Path:
    """Return the external configuration snapshot for one selected project."""
    return app_state_dir(project_root) / "harness.yaml"


def local_state_path() -> Path:
    """Return the external selected-project and task-root index path."""
    return _application_state_root() / "local-state.yaml"


def _application_state_root() -> Path:
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "guardedpy"


def _snapshot_directory(root: Path, state_dir: Path) -> Path:
    root_hash = sha256(str(root).encode()).hexdigest()[:16]
    return state_dir / root_hash


def _snapshot_path(root: Path, state_dir: Path) -> Path:
    return _snapshot_directory(root.resolve(), state_dir) / "harness.yaml"


def _snapshot(config: HarnessConfig) -> dict[str, object]:
    profile = config.profile
    return {
        "profile": {
            "root": str(profile.root),
            "discovery_source": profile.discovery_source,
            "source_dirs": [path.as_posix() for path in profile.source_dirs],
            "test_dirs": [path.as_posix() for path in profile.test_dirs],
            "pytest_command": list(profile.pytest_command),
        },
        "model": config.model,
        "thinking_enabled": config.thinking_enabled,
        "reasoning_effort": config.reasoning_effort,
        "timeout_seconds": config.timeout_seconds,
    }


def _write_config(path: Path, config: HarnessConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(yaml.safe_dump(_snapshot(config), sort_keys=False).encode())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
