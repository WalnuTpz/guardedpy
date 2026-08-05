"""A WSL-native exclusive lease for local GuardedPy mutations."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

from guardedpy.config import app_state_dir


class ExecutionLease:
    """Hold one non-blocking advisory lock for a selected project root."""

    def __init__(self, project_root: Path) -> None:
        self._path = app_state_dir(project_root) / "execution.lock"
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        """Return whether this lease currently owns the lock."""
        return self._fd is not None

    def try_acquire(self) -> bool:
        """Acquire the project lock without waiting for another local runtime."""
        if self._fd is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def release(self) -> None:
        """Release the lock by closing its file descriptor."""
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None
