"""Root-contained filesystem tools for a selected project."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any

from guardedpy.config import HarnessConfig
from guardedpy.feedback import PytestRun


_MAX_READ_LINES = 200
_MAX_LISTED_FILES = 200
_MAX_OUTPUT_CHARS = 32 * 1024
_MAX_PROGRAM_OUTPUT_CHARS = 2_000
_HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)


@dataclass(frozen=True)
class ToolResult:
    """A factual result from a workspace operation."""

    ok: bool
    summary: str
    data: dict[str, Any]


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_count: int
    new_count: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _FilePatch:
    path: PurePosixPath
    create: bool
    hunks: tuple[_Hunk, ...]


class Workspace:
    """Expose filesystem facts without making governance decisions."""

    def __init__(self, root: Path, config: HarnessConfig) -> None:
        self.root = root.resolve()
        self.config = config

    def list_files(self, path: PurePosixPath = PurePosixPath(".")) -> ToolResult:
        target = self._inside_root(path)
        if target is None:
            return self._outside_root()
        if not target.is_dir():
            return ToolResult(False, "Path is not a directory", {"reason": "not_a_directory"})

        files = tuple(
            sorted(
                entry.relative_to(self.root).as_posix()
                for entry in target.rglob("*")
                if entry.is_file()
                and not any(part.startswith(".") for part in entry.relative_to(self.root).parts)
                and entry.resolve().is_relative_to(self.root)
            )
        )[:_MAX_LISTED_FILES]
        return ToolResult(True, "Listed project files", {"files": files})

    def read_file(self, path: PurePosixPath, offset: int, limit: int) -> ToolResult:
        target = self._inside_root(path)
        if target is None:
            return self._outside_root()
        if not target.is_file():
            return ToolResult(False, "Path is not a file", {"reason": "not_a_file"})
        if offset < 0 or limit < 1 or limit > _MAX_READ_LINES:
            return ToolResult(False, "Page is outside supported bounds", {"reason": "invalid_page"})

        content_bytes = target.read_bytes()
        try:
            text = content_bytes.decode()
        except UnicodeDecodeError:
            return ToolResult(False, "Path is not a text file", {"reason": "not_a_text_file"})
        lines = text.splitlines(keepends=True)
        selected = lines[offset : offset + limit]
        page: list[str] = []
        size = 0
        for line in selected:
            encoded = line.encode()
            if size + len(encoded) > _MAX_OUTPUT_CHARS:
                break
            page.append(line)
            size += len(encoded)
        content = "".join(page)
        next_offset = offset + len(page)
        return ToolResult(
            True,
            "Read project file",
            {
                "path": target.relative_to(self.root).as_posix(),
                "content": content,
                "next_offset": next_offset,
                "sha256": sha256(content_bytes).hexdigest(),
                "complete": offset == 0 and next_offset == len(lines),
            },
        )

    def apply_patch(self, diff: str) -> ToolResult:
        parsed = self._parse_patch(diff)
        if parsed is None:
            return ToolResult(False, "Diff is not a supported unified patch", {"reason": "invalid_patch"})

        prepared: dict[Path, str] = {}
        for file_patch in parsed:
            target = self._inside_root(file_patch.path)
            if target is None:
                return self._outside_root()
            if file_patch.create and not self._inside_discovered_write_directory(file_patch.path):
                allowed = tuple(
                    directory.as_posix()
                    for directory in (*self.config.source_dirs, *self.config.test_dirs)
                )
                return ToolResult(
                    False,
                    "New files must be created inside discovered source or test directories: "
                    + ", ".join(allowed),
                    {"reason": "new_file_path_not_allowed", "allowed_directories": allowed},
                )
            if not self._patch_target_allowed(file_patch.path, target, file_patch.create):
                return ToolResult(False, "Patch target is not permitted", {"reason": "patch_invalid"})
            if file_patch.create:
                if target.exists():
                    return ToolResult(False, "New file target already exists; read it before modifying", {"reason": "new_file_target_exists"})
                if not target.parent.is_dir():
                    return ToolResult(False, "New file parent directory does not exist", {"reason": "new_file_parent_missing"})
                original = ""
            else:
                if not target.is_file():
                    return ToolResult(False, "Patch target is not a file", {"reason": "not_a_file"})
                original = prepared.get(target, target.read_text())

            updated = self._apply_hunks(original, file_patch.hunks)
            if updated is None:
                return ToolResult(False, "Patch hunk does not match the current file", {"reason": "hunk_mismatch"})
            prepared[target] = updated

        if not self._atomic_replace(prepared):
            return ToolResult(False, "Patch could not be applied", {"reason": "patch_not_applied"})
        return ToolResult(
            True,
            "Applied unified patch",
            {"files": tuple(path.relative_to(self.root).as_posix() for path in prepared)},
        )

    def delete_path(self, path: PurePosixPath) -> ToolResult:
        target = self._inside_root(path)
        if target is None:
            return self._outside_root()
        if any(part.startswith(".") for part in path.parts) or self._contains_symlink(path) or target.is_symlink():
            return ToolResult(False, "Path is protected", {"reason": "protected_path"})
        if target.is_file():
            target.unlink()
        elif target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        else:
            return ToolResult(False, "Path cannot be deleted", {"reason": "not_deletable"})
        return ToolResult(
            True,
            "Deleted project file",
            {"path": target.relative_to(self.root).as_posix()},
        )

    def run_pytest(self, targets: tuple[str, ...]) -> PytestRun:
        """Run configured pytest from the selected root with test-root-only targets."""
        self._validate_pytest_targets(targets)
        try:
            completed = subprocess.run(
                (*self.config.pytest_command, *targets),
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            return PytestRun(
                -1,
                self._output_text(error.stdout),
                self._output_text(error.stderr),
                True,
            )
        return PytestRun(completed.returncode, completed.stdout, completed.stderr, False)

    def run_python(self, path: PurePosixPath, argv: tuple[str, ...]) -> ToolResult:
        """Run one contained Python program through the interpreter, never a shell."""
        target = self._inside_root(path)
        if target is None:
            return self._outside_root()
        if (
            target.suffix != ".py"
            or not target.is_file()
            or target.is_symlink()
            or self._contains_symlink(path)
            or any(part.startswith(".") for part in path.parts)
        ):
            reason = "not_python_file" if target.suffix != ".py" else "not_a_file"
            return ToolResult(False, "Program target must be a regular project Python file", {"reason": reason})
        try:
            completed = subprocess.run(
                (sys.executable, str(target), *argv),
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            return ToolResult(
                False,
                "Python program timed out",
                {
                    "reason": "program_timeout",
                    "output": self._program_output(error.stdout, error.stderr),
                },
            )
        output = self._program_output(completed.stdout, completed.stderr)
        if completed.returncode:
            return ToolResult(
                False,
                f"Python program exited with code {completed.returncode}",
                {"reason": "program_failed", "exit_code": completed.returncode, "output": output},
            )
        return ToolResult(
            True,
            "Python program completed",
            {
                "path": target.relative_to(self.root).as_posix(),
                "output": output,
            },
        )

    def git_diff(self) -> ToolResult:
        return self._git("diff", "--")

    def git_status(self) -> ToolResult:
        return self._git("status", "--short")

    def _git(self, *arguments: str) -> ToolResult:
        completed = subprocess.run(
            ("git", "-C", str(self.root), *arguments),
            capture_output=True, text=True, timeout=self.config.timeout_seconds,
            check=False, shell=False,
        )
        if completed.returncode:
            output = completed.stderr.lower()
            reason = "not_a_git_repository" if "not a git repository" in output else "git_failed"
            return ToolResult(False, "Git command failed", {"reason": reason})
        return ToolResult(True, "Read Git state", {"output": self._bounded_output(completed.stdout)})

    def _inside_root(self, path: PurePosixPath) -> Path | None:
        if path.is_absolute() or ".." in path.parts or "\\" in path.as_posix():
            return None
        candidate = self.root / Path(path)
        if candidate.resolve().is_relative_to(self.root):
            return candidate
        return None

    def _patch_target_allowed(self, path: PurePosixPath, target: Path, create: bool) -> bool:
        protected_names = {"README", "README.md", "pyproject.toml", "setup.cfg", "tox.ini"}
        if (
            any(part.startswith(".") or part in {"docs", "config"} for part in path.parts)
            or path.name in protected_names
            or self._contains_symlink(path)
            or target.is_symlink()
        ):
            return False
        if not any(path.is_relative_to(directory) for directory in self.config.source_dirs + self.config.test_dirs):
            return False
        if create:
            return target.parent.is_dir()
        return target.is_file()

    def _inside_discovered_write_directory(self, path: PurePosixPath) -> bool:
        return any(
            path.is_relative_to(directory)
            for directory in (*self.config.source_dirs, *self.config.test_dirs)
        )

    def _contains_symlink(self, path: PurePosixPath) -> bool:
        current = self.root
        for part in path.parts:
            current /= part
            if current.is_symlink():
                return True
        return False

    @staticmethod
    def _bounded_output(output: str) -> str:
        limited_lines = output.splitlines(keepends=True)[:200]
        result = "".join(limited_lines)
        return result.encode()[:_MAX_OUTPUT_CHARS].decode(errors="ignore")

    @staticmethod
    def _program_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
        output = "".join(
            part for part in (Workspace._output_text(stdout), Workspace._output_text(stderr)) if part
        )
        return output[:_MAX_PROGRAM_OUTPUT_CHARS]

    @staticmethod
    def _atomic_replace(prepared: dict[Path, str]) -> bool:
        staged: dict[Path, Path] = {}
        backups: dict[Path, Path | None] = {}
        replaced: list[Path] = []
        try:
            for target, content in prepared.items():
                mode = target.stat().st_mode & 0o777 if target.exists() else None
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, mode="w") as handle:
                    handle.write(content)
                    staged[target] = Path(handle.name)
                if mode is not None:
                    os.chmod(staged[target], mode)
                if target.exists():
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                        handle.write(target.read_bytes())
                        backups[target] = Path(handle.name)
                    os.chmod(backups[target], mode)
                else:
                    backups[target] = None
            for target, temporary in staged.items():
                os.replace(temporary, target)
                replaced.append(target)
            return True
        except OSError:
            for target in reversed(replaced):
                backup = backups[target]
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            return False
        finally:
            for temporary in (*staged.values(), *(item for item in backups.values() if item is not None)):
                temporary.unlink(missing_ok=True)

    def _validate_pytest_targets(self, targets: tuple[str, ...]) -> None:
        test_roots = tuple((self.root / path).resolve() for path in self.config.test_dirs)
        for target in targets:
            path_text = target.split("::", 1)[0]
            if target.startswith("-") or not path_text:
                raise ValueError("pytest targets must name files in configured test directories")
            candidate = self._inside_root(PurePosixPath(path_text))
            if candidate is None or not any(candidate.is_relative_to(root) for root in test_roots):
                raise ValueError("pytest targets must stay inside configured test directories")

    @staticmethod
    def _output_text(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value or ""

    @staticmethod
    def _outside_root() -> ToolResult:
        return ToolResult(False, "Path is outside the project root", {"reason": "path_outside_project"})

    def _parse_patch(self, diff: str) -> tuple[_FilePatch, ...] | None:
        lines = diff.splitlines(keepends=True)
        patches: list[_FilePatch] = []
        index = 0
        while index < len(lines):
            if not lines[index].startswith("--- ") or index + 1 >= len(lines):
                return None
            old_path = self._patch_path(lines[index][4:])
            index += 1
            if not lines[index].startswith("+++ "):
                return None
            new_path = self._patch_path(lines[index][4:])
            index += 1
            if old_path is None or new_path is None:
                return None

            create = old_path == "/dev/null"
            if create:
                if new_path == "/dev/null":
                    return None
                patch_path = new_path
            elif new_path == "/dev/null" or old_path != new_path:
                return None
            else:
                patch_path = old_path

            hunks: list[_Hunk] = []
            while index < len(lines):
                if lines[index].startswith("--- ") and index + 1 < len(lines) and lines[index + 1].startswith("+++ "):
                    break
                match = _HUNK_HEADER.match(lines[index].rstrip("\n"))
                if match is None:
                    return None
                old_start, _old_count, _new_start, _new_count = match.groups()
                index += 1
                hunk_lines: list[str] = []
                while index < len(lines):
                    if lines[index].startswith("@@ ") or (lines[index].startswith("--- ") and index + 1 < len(lines) and lines[index + 1].startswith("+++ ")):
                        break
                    if not lines[index].startswith((" ", "+", "-")):
                        return None
                    hunk_lines.append(lines[index])
                    index += 1
                if not hunk_lines:
                    return None
                actual_old_count = sum(line[0] in " -" for line in hunk_lines)
                actual_new_count = sum(line[0] in " +" for line in hunk_lines)
                hunks.append(_Hunk(int(old_start), actual_old_count, actual_new_count, tuple(hunk_lines)))
            if not hunks:
                return None
            patches.append(_FilePatch(PurePosixPath(patch_path), create, tuple(hunks)))
        return tuple(patches)

    @staticmethod
    def _patch_path(header_value: str) -> str | None:
        raw_path = header_value.rstrip("\n").split("\t", 1)[0]
        if raw_path == "/dev/null":
            return raw_path
        if raw_path.startswith(("a/", "b/")):
            raw_path = raw_path[2:]
        return PurePosixPath(raw_path).as_posix()

    @staticmethod
    def _apply_hunks(original: str, hunks: tuple[_Hunk, ...]) -> str | None:
        lines = original.splitlines(keepends=True)
        line_delta = 0
        for hunk in hunks:
            position = (hunk.old_start - 1 if hunk.old_start else 0) + line_delta
            if position < 0 or position > len(lines):
                return None
            expected = [line[1:] for line in hunk.lines if line[0] in " -"]
            replacement = [line[1:] for line in hunk.lines if line[0] in " +"]
            if lines[position : position + len(expected)] != expected:
                return None
            lines[position : position + len(expected)] = replacement
            line_delta += len(replacement) - len(expected)
        return "".join(lines)
