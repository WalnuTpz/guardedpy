"""Root-contained filesystem tools for a selected project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any

from guardedpy.config import HarnessConfig


_MAX_READ_LINES = 200
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
                if entry.is_file() and entry.resolve().is_relative_to(self.root)
            )
        )
        return ToolResult(True, "Listed project files", {"files": files})

    def read_file(self, path: PurePosixPath, offset: int, limit: int) -> ToolResult:
        target = self._inside_root(path)
        if target is None:
            return self._outside_root()
        if not target.is_file():
            return ToolResult(False, "Path is not a file", {"reason": "not_a_file"})
        if offset < 0 or limit < 1 or limit > _MAX_READ_LINES:
            return ToolResult(False, "Page is outside supported bounds", {"reason": "invalid_page"})

        lines = target.read_text().splitlines(keepends=True)
        content = "".join(lines[offset : offset + limit])
        next_offset = offset + len(lines[offset : offset + limit])
        return ToolResult(
            True,
            "Read project file",
            {
                "path": target.relative_to(self.root).as_posix(),
                "content": content,
                "next_offset": next_offset,
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
            if file_patch.create:
                if target.exists() or not target.parent.is_dir():
                    return ToolResult(False, "Patch target cannot be created", {"reason": "invalid_patch"})
                original = ""
            else:
                if not target.is_file():
                    return ToolResult(False, "Patch target is not a file", {"reason": "not_a_file"})
                original = prepared.get(target, target.read_text())

            updated = self._apply_hunks(original, file_patch.hunks)
            if updated is None:
                return ToolResult(False, "Patch hunk does not match the current file", {"reason": "hunk_mismatch"})
            prepared[target] = updated

        for target, content in prepared.items():
            target.write_text(content)
        return ToolResult(
            True,
            "Applied unified patch",
            {"files": tuple(path.relative_to(self.root).as_posix() for path in prepared)},
        )

    def delete_path(self, path: PurePosixPath) -> ToolResult:
        target = self._inside_root(path)
        if target is None:
            return self._outside_root()
        if not target.is_file():
            return ToolResult(False, "Path is not a file", {"reason": "not_a_file"})

        target.unlink()
        return ToolResult(
            True,
            "Deleted project file",
            {"path": target.relative_to(self.root).as_posix()},
        )

    def _inside_root(self, path: PurePosixPath) -> Path | None:
        candidate = (self.root / Path(path)).resolve()
        if candidate.is_relative_to(self.root):
            return candidate
        return None

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
            while index < len(lines) and not lines[index].startswith("--- "):
                match = _HUNK_HEADER.match(lines[index].rstrip("\n"))
                if match is None:
                    return None
                old_start, old_count, _new_start, new_count = match.groups()
                index += 1
                hunk_lines: list[str] = []
                while index < len(lines) and not lines[index].startswith(("@@ ", "--- ")):
                    if not lines[index].startswith((" ", "+", "-")):
                        return None
                    hunk_lines.append(lines[index])
                    index += 1
                if not hunk_lines:
                    return None
                parsed_old_count = int(old_count) if old_count is not None else 1
                parsed_new_count = int(new_count) if new_count is not None else 1
                if sum(line[0] in " -" for line in hunk_lines) != parsed_old_count:
                    return None
                if sum(line[0] in " +" for line in hunk_lines) != parsed_new_count:
                    return None
                hunks.append(
                    _Hunk(
                        int(old_start),
                        parsed_old_count,
                        parsed_new_count,
                        tuple(hunk_lines),
                    )
                )
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
