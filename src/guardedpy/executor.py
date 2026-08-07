"""Schema-checked, factual implementation of the continuous Agent tool set."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePosixPath
from typing import Mapping
from uuid import UUID

from guardedpy.config import HarnessConfig
from guardedpy.conversation import ReadRecord, ToolCall, Turn
from guardedpy.feedback import FeedbackCollector
from guardedpy.workspace import Workspace


@dataclass(frozen=True)
class ToolExecution:
    call_id: str
    verdict: str
    code: str
    summary: str
    provider_result: Mapping[str, object]
    changed_paths: tuple[str, ...] = ()
    feedback: object | None = None
    approval_id: UUID | None = None


class ToolExecutor:
    """Perform only the fixed local tool vocabulary after governance allows it."""

    def __init__(self, root, config: HarnessConfig) -> None:
        self._workspace = Workspace(root, config)
        self._feedback = FeedbackCollector()

    def execute(self, turn: Turn, item_id: UUID, call: ToolCall) -> ToolExecution:
        del item_id
        try:
            arguments = json.loads(call.arguments_json)
        except json.JSONDecodeError:
            return self._failure(call, "invalid_tool_call")
        if not isinstance(arguments, dict):
            return self._failure(call, "invalid_tool_call")
        if call.name == "list_files":
            return self._result(call, self._workspace.list_files(self._path(arguments, "path", ".")))
        if call.name == "read_file":
            return self._read(turn, call, arguments)
        if call.name == "apply_patch":
            return self._patch(turn, call, arguments)
        if call.name == "run_pytest":
            nodes = arguments.get("nodes", [])
            if not isinstance(nodes, list) or not all(isinstance(node, str) for node in nodes) or not len(nodes) <= 20:
                return self._failure(call, "invalid_tool_call")
            run = self._workspace.run_pytest(tuple(nodes))
            feedback = self._feedback.collect(run)
            if not nodes and feedback.kind.value == "passed":
                turn.needs_full_verification = False
            kind = feedback.kind.value
            return ToolExecution(call.id, "allow", kind, "pytest completed", {
                "ok": kind == "passed", "code": kind, "summary": "pytest completed",
                "feedback": {
                    "kind": kind, "node_ids": list(feedback.node_ids),
                    "excerpt": feedback.excerpt,
                },
            }, feedback=feedback)
        if call.name == "git_diff":
            return self._result(call, self._workspace.git_diff())
        if call.name == "git_status":
            return self._result(call, self._workspace.git_status())
        if call.name == "delete_path":
            result = self._workspace.delete_path(self._path(arguments, "path"))
            execution = self._result(call, result)
            if result.ok:
                path = self._path(arguments, "path").as_posix()
                turn.needs_full_verification = True
                return ToolExecution(
                    call.id, "allow", execution.code, execution.summary,
                    execution.provider_result, (path,),
                )
            return execution
        return self._failure(call, "invalid_tool_call")

    def _read(self, turn: Turn, call: ToolCall, arguments: dict[str, object]) -> ToolExecution:
        path = self._path(arguments, "path")
        offset, limit = arguments.get("offset", 0), arguments.get("limit", 200)
        if not isinstance(offset, int) or not isinstance(limit, int):
            return self._failure(call, "invalid_tool_call")
        result = self._workspace.read_file(path, offset, limit)
        execution = self._result(call, result)
        if result.ok:
            turn.reads[path.as_posix()] = ReadRecord(
                path.as_posix(), str(result.data["sha256"]), bool(result.data["complete"])
            )
        return execution

    def _patch(self, turn: Turn, call: ToolCall, arguments: dict[str, object]) -> ToolExecution:
        diff = arguments.get("unified_diff")
        if not isinstance(diff, str) or len(diff.encode()) > 65536:
            return self._failure(call, "invalid_tool_call")
        for path in _existing_patch_paths(diff):
            record = turn.reads.get(path)
            target = self._workspace.root / path
            if record is None or not record.complete:
                return self._failure(call, "read_required")
            if not target.is_file() or sha256(target.read_bytes()).hexdigest() != record.sha256:
                return self._failure(call, "stale_read")
        result = self._workspace.apply_patch(diff)
        execution = self._result(call, result)
        if result.ok:
            changed = tuple(result.data["files"])
            turn.needs_full_verification = True
            return ToolExecution(call.id, "allow", "ok", result.summary, execution.provider_result, changed)
        return execution

    @staticmethod
    def _path(arguments: dict[str, object], name: str, default: str | None = None) -> PurePosixPath:
        value = arguments.get(name, default)
        if not isinstance(value, str) or not value:
            raise ValueError("invalid path")
        return PurePosixPath(value)

    @staticmethod
    def _failure(call: ToolCall, code: str) -> ToolExecution:
        return ToolExecution(call.id, "deny", code, code, {"code": code})

    @staticmethod
    def _result(call: ToolCall, result) -> ToolExecution:
        code = "ok" if result.ok else str(result.data.get("reason", "tool_failed"))
        code = {"invalid_patch": "patch_invalid", "hunk_mismatch": "patch_not_applied"}.get(code, code)
        return ToolExecution(
            call.id, "allow", code, result.summary,
            {"ok": result.ok, "code": code, "summary": result.summary, **result.data},
        )


def _existing_patch_paths(diff: str) -> tuple[str, ...]:
    paths: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("--- "):
            continue
        path = line[4:].split("\t", 1)[0]
        if path == "/dev/null":
            continue
        if path.startswith("a/"):
            path = path[2:]
        paths.append(path)
    return tuple(paths)
