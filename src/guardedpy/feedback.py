"""Deterministic, bounded feedback extracted from one pytest invocation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
import re

_MAX_EXCERPT_CHARS = 800
# Pytest output is untrusted; keep repair state and model context bounded.
_MAX_NORMALIZED_NODE_IDS = 100
_MAX_NORMALIZED_NODE_ID_CHARS = 500
_MAX_NORMALIZED_NODE_CHARS = 20_000
_FAILED_NODE = re.compile(r"^FAILED (?P<node>\S+)(?P<detail>[^\n]*)$", re.MULTILINE)
_COLLECTION_NODE = re.compile(r"^ERROR collecting (?P<node>\S+)(?:\s|$)", re.MULTILINE)
_ERROR_PHASE = re.compile(r"^ERROR(?:\s|$)", re.MULTILINE)
_USEFUL_OUTPUT = re.compile(
    r"^(FAILED |ERROR collecting |INTERNALERROR|E\s{7}|.*(?:AssertionError|assert ).*)"
)
_ASSERTION_EVIDENCE = re.compile(r"^E\s+(?:AssertionError\b|assert\b)", re.MULTILINE)
_ASSERTION_SUMMARY = re.compile(r"(?:AssertionError\b|\bassert(?:ion failed)?\b)", re.IGNORECASE)
_ZERO_COLLECTED = re.compile(r"(?:collected 0 items|no tests ran)", re.IGNORECASE)


class FeedbackKind(StrEnum):
    PASSED = "passed"
    ASSERTION_FAILURE = "assertion_failure"
    COLLECTION_ERROR = "collection_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class PytestRun:
    """The unprocessed result of running the configured pytest command."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


@dataclass(frozen=True)
class PytestFeedback:
    """The bounded pytest facts safe to return to the orchestration loop."""

    kind: FeedbackKind
    node_ids: tuple[str, ...]
    excerpt: str


@dataclass(frozen=True)
class FeedbackProjection:
    """Bounded pytest facts included in a provider tool result."""

    kind: str
    node_ids: tuple[str, ...]
    excerpt: str


class FeedbackCollector:
    """Classify pytest output without returning the full, untrusted report."""

    def collect(self, run: PytestRun) -> PytestFeedback:
        output = "\n".join(part for part in (run.stdout, run.stderr) if part)
        if run.timed_out:
            return PytestFeedback(FeedbackKind.TIMEOUT, (), self._excerpt(output))
        if run.exit_code == 0 and not _ZERO_COLLECTED.search(output):
            return PytestFeedback(FeedbackKind.PASSED, (), "")
        if run.exit_code == 0:
            return PytestFeedback(FeedbackKind.EXECUTION_ERROR, (), self._excerpt(output))

        collection_nodes = self._node_ids(_COLLECTION_NODE, output)
        if run.exit_code == 2:
            return PytestFeedback(FeedbackKind.COLLECTION_ERROR, collection_nodes, self._excerpt(output))
        if collection_nodes:
            return PytestFeedback(
                FeedbackKind.COLLECTION_ERROR, collection_nodes, self._excerpt(output)
            )
        if _ERROR_PHASE.search(output):
            return PytestFeedback(FeedbackKind.EXECUTION_ERROR, (), self._excerpt(output))

        failed_matches = tuple(_FAILED_NODE.finditer(output))
        failed_nodes = tuple(match.group("node") for match in failed_matches)
        if failed_nodes:
            excerpt = self._excerpt(output)
            per_node_assertions = tuple(
                bool(_ASSERTION_SUMMARY.search(match.group("detail")))
                for match in failed_matches
            )
            all_assertions = all(per_node_assertions) or (
                len(failed_nodes) == 1 and _ASSERTION_EVIDENCE.search(excerpt) is not None
            )
            if not all_assertions:
                return PytestFeedback(FeedbackKind.EXECUTION_ERROR, failed_nodes, excerpt)
            return PytestFeedback(
                FeedbackKind.ASSERTION_FAILURE, failed_nodes, excerpt
            )
        return PytestFeedback(FeedbackKind.EXECUTION_ERROR, (), self._excerpt(output))

    @staticmethod
    def normalize_nodes(
        feedback: PytestFeedback,
        project_root: Path,
        test_dirs: tuple[PurePosixPath, ...],
    ) -> PytestFeedback:
        """Keep ordered, existing pytest nodes contained by configured test roots."""
        root = project_root.resolve()
        roots = tuple((root / directory).resolve() for directory in test_dirs)
        normalized: list[str] = []
        for node_id in feedback.node_ids:
            if len(node_id) > _MAX_NORMALIZED_NODE_ID_CHARS:
                return PytestFeedback(FeedbackKind.EXECUTION_ERROR, (), feedback.excerpt)
            path_text, separator, suffix = node_id.partition("::")
            candidate = (root / path_text).resolve()
            if (
                candidate.is_file()
                and candidate.is_relative_to(root)
                and any(candidate.is_relative_to(test_root) for test_root in roots)
            ):
                projected = candidate.relative_to(root).as_posix()
                if separator:
                    projected = f"{projected}::{suffix}"
                if len(projected) > _MAX_NORMALIZED_NODE_ID_CHARS:
                    return PytestFeedback(FeedbackKind.EXECUTION_ERROR, (), feedback.excerpt)
                if projected not in normalized:
                    if (
                        len(normalized) == _MAX_NORMALIZED_NODE_IDS
                        or sum(len(node) for node in normalized) + len(projected)
                        > _MAX_NORMALIZED_NODE_CHARS
                    ):
                        return PytestFeedback(FeedbackKind.EXECUTION_ERROR, (), feedback.excerpt)
                    normalized.append(projected)
        return PytestFeedback(feedback.kind, tuple(normalized), feedback.excerpt)

    @staticmethod
    def _node_ids(pattern: re.Pattern[str], output: str) -> tuple[str, ...]:
        return tuple(match.group("node") for match in pattern.finditer(output))

    @staticmethod
    def _excerpt(output: str) -> str:
        useful_lines = [line for line in output.splitlines() if _USEFUL_OUTPUT.match(line)]
        if not useful_lines:
            useful_lines = output.splitlines()[-1:]
        return "\n".join(useful_lines)[:_MAX_EXCERPT_CHARS]
