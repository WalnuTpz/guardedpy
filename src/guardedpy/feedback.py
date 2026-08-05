"""Deterministic, bounded feedback extracted from one pytest invocation."""

from __future__ import annotations

from dataclasses import dataclass
import re

from guardedpy.domain import FeedbackKind


_MAX_NODE_IDS = 20
_MAX_EXCERPT_CHARS = 800
_FAILED_NODE = re.compile(r"^FAILED (?P<node>\S+)(?:\s|$)", re.MULTILINE)
_COLLECTION_NODE = re.compile(r"^ERROR collecting (?P<node>\S+)(?:\s|$)", re.MULTILINE)
_USEFUL_OUTPUT = re.compile(
    r"^(FAILED |ERROR collecting |INTERNALERROR|E\s{7}|.*(?:AssertionError|assert ).*)"
)
_ASSERTION_EVIDENCE = re.compile(r"^E\s+(?:AssertionError\b|assert\b)", re.MULTILINE)


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


class FeedbackCollector:
    """Classify pytest output without returning the full, untrusted report."""

    def collect(self, run: PytestRun) -> PytestFeedback:
        output = "\n".join(part for part in (run.stdout, run.stderr) if part)
        if run.timed_out:
            return PytestFeedback(FeedbackKind.TIMEOUT, (), self._excerpt(output))
        if run.exit_code == 0:
            return PytestFeedback(FeedbackKind.PASSED, (), "")

        collection_nodes = self._node_ids(_COLLECTION_NODE, output)
        if collection_nodes:
            return PytestFeedback(
                FeedbackKind.COLLECTION_ERROR, collection_nodes, self._excerpt(output)
            )

        failed_nodes = self._node_ids(_FAILED_NODE, output)
        if failed_nodes:
            excerpt = self._excerpt(output)
            if not _ASSERTION_EVIDENCE.search(excerpt):
                return PytestFeedback(FeedbackKind.EXECUTION_ERROR, failed_nodes, excerpt)
            return PytestFeedback(
                FeedbackKind.ASSERTION_FAILURE, failed_nodes, excerpt
            )
        return PytestFeedback(FeedbackKind.EXECUTION_ERROR, (), self._excerpt(output))

    @staticmethod
    def _node_ids(pattern: re.Pattern[str], output: str) -> tuple[str, ...]:
        return tuple(match.group("node") for match in pattern.finditer(output))[:_MAX_NODE_IDS]

    @staticmethod
    def _excerpt(output: str) -> str:
        useful_lines = [line for line in output.splitlines() if _USEFUL_OUTPUT.match(line)]
        if not useful_lines:
            useful_lines = output.splitlines()[-1:]
        return "\n".join(useful_lines)[:_MAX_EXCERPT_CHARS]
