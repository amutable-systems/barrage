# SPDX-License-Identifier: MIT
import asyncio
import io
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from barrage.colorize import (
    colored_captured_header,
    colored_result_line,
    colored_section_header,
    colored_separator,
    colored_summary,
    strip_ansi,
)

if TYPE_CHECKING:
    from barrage.case import AsyncTestCase


class Outcome(Enum):
    PASSED = auto()
    FAILED = auto()
    ERRORED = auto()
    SKIPPED = auto()
    INTERRUPTED = auto()


@dataclass
class TestOutcome:
    """The result of running a single test method."""

    test_id: str
    test_str: str
    outcome: Outcome
    duration: float = 0.0
    message: str = ""
    traceback: str = ""
    stdout: str = ""
    stderr: str = ""


class AsyncTestResult:
    """
    Collects results from concurrent async test runs.

    All mutation is protected by an ``asyncio.Lock`` so that multiple
    concurrent test tasks can safely record outcomes.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.results: list[TestOutcome] = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    async def add_success(
        self,
        test: "AsyncTestCase",
        duration: float,
        stdout: str = "",
        stderr: str = "",
    ) -> TestOutcome:
        outcome = TestOutcome(
            test_id=test.id(),
            test_str=str(test),
            outcome=Outcome.PASSED,
            duration=duration,
            stdout=stdout,
            stderr=stderr,
        )
        async with self._lock:
            self.results.append(outcome)
        return outcome

    async def add_failure(
        self,
        test: "AsyncTestCase",
        exc: BaseException,
        duration: float,
        stdout: str = "",
        stderr: str = "",
        traceback_str: str = "",
    ) -> TestOutcome:
        outcome = TestOutcome(
            test_id=test.id(),
            test_str=str(test),
            outcome=Outcome.FAILED,
            message=str(exc),
            traceback=traceback_str,
            duration=duration,
            stdout=stdout,
            stderr=stderr,
        )
        async with self._lock:
            self.results.append(outcome)
        return outcome

    async def add_error(
        self,
        test: "AsyncTestCase",
        exc: BaseException,
        duration: float,
        stdout: str = "",
        stderr: str = "",
        traceback_str: str = "",
    ) -> TestOutcome:
        outcome = TestOutcome(
            test_id=test.id(),
            test_str=str(test),
            outcome=Outcome.ERRORED,
            message=str(exc),
            traceback=traceback_str,
            duration=duration,
            stdout=stdout,
            stderr=stderr,
        )
        async with self._lock:
            self.results.append(outcome)
        return outcome

    async def add_skip(self, test: "AsyncTestCase", reason: str) -> TestOutcome:
        outcome = TestOutcome(
            test_id=test.id(),
            test_str=str(test),
            outcome=Outcome.SKIPPED,
            message=reason,
        )
        async with self._lock:
            self.results.append(outcome)
        return outcome

    async def add_interrupted(self, test: "AsyncTestCase") -> TestOutcome:
        outcome = TestOutcome(
            test_id=test.id(),
            test_str=str(test),
            outcome=Outcome.INTERRUPTED,
        )
        async with self._lock:
            self.results.append(outcome)
        return outcome

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    @property
    def passed(self) -> list[TestOutcome]:
        return [r for r in self.results if r.outcome is Outcome.PASSED]

    @property
    def failures(self) -> list[TestOutcome]:
        return [r for r in self.results if r.outcome is Outcome.FAILED]

    @property
    def errors(self) -> list[TestOutcome]:
        return [r for r in self.results if r.outcome is Outcome.ERRORED]

    @property
    def skipped(self) -> list[TestOutcome]:
        return [r for r in self.results if r.outcome is Outcome.SKIPPED]

    @property
    def interrupted(self) -> list[TestOutcome]:
        return [r for r in self.results if r.outcome is Outcome.INTERRUPTED]

    @property
    def tests_run(self) -> int:
        return len(self.results)

    @property
    def was_successful(self) -> bool:
        return len(self.failures) == 0 and len(self.errors) == 0 and len(self.interrupted) == 0

    @property
    def total_duration(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    # ------------------------------------------------------------------ #
    # Formatting
    # ------------------------------------------------------------------ #

    def format_report(
        self,
        *,
        verbosity: int = 1,
        show_output: bool = False,
        color: bool = False,
    ) -> str:
        """
        Return a human-readable report string.

        verbosity=0 : one-line summary only
        verbosity=1 : summary + failure/error details (with captured output)
        verbosity=2 : per-test lines + summary + failure/error details

        show_output : when ``True``, captured stdout/stderr is shown for
                      *all* tests (including passes).  When ``False``
                      (the default), captured output is only shown for
                      failures and errors.

        color : when ``True``, ANSI escape sequences in stored
                tracebacks are preserved in the output.  When ``False``
                (the default), they are stripped so the report is safe
                for non-terminal destinations.
        """
        buf = io.StringIO()

        if verbosity >= 2:
            for r in self.results:
                line = colored_result_line(
                    r.outcome.name,
                    r.test_str,
                    r.duration,
                    color=color,
                )
                buf.write(line + "\n")
                if show_output:
                    _write_captured_output(buf, r, indent="      ", color=color)
            buf.write("\n")

        # Print failure / error details
        if verbosity >= 1:
            for section_name, items in [("FAIL", self.failures), ("ERROR", self.errors)]:
                for r in items:
                    buf.write(colored_separator("=", 70, color=color) + "\n")
                    buf.write(colored_section_header(section_name, r.test_str, color=color) + "\n")
                    buf.write(colored_separator("-", 70, color=color) + "\n")
                    tb = r.traceback if color else strip_ansi(r.traceback)
                    buf.write(tb)
                    _write_captured_output(buf, r, indent="", color=color)
                    buf.write("\n")

            # Show captured output for passing tests only if requested
            if show_output:
                for r in self.passed:
                    if r.stdout or r.stderr:
                        buf.write(colored_separator("=", 70, color=color) + "\n")
                        buf.write(f"OUTPUT: {r.test_str}\n")
                        buf.write(colored_separator("-", 70, color=color) + "\n")
                        _write_captured_output(buf, r, indent="", color=color)
                        buf.write("\n")

        # Summary
        buf.write(
            colored_summary(
                was_successful=self.was_successful,
                tests_run=self.tests_run,
                total_duration=self.total_duration,
                n_failures=len(self.failures),
                n_errors=len(self.errors),
                n_skipped=len(self.skipped),
                n_interrupted=len(self.interrupted),
                color=color,
            )
        )

        return buf.getvalue()


def _write_captured_output(
    buf: io.StringIO,
    r: TestOutcome,
    indent: str,
    color: bool = False,
) -> None:
    """Append captured stdout/stderr for *r* to *buf* if non-empty."""
    if r.stdout:
        header = colored_captured_header("Captured stdout:", color=color)
        buf.write(f"{indent}{header}\n")
        for line in r.stdout.splitlines(keepends=True):
            buf.write(f"{indent}  {line}")
        if not r.stdout.endswith("\n"):
            buf.write("\n")
    if r.stderr:
        header = colored_captured_header("Captured stderr:", color=color)
        buf.write(f"{indent}{header}\n")
        for line in r.stderr.splitlines(keepends=True):
            buf.write(f"{indent}  {line}")
        if not r.stderr.endswith("\n"):
            buf.write("\n")


_OUTCOME_SYMBOLS: dict[Outcome, str] = {
    Outcome.PASSED: "✓",
    Outcome.FAILED: "✗",
    Outcome.ERRORED: "E",
    Outcome.SKIPPED: "S",
    Outcome.INTERRUPTED: "C",
}
