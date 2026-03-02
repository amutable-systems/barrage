# SPDX-License-Identifier: MIT
"""
barrage – a concurrent async test framework for Python.

Unlike traditional test frameworks that run tests sequentially, this
framework runs tests concurrently using ``asyncio``.  Whether tests
within the same class run concurrently is configurable on a per-class
basis via ``__init_subclass__``, and the overall level of concurrency
is tuneable via the runner.

Quick start::

    from barrage import AsyncTestCase, AsyncTestRunner

    class MyTests(AsyncTestCase):
        async def setUp(self) -> None:
            self.value = 42

        async def test_example(self) -> None:
            self.assertEqual(self.value, 42)

    if __name__ == "__main__":
        runner = AsyncTestRunner(verbosity=2)
        result = runner.run_classes(MyTests)
        print(result.format_report(verbosity=2))
"""

from barrage.case import AsyncTestCase, MonitoredTestCase
from barrage.discovery import discover, discover_module, resolve_tests
from barrage.result import AsyncTestResult, Outcome, TestOutcome
from barrage.runner import AsyncTestRunner, AsyncTestSuite
from barrage.subprocess import (
    DEVNULL,
    PIPE,
    STDOUT,
    CalledProcessError,
    CompletedProcess,
    run,
    spawn,
)
from barrage.taskgroups import TaskGroup

__all__ = [
    # Core
    "AsyncTestCase",
    "MonitoredTestCase",
    # Runner
    "AsyncTestRunner",
    "AsyncTestSuite",
    # Results
    "AsyncTestResult",
    "TestOutcome",
    "Outcome",
    # Discovery
    "discover",
    "discover_module",
    "resolve_tests",
    # Subprocess
    "spawn",
    "run",
    "CompletedProcess",
    "CalledProcessError",
    "DEVNULL",
    "PIPE",
    "STDOUT",
    # Skip helpers
    "SkipTest",
    # TaskGroup
    "TaskGroup",
]
