# SPDX-License-Identifier: MIT
"""
Self-hosting tests for the async test framework.

Every test here is an ``AsyncTestCase`` subclass that exercises the
framework by running inner test classes through ``AsyncTestRunner`` and
inspecting the ``AsyncTestResult``.

Run with::

    cd Amutable-OS
    python3 -m barrage barrage/tests/

Or run this file directly::

    cd Amutable-OS
    python3 -m barrage barrage/tests/test_framework.py
"""

import asyncio
import io
import sys
import time
from pathlib import Path
from typing import ClassVar

from barrage.case import AsyncTestCase, MonitoredTestCase, SkipTest
from barrage.discovery import discover, discover_module, resolve_tests
from barrage.result import AsyncTestResult
from barrage.runner import AsyncTestRunner, AsyncTestSuite, _collect_test_methods
from barrage.subprocess import PIPE, run

# ===================================================================== #
#  Helpers
# ===================================================================== #


async def _run(
    *classes: type[AsyncTestCase],
    max_concurrency: int | None = None,
    interactive: bool = False,
    show_output: bool = False,
    interactive_stream: io.StringIO | None = None,
    failfast: bool = False,
) -> AsyncTestResult:
    """Run one or more inner test classes and return the result."""
    # When interactive, default to a StringIO sink so that the inner
    # runner's status lines don't leak to the real terminal.  Callers
    # can pass their own stream to inspect the interactive output.
    if interactive and interactive_stream is None:
        interactive_stream = io.StringIO()
    runner = AsyncTestRunner(
        max_concurrency=max_concurrency,
        verbosity=1 if interactive else 0,
        interactive=interactive,
        show_output=show_output,
        interactive_stream=interactive_stream,
        failfast=failfast,
    )
    return await runner.run_classes_async(*classes)


# ===================================================================== #
#  1. Basic execution: pass / fail / error
# ===================================================================== #


class TestBasicExecution(AsyncTestCase, concurrent=True):
    async def test_passing_tests(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_one(self) -> None:
                assert 1 + 1 == 2

            async def test_two(self) -> None:
                assert "hello".upper() == "HELLO"

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)
        self.assertEqual(len(result.passed), 2)
        self.assertEqual(len(result.failures), 0)
        self.assertEqual(len(result.errors), 0)

    async def test_assertion_failure(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_fail(self) -> None:
                raise AssertionError("deliberate failure")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("deliberate failure", result.failures[0].traceback)

    async def test_runtime_error(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_error(self) -> None:
                raise RuntimeError("deliberate error")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("deliberate error", result.errors[0].traceback)

    async def test_mixed_results(self) -> None:
        class _Pass(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_ok_1(self) -> None:
                pass

            async def test_ok_2(self) -> None:
                pass

        class _Fail(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_fail(self) -> None:
                raise AssertionError("bad")

        class _Error(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_error(self) -> None:
                raise RuntimeError("boom")

        result = await _run(_Pass, _Fail, _Error)
        self.assertFalse(result.was_successful)
        self.assertEqual(result.tests_run, 4)
        self.assertEqual(len(result.passed), 2)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(len(result.errors), 1)


# ===================================================================== #
#  2. Lifecycle ordering (sequential class for determinism)
# ===================================================================== #


class TestLifecycle(AsyncTestCase, concurrent=True):
    async def test_lifecycle_order(self) -> None:
        log: list[str] = []

        class _Inner(AsyncTestCase):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                log.append("setUpClass")

            async def setUp(self) -> None:
                log.append("setUp")

            async def test_a(self) -> None:
                log.append("test_a")

            async def test_b(self) -> None:
                log.append("test_b")

            async def tearDown(self) -> None:
                log.append("tearDown")

            @classmethod
            async def tearDownClass(cls) -> None:
                log.append("tearDownClass")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(
            log,
            [
                "setUpClass",
                "setUp",
                "test_a",
                "tearDown",
                "setUp",
                "test_b",
                "tearDown",
                "tearDownClass",
            ],
        )

    async def test_teardown_runs_on_failure(self) -> None:
        teardown_ran = False

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def tearDown(self) -> None:
                nonlocal teardown_ran
                teardown_ran = True

            async def test_fail(self) -> None:
                raise AssertionError("deliberate")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertTrue(teardown_ran)

    async def test_teardown_runs_on_error(self) -> None:
        teardown_ran = False

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def tearDown(self) -> None:
                nonlocal teardown_ran
                teardown_ran = True

            async def test_error(self) -> None:
                raise RuntimeError("kaboom")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertTrue(teardown_ran)


# ===================================================================== #
#  3. setUp / tearDown error handling
# ===================================================================== #


class TestSetUpTearDownErrors(AsyncTestCase, concurrent=True):
    async def test_setup_failure_is_error(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def setUp(self) -> None:
                raise RuntimeError("setUp exploded")

            async def test_never_reached(self) -> None:
                raise AssertionError("should not run")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("setUp exploded", result.errors[0].traceback)

    async def test_teardown_failure_is_error(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def tearDown(self) -> None:
                raise RuntimeError("tearDown exploded")

            async def test_passes(self) -> None:
                assert True

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("tearDown exploded", result.errors[0].traceback)

    async def test_setup_class_failure_errors_all_tests(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                raise RuntimeError("setUpClass kaboom")

            async def test_a(self) -> None:
                pass

            async def test_b(self) -> None:
                pass

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(result.tests_run, 2)
        self.assertEqual(len(result.errors), 2)
        for e in result.errors:
            self.assertIn("setUpClass kaboom", e.traceback)

    async def test_teardown_class_failure_is_error(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            @classmethod
            async def tearDownClass(cls) -> None:
                raise RuntimeError("tearDownClass boom")

            async def test_ok(self) -> None:
                pass

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.passed), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("tearDownClass boom", result.errors[0].traceback)


# ===================================================================== #
#  4. Concurrency: tests within a class actually overlap
# ===================================================================== #


class TestConcurrency(AsyncTestCase, concurrent=True):
    async def test_concurrent_tests_run_in_parallel(self) -> None:
        all_started = asyncio.Event()
        count = 0
        lock = asyncio.Lock()

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_sleep_1(self) -> None:
                nonlocal count
                async with lock:
                    count += 1
                    if count >= 3:
                        all_started.set()
                # If tests run concurrently all three arrive before
                # the event is set; if sequential this times out.
                await asyncio.wait_for(all_started.wait(), timeout=5.0)

            async def test_sleep_2(self) -> None:
                nonlocal count
                async with lock:
                    count += 1
                    if count >= 3:
                        all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5.0)

            async def test_sleep_3(self) -> None:
                nonlocal count
                async with lock:
                    count += 1
                    if count >= 3:
                        all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5.0)

        result = await _run(_Inner)

        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 3)

    async def test_sequential_tests_run_in_order(self) -> None:
        log: list[str] = []

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_sleep_1(self) -> None:
                log.append("1_start")
                await asyncio.sleep(0)
                log.append("1_end")

            async def test_sleep_2(self) -> None:
                log.append("2_start")
                await asyncio.sleep(0)
                log.append("2_end")

            async def test_sleep_3(self) -> None:
                log.append("3_start")
                await asyncio.sleep(0)
                log.append("3_end")

        result = await _run(_Inner)

        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 3)
        # Sequential: each test finishes before the next starts.
        self.assertEqual(
            log,
            ["1_start", "1_end", "2_start", "2_end", "3_start", "3_end"],
        )

    async def test_different_classes_run_concurrently(self) -> None:
        a_started = asyncio.Event()
        b_started = asyncio.Event()

        class _A(AsyncTestCase):
            __test__ = False

            async def test_a(self) -> None:
                a_started.set()
                # Wait for _B to start; if classes run sequentially this
                # will time out because test_b never gets a chance to run.
                await asyncio.wait_for(b_started.wait(), timeout=5.0)

        class _B(AsyncTestCase):
            __test__ = False

            async def test_b(self) -> None:
                b_started.set()
                await asyncio.wait_for(a_started.wait(), timeout=5.0)

        result = await _run(_A, _B)

        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)

    async def test_concurrent_tests_interleave(self) -> None:
        log: list[str] = []
        lock = asyncio.Lock()

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_a(self) -> None:
                async with lock:
                    log.append("a_start")
                await asyncio.sleep(0)
                async with lock:
                    log.append("a_end")

            async def test_b(self) -> None:
                async with lock:
                    log.append("b_start")
                await asyncio.sleep(0)
                async with lock:
                    log.append("b_end")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

        starts = [i for i, v in enumerate(log) if v.endswith("_start")]
        ends = [i for i, v in enumerate(log) if v.endswith("_end")]
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(ends), 2)
        # Second start must come before first end (proves interleaving)
        self.assertLess(starts[1], ends[0])

    async def test_sequential_no_interleaving(self) -> None:
        log: list[str] = []

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_seq_1(self) -> None:
                log.append("seq_1_start")
                await asyncio.sleep(0)
                log.append("seq_1_end")

            async def test_seq_2(self) -> None:
                log.append("seq_2_start")
                await asyncio.sleep(0)
                log.append("seq_2_end")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(
            log,
            [
                "seq_1_start",
                "seq_1_end",
                "seq_2_start",
                "seq_2_end",
            ],
        )


# ===================================================================== #
#  5. Max concurrency (semaphore) enforcement
# ===================================================================== #


class TestMaxConcurrency(AsyncTestCase, concurrent=True):
    async def test_concurrency_limited_by_semaphore(self) -> None:
        peak = 0
        current = 0
        lock = asyncio.Lock()

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def _track(self, _all_arrived: asyncio.Event, _expected: int) -> None:
                nonlocal peak, current
                async with lock:
                    current += 1
                    if current > peak:
                        peak = current
                    if current >= _expected:
                        _all_arrived.set()
                await _all_arrived.wait()
                async with lock:
                    current -= 1

            async def test_1(self) -> None:
                await self._track(_arrived, 2)

            async def test_2(self) -> None:
                await self._track(_arrived, 2)

            async def test_3(self) -> None:
                await self._track(_arrived, 2)

            async def test_4(self) -> None:
                await self._track(_arrived, 2)

        _arrived = asyncio.Event()
        result = await _run(_Inner, max_concurrency=2)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 4)
        self.assertLessEqual(peak, 2)

    async def test_unlimited_concurrency(self) -> None:
        peak = 0
        current = 0
        lock = asyncio.Lock()

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def _track(self, _all_arrived: asyncio.Event, _expected: int) -> None:
                nonlocal peak, current
                async with lock:
                    current += 1
                    if current > peak:
                        peak = current
                    if current >= _expected:
                        _all_arrived.set()
                await _all_arrived.wait()
                async with lock:
                    current -= 1

            async def test_1(self) -> None:
                await self._track(_arrived, 4)

            async def test_2(self) -> None:
                await self._track(_arrived, 4)

            async def test_3(self) -> None:
                await self._track(_arrived, 4)

            async def test_4(self) -> None:
                await self._track(_arrived, 4)

        _arrived = asyncio.Event()
        result = await _run(_Inner, max_concurrency=None)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 4)
        self.assertGreaterEqual(peak, 3)


# ===================================================================== #
#  6. Skip support
# ===================================================================== #


class TestSkip(AsyncTestCase, concurrent=True):
    async def test_skip_programmatic(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_skip_me(self) -> None:
                self.skipTest("runtime skip")
                raise AssertionError("should never run")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].message, "runtime skip")

    async def test_skip_in_setup_class(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                raise SkipTest("class not ready")

            async def test_a(self) -> None:
                pass

            async def test_b(self) -> None:
                pass

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)
        self.assertEqual(len(result.skipped), 2)

    async def test_skip_in_setup(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def setUp(self) -> None:
                raise SkipTest("setUp skip")

            async def test_a(self) -> None:
                raise AssertionError("should not run")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        self.assertEqual(len(result.skipped), 1)


# ===================================================================== #
#  7. Assertion methods
# ===================================================================== #


class TestAssertions(AsyncTestCase, concurrent=True):
    async def test_all_assertions_pass(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_assert_true(self) -> None:
                self.assertTrue(True)

            async def test_assert_false(self) -> None:
                self.assertFalse(False)

            async def test_assert_equal(self) -> None:
                self.assertEqual(1, 1)

            async def test_assert_not_equal(self) -> None:
                self.assertNotEqual(1, 2)

            async def test_assert_is(self) -> None:
                s = object()
                self.assertIs(s, s)

            async def test_assert_is_not(self) -> None:
                self.assertIsNot(object(), object())

            async def test_assert_is_none(self) -> None:
                self.assertIsNone(None)

            async def test_assert_is_not_none(self) -> None:
                self.assertIsNotNone(42)

            async def test_assert_in(self) -> None:
                self.assertIn(2, [1, 2, 3])

            async def test_assert_not_in(self) -> None:
                self.assertNotIn(4, [1, 2, 3])

            async def test_assert_is_instance(self) -> None:
                self.assertIsInstance(42, int)

            async def test_assert_is_instance_tuple(self) -> None:
                self.assertIsInstance(42, (int, str))
                self.assertIsInstance("hello", (int, str))

            async def test_assert_is_instance_returns_true(self) -> None:
                result = self.assertIsInstance(42, int)
                self.assertTrue(result)

            async def test_assert_is_instance_returns_true_tuple(self) -> None:
                result = self.assertIsInstance("hello", (int, str))
                self.assertTrue(result)

            async def test_assert_is_not_instance(self) -> None:
                self.assertIsNotInstance("hello", int)

            async def test_assert_is_not_instance_tuple(self) -> None:
                self.assertIsNotInstance(42, (str, list))

            async def test_assert_greater(self) -> None:
                self.assertGreater(2, 1)

            async def test_assert_greater_equal(self) -> None:
                self.assertGreaterEqual(2, 2)

            async def test_assert_less(self) -> None:
                self.assertLess(1, 2)

            async def test_assert_less_equal(self) -> None:
                self.assertLessEqual(2, 2)

            async def test_assert_greater_time(self) -> None:
                start = time.time()
                self.assertGreater(start + 1.0, start)

            async def test_assert_greater_equal_time(self) -> None:
                now = time.time()
                self.assertGreaterEqual(now, now)

            async def test_assert_less_time(self) -> None:
                start = time.time()
                self.assertLess(start, start + 1.0)

            async def test_assert_less_equal_time(self) -> None:
                now = time.time()
                self.assertLessEqual(now, now)

            async def test_assert_almost_equal(self) -> None:
                self.assertAlmostEqual(1.00000001, 1.00000002)

            async def test_assert_raises(self) -> None:
                with self.assertRaises(ValueError):
                    raise ValueError("boom")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful, result.format_report(verbosity=2))
        self.assertEqual(result.tests_run, 26)

    async def test_assertion_failures_are_failures(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_assert_true_fails(self) -> None:
                self.assertTrue(False)

            async def test_assert_equal_fails(self) -> None:
                self.assertEqual(1, 2)

            async def test_assert_raises_no_exception(self) -> None:
                with self.assertRaises(ValueError):
                    pass

            async def test_fail(self) -> None:
                self.fail("explicit failure")

            async def test_assert_is_instance_fails(self) -> None:
                self.assertIsInstance("hello", int)

            async def test_assert_is_instance_tuple_fails(self) -> None:
                self.assertIsInstance(3.14, (int, str))

            async def test_assert_is_not_instance_fails(self) -> None:
                self.assertIsNotInstance(42, int)

            async def test_assert_is_not_instance_tuple_fails(self) -> None:
                self.assertIsNotInstance("hello", (int, str))

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(result.tests_run, 8)
        self.assertEqual(len(result.failures), 8)

    async def test_assert_is_instance_custom_message(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_custom_msg(self) -> None:
                self.assertIsInstance("hello", int, "expected an int")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("expected an int", result.failures[0].traceback)

    async def test_assert_is_not_instance_custom_message(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_custom_msg(self) -> None:
                self.assertIsNotInstance(42, int, "should not be an int")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("should not be an int", result.failures[0].traceback)


# ===================================================================== #
#  8. __init_subclass__ concurrent configuration
# ===================================================================== #


class TestConcurrentConfig(AsyncTestCase, concurrent=True):
    async def test_default_is_sequential(self) -> None:
        class _Check(AsyncTestCase):
            __test__ = False

        self.assertFalse(_Check.__concurrent__)

    async def test_concurrent_false(self) -> None:
        class _Check(AsyncTestCase):
            __test__ = False

        self.assertFalse(_Check.__concurrent__)

    async def test_concurrent_true(self) -> None:
        class _Check(AsyncTestCase, concurrent=True):
            __test__ = False

        self.assertTrue(_Check.__concurrent__)

    async def test_concurrent_inherited(self) -> None:
        class _Base(AsyncTestCase):
            __test__ = False

        class _Child(_Base):
            __test__ = False

        self.assertFalse(_Child.__concurrent__)

    async def test_concurrent_overridden_in_child(self) -> None:
        class _Base(AsyncTestCase):
            __test__ = False

        class _Child(_Base, concurrent=True):
            __test__ = False

        self.assertFalse(_Base.__concurrent__)
        self.assertTrue(_Child.__concurrent__)


# ===================================================================== #
#  9. Test method collection
# ===================================================================== #


class TestCollection(AsyncTestCase, concurrent=True):
    async def test_collect_test_methods(self) -> None:
        class _Example(AsyncTestCase):
            __test__ = False

            async def test_b(self) -> None:
                pass

            async def test_a(self) -> None:
                pass

            async def helper(self) -> None:
                """Not a test – no test_ prefix."""

            def test_sync(self) -> None:
                """Not a test – not async."""

        methods = _collect_test_methods(_Example)
        self.assertEqual(methods, ["test_a", "test_b"])


# ===================================================================== #
#  10. AsyncTestSuite
# ===================================================================== #


class TestSuite(AsyncTestCase, concurrent=True):
    async def test_explicit_methods(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_one(self) -> None:
                pass

            async def test_two(self) -> None:
                pass

        suite = AsyncTestSuite()
        suite.add_class(_Inner, methods=["test_one"])
        runner = AsyncTestRunner(verbosity=0)
        result = await runner.run_suite_async(suite)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        self.assertIn("test_one", result.passed[0].test_id)

    async def test_auto_discover_methods(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_x(self) -> None:
                pass

            async def test_y(self) -> None:
                pass

        suite = AsyncTestSuite()
        suite.add_class(_Inner)
        entries = suite.entries
        self.assertEqual(len(entries), 1)
        cls, methods = entries[0]
        self.assertIs(cls, _Inner)
        self.assertEqual(sorted(methods), ["test_x", "test_y"])

    async def test_empty_class(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

        suite = AsyncTestSuite()
        suite.add_class(_Inner)
        runner = AsyncTestRunner(verbosity=0)
        result = await runner.run_suite_async(suite)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 0)


# ===================================================================== #
#  11. Result reporting
# ===================================================================== #


class TestReporting(AsyncTestCase, concurrent=True):
    async def test_report_success(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_a(self) -> None:
                pass

            async def test_b(self) -> None:
                pass

        result = await _run(_Inner)
        report = result.format_report(verbosity=1)
        self.assertIn("OK", report)
        self.assertIn("Ran 2 test(s)", report)

    async def test_report_failure(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_fail(self) -> None:
                raise AssertionError("bad")

        result = await _run(_Inner)
        report = result.format_report(verbosity=1)
        self.assertIn("FAILED", report)
        self.assertIn("failures=1", report)

    async def test_report_verbose(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_ok(self) -> None:
                pass

        result = await _run(_Inner)
        report = result.format_report(verbosity=2)
        self.assertIn("✓", report)
        self.assertIn("test_ok", report)

    async def test_report_quiet(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_ok(self) -> None:
                pass

        result = await _run(_Inner)
        report = result.format_report(verbosity=0)
        self.assertIn("Ran 1 test(s)", report)
        self.assertNotIn("✓", report)


# ===================================================================== #
#  12. Duration tracking
# ===================================================================== #


class TestDuration(AsyncTestCase, concurrent=True):
    async def test_duration_tracked(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_sleeper(self) -> None:
                pass

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertGreater(result.passed[0].duration, 0)
        self.assertGreater(result.total_duration, 0)


# ===================================================================== #
#  13. Test isolation: each test gets its own instance
# ===================================================================== #


class TestIsolation(AsyncTestCase, concurrent=True):
    async def test_each_test_gets_own_instance(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def setUp(self) -> None:
                self.value = 0

            async def test_mutate_1(self) -> None:
                self.value += 1
                self.assertEqual(self.value, 1)

            async def test_mutate_2(self) -> None:
                self.value += 10
                self.assertEqual(self.value, 10)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful, result.format_report(verbosity=2))
        self.assertEqual(result.tests_run, 2)


# ===================================================================== #
#  14. Shared class state (ClassVar) across concurrent tests
# ===================================================================== #


class TestSharedState(AsyncTestCase, concurrent=True):
    async def test_class_vars_shared(self) -> None:
        counter = 0
        lock = asyncio.Lock()

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_inc_1(self) -> None:
                nonlocal counter
                async with lock:
                    counter += 1

            async def test_inc_2(self) -> None:
                nonlocal counter
                async with lock:
                    counter += 1

            async def test_inc_3(self) -> None:
                nonlocal counter
                async with lock:
                    counter += 1

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(counter, 3)


# ===================================================================== #
#  15. Test id / repr / str
# ===================================================================== #


class TestRepresentation(AsyncTestCase, concurrent=True):
    async def test_id(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False

        instance = _Inner("test_something")
        self.assertIn("test_something", instance.id())
        self.assertIn("_Inner", instance.id())

    async def test_repr(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False

        instance = _Inner("test_something")
        self.assertIn("test_something", repr(instance))
        self.assertIn("_Inner", repr(instance))

    async def test_str(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False

        instance = _Inner("test_something")
        self.assertIn("test_something", str(instance))
        self.assertIn("_Inner", str(instance))


# ===================================================================== #
#  16. Filesystem discovery
# ===================================================================== #


class TestDiscovery(AsyncTestCase, concurrent=True):
    async def test_discover_from_directory(self) -> None:
        sample_dir = Path(__file__).parent / "_sample_discover"
        if not sample_dir.is_dir():
            self.skipTest(f"sample directory missing: {sample_dir}")

        suite = discover(
            start_dir=str(sample_dir),
            pattern="test_*.py",
            top_level_dir=str(Path(__file__).resolve().parents[2]),
        )

        entries = suite.entries
        class_names = sorted(cls.__name__ for cls, _methods in entries)
        self.assertIn("SamplePassingTests", class_names)
        self.assertIn("SampleSequentialTests", class_names)
        self.assertNotIn("_InternalHelper", class_names)

        runner = AsyncTestRunner(verbosity=0)
        result = await runner.run_suite_async(suite)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 4)

    async def test_discover_respects_pattern(self) -> None:
        sample_dir = Path(__file__).parent / "_sample_discover"
        if not sample_dir.is_dir():
            self.skipTest(f"sample directory missing: {sample_dir}")

        suite = discover(
            start_dir=str(sample_dir),
            pattern="check_*.py",
            top_level_dir=str(Path(__file__).resolve().parents[2]),
        )
        self.assertEqual(len(suite.entries), 0)

    async def test_discover_nonexistent_dir(self) -> None:
        suite = discover("/tmp/_barrage_no_such_dir_12345")
        self.assertEqual(len(suite.entries), 0)

    async def test_discover_module(self) -> None:
        this_module = sys.modules[__name__]

        suite = discover_module(this_module)
        class_names = [cls.__name__ for cls, _methods in suite.entries]
        # This module's own test classes should be discoverable
        self.assertIn("TestBasicExecution", class_names)
        self.assertIn("TestConcurrency", class_names)


# ===================================================================== #
#  16b. resolve_tests() path-spec resolution
# ===================================================================== #


class TestResolveTests(AsyncTestCase, concurrent=True):
    """Tests for the ``resolve_tests()`` path-spec resolution function."""

    _sample_dir: str
    _sample_file: str
    _top_dir: str

    @classmethod
    async def setUpClass(cls) -> None:
        cls._sample_dir = str(Path(__file__).parent / "_sample_discover")
        cls._sample_file = str(Path(__file__).parent / "_sample_discover" / "test_sample.py")
        cls._top_dir = str(Path(__file__).resolve().parents[1])

    # ── directory ─────────────────────────────────────────────────────

    async def test_resolve_directory(self) -> None:
        """A bare directory discovers all test classes inside it."""
        suite = resolve_tests([self._sample_dir])
        class_names = sorted(cls.__name__ for cls, _ in suite.entries)
        self.assertIn("SamplePassingTests", class_names)
        self.assertIn("SampleSequentialTests", class_names)
        self.assertNotIn("_InternalHelper", class_names)

        total_methods = sum(len(m) for _, m in suite.entries)
        self.assertEqual(total_methods, 4)

    async def test_resolve_directory_with_pattern(self) -> None:
        """The pattern parameter filters which files are picked up."""
        suite = resolve_tests([self._sample_dir], pattern="check_*.py")
        self.assertEqual(len(suite.entries), 0)

    # ── single file ───────────────────────────────────────────────────

    async def test_resolve_file(self) -> None:
        """A bare file path discovers all test classes in that file."""
        suite = resolve_tests([self._sample_file])
        class_names = sorted(cls.__name__ for cls, _ in suite.entries)
        self.assertIn("SamplePassingTests", class_names)
        self.assertIn("SampleSequentialTests", class_names)
        self.assertNotIn("_InternalHelper", class_names)

    async def test_resolve_file_runs_successfully(self) -> None:
        """Tests discovered from a file can actually be executed."""
        suite = resolve_tests([self._sample_file])
        runner = AsyncTestRunner(verbosity=0)
        result = await runner.run_suite_async(suite)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 4)

    # ── file::ClassName ───────────────────────────────────────────────

    async def test_resolve_file_class(self) -> None:
        """``file::ClassName`` discovers only the named class."""
        suite = resolve_tests([f"{self._sample_file}::SamplePassingTests"])
        self.assertEqual(len(suite.entries), 1)
        cls, methods = suite.entries[0]
        self.assertEqual(cls.__name__, "SamplePassingTests")
        self.assertEqual(sorted(methods), ["test_add", "test_string"])

    async def test_resolve_file_class_other(self) -> None:
        """Selecting a different class works the same way."""
        suite = resolve_tests([f"{self._sample_file}::SampleSequentialTests"])
        self.assertEqual(len(suite.entries), 1)
        cls, methods = suite.entries[0]
        self.assertEqual(cls.__name__, "SampleSequentialTests")
        self.assertEqual(sorted(methods), ["test_seq_a", "test_seq_b"])

    async def test_resolve_file_class_runs_successfully(self) -> None:
        """A class-filtered suite can actually be executed."""
        suite = resolve_tests([f"{self._sample_file}::SamplePassingTests"])
        runner = AsyncTestRunner(verbosity=0)
        result = await runner.run_suite_async(suite)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)

    # ── file::ClassName::method ───────────────────────────────────────

    async def test_resolve_file_class_method(self) -> None:
        """``file::Class::method`` discovers only that single method."""
        suite = resolve_tests([f"{self._sample_file}::SamplePassingTests::test_add"])
        self.assertEqual(len(suite.entries), 1)
        cls, methods = suite.entries[0]
        self.assertEqual(cls.__name__, "SamplePassingTests")
        self.assertEqual(methods, ["test_add"])

    async def test_resolve_file_class_method_runs_successfully(self) -> None:
        """A method-filtered suite can actually be executed."""
        suite = resolve_tests([f"{self._sample_file}::SamplePassingTests::test_add"])
        runner = AsyncTestRunner(verbosity=0)
        result = await runner.run_suite_async(suite)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)

    # ── multiple paths ────────────────────────────────────────────────

    async def test_resolve_multiple_paths(self) -> None:
        """Multiple path specs are combined into a single suite."""
        suite = resolve_tests(
            [
                f"{self._sample_file}::SamplePassingTests::test_add",
                f"{self._sample_file}::SampleSequentialTests::test_seq_a",
            ]
        )
        self.assertEqual(len(suite.entries), 2)
        all_methods = [m for _, methods in suite.entries for m in methods]
        self.assertIn("test_add", all_methods)
        self.assertIn("test_seq_a", all_methods)
        self.assertEqual(len(all_methods), 2)

    async def test_resolve_multiple_paths_runs_successfully(self) -> None:
        """A suite built from multiple specs can actually be executed."""
        suite = resolve_tests(
            [
                f"{self._sample_file}::SamplePassingTests",
                f"{self._sample_file}::SampleSequentialTests::test_seq_b",
            ]
        )
        runner = AsyncTestRunner(verbosity=0)
        result = await runner.run_suite_async(suite)
        self.assertTrue(result.was_successful)
        # SamplePassingTests has 2 methods + 1 from SampleSequentialTests
        self.assertEqual(result.tests_run, 3)

    # ── error cases ───────────────────────────────────────────────────

    async def test_resolve_nonexistent_path(self) -> None:
        """A path that does not exist causes SystemExit(2)."""
        with self.assertRaises(SystemExit) as ctx:
            resolve_tests(["/tmp/_barrage_no_such_file_99999.py"])
        assert isinstance(ctx.exception, SystemExit)
        self.assertEqual(ctx.exception.code, 2)

    async def test_resolve_nonexistent_class(self) -> None:
        """``file::NoSuchClass`` causes SystemExit(2)."""
        with self.assertRaises(SystemExit) as ctx:
            resolve_tests([f"{self._sample_file}::NoSuchClass"])
        assert isinstance(ctx.exception, SystemExit)
        self.assertEqual(ctx.exception.code, 2)

    async def test_resolve_nonexistent_method(self) -> None:
        """``file::Class::no_such_method`` causes SystemExit(2)."""
        with self.assertRaises(SystemExit) as ctx:
            resolve_tests([f"{self._sample_file}::SamplePassingTests::no_such_method"])
        assert isinstance(ctx.exception, SystemExit)
        self.assertEqual(ctx.exception.code, 2)

    async def test_resolve_class_filter_on_directory(self) -> None:
        """``directory::ClassName`` is not allowed and causes SystemExit(2)."""
        with self.assertRaises(SystemExit) as ctx:
            resolve_tests([f"{self._sample_dir}::SomeClass"])
        assert isinstance(ctx.exception, SystemExit)
        self.assertEqual(ctx.exception.code, 2)

    async def test_resolve_too_many_separators(self) -> None:
        """More than two ``::`` separators causes SystemExit(2)."""
        with self.assertRaises(SystemExit) as ctx:
            resolve_tests([f"{self._sample_file}::SamplePassingTests::test_add::extra"])
        assert isinstance(ctx.exception, SystemExit)
        self.assertEqual(ctx.exception.code, 2)

    async def test_resolve_explicit_test_false_class(self) -> None:
        """``__test__ = False`` classes are excluded from directory discovery
        but *can* be selected explicitly by name."""
        suite = resolve_tests([f"{self._sample_file}::_InternalHelper"])
        self.assertEqual(len(suite.entries), 1)
        cls, methods = suite.entries[0]
        self.assertEqual(cls.__name__, "_InternalHelper")
        self.assertEqual(methods, ["test_should_not_be_found"])

    async def test_resolve_non_test_class(self) -> None:
        """Selecting a name that is not an AsyncTestCase subclass fails."""
        # ``asyncio`` is imported at module level in test_sample.py, so
        # it exists as an attribute but is a module, not an
        # AsyncTestCase subclass.
        with self.assertRaises(SystemExit) as ctx:
            resolve_tests([f"{self._sample_file}::asyncio"])
        assert isinstance(ctx.exception, SystemExit)
        self.assertEqual(ctx.exception.code, 2)


# ===================================================================== #
#  17. CLI (__main__) via subprocess
# ===================================================================== #


class TestCLI(AsyncTestCase, concurrent=True):
    async def test_main_discover_sample_dir(self) -> None:
        sample_dir = str(Path(__file__).parent / "_sample_discover")
        top_dir = str(Path(__file__).resolve().parents[1])

        if not Path(sample_dir).is_dir():
            self.skipTest("sample directory missing")

        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", sample_dir, "-v"],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        stdout_text = result.stdout.decode()
        self.assertEqual(result.returncode, 0, f"stdout:\n{stdout_text}\nstderr:\n{result.stderr.decode()}")
        self.assertIn("Ran 4 test(s)", stdout_text)
        self.assertIn("OK", stdout_text)

    async def test_main_returns_nonzero_on_no_tests(self) -> None:
        top_dir = str(Path(__file__).resolve().parents[1])

        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", "/tmp/_barrage_no_such_dir_99999"],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        self.assertEqual(result.returncode, 2)

    async def test_main_file_with_class(self) -> None:
        """The CLI accepts ``file::Class`` to run a single class."""
        sample_file = str(Path(__file__).parent / "_sample_discover" / "test_sample.py")
        top_dir = str(Path(__file__).resolve().parents[1])

        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", f"{sample_file}::SamplePassingTests", "-v"],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        stdout_text = result.stdout.decode()
        self.assertEqual(result.returncode, 0, f"stdout:\n{stdout_text}\nstderr:\n{result.stderr.decode()}")
        self.assertIn("Ran 2 test(s)", stdout_text)
        self.assertIn("OK", stdout_text)

    async def test_main_file_with_class_and_method(self) -> None:
        """The CLI accepts ``file::Class::method`` to run a single test."""
        sample_file = str(Path(__file__).parent / "_sample_discover" / "test_sample.py")
        top_dir = str(Path(__file__).resolve().parents[1])

        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", f"{sample_file}::SamplePassingTests::test_add", "-v"],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        stdout_text = result.stdout.decode()
        self.assertEqual(result.returncode, 0, f"stdout:\n{stdout_text}\nstderr:\n{result.stderr.decode()}")
        self.assertIn("Ran 1 test(s)", stdout_text)
        self.assertIn("test_add", stdout_text)
        self.assertIn("OK", stdout_text)

    async def test_main_multiple_paths(self) -> None:
        """The CLI accepts multiple path specs."""
        sample_file = str(Path(__file__).parent / "_sample_discover" / "test_sample.py")
        top_dir = str(Path(__file__).resolve().parents[1])

        async with asyncio.timeout(30):
            result = await run(
                [
                    sys.executable,
                    "-m",
                    "barrage",
                    f"{sample_file}::SamplePassingTests::test_add",
                    f"{sample_file}::SampleSequentialTests::test_seq_a",
                    "-v",
                ],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        stdout_text = result.stdout.decode()
        self.assertEqual(result.returncode, 0, f"stdout:\n{stdout_text}\nstderr:\n{result.stderr.decode()}")
        self.assertIn("Ran 2 test(s)", stdout_text)
        self.assertIn("OK", stdout_text)

    async def test_main_nonexistent_class_exits_2(self) -> None:
        """``file::BadClass`` via CLI exits with code 2."""
        sample_file = str(Path(__file__).parent / "_sample_discover" / "test_sample.py")
        top_dir = str(Path(__file__).resolve().parents[1])

        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", f"{sample_file}::NoSuchClass"],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("NoSuchClass", result.stderr.decode())

    async def test_main_nonexistent_method_exits_2(self) -> None:
        """``file::Class::bad_method`` via CLI exits with code 2."""
        sample_file = str(Path(__file__).parent / "_sample_discover" / "test_sample.py")
        top_dir = str(Path(__file__).resolve().parents[1])

        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", f"{sample_file}::SamplePassingTests::no_such_method"],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("no_such_method", result.stderr.decode())


# ===================================================================== #
#  18. Stress test: many tests with limited concurrency
# ===================================================================== #


class TestStress(AsyncTestCase, concurrent=True):
    async def test_many_tests_limited_concurrency(self) -> None:
        completed = 0
        lock = asyncio.Lock()

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

        # Dynamically add 20 test methods
        for i in range(20):

            async def _make_test(self: _Inner, idx: int = i) -> None:
                nonlocal completed
                await asyncio.sleep(0.01)
                async with lock:
                    completed += 1

            setattr(_Inner, f"test_stress_{i:02d}", _make_test)

        result = await _run(_Inner, max_concurrency=5)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 20)
        self.assertEqual(completed, 20)

    async def test_many_tests_unlimited_concurrency(self) -> None:
        completed = 0
        lock = asyncio.Lock()

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

        for i in range(20):

            async def _make_test(self: _Inner, idx: int = i) -> None:
                nonlocal completed
                await asyncio.sleep(0.01)
                async with lock:
                    completed += 1

            setattr(_Inner, f"test_stress_{i:02d}", _make_test)

        result = await _run(_Inner, max_concurrency=None)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 20)
        self.assertEqual(completed, 20)


# ===================================================================== #
#  19. Output capture
# ===================================================================== #


class TestOutputCapture(AsyncTestCase):
    async def test_stdout_captured_on_failure(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_fail(self) -> None:
                print("hello from failing test")
                raise AssertionError("deliberate")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("hello from failing test", result.failures[0].stdout)

    async def test_stderr_captured_on_failure(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_fail(self) -> None:
                import sys as _sys

                _sys.stderr.write("debug info on stderr\n")
                raise AssertionError("deliberate")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("debug info on stderr", result.failures[0].stderr)

    async def test_stdout_captured_on_error(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_error(self) -> None:
                print("output before error")
                raise RuntimeError("kaboom")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("output before error", result.errors[0].stdout)

    async def test_stdout_captured_on_pass(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_ok(self) -> None:
                print("chatty passing test")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(len(result.passed), 1)
        self.assertIn("chatty passing test", result.passed[0].stdout)

    async def test_captured_output_in_report_for_failure(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_fail(self) -> None:
                print("visible in report")
                raise AssertionError("bad")

        result = await _run(_Inner)
        report = result.format_report(verbosity=1)
        self.assertIn("Captured stdout:", report)
        self.assertIn("visible in report", report)

    async def test_captured_output_hidden_for_pass_by_default(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_ok(self) -> None:
                print("hidden by default")

        result = await _run(_Inner)
        report = result.format_report(verbosity=1, show_output=False)
        self.assertNotIn("hidden by default", report)

    async def test_captured_output_shown_for_pass_with_show_output(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_ok(self) -> None:
                print("now visible")

        result = await _run(_Inner)
        report = result.format_report(verbosity=1, show_output=True)
        self.assertIn("now visible", report)

    async def test_capture_includes_setup_and_teardown(self) -> None:
        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def setUp(self) -> None:
                print("in setUp")

            async def tearDown(self) -> None:
                print("in tearDown")

            async def test_fail(self) -> None:
                print("in test")
                raise AssertionError("bad")

        result = await _run(_Inner)
        self.assertFalse(result.was_successful)
        stdout = result.failures[0].stdout
        self.assertIn("in setUp", stdout)
        self.assertIn("in test", stdout)
        self.assertIn("in tearDown", stdout)

    async def test_concurrent_capture_isolation(self) -> None:
        """Output from concurrent tests must not leak across tests."""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_a(self) -> None:
                print("output-from-a")
                await asyncio.sleep(0)

            async def test_b(self) -> None:
                print("output-from-b")
                await asyncio.sleep(0)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)

        outputs = {r.test_str: r.stdout for r in result.passed}
        for test_str, stdout in outputs.items():
            if "test_a" in test_str:
                self.assertIn("output-from-a", stdout)
                self.assertNotIn("output-from-b", stdout)
            else:
                self.assertIn("output-from-b", stdout)
                self.assertNotIn("output-from-a", stdout)

    async def test_no_capture_leaks_to_outer(self) -> None:
        """Captured output must not appear in the outer capture buffer."""
        from barrage.runner import _capture_stdout

        # The outer runner's _run_single_test has set _capture_stdout to
        # an outer buffer for *this* test task.  Grab it so we can check
        # that the inner test's output does NOT leak into it.
        outer_buf = _capture_stdout.get(None)
        outer_before = outer_buf.getvalue() if outer_buf else ""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_noisy(self) -> None:
                print("should-be-captured")

        result = await _run(_Inner)

        self.assertTrue(result.was_successful)
        # The inner test's output must have been captured in its own buffer.
        self.assertIn("should-be-captured", result.passed[0].stdout)
        # And it must NOT have leaked into the outer capture buffer.
        outer_after = outer_buf.getvalue() if outer_buf else ""
        outer_new = outer_after[len(outer_before) :]
        self.assertNotIn("should-be-captured", outer_new)

    async def test_stdin_devnull_in_non_interactive(self) -> None:
        """In non-interactive mode, stdin is replaced with /dev/null."""
        captured_line: list[str] = []

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_read_stdin(self) -> None:
                captured_line.append(sys.stdin.readline())

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        # /dev/null yields an empty string on readline (EOF).
        self.assertEqual(captured_line, [""])

    async def test_stdin_available_in_interactive(self) -> None:
        """In interactive mode, stdin is NOT replaced with /dev/null."""
        original_stdin = sys.stdin

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_stdin_is_real(self) -> None:
                # stdin should still be the original stream, not /dev/null.
                self.assertIs(sys.stdin, original_stdin)

        result = await _run(_Inner, interactive=True)
        self.assertTrue(result.was_successful)


# ===================================================================== #
#  20. Interactive mode
# ===================================================================== #


class TestInteractiveMode(AsyncTestCase, concurrent=True):
    async def test_interactive_forces_sequential(self) -> None:
        """Even a concurrent class runs sequentially in interactive mode."""
        log: list[str] = []

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_a(self) -> None:
                log.append("a_start")
                await asyncio.sleep(0)
                log.append("a_end")

            async def test_b(self) -> None:
                log.append("b_start")
                await asyncio.sleep(0)
                log.append("b_end")

        result = await _run(_Inner, interactive=True)
        self.assertTrue(result.was_successful)
        # Sequential means a finishes before b starts
        self.assertEqual(log, ["a_start", "a_end", "b_start", "b_end"])

    async def test_interactive_no_capture(self) -> None:
        """In interactive mode, output is NOT captured (flows to terminal)."""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_ok(self) -> None:
                pass

        stream = io.StringIO()
        result = await _run(_Inner, interactive=True, interactive_stream=stream)
        self.assertTrue(result.was_successful)
        # Interactive status lines were written to the stream…
        self.assertIn("test_ok", stream.getvalue())
        self.assertIn("ok", stream.getvalue())
        # …but stdout/stderr were not captured into the result.
        self.assertEqual(result.passed[0].stdout, "")
        self.assertEqual(result.passed[0].stderr, "")

    async def test_interactive_live_output(self) -> None:
        """Interactive mode writes live per-test status to stderr."""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_ok(self) -> None:
                pass

            async def test_fail(self) -> None:
                raise AssertionError("bad")

        stream = io.StringIO()
        await _run(_Inner, interactive=True, interactive_stream=stream)

        live_output = stream.getvalue()
        self.assertIn("test_ok", live_output)
        self.assertIn("test_fail", live_output)
        self.assertIn("ok", live_output)
        self.assertIn("FAIL", live_output)

    async def test_interactive_classes_sequential(self) -> None:
        """In interactive mode, classes also run sequentially."""
        log: list[str] = []

        class _A(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_a(self) -> None:
                log.append("A")
                await asyncio.sleep(0)
                log.append("A_done")

        class _B(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_b(self) -> None:
                log.append("B")
                await asyncio.sleep(0)
                log.append("B_done")

        result = await _run(_A, _B, interactive=True)
        self.assertTrue(result.was_successful)
        # Classes must not interleave
        self.assertEqual(log, ["A", "A_done", "B", "B_done"])

    async def test_interactive_cli_flag(self) -> None:
        """The CLI ``-i`` flag triggers interactive mode."""
        sample_dir = str(Path(__file__).parent / "_sample_discover")
        top_dir = str(Path(__file__).resolve().parents[1])

        if not Path(sample_dir).is_dir():
            self.skipTest("sample directory missing")

        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", sample_dir, "-i"],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        self.assertEqual(
            result.returncode, 0, f"stdout:\n{result.stdout.decode()}\nstderr:\n{result.stderr.decode()}"
        )
        # In interactive mode, per-test lines go to stderr
        stderr_text = result.stderr.decode()
        self.assertIn("test_add", stderr_text)
        self.assertIn("ok", stderr_text)

    async def test_interactive_setup_class_traceback(self) -> None:
        """Interactive mode shows traceback when setUpClass fails."""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                raise RuntimeError("setUpClass kaboom")

            async def test_a(self) -> None:
                pass

            async def test_b(self) -> None:
                pass

        stream = io.StringIO()
        result = await _run(_Inner, interactive=True, interactive_stream=stream)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.errors), 2)
        live_output = stream.getvalue()
        self.assertIn("setUpClass kaboom", live_output)
        self.assertIn("RuntimeError", live_output)

    async def test_interactive_setup_traceback(self) -> None:
        """Interactive mode shows traceback when setUp fails."""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def setUp(self) -> None:
                raise RuntimeError("setUp exploded")

            async def test_a(self) -> None:
                pass

        stream = io.StringIO()
        result = await _run(_Inner, interactive=True, interactive_stream=stream)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.errors), 1)
        live_output = stream.getvalue()
        self.assertIn("setUp exploded", live_output)
        self.assertIn("RuntimeError", live_output)

    async def test_interactive_teardown_class_traceback(self) -> None:
        """Interactive mode shows traceback when tearDownClass fails."""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            @classmethod
            async def tearDownClass(cls) -> None:
                raise RuntimeError("tearDownClass boom")

            async def test_ok(self) -> None:
                pass

        stream = io.StringIO()
        result = await _run(_Inner, interactive=True, interactive_stream=stream)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.errors), 1)
        live_output = stream.getvalue()
        self.assertIn("tearDownClass boom", live_output)
        self.assertIn("RuntimeError", live_output)

    async def test_interactive_stdin(self) -> None:
        """In interactive mode the user can interact with a running test via stdin."""
        helper = str(Path(__file__).parent / "_interactive_stdin_helper.py")
        top_dir = str(Path(__file__).resolve().parents[1])

        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", helper, "-i"],
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                input=b"hello from outer test\n",
                cwd=top_dir,
                check=False,
            )
        self.assertEqual(
            result.returncode,
            0,
            f"stdout:\n{result.stdout.decode()}\nstderr:\n{result.stderr.decode()}",
        )
        stderr_text = result.stderr.decode()
        self.assertIn("test_read_from_stdin", stderr_text)
        self.assertIn("ok", stderr_text)


# ===================================================================== #
#  Failfast (-x)
# ===================================================================== #


class TestFailfast(AsyncTestCase):
    """Tests for the ``-x`` / ``--failfast`` stop-on-first-failure feature."""

    async def test_failfast_stops_after_first_failure(self) -> None:
        """With failfast, tests after the first failure are not executed."""

        class _Inner(AsyncTestCase):
            executed: ClassVar[list[str]] = []

            async def test_a_pass(self) -> None:
                _Inner.executed.append("a")

            async def test_b_fail(self) -> None:
                _Inner.executed.append("b")
                self.fail("boom")

            async def test_c_never(self) -> None:
                _Inner.executed.append("c")

        _Inner.executed = []
        result = await _run(_Inner, failfast=True)

        # test_a_pass and test_b_fail ran (sorted order), test_c_never did not
        self.assertIn("a", _Inner.executed)
        self.assertIn("b", _Inner.executed)
        self.assertNotIn("c", _Inner.executed)
        self.assertEqual(len(result.failures), 1)

    async def test_failfast_stops_after_first_error(self) -> None:
        """Failfast also triggers on errors (not just assertion failures)."""

        class _Inner(AsyncTestCase):
            executed: ClassVar[list[str]] = []

            async def test_a_pass(self) -> None:
                _Inner.executed.append("a")

            async def test_b_error(self) -> None:
                _Inner.executed.append("b")
                raise RuntimeError("kaboom")

            async def test_c_never(self) -> None:
                _Inner.executed.append("c")

        _Inner.executed = []
        result = await _run(_Inner, failfast=True)

        self.assertIn("a", _Inner.executed)
        self.assertIn("b", _Inner.executed)
        self.assertNotIn("c", _Inner.executed)
        self.assertEqual(len(result.errors), 1)

    async def test_failfast_skips_subsequent_classes(self) -> None:
        """With failfast, classes after the failing one are skipped."""

        class _First(AsyncTestCase):
            async def test_fail(self) -> None:
                self.fail("first fails")

        class _Second(AsyncTestCase):
            executed: ClassVar[bool] = False

            async def test_never(self) -> None:
                _Second.executed = True

        _Second.executed = False
        result = await _run(_First, _Second, failfast=True, interactive=True)

        self.assertEqual(len(result.failures), 1)
        self.assertFalse(_Second.executed)

    async def test_no_failfast_runs_all(self) -> None:
        """Without failfast, all tests run even after a failure."""

        class _Inner(AsyncTestCase):
            executed: ClassVar[list[str]] = []

            async def test_a_pass(self) -> None:
                _Inner.executed.append("a")

            async def test_b_fail(self) -> None:
                _Inner.executed.append("b")
                self.fail("boom")

            async def test_c_also_runs(self) -> None:
                _Inner.executed.append("c")

        _Inner.executed = []
        result = await _run(_Inner, failfast=False)

        self.assertIn("a", _Inner.executed)
        self.assertIn("b", _Inner.executed)
        self.assertIn("c", _Inner.executed)
        self.assertEqual(len(result.failures), 1)

    async def test_failfast_with_setup_error(self) -> None:
        """Failfast triggers when setUp raises."""

        class _Inner(AsyncTestCase):
            executed: ClassVar[list[str]] = []

            async def setUp(self) -> None:
                raise RuntimeError("setUp exploded")

            async def test_a(self) -> None:
                _Inner.executed.append("a")

            async def test_b(self) -> None:
                _Inner.executed.append("b")

        _Inner.executed = []
        result = await _run(_Inner, failfast=True)

        # setUp error should record an error for the first test and stop
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(_Inner.executed), 0)

    async def test_failfast_cli_flag(self) -> None:
        """The CLI ``-x`` flag triggers failfast mode."""
        sample_dir = str(Path(__file__).parent / "_sample_discover")
        top_dir = str(Path(__file__).resolve().parents[1])

        if not Path(sample_dir).is_dir():
            self.skipTest("sample directory missing")

        # The sample tests should all pass, so -x has no effect — but
        # we verify the flag is accepted without error.
        async with asyncio.timeout(30):
            result = await run(
                [sys.executable, "-m", "barrage", sample_dir, "-x"],
                stdout=PIPE,
                stderr=PIPE,
                cwd=top_dir,
                check=False,
            )
        self.assertEqual(
            result.returncode, 0, f"stdout:\n{result.stdout.decode()}\nstderr:\n{result.stderr.decode()}"
        )

    async def test_failfast_no_task_leaks_concurrent(self) -> None:
        """Failfast with concurrent classes/methods leaves no orphaned tasks."""

        class _Early(AsyncTestCase, concurrent=True):
            """Concurrent class whose first method (sorted) fails fast."""

            async def test_a_fail(self) -> None:
                self.fail("early failure")

            async def test_b_slow(self) -> None:
                await asyncio.sleep(0.2)

            async def test_c_slow(self) -> None:
                await asyncio.sleep(0.2)

        class _Late(AsyncTestCase, concurrent=True):
            executed: ClassVar[bool] = False

            async def test_z(self) -> None:
                _Late.executed = True

        _Late.executed = False

        # Snapshot tasks *before* the inner run.
        tasks_before = asyncio.all_tasks()

        result = await _run(_Early, _Late, failfast=True)

        # Snapshot tasks *after* — any new tasks still pending would be
        # a leak.  Only consider tasks created by the runner (named
        # "class:*" or "ClassName.*") to avoid false positives from
        # unrelated infrastructure like subprocess read streams.
        tasks_after = asyncio.all_tasks()
        leaked = {
            t
            for t in (tasks_after - tasks_before)
            if ("_Early" in (t.get_name() or "") or "_Late" in (t.get_name() or ""))
        }
        self.assertEqual(
            len(leaked),
            0,
            f"Leaked tasks after failfast run: {leaked}",
        )

        # At least the failure was recorded.
        self.assertGreaterEqual(len(result.failures) + len(result.errors), 1)


class TestMonitored(AsyncTestCase):
    """Meta-tests for MonitoredTestCase using AsyncTestRunner."""

    async def test_monitored_concurrent_crash_cancels_all(self) -> None:
        """When concurrent=True and a background task crashes, all running
        test tasks are cancelled and skipped."""

        class ConcurrentCrash(MonitoredTestCase, concurrent=True):
            __test__ = False
            test_a_ran: ClassVar[bool] = False
            test_b_ran: ClassVar[bool] = False
            crash_fut: ClassVar[asyncio.Future[int]]

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()
                cls.crash_fut = asyncio.Future()
                cls.create_task(
                    asyncio.wait_for(cls.crash_fut, timeout=None),
                    name="shared-component",
                )

            async def test_a_sleeps(self) -> None:
                ConcurrentCrash.test_a_ran = True
                # Trigger the crash while both tests are in-flight
                self.crash_fut.set_exception(RuntimeError("concurrent boom"))
                await asyncio.sleep(10)

            async def test_b_sleeps(self) -> None:
                ConcurrentCrash.test_b_ran = True
                await asyncio.sleep(10)

        ConcurrentCrash.test_a_ran = False
        ConcurrentCrash.test_b_ran = False
        result = await _run(ConcurrentCrash)
        # Both tests should have started (concurrent), then been cancelled
        self.assertTrue(ConcurrentCrash.test_a_ran)
        self.assertTrue(ConcurrentCrash.test_b_ran)
        # Both cancelled → skipped
        self.assertEqual(len(result.skipped), 2)
        # tearDownClass raises ExceptionGroup → 1 error
        self.assertEqual(len(result.errors), 1)
        self.assertIn("concurrent boom", result.errors[0].traceback)

    async def test_monitored_monitor_async_context(self) -> None:
        """monitor_async_context() happy path: value is available and cleanup
        runs in tearDownClass."""

        cleanup_ran = False

        class CMScenario(MonitoredTestCase):
            __test__ = False
            cm_value: ClassVar[str | None] = None

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()

                class _CM:
                    async def __aenter__(self) -> str:
                        return "hello"

                    async def __aexit__(self, *args: object) -> None:
                        nonlocal cleanup_ran
                        cleanup_ran = True
                        # Block until cancelled so the task stays alive
                        try:
                            await asyncio.sleep(1e9)
                        except asyncio.CancelledError:
                            pass

                cls.cm_value, _ = await cls.monitor_async_context(_CM())

            async def test_value_available(self) -> None:
                self.assertEqual(type(self).cm_value, "hello")

        result = await _run(CMScenario)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        self.assertTrue(cleanup_ran)

    async def test_monitored_monitor_async_context_crash(self) -> None:
        """When the coroutine behind monitor_async_context fails, tests abort."""

        class CMCrash(MonitoredTestCase):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()

                class _CM:
                    async def __aenter__(self) -> str:
                        return "ok"

                    async def __aexit__(self, *args: object) -> None:
                        # Simulate the component crashing after enter
                        await asyncio.sleep(0.01)
                        raise RuntimeError("context manager exploded")

                await cls.monitor_async_context(_CM())

            async def test_should_be_skipped(self) -> None:
                await asyncio.sleep(10)

        result = await _run(CMCrash)
        # The test should be skipped (cancelled by crash)
        self.assertEqual(len(result.skipped), 1)
        # tearDownClass reports the error
        self.assertEqual(len(result.errors), 1)
        self.assertIn("context manager exploded", result.errors[0].traceback)

    async def test_monitored_multiple_errors(self) -> None:
        """When multiple background tasks fail, all errors are collected in
        the ExceptionGroup raised by tearDownClass."""

        class MultiError(MonitoredTestCase):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()

                async def fail_1() -> None:
                    raise RuntimeError("error-alpha")

                async def fail_2() -> None:
                    raise RuntimeError("error-bravo")

                cls.create_task(fail_1(), name="task-1")
                cls.create_task(fail_2(), name="task-2")

            async def test_placeholder(self) -> None:
                await asyncio.sleep(10)

        result = await _run(MultiError)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(len(result.errors), 1)
        tb = result.errors[0].traceback
        self.assertIn("error-alpha", tb)
        self.assertIn("error-bravo", tb)

    async def test_monitored_no_leaked_tasks(self) -> None:
        """After a crash scenario completes, no monitored tasks are left."""

        class LeakCheck(MonitoredTestCase):
            __test__ = False
            fut: ClassVar[asyncio.Future[int]]

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()
                cls.fut = asyncio.Future()
                cls.create_task(
                    asyncio.wait_for(cls.fut, timeout=None),
                    name="leak-check-component",
                )

            async def test_crash(self) -> None:
                self.fut.set_exception(RuntimeError("leak test crash"))
                await asyncio.sleep(10)

        tasks_before = asyncio.all_tasks()
        await _run(LeakCheck)
        tasks_after = asyncio.all_tasks()

        leaked = {
            t
            for t in (tasks_after - tasks_before)
            if not t.done()
            and (t.get_name() or "").startswith(("leak-check-component", "LeakCheck.", "class:LeakCheck"))
        }
        self.assertEqual(len(leaked), 0, f"Leaked tasks: {leaked}")

    async def test_monitored_task_cleanup_on_success(self) -> None:
        """Even when all tests pass, background tasks are cancelled and
        cleaned up during tearDownClass."""

        class CleanupCheck(MonitoredTestCase):
            __test__ = False
            bg_task: ClassVar[asyncio.Task[object] | None] = None

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()

                async def long_running() -> None:
                    await asyncio.Event().wait()  # block forever

                cls.bg_task = cls.create_task(long_running(), name="long-running")

            async def test_ok(self) -> None:
                # The background task should still be running during the test.
                bg_task = type(self).bg_task
                assert bg_task is not None
                self.assertFalse(bg_task.done())

        result = await _run(CleanupCheck)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        # After tearDownClass, the background task must have been cleaned up.
        assert CleanupCheck.bg_task is not None
        self.assertTrue(CleanupCheck.bg_task.done())

    async def test_monitored_create_task_while_aborting(self) -> None:
        """Calling create_task() after a crash raises RuntimeError."""

        captured_exc: RuntimeError | None = None

        class AbortCreate(MonitoredTestCase):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()

                async def crash_immediately() -> None:
                    raise RuntimeError("instant crash")

                cls.create_task(crash_immediately(), name="crash")

            async def test_tries_create(self) -> None:
                # By the time this test runs (if it gets past setUp at
                # all), _aborting should be True.  But setUp will skip
                # because _aborting is True, so we test this from
                # setUpClass instead.
                pass

        # The above test will be skipped via setUp.  To actually
        # exercise create_task-while-aborting, we call it directly
        # on a class that is already in the aborting state.
        class DirectAbort(MonitoredTestCase):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()

                # Trigger _aborting by crashing a task
                async def crash() -> None:
                    raise RuntimeError("boom")

                cls.create_task(crash(), name="crash")
                # Yield so the crash callback fires
                await asyncio.sleep(0.01)

                nonlocal captured_exc
                try:

                    async def noop() -> None:
                        pass

                    cls.create_task(noop(), name="should-fail")
                except RuntimeError as exc:
                    captured_exc = exc

            async def test_placeholder(self) -> None:
                await asyncio.sleep(0)

        await _run(DirectAbort)
        self.assertIsNotNone(captured_exc)
        self.assertIn("shutting down", str(captured_exc))

    async def test_monitored_crash_aborts(self) -> None:
        class CrashScenario(MonitoredTestCase):
            __test__ = False

            async def test_crash(self) -> None:
                crash_component: asyncio.Future[int] = asyncio.Future()
                non_crashing_component: asyncio.Future[int] = asyncio.Future()
                self.create_task(
                    asyncio.wait_for(crash_component, timeout=None),
                    name="crashing test-component",
                )
                self.create_task(
                    asyncio.wait_for(non_crashing_component, timeout=None),
                    name="non-crashing test-component",
                )

                async def crash_later() -> None:
                    await asyncio.sleep(0.01)
                    crash_component.set_exception(
                        RuntimeError("WARNING: component crashed: ['crashing test-component']")
                    )

                # we trigger the crash with a delay which aborts the test
                asyncio.create_task(crash_later())
                await asyncio.sleep(10)
                raise ValueError("The test should abort and never reach this")

        result = await _run(CrashScenario)
        # test_crash is skipped (CancelledError → skipTest) and
        # tearDownClass raises ExceptionGroup → recorded as an extra error
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(len(result.errors), 1)
        err = result.errors[0]
        self.assertIn("RuntimeError", err.traceback)
        self.assertIn(
            "WARNING: component crashed: ['crashing test-component']",
            err.traceback,
        )
        # the non-crashing component should not appear in the error
        self.assertNotIn("non-crashing", err.traceback)

    async def test_monitored_no_crash_succeeds(self) -> None:
        class NoCrashScenario(MonitoredTestCase):
            __test__ = False
            test_no_crash_finished = 0

            async def test_no_crash(self) -> None:
                component: asyncio.Future[int] = asyncio.Future()
                self.create_task(
                    asyncio.wait_for(component, timeout=None),
                    name="test-component",
                )
                await asyncio.sleep(0.01)
                NoCrashScenario.test_no_crash_finished += 1

        result = await _run(NoCrashScenario)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)
        self.assertEqual(NoCrashScenario.test_no_crash_finished, 1)

    async def test_monitored_skips_after_crash(self) -> None:
        class SkipTestAfterCrash(MonitoredTestCase):
            __test__ = False
            test_1_ran = False
            test_2_ran = False
            crash_component: asyncio.Future[int]

            @classmethod
            async def setUpClass(cls) -> None:
                await super().setUpClass()
                cls.crash_component = asyncio.Future()
                cls.create_task(
                    asyncio.wait_for(cls.crash_component, timeout=None),
                    name="vm",
                )

            async def test_1_crash(self) -> None:
                SkipTestAfterCrash.test_1_ran = True
                self.crash_component.set_exception(RuntimeError("vm crashed"))
                # this sleep will be cancelled immediately because
                # the crash_component raises an exception above
                await asyncio.sleep(10)
                raise Exception("never reached, test gets canceled before")

            async def test_2_skipped(self) -> None:
                SkipTestAfterCrash.test_2_ran = True

        result = await _run(SkipTestAfterCrash)
        # test_1_crash → skip (cancelled), test_2_skipped → skip (aborting)
        self.assertEqual(len(result.skipped), 2)
        # tearDownClass raises ExceptionGroup → 1 error
        self.assertEqual(len(result.errors), 1)
        self.assertTrue(SkipTestAfterCrash.test_1_ran)
        self.assertFalse(SkipTestAfterCrash.test_2_ran)


# ===================================================================== #
#  Self-hosting entry point
# ===================================================================== #

if __name__ == "__main__":
    from barrage.colorize import should_colorize

    runner = AsyncTestRunner(
        max_concurrency=None,
        verbosity=2,
    )
    suite = discover_module(sys.modules[__name__])
    result = runner.run_suite(suite)
    report = result.format_report(verbosity=2, color=should_colorize(sys.stdout))
    print(report, end="")
    sys.exit(0 if result.was_successful else 1)
