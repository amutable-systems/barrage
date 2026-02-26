# SPDX-License-Identifier: MIT
"""
Tests for ``TaskGroup.monitor_async_context()``.

Run with::

    python3 -m barrage tests/test_taskgroups.py
"""

import asyncio
import sys

from barrage.case import AsyncTestCase
from barrage.discovery import discover_module
from barrage.runner import AsyncTestRunner
from barrage.taskgroups import TaskGroup


class TestMonitorAsyncContext(AsyncTestCase):
    """Direct unit tests for TaskGroup.monitor_async_context()."""

    async def test_returns_aenter_value(self) -> None:
        """The value produced by __aenter__ is returned to the caller."""

        class _CM:
            async def __aenter__(self) -> str:
                return "hello"

            async def __aexit__(self, *args: object) -> None:
                pass

        async with TaskGroup() as tg:
            value, _ = await tg.monitor_async_context(_CM())

        self.assertEqual(value, "hello")

    async def test_returns_none_value(self) -> None:
        """Works correctly when __aenter__ returns None."""

        class _CM:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, *args: object) -> None:
                pass

        async with TaskGroup() as tg:
            await tg.monitor_async_context(_CM())

    async def test_complex_value_type(self) -> None:
        """The generic type parameter works with complex types."""

        class _CM:
            async def __aenter__(self) -> dict[str, list[int]]:
                return {"a": [1, 2], "b": [3]}

            async def __aexit__(self, *args: object) -> None:
                pass

        async with TaskGroup() as tg:
            value, _ = await tg.monitor_async_context(_CM())

        self.assertEqual(value, {"a": [1, 2], "b": [3]})

    async def test_quick_exit_cm(self) -> None:
        """A CM whose __aexit__ returns immediately completes without error."""

        class _CM:
            async def __aenter__(self) -> list[int]:
                return [1, 2, 3]

            async def __aexit__(self, *args: object) -> None:
                pass

        async with TaskGroup() as tg:
            value, _ = await tg.monitor_async_context(_CM())

        self.assertEqual(value, [1, 2, 3])

    async def test_aexit_runs_on_group_exit(self) -> None:
        """__aexit__ is invoked when the background task finishes."""
        cleanup_ran = False

        class _CM:
            async def __aenter__(self) -> str:
                return "value"

            async def __aexit__(self, *args: object) -> None:
                nonlocal cleanup_ran
                cleanup_ran = True

        async with TaskGroup() as tg:
            await tg.monitor_async_context(_CM())

        self.assertTrue(cleanup_ran)

    async def test_aexit_receives_cancellation(self) -> None:
        """When a sibling task crashes, the CM's __aexit__ receives
        CancelledError as the TaskGroup tears down."""
        got_cancelled = False

        class _CM:
            async def __aenter__(self) -> int:
                return 42

            async def __aexit__(self, *args: object) -> None:
                nonlocal got_cancelled
                try:
                    await asyncio.sleep(1e9)
                except asyncio.CancelledError:
                    got_cancelled = True

        async def crash() -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("trigger cancel")

        with self.assertRaises(ExceptionGroup):
            async with TaskGroup() as tg:
                await tg.monitor_async_context(_CM())
                tg.create_task(crash())
                await asyncio.sleep(1e9)

        self.assertTrue(got_cancelled)

    async def test_aexit_cleanup_on_cancellation(self) -> None:
        """__aexit__ can perform cleanup work when cancelled."""
        cleanup_order: list[str] = []

        class _CM:
            async def __aenter__(self) -> str:
                return "resource"

            async def __aexit__(self, *args: object) -> None:
                try:
                    await asyncio.sleep(1e9)
                except asyncio.CancelledError:
                    cleanup_order.append("cleanup-start")
                    await asyncio.sleep(0)  # yield once to simulate async cleanup
                    cleanup_order.append("cleanup-end")

        async def crash() -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("boom")

        with self.assertRaises(ExceptionGroup):
            async with TaskGroup() as tg:
                await tg.monitor_async_context(_CM())
                tg.create_task(crash())
                await asyncio.sleep(1e9)

        self.assertEqual(cleanup_order, ["cleanup-start", "cleanup-end"])

    async def test_aenter_exception_propagates(self) -> None:
        """If __aenter__ raises, the exception propagates through the
        TaskGroup as an ExceptionGroup."""

        class _CM:
            async def __aenter__(self) -> str:
                raise RuntimeError("enter failed")

            async def __aexit__(self, *args: object) -> None:
                pass

        with self.assertRaises(ExceptionGroup) as ctx:
            async with TaskGroup() as tg:
                await tg.monitor_async_context(_CM())

        exc = ctx.exception
        assert isinstance(exc, ExceptionGroup)
        self.assertEqual(len(exc.exceptions), 1)
        self.assertIsInstance(exc.exceptions[0], RuntimeError)
        self.assertEqual(str(exc.exceptions[0]), "enter failed")

    async def test_aenter_exception_does_not_invoke_aexit(self) -> None:
        """When __aenter__ raises, __aexit__ must not be called (standard
        context-manager protocol)."""
        aexit_called = False

        class _CM:
            async def __aenter__(self) -> str:
                raise RuntimeError("enter boom")

            async def __aexit__(self, *args: object) -> None:
                nonlocal aexit_called
                aexit_called = True

        with self.assertRaises(ExceptionGroup):
            async with TaskGroup() as tg:
                await tg.monitor_async_context(_CM())

        self.assertFalse(aexit_called)

    async def test_aexit_crash_propagates(self) -> None:
        """If __aexit__ raises after a successful enter, the TaskGroup
        surfaces the error."""

        class _CM:
            async def __aenter__(self) -> str:
                return "ok"

            async def __aexit__(self, *args: object) -> None:
                await asyncio.sleep(0.01)
                raise RuntimeError("exit exploded")

        with self.assertRaises(ExceptionGroup) as ctx:
            async with TaskGroup() as tg:
                await tg.monitor_async_context(_CM())
                # Keep the group body alive long enough for __aexit__
                # to crash (it runs in a background task).
                await asyncio.sleep(1e9)

        exc = ctx.exception
        assert isinstance(exc, ExceptionGroup)
        exceptions = exc.exceptions
        self.assertEqual(len(exceptions), 1)
        self.assertIsInstance(exceptions[0], RuntimeError)
        self.assertEqual(str(exceptions[0]), "exit exploded")

    async def test_aexit_crash_cancels_sibling_tasks(self) -> None:
        """A CM crash cancels other tasks in the same TaskGroup."""
        sibling_cancelled = False

        class _CM:
            async def __aenter__(self) -> str:
                return "ok"

            async def __aexit__(self, *args: object) -> None:
                await asyncio.sleep(0.01)
                raise RuntimeError("cm crash")

        async def sibling() -> None:
            nonlocal sibling_cancelled
            try:
                await asyncio.sleep(1e9)
            except asyncio.CancelledError:
                sibling_cancelled = True
                raise

        with self.assertRaises(ExceptionGroup):
            async with TaskGroup() as tg:
                await tg.monitor_async_context(_CM())
                tg.create_task(sibling())
                await asyncio.sleep(1e9)

        self.assertTrue(sibling_cancelled)

    async def test_multiple_contexts_values(self) -> None:
        """Multiple CMs can be monitored; all values are returned."""

        class _CM:
            def __init__(self, label: str) -> None:
                self.label = label

            async def __aenter__(self) -> str:
                return self.label

            async def __aexit__(self, *args: object) -> None:
                pass

        async with TaskGroup() as tg:
            v1, _ = await tg.monitor_async_context(_CM("first"))
            v2, _ = await tg.monitor_async_context(_CM("second"))
            v3, _ = await tg.monitor_async_context(_CM("third"))

        self.assertEqual(v1, "first")
        self.assertEqual(v2, "second")
        self.assertEqual(v3, "third")

    async def test_multiple_contexts_all_cleaned_up(self) -> None:
        """When cancelled, all monitored CMs receive cleanup."""
        cleanups: list[str] = []

        class _CM:
            def __init__(self, label: str) -> None:
                self.label = label

            async def __aenter__(self) -> str:
                return self.label

            async def __aexit__(self, *args: object) -> None:
                try:
                    await asyncio.sleep(1e9)
                except asyncio.CancelledError:
                    cleanups.append(self.label)

        async def crash() -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("trigger cleanup")

        with self.assertRaises(ExceptionGroup):
            async with TaskGroup() as tg:
                await tg.monitor_async_context(_CM("first"))
                await tg.monitor_async_context(_CM("second"))
                await tg.monitor_async_context(_CM("third"))
                tg.create_task(crash())
                await asyncio.sleep(1e9)

        self.assertEqual(sorted(cleanups), ["first", "second", "third"])

    async def test_name_forwarded_to_task(self) -> None:
        """The *name* parameter is forwarded to the underlying task."""
        task_name: str | None = None

        class _CM:
            async def __aenter__(self) -> str:
                return "val"

            async def __aexit__(self, *args: object) -> None:
                nonlocal task_name
                t = asyncio.current_task()
                assert t is not None
                task_name = t.get_name()

        async with TaskGroup() as tg:
            await tg.monitor_async_context(_CM(), name="my-component")

        self.assertEqual(task_name, "my-component")

    async def test_name_defaults_to_none(self) -> None:
        """When *name* is omitted the task still works (default name)."""

        class _CM:
            async def __aenter__(self) -> int:
                return 1

            async def __aexit__(self, *args: object) -> None:
                pass

        async with TaskGroup() as tg:
            value, _ = await tg.monitor_async_context(_CM())

        self.assertEqual(value, 1)

    async def test_structured_concurrency_with_inner_taskgroup(self) -> None:
        """A CM that uses an inner TaskGroup works correctly because
        __aenter__ and __aexit__ run in the same background task,
        preserving structured concurrency."""
        inner_task_ran = False
        cleanup_ran = False

        class _CM:
            async def __aenter__(self) -> str:
                return "structured"

            async def __aexit__(self, *args: object) -> None:
                nonlocal inner_task_ran, cleanup_ran

                async def inner_work() -> None:
                    nonlocal inner_task_ran
                    inner_task_ran = True
                    await asyncio.sleep(1e9)

                try:
                    async with asyncio.TaskGroup() as inner_tg:
                        inner_tg.create_task(inner_work())
                        await asyncio.sleep(1e9)
                except asyncio.CancelledError:
                    cleanup_ran = True

        async def crash() -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("trigger teardown")

        with self.assertRaises(ExceptionGroup):
            async with TaskGroup() as tg:
                val, _ = await tg.monitor_async_context(_CM())
                tg.create_task(crash())
                await asyncio.sleep(1e9)

        self.assertEqual(val, "structured")
        self.assertTrue(inner_task_ran)
        self.assertTrue(cleanup_ran)

    async def test_aenter_and_aexit_share_same_task(self) -> None:
        """Both __aenter__ and __aexit__ execute in the same background
        task, which is the key invariant for structured concurrency."""
        enter_task: asyncio.Task[object] | None = None
        exit_task: asyncio.Task[object] | None = None

        class _CM:
            async def __aenter__(self) -> str:
                nonlocal enter_task
                enter_task = asyncio.current_task()
                return "ok"

            async def __aexit__(self, *args: object) -> None:
                nonlocal exit_task
                exit_task = asyncio.current_task()

        async with TaskGroup() as tg:
            await tg.monitor_async_context(_CM())

        self.assertIsNotNone(enter_task)
        self.assertIs(enter_task, exit_task)


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
