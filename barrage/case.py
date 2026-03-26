# SPDX-License-Identifier: MIT
import asyncio
import functools
import inspect
from collections.abc import Callable, Container, Coroutine
from contextlib import AbstractAsyncContextManager
from typing import Any, ClassVar, TypeGuard, overload

import barrage.assertions as Assert
from barrage.assertions import (
    _SupportsDunderGE,
    _SupportsDunderGT,
    _SupportsDunderLE,
    _SupportsDunderLT,
)


class AsyncTestCase:
    """
    Base class for async tests that can run concurrently.

    Subclass this and define ``async def test_*`` methods. By default,
    tests within a class run **sequentially**. Pass ``concurrent=True``
    to ``__init_subclass__`` (or set the class variable) to run them
    concurrently instead.

    Example::

        class MyTests(AsyncTestCase):
            async def setUp(self) -> None:
                self.value = 42

            async def test_something(self) -> None:
                self.assertEqual(self.value, 42)

        class MyConcurrentTests(AsyncTestCase, concurrent=True):
            async def test_first(self) -> None:
                await asyncio.sleep(0.1)

            async def test_second(self) -> None:
                await asyncio.sleep(0.1)
    """

    # Whether tests within this class run concurrently. Subclasses
    # override this via ``__init_subclass__(concurrent=...)``.
    __concurrent__: ClassVar[bool] = False

    # If set on a class or method, that item is skipped with this reason.
    __skip_reason__: ClassVar[str | None] = None

    # The name of the test method this instance will run.
    _test_method_name: str

    def __init__(self, method_name: str = "runTest") -> None:
        self._test_method_name = method_name

    def __init_subclass__(cls, concurrent: bool | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if concurrent is not None:
            cls.__concurrent__ = concurrent

    # ------------------------------------------------------------------ #
    # Lifecycle hooks (all async)
    # ------------------------------------------------------------------ #

    @classmethod
    async def setUpClass(cls) -> None:
        """Called once before any test in the class runs."""

    async def setUp(self) -> None:
        """Called before each test method."""

    async def tearDown(self) -> None:
        """Called after each test method (even if it failed)."""

    @classmethod
    async def tearDownClass(cls) -> None:
        """Called once after all tests in the class have finished."""

    # ------------------------------------------------------------------ #
    # Assertion helpers
    #
    # All assertions delegate to :mod:`barrage.assertions` so that
    # the same logic is available to standalone test functions via
    # ``import barrage.assertions as A``.
    # ------------------------------------------------------------------ #

    def fail(self, msg: str | None = None) -> None:
        Assert.fail(msg)

    def assertTrue(self, expr: object, msg: str | None = None) -> None:
        Assert.true(expr, msg)

    def assertFalse(self, expr: object, msg: str | None = None) -> None:
        Assert.false(expr, msg)

    def assertEqual(self, first: object, second: object, msg: str | None = None) -> None:
        Assert.eq(first, second, msg)

    def assertNotEqual(self, first: object, second: object, msg: str | None = None) -> None:
        Assert.ne(first, second, msg)

    def assertIs(self, first: object, second: object, msg: str | None = None) -> None:
        Assert.is_(first, second, msg)

    def assertIsNot(self, first: object, second: object, msg: str | None = None) -> None:
        Assert.is_not(first, second, msg)

    def assertIsNone(self, expr: object, msg: str | None = None) -> None:
        Assert.none(expr, msg)

    def assertIsNotNone(self, expr: object, msg: str | None = None) -> None:
        Assert.not_none(expr, msg)

    def assertIn(self, member: object, container: Container[object], msg: str | None = None) -> None:
        Assert.in_(member, container, msg)

    def assertNotIn(self, member: object, container: Container[object], msg: str | None = None) -> None:
        Assert.not_in(member, container, msg)

    @overload
    def assertIsInstance[T](self, obj: object, cls: type[T], msg: str | None = None) -> TypeGuard[T]: ...
    @overload
    def assertIsInstance[T](
        self, obj: object, cls: tuple[type[T], ...], msg: str | None = None
    ) -> TypeGuard[T]: ...
    def assertIsInstance(self, obj: object, cls: type | tuple[type, ...], msg: str | None = None) -> bool:
        return Assert.isinstance_(obj, cls, msg)

    def assertIsNotInstance(self, obj: object, cls: type | tuple[type, ...], msg: str | None = None) -> None:
        Assert.not_isinstance(obj, cls, msg)

    @overload
    def assertGreater[T](self, first: _SupportsDunderGT[T], second: T, msg: str | None = None) -> None: ...
    @overload
    def assertGreater[T](self, first: T, second: _SupportsDunderLT[T], msg: str | None = None) -> None: ...
    def assertGreater(self, first: Any, second: Any, msg: str | None = None) -> None:
        Assert.gt(first, second, msg)

    @overload
    def assertGreaterEqual[T](
        self, first: _SupportsDunderGE[T], second: T, msg: str | None = None
    ) -> None: ...
    @overload
    def assertGreaterEqual[T](
        self, first: T, second: _SupportsDunderLE[T], msg: str | None = None
    ) -> None: ...
    def assertGreaterEqual(self, first: Any, second: Any, msg: str | None = None) -> None:
        Assert.ge(first, second, msg)

    @overload
    def assertLess[T](self, first: _SupportsDunderLT[T], second: T, msg: str | None = None) -> None: ...
    @overload
    def assertLess[T](self, first: T, second: _SupportsDunderGT[T], msg: str | None = None) -> None: ...
    def assertLess(self, first: Any, second: Any, msg: str | None = None) -> None:
        Assert.lt(first, second, msg)

    @overload
    def assertLessEqual[T](self, first: _SupportsDunderLE[T], second: T, msg: str | None = None) -> None: ...
    @overload
    def assertLessEqual[T](self, first: T, second: _SupportsDunderGE[T], msg: str | None = None) -> None: ...
    def assertLessEqual(self, first: Any, second: Any, msg: str | None = None) -> None:
        Assert.le(first, second, msg)

    def assertAlmostEqual(
        self, first: float, second: float, places: int = 7, msg: str | None = None
    ) -> None:
        Assert.almost_eq(first, second, places, msg)

    def assertRaises(self, exc_type: type[BaseException]) -> Assert._RaisesContext:
        return Assert.raises(exc_type)

    def skipTest(self, reason: str = "") -> None:
        Assert.skip(reason)

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #

    def id(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}.{self._test_method_name}"

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__} testMethod={self._test_method_name}>"

    def __str__(self) -> str:
        return f"{self._test_method_name} ({type(self).__qualname__})"


class MonitoredTestCase(AsyncTestCase):
    """
    MonitoredTestCase extends AsyncTestCase with crash monitoring.

    Register coroutines via create_task(). If any fail unexpectedly, all
    currently running tests are cancelled and any remaining tests are
    skipped (and the tearDownClass() method will report the background
    task exception(s)). Useful for detecting crashes of external
    components like VMs or helper services.

    Both concurrent and non-concurrent execution are supported.  Pass
    ``concurrent=True`` or ``concurrent=False`` on the subclass as with
    any ``AsyncTestCase``.
    """

    _loop: ClassVar[asyncio.AbstractEventLoop]
    _tasks: ClassVar[set[asyncio.Task[object]]]
    _errors: ClassVar[list[Exception]]
    _base_error: ClassVar[BaseException | None]
    _aborting: ClassVar[bool]
    _on_completed_fut: ClassVar[asyncio.Future[object] | None]
    _current_test_tasks: ClassVar[set[asyncio.Task[object]]]

    def __init_subclass__(cls, concurrent: bool | None = None, **kwargs: object) -> None:
        super().__init_subclass__(concurrent=concurrent, **kwargs)
        # Wrap all test_* methods so that task cancellation from a
        # monitored background task is converted into a skip rather
        # than propagating as an unhandled CancelledError.
        for name in list(vars(cls)):
            if name.startswith("test_") and inspect.iscoroutinefunction(vars(cls)[name]):
                original = vars(cls)[name]

                @functools.wraps(original)
                async def wrapped(
                    self: "MonitoredTestCase",
                    _orig: Callable[..., Coroutine[object, object, None]] = original,
                ) -> None:
                    task = asyncio.current_task()
                    assert task is not None
                    type(self)._current_test_tasks.add(task)
                    try:
                        await _orig(self)
                        return
                    except asyncio.CancelledError:
                        self.skipTest("Skipping test as background task failed")
                    finally:
                        type(self)._current_test_tasks.discard(task)

                setattr(cls, name, wrapped)

    @classmethod
    async def setUpClass(cls) -> None:
        await super().setUpClass()
        cls._loop = asyncio.get_running_loop()
        cls._tasks = set()
        cls._errors = []
        # This stores any BaseException that is not a regular Exception
        # (e.g. SystemExit, KeyboardInterrupt) which has to be reraised
        # directly instead of being wrapped in an ExceptionGroup.
        cls._base_error = None
        cls._aborting = False
        cls._on_completed_fut = None
        cls._current_test_tasks = set()

    @classmethod
    async def tearDownClass(cls) -> None:
        for t in cls._tasks:
            if not t.done():
                t.cancel()

        while cls._tasks:
            # _on_completed_fut is resolved in _on_task_done() further down.
            # We use a future + while loop instead of asyncio.wait() so that
            # we can ignore cancellations until all tasks have finished.
            if cls._on_completed_fut is None:
                cls._on_completed_fut = cls._loop.create_future()

            try:
                await cls._on_completed_fut
            except asyncio.CancelledError:
                pass

            cls._on_completed_fut = None

        assert len(cls._tasks) == 0

        # Finish cleaning up before we reraise any exception from
        # a background task.
        await super().tearDownClass()

        if cls._base_error is not None:
            raise cls._base_error

        if cls._errors:
            raise ExceptionGroup(
                "unhandled errors in a MonitoredTestCase",
                cls._errors,
            )

    async def setUp(self) -> None:
        if type(self)._aborting:
            self.skipTest("Skipping test as background task failed earlier")
        await super().setUp()

    @classmethod
    def create_task[T](
        cls,
        coro: Coroutine[object, object, T],
        *,
        name: str | None = None,
    ) -> asyncio.Task[T]:
        """Register a future to monitor. When it completes, tests abort fast."""

        if cls._aborting:
            coro.close()
            raise RuntimeError(f"Cannot create task as MonitoredTestCase {cls!r} is shutting down")

        task = cls._loop.create_task(coro, name=name)
        cls._tasks.add(task)
        task.add_done_callback(cls._on_task_done)
        return task

    @classmethod
    async def monitor_async_context[T](
        cls, cm: AbstractAsyncContextManager[T], name: str | None = None
    ) -> tuple[T, asyncio.Task[None]]:
        """Enter an async context manager and monitor it for failures.

        The context manager's ``__aenter__`` and ``__aexit__`` both run
        inside a single monitored background task. This is important
        because async context managers that use ``asyncio.TaskGroup``
        internally record the *current task* as the parent task during
        ``__aenter__``. By running both methods in the same background
        task, structured concurrency is preserved: if a child task in
        the TaskGroup fails, the TaskGroup cancels its parent (the
        background task), which triggers ``__aexit__`` and propagates
        the error through ``_on_task_done``, aborting the current test.

        The context manager's ``__aexit__`` must block until all of its
        background work is finished (e.g. by awaiting a
        ``TaskGroup``).  This is what keeps the monitored background
        task alive for the lifetime of the component.  When the
        background task is cancelled during teardown, ``__aexit__``
        receives the cancellation and can clean up its resources.
        """
        ready: asyncio.Future[T] = cls._loop.create_future()

        async def run_cm() -> None:
            try:
                async with cm as value:
                    ready.set_result(value)
            except BaseException as ex:
                if not ready.done():
                    ready.set_exception(ex)
                    return
                raise

        task = cls.create_task(run_cm(), name=name)
        return await ready, task

    @classmethod
    def _on_task_done(cls, task: asyncio.Task[object]) -> None:
        cls._tasks.discard(task)

        if cls._on_completed_fut is not None and len(cls._tasks) == 0:
            if not cls._on_completed_fut.done():
                cls._on_completed_fut.set_result(True)

        if task.cancelled():
            return

        exc = task.exception()
        if exc is None:
            return

        # If a background task fails, we want to abort execution of
        # all currently running tests and all further tests in this
        # class. We save the error(s) so we can reraise them as an
        # ExceptionGroup from tearDownClass().

        if isinstance(exc, Exception):
            cls._errors.append(exc)
        else:
            if cls._base_error is None:
                cls._base_error = exc

        if not cls._aborting:
            cls._aborting = True

            # Cancel every currently executing test task. The remaining
            # background tasks will be cancelled in tearDownClass().
            for test_task in cls._current_test_tasks:
                test_task.cancel()
