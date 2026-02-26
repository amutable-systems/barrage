# SPDX-License-Identifier: MIT
import asyncio
import functools
import inspect
from collections.abc import Callable, Container, Coroutine
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, ClassVar, Protocol, Self, TypeGuard, overload


class _SupportsDunderGT[T_contra](Protocol):
    def __gt__(self, other: T_contra, /) -> object: ...


class _SupportsDunderGE[T_contra](Protocol):
    def __ge__(self, other: T_contra, /) -> object: ...


class _SupportsDunderLT[T_contra](Protocol):
    def __lt__(self, other: T_contra, /) -> object: ...


class _SupportsDunderLE[T_contra](Protocol):
    def __le__(self, other: T_contra, /) -> object: ...


class SkipTest(Exception):
    """Raised to skip a test."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason)
        self.reason = reason


class _AssertRaisesContext:
    """Context manager for ``assertRaises``."""

    def __init__(self, exc_type: type[BaseException]) -> None:
        self.exc_type = exc_type
        self.exception: BaseException | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, tb: TracebackType | None
    ) -> bool:
        if exc_type is None:
            raise AssertionError(f"{self.exc_type.__name__} not raised")
        if not issubclass(exc_type, self.exc_type):
            return False  # re-raise unexpected exception
        self.exception = exc_val
        return True  # suppress the expected exception


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
    # ------------------------------------------------------------------ #

    def fail(self, msg: str | None = None) -> None:
        raise AssertionError(msg or "test failed explicitly")

    def assertTrue(self, expr: object, msg: str | None = None) -> None:
        if not expr:
            raise AssertionError(msg or f"expected truthy, got {expr!r}")

    def assertFalse(self, expr: object, msg: str | None = None) -> None:
        if expr:
            raise AssertionError(msg or f"expected falsy, got {expr!r}")

    def assertEqual(self, first: object, second: object, msg: str | None = None) -> None:
        if first != second:
            raise AssertionError(msg or f"{first!r} != {second!r}")

    def assertNotEqual(self, first: object, second: object, msg: str | None = None) -> None:
        if first == second:
            raise AssertionError(msg or f"{first!r} == {second!r}")

    def assertIs(self, first: object, second: object, msg: str | None = None) -> None:
        if first is not second:
            raise AssertionError(msg or f"{first!r} is not {second!r}")

    def assertIsNot(self, first: object, second: object, msg: str | None = None) -> None:
        if first is second:
            raise AssertionError(msg or f"{first!r} is {second!r}")

    def assertIsNone(self, expr: object, msg: str | None = None) -> None:
        if expr is not None:
            raise AssertionError(msg or f"expected None, got {expr!r}")

    def assertIsNotNone(self, expr: object, msg: str | None = None) -> None:
        if expr is None:
            raise AssertionError(msg or "expected non-None value")

    def assertIn(self, member: object, container: Container[object], msg: str | None = None) -> None:
        if member not in container:
            raise AssertionError(msg or f"{member!r} not in {container!r}")

    def assertNotIn(self, member: object, container: Container[object], msg: str | None = None) -> None:
        if member in container:
            raise AssertionError(msg or f"{member!r} unexpectedly in {container!r}")

    @overload
    def assertIsInstance[T](self, obj: object, cls: type[T], msg: str | None = None) -> TypeGuard[T]: ...
    @overload
    def assertIsInstance[T](
        self, obj: object, cls: tuple[type[T], ...], msg: str | None = None
    ) -> TypeGuard[T]: ...
    def assertIsInstance(self, obj: object, cls: type | tuple[type, ...], msg: str | None = None) -> bool:
        if not isinstance(obj, cls):
            raise AssertionError(msg or f"{obj!r} is not an instance of {cls!r}")
        return True

    def assertIsNotInstance(self, obj: object, cls: type | tuple[type, ...], msg: str | None = None) -> None:
        if isinstance(obj, cls):
            raise AssertionError(msg or f"{obj!r} is unexpectedly an instance of {cls!r}")

    # The comparison assertions use two overloads each, mirroring
    # typeshed's approach for unittest.TestCase. The first overload
    # checks whether ``first`` supports the forward dunder (e.g.
    # ``__gt__``), the second checks whether ``second`` supports the
    # reflected dunder (e.g. ``__lt__``). This is needed because type
    # checkers may widen ``float`` to ``int | float`` (PEP 484 numeric
    # tower), and ``int`` alone does not satisfy e.g.
    # ``_SupportsDunderGT[int | float]`` since ``int.__gt__`` only
    # accepts ``int``. The two-overload pattern lets at least one
    # overload match in all practical cases.

    @overload
    def assertGreater[T](self, first: _SupportsDunderGT[T], second: T, msg: str | None = None) -> None: ...
    @overload
    def assertGreater[T](self, first: T, second: _SupportsDunderLT[T], msg: str | None = None) -> None: ...
    def assertGreater(self, first: Any, second: Any, msg: str | None = None) -> None:
        if not first > second:
            raise AssertionError(msg or f"{first!r} is not greater than {second!r}")

    @overload
    def assertGreaterEqual[T](
        self, first: _SupportsDunderGE[T], second: T, msg: str | None = None
    ) -> None: ...
    @overload
    def assertGreaterEqual[T](
        self, first: T, second: _SupportsDunderLE[T], msg: str | None = None
    ) -> None: ...
    def assertGreaterEqual(self, first: Any, second: Any, msg: str | None = None) -> None:
        if not first >= second:
            raise AssertionError(msg or f"{first!r} is not greater than or equal to {second!r}")

    @overload
    def assertLess[T](self, first: _SupportsDunderLT[T], second: T, msg: str | None = None) -> None: ...
    @overload
    def assertLess[T](self, first: T, second: _SupportsDunderGT[T], msg: str | None = None) -> None: ...
    def assertLess(self, first: Any, second: Any, msg: str | None = None) -> None:
        if not first < second:
            raise AssertionError(msg or f"{first!r} is not less than {second!r}")

    @overload
    def assertLessEqual[T](self, first: _SupportsDunderLE[T], second: T, msg: str | None = None) -> None: ...
    @overload
    def assertLessEqual[T](self, first: T, second: _SupportsDunderGE[T], msg: str | None = None) -> None: ...
    def assertLessEqual(self, first: Any, second: Any, msg: str | None = None) -> None:
        if not first <= second:
            raise AssertionError(msg or f"{first!r} is not less than or equal to {second!r}")

    def assertAlmostEqual(
        self, first: float, second: float, places: int = 7, msg: str | None = None
    ) -> None:
        if round(abs(second - first), places) != 0:
            raise AssertionError(msg or f"{first!r} != {second!r} within {places} places")

    def assertRaises(self, exc_type: type[BaseException]) -> _AssertRaisesContext:
        return _AssertRaisesContext(exc_type)

    def skipTest(self, reason: str = "") -> None:
        raise SkipTest(reason)

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
