# SPDX-License-Identifier: MIT

"""
Singleton support for barrage.

A *singleton* is a resource that is expensive to create and tear down
(e.g. a pool of virtual machines, a resource monitor) and should
therefore be shared across many test classes -- and, in watch mode,
across many successive test runs.

Singletons are classes that implement the async context manager
protocol (``__aenter__``/``__aexit__``).  Dependencies between
singletons are declared using the :class:`singleton` descriptor,
the same mechanism used to declare singletons on test classes.

Singletons are declared on test classes using the :class:`singleton`
descriptor and managed by a :class:`SingletonManager` which owns
their lifecycle.

Quick start::

    from typing import Self

    from barrage import AsyncTestCase
    from barrage.singleton import Singleton, singleton

    class VMManager(Singleton):
        async def __aenter__(self) -> Self:
            await self.boot()
            return self

        async def __aexit__(self, *exc: object) -> None:
            await self.shutdown()

    class MyTests(AsyncTestCase):
        manager = singleton(VMManager)

        async def test_something(self) -> None:
            vm = await self.manager.acquire()
            result = await vm.run("uname -r")
            self.assertIn("6.", result)

Dependencies between singletons are declared using the same
:class:`singleton` descriptor::

    class ResourceMonitor(Singleton):
        async def __aenter__(self) -> Self:
            ...

    class VMManager(Singleton):
        monitor = singleton(ResourceMonitor)

        async def __aenter__(self) -> Self:
            ...

    class MyTests(AsyncTestCase):
        monitor = singleton(ResourceMonitor)
        manager = singleton(VMManager)  # ResourceMonitor created first
"""

import asyncio
import logging
import types
from contextlib import AbstractAsyncContextManager
from typing import Any, Self, cast, overload

from barrage.case import AsyncTestCase
from barrage.taskgroups import TaskGroup

log = logging.getLogger(__name__)


# ===================================================================== #
#  Base class
# ===================================================================== #


class Singleton(AbstractAsyncContextManager["Singleton"]):
    """Base class for singletons managed by :class:`SingletonManager`.

    Singletons live inside a monitored background task for the entire
    test session.  The ``__aexit__`` of the context manager must
    therefore **block** until the singleton is ready to be torn down —
    otherwise the background task completes immediately and the
    singleton becomes unmonitored.

    This base class provides a default ``__aexit__`` that blocks
    forever (via an unresolved ``Future``), keeping the background
    task alive until the ``SingletonManager`` cancels it during
    ordered teardown.

    Subclasses that need custom cleanup should override
    ``__aexit__``.  The override must block until teardown is
    complete.  Subclasses that have their own long-running background
    work (e.g. an internal ``TaskGroup``) can block on that instead.

    Quick start::

        from barrage.singleton import Singleton

        class VMPool(Singleton):
            async def __aenter__(self) -> Self:
                self.pool = await create_pool()
                return self

            async def __aexit__(self, *exc: object) -> None:
                await self.pool.shutdown()
    """

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.Future()


# ===================================================================== #
#  Descriptor
# ===================================================================== #


class singleton[T: Singleton]:
    """Descriptor that declares a singleton dependency.

    Can be used on both test classes and singleton classes.  The
    :class:`SingletonManager` will instantiate the referenced class
    (recursively resolving any of *its* ``singleton`` descriptors
    first), enter it as an async context manager, and replace the
    descriptor with the resulting instance.  On test classes this
    happens before ``setUpClass``; on singleton classes it happens
    before ``__aenter__``.

    After injection the descriptor is gone -- attribute access returns
    the plain value through normal Python lookup.  If the descriptor
    is accessed *before* injection (e.g. when running without a
    session-aware runner), ``__get__`` raises :class:`RuntimeError`
    with a helpful message.

    Typing works automatically: ``singleton(SomeClass)`` infers ``T``
    from the class, so ``self.my_singleton`` is seen as ``T`` by type
    checkers.
    """

    def __init__(self, cls: type[T]) -> None:
        if not isinstance(cls, type):
            raise TypeError(f"singleton() expects a class, got {type(cls).__name__}: {cls!r}")
        if not issubclass(cls, Singleton):
            raise TypeError(
                f"singleton() expects a Singleton subclass, "
                f"but {cls.__qualname__!r} does not inherit from Singleton"
            )
        self.cls = cls
        self._attr_name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = name

    # The two overloads let type checkers distinguish class access
    # (returns the descriptor itself -- useful for framework
    # introspection) from instance access (returns T).

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...

    @overload
    def __get__(self, obj: object, objtype: type | None = None) -> T: ...

    def __get__(self, obj: object | None, objtype: type | None = None) -> Self | T:
        # Before injection this descriptor is still present on the
        # class.  Give the user an actionable error message.
        raise RuntimeError(
            f"Singleton {self._attr_name!r} has not been injected. "
            f"Are you running with a session-aware runner?"
        )

    def __repr__(self) -> str:
        qname = getattr(self.cls, "__qualname__", None) or repr(self.cls)
        return f"singleton({qname!s}, attr={self._attr_name!r})"


# ===================================================================== #
#  Helpers
# ===================================================================== #


def singleton_key(descriptor: singleton[Any]) -> str:
    """Return a stable identity string for *descriptor*'s class.

    The key is based on the class's ``__module__`` and
    ``__qualname__`` so that it survives module reloads (where the
    class *object* changes but its qualified name does not).
    """
    cls = descriptor.cls
    module = getattr(cls, "__module__", None) or "<unknown>"
    qualname = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", repr(cls))
    return f"{module}.{qualname}"


def discover_singletons(
    cls: type,
) -> dict[str, singleton[Any]]:
    """Find all :class:`singleton` descriptors on *cls* and its bases.

    Works on both :class:`AsyncTestCase` and :class:`Singleton`
    subclasses.  Walks the MRO in reverse so that subclass overrides
    win.  Uses ``vars(klass)`` to retrieve the raw descriptor without
    triggering ``__get__``.
    """
    singletons: dict[str, singleton[Any]] = {}
    for klass in reversed(cls.__mro__):
        for name, value in vars(klass).items():
            if isinstance(value, singleton):
                singletons[name] = value
    return singletons


def discover_singletons_from_suite(
    entries: list[tuple[type[AsyncTestCase], list[str]]],
) -> dict[str, singleton[Any]]:
    """Discover all unique singletons referenced by a resolved suite.

    Recursively discovers singletons declared on singleton classes
    themselves, not just on test classes.

    Returns a mapping of *singleton_key* -> *descriptor*.  When the
    same class is referenced by multiple test classes the first
    descriptor encountered wins (they are all equivalent).
    """
    seen: dict[str, singleton[Any]] = {}

    def _collect(cls: type) -> None:
        for _attr, descriptor in discover_singletons(cls).items():
            key = singleton_key(descriptor)
            if key not in seen:
                seen[key] = descriptor
                _collect(descriptor.cls)

    for cls, _methods in entries:
        _collect(cls)

    return seen


# ===================================================================== #
#  Singleton manager
# ===================================================================== #


class SingletonManager:
    """Manages the lifecycle of singletons.

    The manager caches singleton instances by their
    :func:`singleton_key` so that multiple test classes (or multiple
    runs in watch mode) that reference the same class share a single
    instance.

    Dependencies between singletons are resolved automatically from
    :class:`singleton` descriptors declared on the singleton class.

    Each singleton's ``__aenter__`` and ``__aexit__`` run inside a
    monitored background task within an internal
    :class:`~barrage.taskgroups.TaskGroup`.  If a singleton's
    background work crashes, the task group cancels all sibling
    tasks (triggering ``__aexit__`` on other singletons) and
    propagates the error.

    On normal teardown, singletons are cancelled in reverse creation
    order so that dependents are torn down before their dependencies.

    Typical usage::

        async with SingletonManager() as sm:
            suite = resolve_tests(...)
            entries = suite.entries
            await sm.inject(entries)
            result = await runner.run_suite_async(suite)
    """

    def __init__(self) -> None:
        self._tg = TaskGroup()
        # key -> value, for deduplication and cache lookup.
        self._active: dict[str, Any] = {}
        # (key, task) pairs in creation order, for ordered teardown.
        self._tasks: list[tuple[str, asyncio.Task[None]]] = []

    async def __aenter__(self) -> Self:
        await self._tg.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool:
        try:
            # Cancel singletons in reverse creation order so that
            # dependents are torn down before their dependencies.
            for key, task in reversed(self._tasks):
                log.debug("tearing down singleton %s", key)
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            await self._tg.__aexit__(exc_type, exc_val, exc_tb)
        return False

    # ── core API ─────────────────────────────────────────────────────

    async def get_or_create[T: Singleton](self, descriptor: singleton[T]) -> T:
        """Return the singleton instance, creating it on first access.

        Dependencies declared via :class:`singleton` descriptors on
        the singleton class are resolved and created recursively
        before the singleton itself.
        """
        key = singleton_key(descriptor)
        if key in self._active:
            return cast(T, self._active[key])

        return await self._create_singleton(descriptor, frozenset(), ())

    async def _create_singleton[T: Singleton](
        self,
        descriptor: singleton[T],
        creating: frozenset[str],
        creation_path: tuple[str, ...],
    ) -> T:
        """Recursively create a singleton and its dependencies."""
        key = singleton_key(descriptor)

        if key in self._active:
            return cast(T, self._active[key])

        # Cycle detection.
        if key in creating:
            cycle = creation_path[creation_path.index(key) :] + (key,)
            raise RuntimeError(f"Circular singleton dependency detected: {' -> '.join(cycle)}")
        creating = creating | {key}
        creation_path = creation_path + (key,)

        # Resolve dependencies from singleton descriptors on the class.
        deps = discover_singletons(descriptor.cls)

        for attr_name, dep_descriptor in deps.items():
            value = await self._create_singleton(
                dep_descriptor,
                creating,
                creation_path,
            )
            setattr(descriptor.cls, attr_name, value)

        # All dependencies resolved and injected.  Create the singleton.
        log.debug("entering singleton %s", key)
        instance = descriptor.cls()
        value, task = await self._tg.monitor_async_context(
            instance,
            name=f"singleton:{key}",
        )
        self._tasks.append((key, task))
        self._active[key] = value

        return cast(T, value)

    async def inject(
        self,
        entries: list[tuple[type[AsyncTestCase], list[str]]],
    ) -> None:
        """Discover and inject all singletons for the given suite entries.

        For every test class that has :class:`singleton` descriptors,
        the corresponding instances are obtained (creating them if
        necessary) and injected via ``setattr``, replacing the
        descriptors with plain values.  Dependencies between
        singletons are resolved recursively via their own
        :class:`singleton` descriptors.
        """
        for cls, _methods in entries:
            for attr_name, descriptor in discover_singletons(cls).items():
                value = await self.get_or_create(descriptor)
                setattr(cls, attr_name, value)

    @property
    def active_keys(self) -> frozenset[str]:
        """The set of singleton keys that are currently active."""
        return frozenset(self._active)
