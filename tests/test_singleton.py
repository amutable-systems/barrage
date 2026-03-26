# SPDX-License-Identifier: MIT

"""
Tests for singleton support.

Run with::

    python3 -m barrage tests/test_singleton.py
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, ClassVar, Self

from barrage.case import AsyncTestCase
from barrage.result import AsyncTestResult
from barrage.runner import AsyncTestRunner, AsyncTestSuite
from barrage.singleton import (
    Singleton,
    SingletonManager,
    _SingletonDescriptor,
    discover_singletons,
    discover_singletons_from_function,
    discover_singletons_from_suite,
    singleton,
    singleton_key,
)

# ===================================================================== #
#  Helpers
# ===================================================================== #


async def _run(
    *classes: type[AsyncTestCase],
    max_concurrency: int | None = None,
    failfast: bool = False,
) -> AsyncTestResult:
    """Run one or more inner test classes and return the result."""
    runner = AsyncTestRunner(
        max_concurrency=max_concurrency,
        verbosity=0,
        failfast=failfast,
    )
    return await runner.run_classes_async(*classes)


async def _run_suite(
    suite: AsyncTestSuite,
) -> AsyncTestResult:
    runner = AsyncTestRunner(
        verbosity=0,
    )
    return await runner.run_suite_async(suite)


# ── Sample singletons ───────────────────────────────────────────────


class Counter(Singleton):
    """A simple singleton that tracks creation/teardown."""

    instances_created: ClassVar[int] = 0
    instances_torn_down: ClassVar[int] = 0

    def __init__(self) -> None:
        Counter.instances_created += 1
        self.value = 0
        self.torn_down = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await asyncio.Future()
        finally:
            self.torn_down = True
            Counter.instances_torn_down += 1

    def increment(self) -> None:
        self.value += 1

    @classmethod
    def reset(cls) -> None:
        cls.instances_created = 0
        cls.instances_torn_down = 0


class ConfigurableCounter(Singleton):
    """A singleton that uses __init_subclass__ for configuration."""

    start: int = 0
    step: int = 1

    def __init_subclass__(cls, *, start: int = 0, step: int = 1, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.start = start
        cls.step = step

    def __init__(self) -> None:
        self.value = self.start

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.Future()

    def increment(self) -> None:
        self.value += self.step


class CounterFrom10(ConfigurableCounter, start=10):
    """A parameterised singleton: Counter starting at 10."""


class CounterBy5(ConfigurableCounter, step=5):
    """A parameterised singleton: Counter stepping by 5."""


class Greeter(Singleton):
    def __init__(self) -> None:
        self.greeting = "Hello"

    def greet(self, name: str) -> str:
        return f"{self.greeting}, {name}!"


class AsyncGreeter(Singleton):
    """A singleton that does async work during setup/teardown."""

    def __init__(self) -> None:
        self.greeting = "Async Hello"

    async def __aenter__(self) -> Self:
        await asyncio.sleep(0)
        return self

    def greet(self, name: str) -> str:
        return f"{self.greeting}, {name}!"


# ===================================================================== #
#  Descriptor tests
# ===================================================================== #


class TestSingletonDescriptor(AsyncTestCase, concurrent=True):
    async def test_descriptor_raises_before_injection(self) -> None:
        """Accessing a singleton before injection raises RuntimeError."""

        class _Inner(AsyncTestCase):
            __test__ = False
            counter = singleton(Counter)

        with self.assertRaises(RuntimeError) as ctx:
            _Inner("runTest").counter  # noqa: B018

        self.assertIn("has not been injected", str(ctx.exception))
        self.assertIn("counter", str(ctx.exception))

    async def test_descriptor_class_access_raises_before_injection(self) -> None:
        """Class-level access also raises before injection."""

        class _Inner(AsyncTestCase):
            __test__ = False
            counter = singleton(Counter)

        with self.assertRaises(RuntimeError):
            _Inner.counter  # noqa: B018

    async def test_descriptor_set_name(self) -> None:
        """__set_name__ records the attribute name."""

        class _Inner(AsyncTestCase):
            __test__ = False
            my_counter = singleton(Counter)

        desc = vars(_Inner)["my_counter"]
        self.assertIsInstance(desc, _SingletonDescriptor)
        self.assertEqual(desc._attr_name, "my_counter")

    async def test_descriptor_repr(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        desc = vars(_Inner)["ctr"]
        r = repr(desc)
        self.assertIn("singleton", r)
        self.assertIn("Counter", r)
        self.assertIn("ctr", r)

    async def test_descriptor_cls_stored(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        desc = vars(_Inner)["ctr"]
        self.assertIs(desc.cls, Counter)

    async def test_descriptor_rejects_non_type(self) -> None:
        """singleton() raises TypeError if given a non-type."""

        def not_a_class() -> None:
            pass

        with self.assertRaises(TypeError) as ctx:
            singleton(not_a_class)  # type: ignore[arg-type,type-var]

        self.assertIn("expects a class", str(ctx.exception))

    async def test_descriptor_rejects_non_singleton(self) -> None:
        """singleton() raises TypeError if the class doesn't inherit from Singleton."""

        class NotASingleton:
            pass

        with self.assertRaises(TypeError) as ctx:
            singleton(NotASingleton)  # type: ignore[type-var]

        self.assertIn("Singleton subclass", str(ctx.exception))
        self.assertIn("NotASingleton", str(ctx.exception))


# ===================================================================== #
#  Discovery tests
# ===================================================================== #


class TestSingletonDiscovery(AsyncTestCase, concurrent=True):
    async def test_discover_singletons_on_class(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        found = discover_singletons(_Inner)
        self.assertEqual(set(found.keys()), {"ctr"})
        self.assertIs(found["ctr"].cls, Counter)

    async def test_discover_multiple_singletons(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)
            greet = singleton(Greeter)

        found = discover_singletons(_Inner)
        self.assertEqual(set(found.keys()), {"ctr", "greet"})

    async def test_discover_no_singletons(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_ok(self) -> None:
                pass

        found = discover_singletons(_Inner)
        self.assertEqual(len(found), 0)

    async def test_discover_singletons_inherited(self) -> None:
        """Singletons declared on a base class are discovered on subclasses."""

        class _Base(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        class _Child(_Base):
            __test__ = False

        found = discover_singletons(_Child)
        self.assertEqual(set(found.keys()), {"ctr"})

    async def test_discover_singletons_override_in_subclass(self) -> None:
        """A subclass can override a singleton with a different class."""

        class _Base(AsyncTestCase):
            __test__ = False
            greet = singleton(Greeter)

        class _Child(_Base):
            __test__ = False
            greet = singleton(AsyncGreeter)  # type: ignore[assignment]

        found = discover_singletons(_Child)
        self.assertEqual(set(found.keys()), {"greet"})
        self.assertIs(found["greet"].cls, AsyncGreeter)

    async def test_discover_singletons_from_suite(self) -> None:
        class _A(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_a(self) -> None:
                pass

        class _B(AsyncTestCase):
            __test__ = False
            greet = singleton(Greeter)

            async def test_b(self) -> None:
                pass

        suite = AsyncTestSuite()
        suite.add_class(_A)
        suite.add_class(_B)

        found = discover_singletons_from_suite(suite.entries)
        ctr_key = singleton_key(vars(_A)["ctr"])
        greet_key = singleton_key(vars(_B)["greet"])
        self.assertIn(ctr_key, found)
        self.assertIn(greet_key, found)

    async def test_discover_singletons_from_suite_deduplicates(self) -> None:
        """Same class used by multiple test classes appears once."""

        class _A(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_a(self) -> None:
                pass

        class _B(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_b(self) -> None:
                pass

        suite = AsyncTestSuite()
        suite.add_class(_A)
        suite.add_class(_B)

        found = discover_singletons_from_suite(suite.entries)
        self.assertEqual(len(found), 1)

    async def test_discover_from_annotation(self) -> None:
        """A bare Singleton subclass annotation is discovered."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr: Counter

        found = discover_singletons(_Inner)
        self.assertEqual(set(found.keys()), {"ctr"})
        self.assertIs(found["ctr"].cls, Counter)

    async def test_discover_annotation_and_descriptor(self) -> None:
        """An explicit descriptor wins over a bare annotation."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr: Counter = singleton(Counter)

        found = discover_singletons(_Inner)
        self.assertEqual(set(found.keys()), {"ctr"})

    async def test_discover_annotation_with_non_singleton_ignored(self) -> None:
        """Non-Singleton annotations are ignored."""

        class _Inner(AsyncTestCase):
            __test__ = False
            x: int
            name: str

        found = discover_singletons(_Inner)
        self.assertEqual(len(found), 0)

    async def test_discover_annotation_inherited(self) -> None:
        """Annotated singletons are inherited from base classes."""

        class _Base(AsyncTestCase):
            __test__ = False
            ctr: Counter

        class _Child(_Base):
            __test__ = False

        found = discover_singletons(_Child)
        self.assertEqual(set(found.keys()), {"ctr"})

    async def test_discover_annotation_with_value_ignored(self) -> None:
        """An annotation with a non-descriptor value is not treated as a singleton."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr: Counter = None  # type: ignore[assignment]

        found = discover_singletons(_Inner)
        self.assertEqual(len(found), 0)

    async def test_discover_mixed_annotation_and_descriptor(self) -> None:
        """A class can mix annotation-only and explicit singletons."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr: Counter
            greet = singleton(Greeter)

        found = discover_singletons(_Inner)
        self.assertEqual(set(found.keys()), {"ctr", "greet"})
        self.assertIs(found["ctr"].cls, Counter)
        self.assertIs(found["greet"].cls, Greeter)


# ===================================================================== #
#  Singleton key tests
# ===================================================================== #


class TestSingletonKey(AsyncTestCase, concurrent=True):
    async def test_key_uses_module_and_qualname(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        desc = vars(_Inner)["ctr"]
        key = singleton_key(desc)
        self.assertIn("Counter", key)
        self.assertIn(__name__, key)

    async def test_same_class_same_key(self) -> None:
        class _A(AsyncTestCase):
            __test__ = False
            x = singleton(Counter)

        class _B(AsyncTestCase):
            __test__ = False
            y = singleton(Counter)

        key_a = singleton_key(vars(_A)["x"])
        key_b = singleton_key(vars(_B)["y"])
        self.assertEqual(key_a, key_b)

    async def test_different_class_different_key(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)
            greet = singleton(Greeter)

        key_ctr = singleton_key(vars(_Inner)["ctr"])
        key_greet = singleton_key(vars(_Inner)["greet"])
        self.assertNotEqual(key_ctr, key_greet)


# ===================================================================== #
#  SingletonManager tests
# ===================================================================== #


class TestSingletonManager(AsyncTestCase):
    async def test_get_or_create_enters_context_manager(self) -> None:
        class _Singleton(Singleton):
            def __init__(self) -> None:
                self.torn_down = False

            async def __aexit__(self, *exc: object) -> None:
                try:
                    await asyncio.Future()
                finally:
                    self.torn_down = True

        class _Inner(AsyncTestCase):
            __test__ = False
            s = singleton(_Singleton)

        desc = vars(_Inner)["s"]

        async with SingletonManager() as sm:
            value = await sm.get_or_create(desc)
            self.assertIsInstance(value, _Singleton)
            self.assertFalse(value.torn_down)

        # After exiting the context manager, teardown should have run
        self.assertTrue(value.torn_down)

    async def test_get_or_create_returns_cached(self) -> None:
        """Calling get_or_create twice with the same class returns the same value."""

        class _A(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        class _B(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        desc_a = vars(_A)["ctr"]
        desc_b = vars(_B)["ctr"]

        async with SingletonManager() as sm:
            val_a = await sm.get_or_create(desc_a)
            val_b = await sm.get_or_create(desc_b)
            self.assertIs(val_a, val_b)

    async def test_get_or_create_different_classes(self) -> None:
        """Different classes create distinct singleton instances."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)
            greet = singleton(Greeter)

        desc_ctr = vars(_Inner)["ctr"]
        desc_greet = vars(_Inner)["greet"]

        async with SingletonManager() as sm:
            val_ctr = await sm.get_or_create(desc_ctr)
            val_greet = await sm.get_or_create(desc_greet)
            self.assertIsInstance(val_ctr, Counter)
            self.assertIsInstance(val_greet, Greeter)

    async def test_inject_replaces_descriptors(self) -> None:
        """After inject(), descriptors are replaced with actual values."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_ok(self) -> None:
                pass

        suite = AsyncTestSuite()
        suite.add_class(_Inner)
        entries = suite.entries

        async with SingletonManager() as sm:
            await sm.inject(entries)

            # Descriptor should be replaced
            self.assertFalse(isinstance(vars(_Inner).get("ctr"), _SingletonDescriptor))
            self.assertIsInstance(_Inner.ctr, Counter)  # type: ignore[attr-defined]

            # Instance access works
            instance = _Inner("test_ok")
            self.assertIsInstance(instance.ctr, Counter)  # type: ignore[attr-defined]

    async def test_inject_multiple_classes_share_singleton(self) -> None:
        class _A(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_a(self) -> None:
                pass

        class _B(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_b(self) -> None:
                pass

        suite = AsyncTestSuite()
        suite.add_class(_A)
        suite.add_class(_B)

        async with SingletonManager() as sm:
            await sm.inject(suite.entries)
            self.assertIs(_A.ctr, _B.ctr)  # type: ignore[attr-defined]

    async def test_teardown_reverse_order(self) -> None:
        """Singletons are torn down in reverse creation order."""
        teardown_order: list[str] = []

        class _SingletonA(Singleton):
            async def __aexit__(self, *exc: object) -> None:
                try:
                    await asyncio.Future()
                finally:
                    teardown_order.append("a")

        class _SingletonB(Singleton):
            async def __aexit__(self, *exc: object) -> None:
                try:
                    await asyncio.Future()
                finally:
                    teardown_order.append("b")

        class _Inner(AsyncTestCase):
            __test__ = False
            a = singleton(_SingletonA)
            b = singleton(_SingletonB)

        desc_a = vars(_Inner)["a"]
        desc_b = vars(_Inner)["b"]

        async with SingletonManager() as sm:
            await sm.get_or_create(desc_a)
            await sm.get_or_create(desc_b)

        self.assertEqual(teardown_order, ["b", "a"])

    async def test_teardown_continues_on_error(self) -> None:
        """Even if one singleton's teardown fails, the rest are still torn down."""
        torn_down: list[str] = []

        class _GoodSingleton(Singleton):
            async def __aexit__(self, *exc: object) -> None:
                try:
                    await asyncio.Future()
                finally:
                    torn_down.append("good")

        class _BadSingleton(Singleton):
            async def __aexit__(self, *exc: object) -> None:
                try:
                    await asyncio.Future()
                finally:
                    torn_down.append("bad")
                    raise RuntimeError("teardown failed")

        class _Inner(AsyncTestCase):
            __test__ = False
            good = singleton(_GoodSingleton)
            bad = singleton(_BadSingleton)

        try:
            async with SingletonManager() as sm:
                await sm.get_or_create(vars(_Inner)["good"])
                await sm.get_or_create(vars(_Inner)["bad"])
        except BaseException:
            pass

        self.assertIn("good", torn_down)
        self.assertIn("bad", torn_down)

    async def test_active_keys(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)
            greet = singleton(Greeter)

        desc_ctr = vars(_Inner)["ctr"]
        desc_greet = vars(_Inner)["greet"]
        key_ctr = singleton_key(desc_ctr)
        key_greet = singleton_key(desc_greet)

        async with SingletonManager() as sm:
            self.assertEqual(sm.active_keys, frozenset())

            await sm.get_or_create(desc_ctr)
            self.assertEqual(sm.active_keys, frozenset({key_ctr}))

            await sm.get_or_create(desc_greet)
            self.assertEqual(sm.active_keys, frozenset({key_ctr, key_greet}))

    async def test_context_manager_protocol(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        desc = vars(_Inner)["ctr"]

        async with SingletonManager() as sm:
            value = await sm.get_or_create(desc)
            self.assertFalse(value.torn_down)

        self.assertTrue(value.torn_down)


# ===================================================================== #
#  Dependency resolution tests
# ===================================================================== #


class TestSingletonDependencies(AsyncTestCase):
    async def test_dependency_resolved_before_dependent(self) -> None:
        """A singleton's descriptor dependencies are resolved first."""
        creation_order: list[str] = []

        class _Dep(Singleton):
            async def __aenter__(self) -> Self:
                creation_order.append("dep")
                return self

        class _Main(Singleton):
            dep = singleton(_Dep)

            async def __aenter__(self) -> Self:
                creation_order.append("main")
                return self

        class _Inner(AsyncTestCase):
            __test__ = False
            dep = singleton(_Dep)
            main = singleton(_Main)

            async def test_ok(self) -> None:
                self.assertIsInstance(self.main.dep, _Dep)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(creation_order, ["dep", "main"])

    async def test_dependency_chain(self) -> None:
        """A -> B -> C chain resolves correctly."""
        creation_order: list[str] = []

        class _C(Singleton):
            async def __aenter__(self) -> Self:
                creation_order.append("C")
                return self

        class _B(Singleton):
            c = singleton(_C)

            async def __aenter__(self) -> Self:
                creation_order.append("B")
                return self

        class _A(Singleton):
            b = singleton(_B)

            async def __aenter__(self) -> Self:
                creation_order.append("A")
                return self

        class _Inner(AsyncTestCase):
            __test__ = False
            a = singleton(_A)
            b = singleton(_B)
            c = singleton(_C)

            async def test_chain(self) -> None:
                self.assertIs(self.a.b, self.b)
                self.assertIs(self.b.c, self.c)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(creation_order, ["C", "B", "A"])

    async def test_diamond_dependency(self) -> None:
        """Diamond: A and B both depend on C; C is created once."""

        class _C(Singleton):
            instances: ClassVar[int] = 0

            def __init__(self) -> None:
                _C.instances += 1

        class _A(Singleton):
            c = singleton(_C)

        class _B(Singleton):
            c = singleton(_C)

        _C.instances = 0

        class _Inner(AsyncTestCase):
            __test__ = False
            a = singleton(_A)
            b = singleton(_B)
            c = singleton(_C)

            async def test_shared(self) -> None:
                self.assertIs(self.a.c, self.b.c)
                self.assertIs(self.a.c, self.c)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(_C.instances, 1)

    async def test_circular_dependency_detected(self) -> None:
        """Circular dependency A -> B -> A raises RuntimeError."""

        class _B(Singleton):
            pass

        class _A(Singleton):
            b = singleton(_B)

        # Patch _B to create the cycle now that _A exists.
        _B.a = singleton(_A)  # type: ignore[attr-defined]

        class _Inner(AsyncTestCase):
            __test__ = False
            a = singleton(_A)
            b = singleton(_B)

            async def test_never(self) -> None:
                pass

        with self.assertRaises(ExceptionGroup) as ctx:
            await _run(_Inner)

        assert isinstance(ctx.exception, ExceptionGroup)
        self.assertEqual(len(ctx.exception.exceptions), 1)
        self.assertIsInstance(ctx.exception.exceptions[0], RuntimeError)
        self.assertIn("Circular", str(ctx.exception.exceptions[0]))

    async def test_circular_dependency_three_way(self) -> None:
        """Three-way cycle A -> B -> C -> A is detected."""

        class _C(Singleton):
            pass

        class _B(Singleton):
            c = singleton(_C)

        class _A(Singleton):
            b = singleton(_B)

        # Patch _C to close the cycle now that _A exists.
        _C.a = singleton(_A)  # type: ignore[attr-defined]

        class _Inner(AsyncTestCase):
            __test__ = False
            a = singleton(_A)
            b = singleton(_B)
            c = singleton(_C)

            async def test_never(self) -> None:
                pass

        with self.assertRaises(ExceptionGroup) as ctx:
            await _run(_Inner)

        assert isinstance(ctx.exception, ExceptionGroup)
        self.assertEqual(len(ctx.exception.exceptions), 1)
        self.assertIsInstance(ctx.exception.exceptions[0], RuntimeError)
        self.assertIn("Circular", str(ctx.exception.exceptions[0]))

    async def test_teardown_order_with_dependencies(self) -> None:
        """Dependents are torn down before their dependencies."""
        teardown_order: list[str] = []

        class _Dep(Singleton):
            async def __aexit__(self, *exc: object) -> None:
                try:
                    await asyncio.Future()
                finally:
                    teardown_order.append("dep")

        class _Main(Singleton):
            dep = singleton(_Dep)

            async def __aexit__(self, *exc: object) -> None:
                try:
                    await asyncio.Future()
                finally:
                    teardown_order.append("main")

        class _Inner(AsyncTestCase):
            __test__ = False
            dep = singleton(_Dep)
            main = singleton(_Main)

            async def test_ok(self) -> None:
                pass

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        # main depends on dep, so main should be torn down first
        self.assertEqual(teardown_order, ["main", "dep"])

    async def test_transitive_dependency_not_on_test_class(self) -> None:
        """A dependency only declared on a singleton (not the test class) is still created."""

        class _Transitive(Singleton):
            pass

        class _Main(Singleton):
            t = singleton(_Transitive)

        class _Inner(AsyncTestCase):
            __test__ = False
            main = singleton(_Main)

            async def test_ok(self) -> None:
                self.assertIsInstance(self.main.t, _Transitive)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)


# ===================================================================== #
#  Integration: singletons with the runner
# ===================================================================== #


class TestSingletonIntegration(AsyncTestCase):
    async def test_basic_singleton_injection(self) -> None:
        """Tests can access injected singleton instances."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_use_singleton(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                self.ctr.increment()
                self.assertEqual(self.ctr.value, 1)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)

    async def test_singleton_shared_across_tests_in_class(self) -> None:
        """All test methods in a class see the same singleton instance."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_a(self) -> None:
                self.ctr.increment()

            async def test_b(self) -> None:
                self.ctr.increment()

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)

    async def test_singleton_shared_across_classes(self) -> None:
        """Two classes using the same singleton class share the instance."""
        saw: list[object] = []

        class _A(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_a(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                saw.append(self.ctr)

        class _B(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_b(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                saw.append(self.ctr)

        result = await _run(_A, _B)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)
        # Both classes should share a single Counter instance
        self.assertIs(saw[0], saw[1])

    async def test_singleton_torn_down_after_run(self) -> None:
        """Without an external manager, singletons are torn down after the run."""
        captured: list[Counter] = []

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_ok(self) -> None:
                self.assertFalse(self.ctr.torn_down)
                captured.append(self.ctr)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].torn_down)

    async def test_multiple_singletons_on_one_class(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)
            greet = singleton(Greeter)

            async def test_both(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                self.assertIsInstance(self.greet, Greeter)
                self.assertEqual(self.greet.greet("World"), "Hello, World!")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_singleton_with_async_setup_teardown(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False
            greet = singleton(AsyncGreeter)

            async def test_use(self) -> None:
                self.assertEqual(self.greet.greet("Test"), "Async Hello, Test!")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_singleton_available_in_setup_class(self) -> None:
        """Singletons are injected before setUpClass runs."""
        setup_class_saw_singleton = False

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            @classmethod
            async def setUpClass(cls) -> None:
                nonlocal setup_class_saw_singleton
                await super().setUpClass()
                setup_class_saw_singleton = isinstance(cls.ctr, Counter)  # type: ignore[arg-type]

            async def test_ok(self) -> None:
                pass

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertTrue(setup_class_saw_singleton)

    async def test_singleton_available_in_setup(self) -> None:
        """Singletons are available in per-test setUp."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def setUp(self) -> None:
                await super().setUp()
                self.ctr.increment()

            async def test_check(self) -> None:
                self.assertGreaterEqual(self.ctr.value, 1)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_inherited_singleton(self) -> None:
        """Subclass inherits singleton from base without redeclaring it."""

        class _Base(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

        class _Child(_Base):
            __test__ = False

            async def test_inherited(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                self.ctr.increment()

        result = await _run(_Child)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)

    async def test_no_singletons_runs_normally(self) -> None:
        """Classes without singletons still work as before."""

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_plain(self) -> None:
                self.assertEqual(1 + 1, 2)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)

    async def test_concurrent_tests_with_singleton(self) -> None:
        """Singleton works with concurrent test execution."""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False
            ctr = singleton(Counter)

            async def test_a(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                await asyncio.sleep(0.01)

            async def test_b(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                await asyncio.sleep(0.01)

            async def test_c(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                await asyncio.sleep(0.01)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 3)

    async def test_singleton_creation_error_propagates(self) -> None:
        """If a singleton raises during __aenter__, it propagates."""

        class _Broken(Singleton):
            async def __aenter__(self) -> Self:
                raise RuntimeError("boot failed")

        class _Inner(AsyncTestCase):
            __test__ = False
            broken = singleton(_Broken)

            async def test_never(self) -> None:
                pass

        with self.assertRaises(ExceptionGroup) as ctx:
            await _run(_Inner)

        assert isinstance(ctx.exception, ExceptionGroup)
        self.assertEqual(len(ctx.exception.exceptions), 1)
        self.assertIsInstance(ctx.exception.exceptions[0], RuntimeError)
        self.assertIn("boot failed", str(ctx.exception.exceptions[0]))

    async def test_mixed_singleton_and_non_singleton_classes(self) -> None:
        """A suite with both singleton-using and plain classes works."""

        class _WithSingleton(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_singleton(self) -> None:
                self.assertIsInstance(self.ctr, Counter)

        class _Plain(AsyncTestCase):
            __test__ = False

            async def test_plain(self) -> None:
                self.assertEqual(2 + 2, 4)

        result = await _run(_WithSingleton, _Plain)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)

    async def test_failfast_still_tears_down_singletons(self) -> None:
        """With failfast, singletons are still cleaned up."""
        captured: list[Counter] = []

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_fail(self) -> None:
                captured.append(self.ctr)
                self.fail("intentional failure")

            async def test_ok(self) -> None:
                pass

        result = await _run(_Inner, failfast=True)
        self.assertFalse(result.was_successful)
        # The singleton should still be torn down (no external manager)
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].torn_down)

    async def test_dependent_singleton_integration(self) -> None:
        """Full integration: dependent singletons work through the runner."""

        class _Monitor(Singleton):
            def __init__(self) -> None:
                self.started = False

            async def __aenter__(self) -> Self:
                self.started = True
                return self

            async def __aexit__(self, *exc: object) -> None:
                try:
                    await asyncio.Future()
                finally:
                    self.started = False

        class _Manager(Singleton):
            monitor = singleton(_Monitor)

        class _Inner(AsyncTestCase):
            __test__ = False
            monitor = singleton(_Monitor)
            manager = singleton(_Manager)

            async def test_dependency(self) -> None:
                self.assertIs(self.manager.monitor, self.monitor)
                self.assertTrue(self.monitor.started)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_dependent_singleton_shared_across_classes(self) -> None:
        """Dependent singletons are shared across test classes."""

        class _Dep(Singleton):
            instances: ClassVar[int] = 0

            def __init__(self) -> None:
                _Dep.instances += 1

        class _Main(Singleton):
            dep = singleton(_Dep)

        _Dep.instances = 0

        class _A(AsyncTestCase):
            __test__ = False
            dep = singleton(_Dep)
            main = singleton(_Main)

            async def test_a(self) -> None:
                self.assertIsInstance(self.main.dep, _Dep)

        class _B(AsyncTestCase):
            __test__ = False
            dep = singleton(_Dep)
            main = singleton(_Main)

            async def test_b(self) -> None:
                self.assertIsInstance(self.main.dep, _Dep)

        result = await _run(_A, _B)
        self.assertTrue(result.was_successful)
        self.assertEqual(_Dep.instances, 1)

    async def test_annotation_based_injection(self) -> None:
        """A bare type annotation on a class triggers singleton injection."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr: Counter

            async def test_use(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                self.ctr.increment()
                self.assertEqual(self.ctr.value, 1)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)

    async def test_annotation_shared_across_classes(self) -> None:
        """Annotation-based singletons are shared across classes."""
        saw: list[object] = []

        class _A(AsyncTestCase):
            __test__ = False
            ctr: Counter

            async def test_a(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                saw.append(self.ctr)

        class _B(AsyncTestCase):
            __test__ = False
            ctr: Counter

            async def test_b(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                saw.append(self.ctr)

        result = await _run(_A, _B)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)
        # Both classes should have received the same instance.
        self.assertEqual(len(saw), 2)
        self.assertIs(saw[0], saw[1])

    async def test_annotation_and_descriptor_mixed(self) -> None:
        """Annotation and descriptor singletons coexist on one class."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr: Counter
            greet = singleton(Greeter)

            async def test_both(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                self.assertIsInstance(self.greet, Greeter)
                self.assertEqual(self.greet.greet("World"), "Hello, World!")

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_annotation_dependency_between_singletons(self) -> None:
        """Singletons can declare dependencies via annotations too."""

        class _Dep(Singleton):
            pass

        class _Main(Singleton):
            dep: _Dep

        class _Inner(AsyncTestCase):
            __test__ = False
            dep: _Dep
            main: _Main

            async def test_dep(self) -> None:
                self.assertIsInstance(self.main.dep, _Dep)
                self.assertIs(self.main.dep, self.dep)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)


# ===================================================================== #
#  Singleton inheritance tests
# ===================================================================== #


class TestSingletonInheritance(AsyncTestCase):
    async def test_child_singleton_adds_dependency(self) -> None:
        """A child singleton can add its own dependencies via descriptors."""
        creation_order: list[str] = []

        class _Dep(Singleton):
            async def __aenter__(self) -> Self:
                creation_order.append("dep")
                return self

        class _Base(Singleton):
            async def __aenter__(self) -> Self:
                creation_order.append("base")
                return self

        class _Child(_Base):
            dep = singleton(_Dep)

            async def __aenter__(self) -> Self:
                creation_order.append("child")
                return self

        class _Inner(AsyncTestCase):
            __test__ = False
            dep = singleton(_Dep)
            child = singleton(_Child)

            async def test_ok(self) -> None:
                self.assertIsInstance(self.child.dep, _Dep)
                self.assertIs(self.child.dep, self.dep)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(creation_order, ["dep", "child"])

    async def test_child_singleton_inherits_dependency(self) -> None:
        """A child singleton inherits dependencies declared on the parent."""
        creation_order: list[str] = []

        class _Dep(Singleton):
            async def __aenter__(self) -> Self:
                creation_order.append("dep")
                return self

        class _Base(Singleton):
            dep = singleton(_Dep)

            async def __aenter__(self) -> Self:
                creation_order.append("base")
                return self

        class _Child(_Base):
            async def __aenter__(self) -> Self:
                creation_order.append("child")
                return self

        class _Inner(AsyncTestCase):
            __test__ = False
            dep = singleton(_Dep)
            child = singleton(_Child)

            async def test_ok(self) -> None:
                self.assertIsInstance(self.child.dep, _Dep)
                self.assertIs(self.child.dep, self.dep)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(creation_order, ["dep", "child"])

    async def test_child_singleton_overrides_dependency(self) -> None:
        """A child can override a parent's dependency with a different class."""

        class _DepA(Singleton):
            pass

        class _DepB(Singleton):
            pass

        class _Base(Singleton):
            dep = singleton(_DepA)

        class _Child(_Base):
            dep = singleton(_DepB)  # type: ignore[assignment]

        class _Inner(AsyncTestCase):
            __test__ = False
            b = singleton(_DepB)
            child = singleton(_Child)

            async def test_ok(self) -> None:
                self.assertIsInstance(self.child.dep, _DepB)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_child_singleton_no_init(self) -> None:
        """A singleton hierarchy where neither parent nor child has dependencies."""

        class _Base(Singleton):
            async def __aenter__(self) -> Self:
                return self

        class _Child(_Base):
            async def __aenter__(self) -> Self:
                return self

        class _Inner(AsyncTestCase):
            __test__ = False
            child = singleton(_Child)

            async def test_ok(self) -> None:
                self.assertIsInstance(self.child, _Child)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)


# ===================================================================== #
#  Parameterised singleton tests (subclass with fixed args)
# ===================================================================== #


class TestParameterisedSingletonSubclass(AsyncTestCase):
    async def test_subclass_receives_constructor_args(self) -> None:
        """A subclass that passes fixed args to super().__init__() works."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(CounterFrom10)

            async def test_value(self) -> None:
                self.assertEqual(self.ctr.value, 10)
                self.assertEqual(self.ctr.step, 1)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_subclass_receives_keyword_args(self) -> None:
        """A subclass that passes keyword args to super().__init__() works."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(CounterBy5)

            async def test_step(self) -> None:
                self.ctr.increment()
                self.assertEqual(self.ctr.value, 5)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_different_subclasses_are_distinct(self) -> None:
        """Two subclasses of the same base create separate instances."""

        class _Inner(AsyncTestCase):
            __test__ = False
            from10 = singleton(CounterFrom10)
            by5 = singleton(CounterBy5)

            async def test_distinct(self) -> None:
                self.assertIsNot(self.from10, self.by5)
                self.assertEqual(self.from10.value, 10)
                self.assertEqual(self.by5.step, 5)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_subclass_shared_across_classes(self) -> None:
        """The same parameterised subclass is shared across test classes."""

        class _A(AsyncTestCase):
            __test__ = False
            ctr = singleton(CounterFrom10)

            async def test_a(self) -> None:
                self.assertIsInstance(self.ctr, CounterFrom10)

        class _B(AsyncTestCase):
            __test__ = False
            ctr = singleton(CounterFrom10)

            async def test_b(self) -> None:
                self.assertIsInstance(self.ctr, CounterFrom10)
                self.assertIs(self.ctr, _A.ctr)  # type: ignore[attr-defined]

        result = await _run(_A, _B)
        self.assertTrue(result.was_successful)

    async def test_subclass_and_base_are_distinct(self) -> None:
        """A parameterised subclass and its base class are separate singletons."""

        class _Inner(AsyncTestCase):
            __test__ = False
            base = singleton(ConfigurableCounter)
            from10 = singleton(CounterFrom10)

            async def test_distinct(self) -> None:
                self.assertIsNot(self.base, self.from10)
                self.assertEqual(self.base.value, 0)
                self.assertEqual(self.from10.value, 10)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_subclass_as_dependency(self) -> None:
        """A parameterised singleton can be used as a dependency of another singleton."""

        class _Consumer(Singleton):
            ctr = singleton(CounterFrom10)

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(CounterFrom10)
            consumer = singleton(_Consumer)

            async def test_dep(self) -> None:
                self.assertIs(self.consumer.ctr, self.ctr)
                self.assertEqual(self.ctr.value, 10)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_inline_subclass(self) -> None:
        """A parameterised singleton defined inline in a test works."""

        class _CounterFrom42(ConfigurableCounter, start=42, step=3):
            pass

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(_CounterFrom42)

            async def test_inline(self) -> None:
                self.assertEqual(self.ctr.value, 42)
                self.ctr.increment()
                self.assertEqual(self.ctr.value, 45)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)


# ===================================================================== #
#  Parameterised singleton tests (inline args)
# ===================================================================== #


class TestParameterisedSingletonInline(AsyncTestCase):
    async def test_kwargs_forwarded_to_init_subclass(self) -> None:
        """singleton(cls, kwarg=value) forwards kwargs to __init_subclass__."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(ConfigurableCounter, start=10)

            async def test_value(self) -> None:
                self.assertEqual(self.ctr.value, 10)
                self.assertEqual(self.ctr.step, 1)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_multiple_kwargs(self) -> None:
        """Multiple keyword args are forwarded correctly."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(ConfigurableCounter, start=0, step=5)

            async def test_step(self) -> None:
                self.ctr.increment()
                self.assertEqual(self.ctr.value, 5)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_same_kwargs_shared_across_classes(self) -> None:
        """Same class + kwargs across test classes share one instance."""

        class _A(AsyncTestCase):
            __test__ = False
            ctr = singleton(ConfigurableCounter, start=10)

            async def test_a(self) -> None:
                pass

        class _B(AsyncTestCase):
            __test__ = False
            ctr = singleton(ConfigurableCounter, start=10)

            async def test_b(self) -> None:
                self.assertIs(self.ctr, _A.ctr)  # type: ignore[attr-defined]

        result = await _run(_A, _B)
        self.assertTrue(result.was_successful)

    async def test_different_kwargs_create_distinct_instances(self) -> None:
        """Different kwargs for the same class create separate instances."""

        class _Inner(AsyncTestCase):
            __test__ = False
            a = singleton(ConfigurableCounter, start=10)
            b = singleton(ConfigurableCounter, start=20)

            async def test_distinct(self) -> None:
                self.assertIsNot(self.a, self.b)
                self.assertEqual(self.a.value, 10)
                self.assertEqual(self.b.value, 20)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_inline_and_bare_are_distinct(self) -> None:
        """singleton(cls, kwarg=v) and singleton(cls) create separate instances."""

        class _Inner(AsyncTestCase):
            __test__ = False
            bare = singleton(ConfigurableCounter)
            from10 = singleton(ConfigurableCounter, start=10)

            async def test_distinct(self) -> None:
                self.assertIsNot(self.bare, self.from10)
                self.assertEqual(self.bare.value, 0)
                self.assertEqual(self.from10.value, 10)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_inline_as_dependency(self) -> None:
        """An inline-parameterised singleton can be a dependency."""

        class _Consumer(Singleton):
            ctr = singleton(ConfigurableCounter, start=10)

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(ConfigurableCounter, start=10)
            consumer = singleton(_Consumer)

            async def test_dep(self) -> None:
                self.assertIs(self.consumer.ctr, self.ctr)
                self.assertEqual(self.ctr.value, 10)

        result = await _run(_Inner)
        self.assertTrue(result.was_successful)

    async def test_unhashable_kwarg_raises(self) -> None:
        """Unhashable keyword arguments raise TypeError."""
        with self.assertRaises(TypeError) as ctx:
            singleton(ConfigurableCounter, start=[1, 2])  # type: ignore[arg-type]
        self.assertIn("not hashable", str(ctx.exception))
        self.assertIn("'start'", str(ctx.exception))

    async def test_generated_type_has_readable_qualname(self) -> None:
        """The generated subclass has a qualname that includes the arguments."""
        desc = _SingletonDescriptor(ConfigurableCounter, start=10, step=2)
        self.assertIn("ConfigurableCounter", desc.cls.__qualname__)
        self.assertIn("start=10", desc.cls.__qualname__)
        self.assertIn("step=2", desc.cls.__qualname__)


# ===================================================================== #
#  Singleton crash propagation tests
# ===================================================================== #


class TestSingletonCrashPropagation(AsyncTestCase):
    async def test_singleton_background_crash_propagates(self) -> None:
        """A singleton whose background work crashes surfaces the error."""

        class _CrashingSingleton(Singleton):
            async def __aexit__(self, *exc: object) -> None:
                # Simulate background work that crashes after setup.
                raise RuntimeError("background crash")

        class _Inner(AsyncTestCase):
            __test__ = False
            s = singleton(_CrashingSingleton)

        desc = vars(_Inner)["s"]

        with self.assertRaises(BaseException) as ctx:
            async with SingletonManager() as sm:
                await sm.get_or_create(desc)

        # The TaskGroup wraps the error in an ExceptionGroup.
        assert isinstance(ctx.exception, ExceptionGroup)
        self.assertEqual(len(ctx.exception.exceptions), 1)
        self.assertIsInstance(ctx.exception.exceptions[0], RuntimeError)
        self.assertIn("background crash", str(ctx.exception.exceptions[0]))

    async def test_singleton_background_crash_cancels_siblings(self) -> None:
        """When one singleton crashes, other singletons are torn down."""
        torn_down = False

        class _GoodSingleton(Singleton):
            async def __aexit__(self, *exc: object) -> None:
                nonlocal torn_down
                try:
                    await asyncio.Future()
                finally:
                    torn_down = True

        class _CrashingSingleton(Singleton):
            async def __aexit__(self, *exc: object) -> None:
                raise RuntimeError("crash")

        class _Inner(AsyncTestCase):
            __test__ = False
            good = singleton(_GoodSingleton)
            crash = singleton(_CrashingSingleton)

        desc_good = vars(_Inner)["good"]
        desc_crash = vars(_Inner)["crash"]

        try:
            async with SingletonManager() as sm:
                await sm.get_or_create(desc_good)
                await sm.get_or_create(desc_crash)
        except BaseException:
            pass

        # The good singleton should have been torn down when the
        # crashing singleton caused the TaskGroup to cancel.
        self.assertTrue(torn_down)


# ===================================================================== #
#  Function singleton tests
# ===================================================================== #


async def _run_functions(
    *funcs: Callable[..., Coroutine[Any, Any, None]],
    max_concurrency: int | None = None,
    failfast: bool = False,
) -> AsyncTestResult:
    """Run one or more standalone test functions and return the result."""
    runner = AsyncTestRunner(
        max_concurrency=max_concurrency,
        verbosity=0,
        failfast=failfast,
    )
    return await runner.run_functions_async(*funcs)


class TestFunctionSingletonDiscovery(AsyncTestCase, concurrent=True):
    """Tests for discover_singletons_from_function."""

    async def test_discover_from_annotation(self) -> None:
        """A bare Singleton subclass annotation is discovered."""

        async def test_fn(ctr: Counter) -> None:
            pass

        found = discover_singletons_from_function(test_fn)
        self.assertEqual(set(found.keys()), {"ctr"})
        self.assertIs(found["ctr"].cls, Counter)

    async def test_discover_from_default(self) -> None:
        """An explicit singleton() default is discovered."""

        async def test_fn(ctr: object = singleton(Counter)) -> None:
            pass

        found = discover_singletons_from_function(test_fn)
        self.assertEqual(set(found.keys()), {"ctr"})
        self.assertIs(found["ctr"].cls, Counter)

    async def test_default_wins_over_annotation(self) -> None:
        """When both annotation and default exist, the default wins."""

        async def test_fn(
            ctr: ConfigurableCounter = singleton(ConfigurableCounter, start=10),
        ) -> None:
            pass

        found = discover_singletons_from_function(test_fn)
        self.assertEqual(set(found.keys()), {"ctr"})
        # The descriptor should be the parameterised one from the default.
        self.assertIn("start=10", found["ctr"].cls.__qualname__)

    async def test_discover_multiple_annotations(self) -> None:
        async def test_fn(ctr: Counter, greet: Greeter) -> None:
            pass

        found = discover_singletons_from_function(test_fn)
        self.assertEqual(set(found.keys()), {"ctr", "greet"})

    async def test_discover_no_singletons(self) -> None:
        async def test_fn() -> None:
            pass

        found = discover_singletons_from_function(test_fn)
        self.assertEqual(found, {})

    async def test_non_singleton_annotations_ignored(self) -> None:
        """Non-Singleton type annotations are ignored."""

        async def test_fn(x: int = 42, name: str = "hello") -> None:
            pass

        found = discover_singletons_from_function(test_fn)
        self.assertEqual(found, {})

    async def test_discover_mixed(self) -> None:
        """Annotation-based and non-singleton params coexist."""

        async def test_fn(x: int, ctr: Counter) -> None:
            pass

        found = discover_singletons_from_function(test_fn)
        self.assertEqual(set(found.keys()), {"ctr"})

    async def test_discover_from_suite_with_functions(self) -> None:
        """discover_singletons_from_suite finds singletons in functions too."""

        async def test_fn(ctr: Counter) -> None:
            pass

        result = discover_singletons_from_suite([], functions=[test_fn])
        key = singleton_key(_SingletonDescriptor(Counter))
        self.assertIn(key, result)


class TestFunctionSingletonInjection(AsyncTestCase):
    async def test_inject_function_resolves_singletons(self) -> None:
        """SingletonManager.inject_function returns resolved kwargs."""

        async def test_fn(ctr: Counter) -> None:
            pass

        async with SingletonManager() as sm:
            kwargs = await sm.inject_function(test_fn)
            self.assertIn("ctr", kwargs)
            self.assertIsInstance(kwargs["ctr"], Counter)

    async def test_inject_function_shares_with_class(self) -> None:
        """Functions and classes share the same singleton instance."""

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_ok(self) -> None:
                pass

        async def test_fn(ctr: Counter) -> None:
            pass

        suite = AsyncTestSuite()
        suite.add_class(_Inner)
        suite.add_function(test_fn)

        async with SingletonManager() as sm:
            await sm.inject(suite.entries)
            kwargs = await sm.inject_function(test_fn)
            self.assertIs(_Inner.ctr, kwargs["ctr"])


class TestFunctionSingletonIntegration(AsyncTestCase):
    async def test_basic_singleton_in_function(self) -> None:
        """A function test receives its singleton via annotation."""

        async def test_use_counter(ctr: Counter) -> None:
            ctr.increment()
            assert ctr.value >= 1

        result = await _run_functions(test_use_counter)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 1)

    async def test_multiple_singletons_in_function(self) -> None:
        async def test_both(ctr: Counter, greet: Greeter) -> None:
            assert greet.greet("World") == "Hello, World!"

        result = await _run_functions(test_both)
        self.assertTrue(result.was_successful)

    async def test_parameterised_singleton_via_subclass(self) -> None:
        """Parameterised singletons work via subclass annotation."""

        async def test_from10(ctr: CounterFrom10) -> None:
            assert ctr.value == 10

        result = await _run_functions(test_from10)
        self.assertTrue(result.was_successful)

    async def test_inline_parameterised_singleton_in_function(self) -> None:
        """Inline parameterised singletons use default values."""

        async def test_from42(
            ctr: ConfigurableCounter = singleton(ConfigurableCounter, start=42),
        ) -> None:
            assert ctr.value == 42

        result = await _run_functions(test_from42)
        self.assertTrue(result.was_successful)

    async def test_singleton_shared_between_function_and_class(self) -> None:
        """A mixed suite shares singleton instances between classes and functions."""
        saw: list[object] = []

        class _Inner(AsyncTestCase):
            __test__ = False
            ctr = singleton(Counter)

            async def test_class(self) -> None:
                self.assertIsInstance(self.ctr, Counter)
                saw.append(self.ctr)

        async def test_func(ctr: Counter) -> None:
            assert isinstance(ctr, Counter)
            saw.append(ctr)

        suite = AsyncTestSuite()
        suite.add_class(_Inner)
        suite.add_function(test_func)
        runner = AsyncTestRunner(verbosity=0)
        result = await runner.run_suite_async(suite)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)
        # Class and function should share the same instance
        self.assertEqual(len(saw), 2)
        self.assertIs(saw[0], saw[1])

    async def test_function_without_singletons_still_works(self) -> None:
        async def test_plain() -> None:
            assert 1 + 1 == 2

        result = await _run_functions(test_plain)
        self.assertTrue(result.was_successful)
