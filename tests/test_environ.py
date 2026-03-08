# SPDX-License-Identifier: MIT

"""
Tests for per-test environment variable isolation.

Run with::

    python3 -m barrage tests/test_environ.py
"""

import asyncio
import os

from barrage.case import AsyncTestCase
from barrage.result import AsyncTestResult
from barrage.runner import AsyncTestRunner


async def _run(
    *classes: type[AsyncTestCase],
) -> AsyncTestResult:
    runner = AsyncTestRunner(verbosity=0)
    return await runner.run_classes_async(*classes)


class TestEnvironIsolation(AsyncTestCase, concurrent=True):
    async def test_mutation_does_not_leak(self) -> None:
        """Setting an env var in one test does not affect another."""

        marker = "_BARRAGE_TEST_ENVIRON_LEAK"

        class _Setter(AsyncTestCase):
            __test__ = False

            async def test_set_var(self) -> None:
                os.environ[marker] = "leaked"
                self.assertEqual(os.environ[marker], "leaked")

        class _Reader(AsyncTestCase):
            __test__ = False

            async def test_var_absent(self) -> None:
                self.assertNotIn(marker, os.environ)

        # Run setter first, then reader.  If isolation works the
        # reader should not see the variable.
        result = await _run(_Setter)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)

        result = await _run(_Reader)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)

    async def test_concurrent_independence(self) -> None:
        """Concurrent tests each get their own snapshot."""

        marker = "_BARRAGE_TEST_ENVIRON_CONCURRENT"
        results: dict[str, str | None] = {}

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_a(self) -> None:
                os.environ[marker] = "a"
                await asyncio.sleep(0.05)
                results["a"] = os.environ.get(marker)

            async def test_b(self) -> None:
                os.environ[marker] = "b"
                await asyncio.sleep(0.05)
                results["b"] = os.environ.get(marker)

        result = await _run(_Inner)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)

        # Each test should see only its own value.
        self.assertEqual(results["a"], "a")
        self.assertEqual(results["b"], "b")

        # And the marker should not be in the real environ.
        self.assertNotIn(marker, os.environ)

    async def test_delete_is_isolated(self) -> None:
        """Deleting an env var in a test does not affect the real environ."""

        marker = "_BARRAGE_TEST_ENVIRON_DELETE"
        os.environ[marker] = "original"

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_delete(self) -> None:
                del os.environ[marker]
                self.assertNotIn(marker, os.environ)

        result = await _run(_Inner)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)

        # Original value should still be there.
        self.assertEqual(os.environ.get(marker), "original")
        del os.environ[marker]

    async def test_mapping_operations(self) -> None:
        """get(), in, iter, len all work with the snapshot."""

        marker = "_BARRAGE_TEST_ENVIRON_MAPPING"
        observed: dict[str, object] = {}

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_ops(self) -> None:
                os.environ[marker] = "value"

                observed["getitem"] = os.environ[marker]
                observed["get"] = os.environ.get(marker)
                observed["contains"] = marker in os.environ
                observed["iter"] = marker in list(os.environ)
                observed["len"] = len(os.environ) > 0

        result = await _run(_Inner)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)

        self.assertEqual(observed["getitem"], "value")
        self.assertEqual(observed["get"], "value")
        self.assertTrue(observed["contains"])
        self.assertTrue(observed["iter"])
        self.assertTrue(observed["len"])

        # Not leaked.
        self.assertNotIn(marker, os.environ)

    async def test_setup_class_visible_to_tests(self) -> None:
        """Env vars set in setUpClass are visible to all tests in the class."""

        marker = "_BARRAGE_TEST_ENVIRON_SETUP_CLASS"
        observed: dict[str, str | None] = {}

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                os.environ[marker] = "from_setup_class"

            async def test_a(self) -> None:
                observed["a"] = os.environ.get(marker)

            async def test_b(self) -> None:
                observed["b"] = os.environ.get(marker)

        result = await _run(_Inner)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)

        # Both tests should see the value set in setUpClass.
        self.assertEqual(observed["a"], "from_setup_class")
        self.assertEqual(observed["b"], "from_setup_class")

        # Not leaked outside the class.
        self.assertNotIn(marker, os.environ)

    async def test_setup_class_mutation_isolated_per_test(self) -> None:
        """A test overriding a setUpClass env var does not affect siblings."""

        marker = "_BARRAGE_TEST_ENVIRON_SETUP_OVERRIDE"
        observed: dict[str, str | None] = {}

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            @classmethod
            async def setUpClass(cls) -> None:
                os.environ[marker] = "original"

            async def test_override(self) -> None:
                os.environ[marker] = "overridden"
                await asyncio.sleep(0.05)
                observed["override"] = os.environ.get(marker)

            async def test_reader(self) -> None:
                await asyncio.sleep(0.05)
                observed["reader"] = os.environ.get(marker)

        result = await _run(_Inner)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)

        # Each test sees its own snapshot.
        self.assertEqual(observed["override"], "overridden")
        self.assertEqual(observed["reader"], "original")

    async def test_singleton_env_visible_to_tests(self) -> None:
        """Env vars set during singleton creation are visible to tests."""

        from barrage.singleton import Singleton, singleton

        marker = "_BARRAGE_TEST_ENVIRON_SINGLETON"

        class _EnvSetter(Singleton):
            async def __aenter__(self) -> "_EnvSetter":
                os.environ[marker] = "from_singleton"
                return self

            async def __aexit__(self, *exc: object) -> None:
                await asyncio.Future()

        class _Inner(AsyncTestCase):
            __test__ = False
            sng = singleton(_EnvSetter)

            async def test_sees_singleton_env(self) -> None:
                self.assertEqual(os.environ.get(marker), "from_singleton")

        result = await _run(_Inner)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.failures), 0)

        # Not leaked.
        self.assertNotIn(marker, os.environ)
