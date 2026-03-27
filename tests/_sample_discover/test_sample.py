# SPDX-License-Identifier: MIT
"""
Sample test file used by the discovery tests in test_framework.py.

This file lives in a subdirectory so that ``discover()`` can be pointed
at it without accidentally picking up the main test suite.
"""

import asyncio

from barrage.case import AsyncTestCase


class SamplePassingTests(AsyncTestCase, concurrent=True):
    async def test_add(self) -> None:
        self.assertEqual(1 + 1, 2)

    async def test_string(self) -> None:
        self.assertIn("oo", "foobar")

    async def test_shared_name(self) -> None:
        self.assertTrue(True)


class SampleSequentialTests(AsyncTestCase):
    async def test_seq_a(self) -> None:
        await asyncio.sleep(0)
        self.assertTrue(True)

    async def test_seq_b(self) -> None:
        await asyncio.sleep(0)
        self.assertIsNotNone(42)

    async def test_shared_name(self) -> None:
        self.assertTrue(True)


class _InternalHelper(AsyncTestCase):
    """
    Has ``__test__ = False`` so discovery should skip it.
    """

    __test__ = False

    async def test_should_not_be_found(self) -> None:
        raise AssertionError("must not be discovered")
