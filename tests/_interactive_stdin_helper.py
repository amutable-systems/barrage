# SPDX-License-Identifier: MIT
"""
Helper script for ``TestInteractiveMode.test_interactive_stdin``.

Defines a small test that reads a line from ``sys.stdin`` and asserts
it matches the expected value.  The outer test launches this script as
a subprocess in interactive mode (``-i``) with ``stdin=PIPE`` and
writes a known string.  If barrage's interactive mode correctly leaves
stdin connected, the inner test reads the string and passes.
"""

import asyncio
import sys

from barrage.case import AsyncTestCase


class InteractiveStdinTest(AsyncTestCase):
    async def test_read_from_stdin(self) -> None:
        line = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
        assert line.strip() == "hello from outer test", f"unexpected stdin: {line!r}"
