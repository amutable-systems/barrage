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
import os
import sys

from barrage.case import AsyncTestCase


class InteractiveStdinTest(AsyncTestCase):
    async def test_read_from_stdin(self) -> None:
        devnull_fd = os.open(os.devnull, os.O_RDONLY)
        try:
            is_devnull = os.path.sameopenfile(sys.stdin.fileno(), devnull_fd)
        finally:
            os.close(devnull_fd)
        if is_devnull:
            line = b""
        else:
            loop = asyncio.get_running_loop()
            reader = asyncio.StreamReader()
            await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
            line = await reader.readline()
        assert line.strip() == b"hello from outer test", f"unexpected stdin: {line!r}"
